#!/usr/bin/env python3
"""
SugarJepa architecture module — SugarOne + a frozen/fine-tunable CGM-JEPA
embedding as a 4th cross-attention auxiliary stream.

Covariates: Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g), plus a
288-step (24h) glucose-only lookback encoded by a pretrained CGM-JEPA encoder
(see scripts/sugar_jepa/vendor/cgm_jepa/, vendored from
https://github.com/cruiseresearchgroup/CGM-JEPA, MIT license).

Architecture:
  Identical to SugarOne (scripts/sugar_one/sugar_one_model.py) — parallel
  cross-attention + multi-scale self-attention — except the cross-attention
  block now mixes FOUR auxiliaries (basal, bolus, carbs, jepa) instead of
  three, still via a learnable softmax mixing weight. The JEPA auxiliary is
  a sequence of 24 patch embeddings (dim 96) from a longer, glucose-only 24h
  lookback window, projected into d_model — independent of and longer than
  the model's own `input_steps` window, which is unchanged from SugarOne.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.common.network import apply_windows_tls_workarounds
from scripts.sugar_jepa.vendor.cgm_jepa.encoder import Encoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (identical to base GluMind/SugarOne)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class JepaEncoderWrapper(nn.Module):
    """
    Wraps the vendored, pretrained CGM-JEPA `Encoder` to turn a raw
    (batch, jepa_window) glucose-only lookback into a (seq, batch, d_model)
    cross-attention K/V stream.

    Frozen by default (encoder runs under torch.no_grad()); pass
    freeze=False to fine-tune the encoder's own weights alongside the rest
    of the model (train_sugar_jepa.py then puts its params in a separate,
    smaller-LR optimizer group).
    """

    def __init__(
        self,
        d_model: int,
        weights_dir: str,
        patch_size: int = 12,
        freeze: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.freeze = freeze
        # Harmless when weights_dir is a local path (the default); makes
        # weights_dir="CRUISEResearchGroup/CGM-JEPA"-style Hub ids work too
        # on Windows machines hitting the TLS issues in scripts/common/network.py.
        apply_windows_tls_workarounds()
        self.encoder = Encoder.from_pretrained(weights_dir)
        self.embed_dim = self.encoder.embed_dim
        self.proj = nn.Linear(self.embed_dim, d_model)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def forward(self, glucose_window: torch.Tensor) -> torch.Tensor:
        """glucose_window: (batch, jepa_window), z-score normalized.

        Returns (seq=num_patches, batch, d_model) for nn.MultiheadAttention.
        """
        batch, total_steps = glucose_window.shape
        num_patches = total_steps // self.patch_size
        patches = glucose_window[:, : num_patches * self.patch_size].view(
            batch, num_patches, self.patch_size
        )
        if self.freeze:
            self.encoder.eval()
            with torch.no_grad():
                hidden, _ = self.encoder(patches, x_mark=None, mask=None)
        else:
            hidden, _ = self.encoder(patches, x_mark=None, mask=None)
        # hidden: (batch, num_patches, embed_dim) -> (num_patches, batch, d_model)
        return self.proj(hidden).permute(1, 0, 2)

class JepaBlock(nn.Module):
    """Pre-norm transformer block — CGM-JEPA's utils/modules.py:Block, reimplemented."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class JepaEncoder(nn.Module):
    """Our own JEPA encoder — replaces JepaEncoderWrapper (no pretrained CGM-JEPA).

    glucose (batch, n_time_steps)  ->  patch embeddings (batch, n_patches, embed_dim)

    Differences from the vendored CGM-JEPA Encoder it is modelled on:
      - one Conv1d does patchify + embed in a single op (upstream's ValueEmbedding
        convolves *within* each patch and then flattens through a Linear — a shape
        that only exists to match their checkpoint);
      - no time-feature embedding and no mask branch (upstream never runs either
        at inference: x_mark=None and jepa=False);
      - no `proj` MLP head — that is the SSL projection, dead weight for forecasting.

    Normalisation is a per-window instance z-score, so the encoder is invariant to
    the global MinMax scaling applied by SugarOneWindowDataset. That is what lets
    the same weights be pretrained on raw mg/dL and then fine-tuned on MinMax-scaled
    x[..., 0] without a distribution shift.
    """

    def __init__(
        self,
        n_time_steps: int,
        patch_size: int = 8,
        embed_dim: int = 96,
        n_layers: int = 3,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm: str = "instance",
    ):
        super().__init__()

        if n_time_steps % patch_size != 0:
            raise ValueError(
                f"n_time_steps ({n_time_steps}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        if norm not in ("instance", "none"):
            raise ValueError(f"norm must be 'instance' or 'none', got {norm!r}")

        self.n_patches = n_time_steps // patch_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm_mode = norm

        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=self.n_patches)
        self.blocks = nn.ModuleList(
            [JepaBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, glucose: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """glucose: (batch, n_time_steps). keep: (batch, k) patch indices, or None.

        `keep` selects the context patches and is only used by self-supervised
        pretraining; the forecaster always passes None (encode every patch).
        """
        if self.norm_mode == "instance":
            mean = glucose.mean(dim=1, keepdim=True)
            std = glucose.std(dim=1, keepdim=True).clamp_min(1e-6)
            glucose = (glucose - mean) / std

        x = self.patch_embed(glucose.unsqueeze(1)).transpose(1, 2)  # (B, n_patches, embed_dim)
        x = self.pos_enc(x)                                          # + sinusoidal positions

        if keep is not None:
            x = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)                                          # (B, n_patches, embed_dim)


class CrossAttentionSugarJepaBlock(nn.Module):
    """
    Four-stream cross-attention: glucose (Q) attends to basal, bolus, carbs,
    and the CGM-JEPA embedding (each as K/V) via separate attention heads.

    Outputs are merged with a learnable softmax weight (extended from
    SugarOne's 3-way mix to 4-way) so the model can learn how much weight
    to give the foundation-model glucose embedding relative to the
    pharmacological covariates.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        # Attention always runs batch-first internally; `batch_first=False` only
        # means the block's *external* contract is (seq, batch, d_model), which
        # is what SugarJepaModel passes. Parameters are identical either way.
        self.batch_first = batch_first
        self.attn_basal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_bolus = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_carbs = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_jepa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Learnable 4-way mixing: after softmax these become non-negative and sum to 1.
        self.mix_logits = nn.Parameter(torch.zeros(4))

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    def forward(
        self,
        glucose: torch.Tensor,   # (batch, seq, d_model) if batch_first else (seq, batch, d_model)
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
        jepa: torch.Tensor,      # same layout; jepa_seq may differ from seq
    ) -> torch.Tensor:
        if not self.batch_first:
            glucose, basal, bolus, carbs, jepa = (
                t.transpose(0, 1) for t in (glucose, basal, bolus, carbs, jepa)
            )

        out_basal, _ = self.attn_basal(glucose, basal, basal)
        out_bolus, _ = self.attn_bolus(glucose, bolus, bolus)
        out_carbs, _ = self.attn_carbs(glucose, carbs, carbs)
        out_jepa, _ = self.attn_jepa(glucose, jepa, jepa)

        res_basal = self.ln1(glucose + self.dropout(out_basal))
        res_bolus = self.ln1(glucose + self.dropout(out_bolus))
        res_carbs = self.ln1(glucose + self.dropout(out_carbs))
        res_jepa = self.ln1(glucose + self.dropout(out_jepa))

        w = F.softmax(self.mix_logits, dim=0)
        merged = w[0] * res_basal + w[1] * res_bolus + w[2] * res_carbs + w[3] * res_jepa

        ff = self.ffn(merged)
        out = self.ln2(merged + self.dropout(ff))
        return out if self.batch_first else out.transpose(0, 1)


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale self-attention at 3 resolutions: DS=1, DS=2, DS=4.
    Low-resolution outputs are upsampled back and summed (identical to base GluMind/SugarOne).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_low2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_low4 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    @staticmethod
    def _downscale_attend_upscale(
        xt: torch.Tensor,                 # (batch, d_model, seq) — channels-first
        pool: nn.AvgPool1d,
        attn: nn.MultiheadAttention,
        seq_len: int,
    ) -> torch.Tensor:
        """Pool to a coarser resolution, self-attend there, upsample back.

        The transposes are the AvgPool1d/interpolate <-> attention boundary:
        the conv-style ops are channels-first, attention is (batch, seq, d_model).
        """
        low = pool(xt).transpose(1, 2)                   # (batch, seq//k, d_model)
        out, _ = attn(low, low, low)
        return F.interpolate(
            out.transpose(1, 2), size=seq_len, mode="nearest"
        ).transpose(1, 2)                                # (batch, seq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) if batch_first else (seq, batch, d_model)."""
        if not self.batch_first:
            x = x.transpose(0, 1)
        seq_len = x.size(1)

        high_out, _ = self.attn_high(x, x, x)
        high = self.ln1(x + self.dropout(high_out))

        xt = high.transpose(1, 2)
        up2 = self._downscale_attend_upscale(xt, self.pool2, self.attn_low2, seq_len)
        up4 = self._downscale_attend_upscale(xt, self.pool4, self.attn_low4, seq_len)

        fused = high + self.dropout(up2) + self.dropout(up4)
        ff = self.ffn(fused)
        out = self.ln2(fused + self.dropout(ff))
        return out if self.batch_first else out.transpose(0, 1)


class SugarJepaParallelBlock(nn.Module):
    """
    One SugarJepa block: cross-attention (4-aux, incl. JEPA) and multi-scale
    run IN PARALLEL, outputs summed — same philosophy as SugarOne/GluMind.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.cross_attn = CrossAttentionSugarJepaBlock(
            d_model, n_heads, ff_units, dropout, batch_first=batch_first
        )
        self.multiscale = MultiScaleAttentionBlock(
            d_model, n_heads, ff_units, dropout, batch_first=batch_first
        )
        self.ln_fuse = nn.LayerNorm(d_model)

    def forward(
        self,
        glucose: torch.Tensor,
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
        jepa: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs share the block's layout — (batch, seq, d_model) when
        batch_first, else (seq, batch, d_model). `jepa` may have a different
        seq length. Returns glucose's shape."""
        cross_out = self.cross_attn(glucose, basal, bolus, carbs, jepa)
        ms_out = self.multiscale(glucose)
        return self.ln_fuse(cross_out + ms_out)


class SugarJepaModel(nn.Module):
    """
    SugarJepa: SugarOne's Multimodal Parallel-Attention Transformer, plus a
    pretrained CGM-JEPA glucose embedding as a 4th cross-attention auxiliary.

    Inputs:
      x:             (batch, n_time_steps, 4)  — [glucose, basal_rate, bolus_insulin, carbohydrates]
      glucose_jepa:  (batch, jepa_window)      — raw glucose, z-score normalized, jepa_window steps
                                                  (default 288 = 24h @ 5min), independent of n_time_steps.
    Output: (batch, horizon) — predicted glucose values

    Differences vs SugarOne:
      - CrossAttentionSugarJepaBlock uses 4 auxiliaries (basal/bolus/carbs/jepa) with
        a learnable 4-way mixing weight instead of SugarOne's 3-way mix.
      - JepaEncoderWrapper runs a pretrained, frozen-by-default CGM-JEPA encoder over
        a separate, longer glucose-only lookback window each forward pass.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_features: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        ff_units: int = 128,
        n_blocks: int = 3,
        prediction_horizon: int = 12,
        dropout: float = 0.1,
        jepa_weights_dir: str = "scripts/sugar_jepa/pretrained/cgm_jepa",
        jepa_patch_size: int = 12,
        jepa_freeze: bool = True,
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.d_model = d_model
        self.n_features = n_features

        self.embed_glucose = nn.Linear(1, d_model)
        self.embed_basal = nn.Linear(1, d_model)
        self.embed_bolus = nn.Linear(1, d_model)
        self.embed_carbs = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=n_time_steps)

        self.jepa_encoder = JepaEncoderWrapper(
            d_model=d_model,
            weights_dir=jepa_weights_dir,
            patch_size=jepa_patch_size,
            freeze=jepa_freeze,
        )

        self.blocks = nn.ModuleList(
            [
                SugarJepaParallelBlock(d_model, n_heads, ff_units, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, glucose_jepa: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, 4) — glucose, basal, bolus, carbs. glucose_jepa: (batch, jepa_window)."""
        g = x[..., 0:1]  # (batch, seq, 1)
        b = x[..., 1:2]  # basal rate
        bo = x[..., 2:3]  # bolus insulin
        c = x[..., 3:4]  # carbohydrates

        g_e = self.pos_enc(self.embed_glucose(g))    # (batch, seq, d_model)
        b_e = self.pos_enc(self.embed_basal(b))
        bo_e = self.pos_enc(self.embed_bolus(bo))
        c_e = self.pos_enc(self.embed_carbs(c))

        # (seq, batch, d_model) for nn.MultiheadAttention
        g_e = g_e.permute(1, 0, 2)
        b_e = b_e.permute(1, 0, 2)
        bo_e = bo_e.permute(1, 0, 2)
        c_e = c_e.permute(1, 0, 2)

        jepa_e = self.jepa_encoder(glucose_jepa)  # (jepa_seq, batch, d_model)

        out = g_e
        for block in self.blocks:
            out = block(out, b_e, bo_e, c_e, jepa_e)

        out = out.permute(1, 2, 0)          # (batch, d_model, seq)
        out = out.reshape(out.size(0), -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)             # (batch, horizon)

class SugarJepaModel2(nn.Module):
    """
    SugarJepa: SugarOne's Multimodal Parallel-Attention Transformer, plus
    JEPA glucose encoder as a 4th cross-attention auxiliary.

    Input:  x (batch, lookback, 4) — [glucose, basal, bolus, carbs]
    Output: (batch, prediction_horizon)

    Two windows, one tensor
    ----------------------
    The JEPA branch may read a LONGER glucose-only lookback than the backbone
    (`jepa_window`, default = `n_time_steps`). Both views end at the same instant
    — "now", the step before the first forecast — so rather than carrying a second
    tensor through the dataset and the training loop, the caller passes one window
    of `lookback = max(n_time_steps, jepa_window)` steps and this module takes the
    trailing slice each branch needs:

        x            |<-------------- lookback = 288 -------------->| now
        JEPA branch  |<-------------- jepa_window = 288 ----------->|
        backbone                     |<-- n_time_steps = 128 ------>|

    That keeps the dataset contract SugarOne's plain `(x, y)` — no second tensor,
    no second scaler, no bespoke training loop — at the cost of carrying the three
    covariate channels over the extra steps, where only glucose is read.

    Defaulting `jepa_window` to `n_time_steps` makes the single-window behaviour
    (and every checkpoint trained under it) exactly what it was before.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_features: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        ff_units: int = 128,
        n_blocks: int = 3,
        prediction_horizon: int = 12,
        dropout: float = 0.1,
        jepa_window: int | None = None,
        jepa_patch_size: int = 8,
        jepa_heads: int = 6,
        jepa_layers: int = 3,
        jepa_embed_dim: int = 96,
        jepa_mlp_ratio: float = 4.0,
        jepa_dropout: float = 0.0,
        jepa_norm: str = "instance",
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.jepa_window = n_time_steps if jepa_window is None else jepa_window
        # What __getitem__ must hand us: enough history for whichever view is longer.
        self.lookback = max(n_time_steps, self.jepa_window)
        self.d_model = d_model
        self.n_features = n_features

        self.embed_glucose = nn.Linear(1, d_model)
        self.embed_basal = nn.Linear(1, d_model)
        self.embed_bolus = nn.Linear(1, d_model)
        self.embed_carbs = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=n_time_steps)

        self.jepa_encoder = JepaEncoder(
            n_time_steps=self.jepa_window,
            patch_size=jepa_patch_size,
            embed_dim=jepa_embed_dim,
            n_layers=jepa_layers,
            n_heads=jepa_heads,
            mlp_ratio=jepa_mlp_ratio,
            dropout=jepa_dropout,
            norm=jepa_norm,
        )

        # Patch embeddings -> K/V stream at the backbone's width.
        self.jepa_proj = nn.Linear(jepa_embed_dim, d_model)

        # Batch-first throughout, unlike SugarJepaModel — no permutes anywhere.
        self.blocks = nn.ModuleList(
            [
                SugarJepaParallelBlock(d_model, n_heads, ff_units, dropout, batch_first=True)
                for _ in range(n_blocks)
            ]
        )

        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, lookback, 4) — glucose, basal, bolus, carbs.

        `lookback` is max(n_time_steps, jepa_window); both views are trailing
        slices of it, so they end at the same instant.
        """
        if x.size(1) != self.lookback:
            raise ValueError(
                f"expected {self.lookback} steps (max of n_time_steps="
                f"{self.n_time_steps} and jepa_window={self.jepa_window}), "
                f"got {x.size(1)} — build the dataset with input_steps=lookback."
            )

        ctx = x[:, -self.n_time_steps :, :]  # the backbone's own, shorter window
        g = ctx[..., 0:1]  # (batch, n_time_steps, 1)
        b = ctx[..., 1:2]  # basal rate
        bo = ctx[..., 2:3]  # bolus insulin
        c = ctx[..., 3:4]  # carbohydrates

        g_e = self.pos_enc(self.embed_glucose(g))    # (batch, n_time_steps, d_model)
        b_e = self.pos_enc(self.embed_basal(b))
        bo_e = self.pos_enc(self.embed_bolus(bo))
        c_e = self.pos_enc(self.embed_carbs(c))

        # JEPA reads its own (possibly longer) trailing window, glucose channel
        # only, and yields one K/V position per patch — a different sequence
        # length from the query, which cross-attention has always allowed.
        jepa_e = self.jepa_encoder(x[:, -self.jepa_window :, 0])  # (batch, n_patches, embed_dim)
        jepa_e = self.jepa_proj(jepa_e)                           # (batch, n_patches, d_model)

        out = g_e
        for block in self.blocks:
            out = block(out, b_e, bo_e, c_e, jepa_e)

        out = out.transpose(1, 2)           # (batch, d_model, seq)
        out = out.reshape(out.size(0), -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)             # (batch, horizon)
