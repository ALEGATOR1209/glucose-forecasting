#!/usr/bin/env python3
"""
Self-supervised pretraining for `JepaEncoder` (Stage B, part 1).

The JEPA recipe, scaled down to a 128-step / 16-patch glucose window:

  context encoder  E_theta   = JepaEncoder                     (trains)
  target encoder   E_xi      = EMA copy of E_theta             (no grad)
  predictor        P         = narrow transformer

  1. Encode ALL patches with E_xi           -> target latents  (stop-grad)
  2. Mask M contiguous blocks of patches; encode the REMAINING (context)
     patches with E_theta
  3. P(context latents + mask tokens carrying the *target* positions)
     must predict the target latents, in LATENT space.

Loss is smooth-L1 between predicted and target latents. There is deliberately
NO reconstruction of glucose values — predicting the representation rather than
the signal is the whole point of JEPA.

Data: glucose only, from the **train split only** of the Loop CSV. Val/test rows
never touch this stage; if they did, every downstream forecasting number would be
leakage-contaminated. A slice of the *train* series is held out as an SSL
validation set so we can watch the objective without touching val/test.

Normalisation: none here. `JepaEncoder` instance-normalises each window inside
its own forward pass, and a per-window z-score is invariant to the global MinMax
scaling the forecasting dataset applies — so an encoder pretrained here on raw
mg/dL sees an identical input distribution when fine-tuned on `x[..., 0]`.

READ THE COLLAPSE METRICS. Representation collapse (the encoder emitting nearly
the same vector for every window) drives this loss toward zero and looks like a
triumph. `latent_std` and `eff_rank` are logged every epoch for exactly that
reason; a run whose std flat-lines near zero is dead, and its encoder is noise.

Output: one timestamped directory per run, like the trainers —
`runs/jepa_encoder/jepa_encoder_w128_p8_d96_l3_h6_<timestamp>/` holding
{config.json, encoder.pt, encoder_best.pt, pretrain_metrics.csv, plots/}. A
`latest.txt` in the parent names the most recent run. `encoder.pt` is a plain
state_dict that loads into SugarJepaModel2.jepa_encoder via
`train_sugar_jepa.py --jepa-init`.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, Dataset

from common.checkpoint import update_latest_symlink
from common.data.loading import limit_series, resolve_num_workers
from common.metrics_log import EpochMetricsWriter
from sugar_jepa.encoder_plots import plot_encoder_diagnostics, window_trend
from sugar_one.console_log import echo_plain
from sugar_jepa.sugar_jepa_model import JepaBlock, JepaEncoder

# Glucose-only, so we reuse SugarOne's CSV reader and imputation policy as-is.
from sugar_one.train_sugar_one import impute_and_sort, load_splits_streaming

app = typer.Typer(
    name="jepa_pretrain",
    add_completion=False,
    help="Self-supervised (JEPA) pretraining for the SugarJepa glucose encoder.",
)


# ============================================================================
#  DATA — glucose-only sliding windows
# ============================================================================

class GlucoseWindowDataset(Dataset):
    """Sliding windows of raw glucose, one series at a time.

    No scaler: JepaEncoder instance-norms each window itself. Windows never
    cross a series boundary (the frame is cut per unique_id).
    """

    def __init__(self, df: pl.DataFrame, window: int, stride: int = 1):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.window = window

        self._series: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        n_skipped = 0

        for (_uid,), grp in (
            df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True)
        ):
            g = grp["glucose"].to_numpy().astype(np.float32)
            n_windows = len(g) - window + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            si = len(self._series)
            self._series.append(g)
            for start in range(0, n_windows, stride):
                self._index.append((si, start))

        if n_skipped:
            echo_plain(f"  Note: skipped {n_skipped} series shorter than {window} steps.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        si, start = self._index[idx]
        return torch.from_numpy(self._series[si][start : start + self.window])


# ============================================================================
#  MASKING — I-JEPA multi-block, scaled down to a 16-patch sequence
# ============================================================================

def sample_block_mask(
    n_patches: int,
    n_targets: int,
    min_block: int,
    max_block: int,
    rng: random.Random,
    max_attempts: int = 100,
) -> tuple[list[int], list[int]]:
    """Sample `n_targets` disjoint contiguous target blocks; context is the rest.

    Returns (context_idx, target_idx), both sorted, disjoint, and both non-empty.
    One mask is drawn per batch (not per sample) so the tensors stay rectangular
    — this is what I-JEPA does too.
    """
    for _ in range(max_attempts):
        occupied: set[int] = set()
        placed = 0
        for _ in range(n_targets):
            size = rng.randint(min_block, max_block)
            for _ in range(20):
                start = rng.randint(0, n_patches - size)
                block = set(range(start, start + size))
                if not (block & occupied):
                    occupied |= block
                    placed += 1
                    break
        # Every requested block must land. Accepting a short draw would silently
        # train on a weaker objective than the config says.
        context = [i for i in range(n_patches) if i not in occupied]
        if placed == n_targets and context:
            return context, sorted(occupied)

    raise RuntimeError(
        f"Could not place {n_targets} blocks of {min_block}-{max_block} patches "
        f"in {n_patches} patches while leaving a non-empty context. Reduce "
        f"--n-targets or --max-block."
    )


# ============================================================================
#  PREDICTOR
# ============================================================================

def _sinusoidal_table(n_positions: int, dim: int) -> torch.Tensor:
    """(n_positions, dim) — same formulation as PositionalEncoding, but indexable
    by arbitrary patch positions, which is what the mask tokens need."""
    pe = torch.zeros(n_positions, dim)
    pos = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class JepaPredictor(nn.Module):
    """Predicts target-block latents from context latents + target positions.

    Deliberately narrow (embed_dim // 2): the predictor is a throwaway head. If
    it is as strong as the encoder it can solve the task on its own and the
    encoder learns nothing.
    """

    def __init__(
        self,
        embed_dim: int,
        n_patches: int,
        pred_dim: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        pred_dim = pred_dim or max(embed_dim // 2, n_heads)
        self.pred_dim = pred_dim

        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_table", _sinusoidal_table(n_patches, pred_dim))

        self.blocks = nn.ModuleList(
            [JepaBlock(pred_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)

    def forward(
        self,
        context: torch.Tensor,      # (B, C, embed_dim) — context latents from E_theta
        context_idx: torch.Tensor,  # (C,) patch positions of the context
        target_idx: torch.Tensor,   # (T,) patch positions to predict
    ) -> torch.Tensor:
        """Returns (B, T, embed_dim) — predicted target latents."""
        batch, n_ctx, _ = context.shape
        n_tgt = target_idx.numel()

        ctx = self.in_proj(context) + self.pos_table[context_idx].unsqueeze(0)
        # Mask tokens are identical apart from the position they carry — the
        # position is the only thing telling the predictor *what* to predict.
        tgt = self.mask_token.expand(batch, n_tgt, -1) + self.pos_table[target_idx].unsqueeze(0)

        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, n_ctx:, :])
        return self.out_proj(x)


# ============================================================================
#  EMA TARGET ENCODER
# ============================================================================

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)


def momentum_at(step: int, total_steps: int, base: float, final: float = 1.0) -> float:
    """Cosine ramp base -> final over training, as in BYOL/I-JEPA."""
    if total_steps <= 1:
        return final
    progress = min(step / (total_steps - 1), 1.0)
    return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2


# ============================================================================
#  COLLAPSE DIAGNOSTICS — the thing that must not be skipped
# ============================================================================

@torch.no_grad()
def collapse_metrics(latents: torch.Tensor) -> tuple[float, float]:
    """(latent_std, effective_rank) of the target encoder's output.

    latent_std: mean per-dimension std across samples. Trending to 0 means every
        window maps to the same vector — the encoder has collapsed and the loss
        is meaningless.
    effective_rank: participation ratio of the covariance eigenvalues,
        (sum l)^2 / sum(l^2). Bounded by embed_dim. A collapse to a handful of
        directions shows up here before latent_std fully flat-lines.
    """
    z = latents.reshape(-1, latents.size(-1)).float()
    std = z.std(dim=0).mean().item()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    denom = (ev**2).sum()
    eff_rank = (ev.sum() ** 2 / denom).item() if denom > 0 else 0.0
    return std, eff_rank


# ============================================================================
#  TRAIN / EVAL STEPS
# ============================================================================

def variance_penalty(latents: torch.Tensor, target_std: float) -> torch.Tensor:
    """VICReg-style hinge that makes representation collapse *actively costly*.

    The plain JEPA objective has a trivial solution: emit the same vector for
    every window, and the predictor's job becomes free. The EMA target is the
    only thing standing against that, and it is a weak guard — its strength
    depends on how slowly the target trails the online encoder *per step*, so
    the same `ema_base` that is safe on a small dataset can be far too fast once
    an epoch contains ten times as many steps. That is exactly how a run slides
    into collapse while its loss looks better and better.

    This adds a floor. For each latent dimension, take its std across the batch
    and penalise anything below `target_std`:

        mean_j( relu(target_std - std_j) )

    Zero — literally no gradient — while the representation is healthy, and it
    grows the moment dimensions start contracting. `target_std` is set *below*
    the healthy band (~0.65-0.9 for our encoder) so this is a safety net, not a
    force pulling the representation around during normal training.
    """
    z = latents.reshape(-1, latents.size(-1))
    std = torch.sqrt(z.var(dim=0) + 1e-8)
    return F.relu(target_std - std).mean()


def _forward_loss(
    encoder: JepaEncoder,
    target_encoder: JepaEncoder,
    predictor: JepaPredictor,
    glucose: torch.Tensor,
    ctx_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    var_weight: float = 0.0,
    var_target: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, pred_loss, var_loss, target_latents).

    target_latents is detached, for the collapse diagnostics.
    """
    batch = glucose.size(0)

    with torch.no_grad():
        full = target_encoder(glucose)                       # (B, n_patches, E)
        targets = full[:, tgt_idx, :].detach()               # (B, T, E)

    keep = ctx_idx.unsqueeze(0).expand(batch, -1)            # (B, C)
    context = encoder(glucose, keep=keep)                    # (B, C, E)
    pred = predictor(context, ctx_idx, tgt_idx)              # (B, T, E)

    pred_loss = F.smooth_l1_loss(pred, targets)

    # Penalise the ONLINE encoder's output — that is the thing with gradients,
    # and the thing whose collapse the EMA target would then inherit.
    if var_weight > 0.0:
        var_loss = variance_penalty(context, var_target)
    else:
        var_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)

    return pred_loss + var_weight * var_loss, pred_loss.detach(), var_loss.detach(), full.detach()


# ============================================================================
#  CLI
# ============================================================================

@app.command()
def main(
    csv: Path = typer.Option(..., help="Path to loop_ai_ready_joined2*.csv."),
    unique_id: str = typer.Option("sequence_id", help="sequence_id or user_id."),
    drop_interpolated: bool = typer.Option(False, help="Drop Interpolated rows."),
    max_series: int = typer.Option(0, help="Limit training series (0 = all)."),
    holdout_frac: float = typer.Option(
        0.05, help="Fraction of TRAIN series held out to watch the SSL objective."
    ),
    window_stride: int = typer.Option(
        4, help="Sliding-window stride. >1 cuts the huge overlap between adjacent windows."
    ),
    # --- encoder (must match the forecaster it will initialise) --------------
    input_steps: int = typer.Option(128, help="Window steps — must match --input-steps at fine-tune."),
    patch_size: int = typer.Option(8, help="Steps per patch; input_steps must divide by it."),
    embed_dim: int = typer.Option(96, help="Encoder width."),
    n_layers: int = typer.Option(3, help="Encoder blocks."),
    n_heads: int = typer.Option(6, help="Encoder attention heads."),
    # --- SSL objective -------------------------------------------------------
    n_targets: int = typer.Option(
        4,
        help="Target blocks masked in each sequence. Block positions are drawn once "
             "per batch and shared by every sequence in it (as in I-JEPA), which keeps "
             "the context/target tensors rectangular.",
    ),
    min_block: int = typer.Option(2, help="Min patches per target block."),
    max_block: int = typer.Option(4, help="Max patches per target block."),
    pred_dim: int = typer.Option(0, help="Predictor width (0 = embed_dim // 2)."),
    pred_layers: int = typer.Option(2, help="Predictor blocks."),
    pred_heads: int = typer.Option(4, help="Predictor attention heads."),
    ema_base: float = typer.Option(
        0.999,
        help="Initial target-encoder momentum, cosine-ramped to 1.0. This is a PER-STEP rate: "
             "a bigger dataset has far more steps per epoch, so the target chases the online "
             "encoder faster in wall-clock terms and the anti-collapse asymmetry weakens. "
             "0.996 is only safe on small data; use 0.999-0.9995 on the full CSV.",
    ),
    var_reg_weight: float = typer.Option(
        25.0,
        help="Weight of the VICReg-style variance floor that makes collapse actively costly "
             "(0 = off, leaving the EMA as the only guard). VICReg's coefficient: the hinge's "
             "gradient is divided by both batch size and embed_dim, so it needs a large weight "
             "to compete with the prediction term. Harmless when healthy — the hinge is exactly "
             "zero above --var-reg-target, so a big weight costs nothing until collapse starts.",
    ),
    var_reg_target: float = typer.Option(
        0.5,
        help="Per-dimension std the encoder must maintain. Set below the healthy band "
             "(~0.65-0.9) so this is a floor, not a force acting during normal training.",
    ),
    # --- optimisation --------------------------------------------------------
    epochs: int = typer.Option(30, help="Training epochs."),
    batch_size: int = typer.Option(256, help="Batch size."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    weight_decay: float = typer.Option(0.04, help="Weight decay."),
    warmup_epochs: int = typer.Option(2, help="Linear LR warmup epochs."),
    num_workers: int = typer.Option(0, help="DataLoader workers (-1 = auto)."),
    precision: str = typer.Option("bf16", help="fp32 | bf16."),
    device_name: str = typer.Option("cuda", "--device", help="cpu | cuda."),
    plot_every: int = typer.Option(
        1, help="Encoder diagnostic figure every N epochs -> <run-dir>/plots (0 = off)."
    ),
    seed: int = typer.Option(42, help="Random seed."),
    out_dir: Path = typer.Option(
        Path("runs/jepa_encoder"),
        help="Parent directory. Each run gets its own timestamped subdirectory inside it.",
    ),
) -> None:
    """Pretrain JepaEncoder with the JEPA objective on the CSV's TRAIN split."""
    if input_steps % patch_size != 0:
        raise typer.BadParameter(
            f"--input-steps ({input_steps}) must be divisible by --patch-size ({patch_size})."
        )
    n_patches = input_steps // patch_size
    if max_block > n_patches or min_block > max_block:
        raise typer.BadParameter(
            f"Need min_block <= max_block <= n_patches ({n_patches})."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    if device_name == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    typer.echo(f"Device: {device} | {n_patches} patches of {patch_size} steps")

    # --- data: TRAIN SPLIT ONLY. val/test never enter the SSL stage. ---------
    train_df, _val_df, _test_df = load_splits_streaming(csv, unique_id, drop_interpolated)
    typer.echo(f"Train rows: {len(train_df):,} (val/test deliberately unused)")
    if max_series > 0:
        train_df = limit_series(train_df, max_series)
    train_df = impute_and_sort(train_df)

    series = train_df["unique_id"].unique(maintain_order=True).to_list()
    rng.shuffle(series)
    n_holdout = max(1, int(len(series) * holdout_frac)) if holdout_frac > 0 else 0
    holdout_ids, fit_ids = series[:n_holdout], series[n_holdout:]

    fit_ds = GlucoseWindowDataset(
        train_df.filter(pl.col("unique_id").is_in(fit_ids)), input_steps, window_stride
    )
    hold_ds = (
        GlucoseWindowDataset(
            train_df.filter(pl.col("unique_id").is_in(holdout_ids)), input_steps, window_stride
        )
        if n_holdout
        else None
    )
    typer.echo(
        f"SSL windows: fit={len(fit_ds):,} ({len(fit_ids)} series) | "
        f"holdout={len(hold_ds) if hold_ds else 0:,} ({n_holdout} series)"
    )
    if len(fit_ds) == 0:
        raise typer.BadParameter("No SSL windows — series are shorter than --input-steps.")

    workers = resolve_num_workers(num_workers, device)
    fit_loader = DataLoader(
        fit_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    hold_loader = (
        DataLoader(hold_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        if hold_ds is not None and len(hold_ds) > 0
        else None
    )
    # drop_last silently empties the loader when windows < batch_size, and the
    # epoch loop would then "train" on nothing and report loss=0.
    if len(fit_loader) == 0:
        raise typer.BadParameter(
            f"{len(fit_ds):,} SSL windows < --batch-size ({batch_size}) with drop_last, "
            "so there are zero batches. Lower --batch-size or --window-stride, or "
            "raise --max-series."
        )

    # --- model ---------------------------------------------------------------
    enc_kwargs = dict(
        n_time_steps=input_steps, patch_size=patch_size, embed_dim=embed_dim,
        n_layers=n_layers, n_heads=n_heads,
    )
    encoder = JepaEncoder(**enc_kwargs).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = JepaPredictor(
        embed_dim=embed_dim, n_patches=n_patches, pred_dim=pred_dim or None,
        n_layers=pred_layers, n_heads=pred_heads,
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    echo_plain(f"  Encoder {n_enc:,} params | predictor {n_pred:,} params (discarded after SSL)")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = max(epochs * len(fit_loader), 1)
    warmup_steps = warmup_epochs * len(fit_loader)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    use_amp = device.type == "cuda" and precision == "bf16"

    # One directory per run, stamped with the encoder shape and the time — so a
    # second pretrain cannot silently overwrite the encoder that a fine-tuned
    # model was initialised from. Mirrors the trainers' runs/<name>_<ts>/ layout.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / (
        f"jepa_encoder_w{input_steps}_p{patch_size}_d{embed_dim}"
        f"_l{n_layers}_h{n_heads}_{ts}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    update_latest_symlink(run_dir, out_dir, log_fn=typer.echo)

    config = {
        **enc_kwargs, "norm": "instance",
        "n_targets": n_targets, "min_block": min_block, "max_block": max_block,
        "pred_dim": predictor.pred_dim, "pred_layers": pred_layers, "pred_heads": pred_heads,
        "ema_base": ema_base, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
        "window_stride": window_stride, "csv": str(csv), "split": "train-only",
        "holdout_frac": holdout_frac, "seed": seed,
        "fit_windows": len(fit_ds), "holdout_windows": len(hold_ds) if hold_ds else 0,
        "start_time": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    typer.echo(f"\n{'=' * 60}\nJEPA pretraining -> {run_dir}\n{'=' * 60}")

    metrics_csv = run_dir / "pretrain_metrics.csv"
    metrics = EpochMetricsWriter(metrics_csv)

    # A FIXED batch of windows, held constant across epochs, so successive plots
    # are comparable — resampling each epoch would make "the encoder changed"
    # indistinguishable from "the sample changed".
    plot_source = hold_ds if (hold_ds is not None and len(hold_ds) >= 3) else fit_ds
    plot_idx = np.linspace(0, len(plot_source) - 1, min(512, len(plot_source)), dtype=int)
    plot_windows = torch.stack([plot_source[int(i)] for i in plot_idx]).to(device)
    # Instance-norm strips level and amplitude, so colour by window *shape*, not
    # by mean glucose — the latter would be null by construction.
    plot_colors = window_trend(plot_windows.cpu().numpy())

    @torch.no_grad()
    def _plot_epoch(epoch: int) -> None:
        if plot_every <= 0 or (epoch % plot_every and epoch != epochs):
            return
        was_training = encoder.training
        encoder.eval()
        latents = encoder(plot_windows).float()   # (N, n_patches, embed_dim)
        encoder.train(was_training)
        # Recomputed on exactly the latents being drawn, so the caption and the
        # panels cannot disagree. (The CSV's latent_std comes from the *target*
        # encoder on the last training batch — a different encoder and sample.)
        std, rank = collapse_metrics(latents)
        plot_encoder_diagnostics(
            latents.cpu().numpy(),
            run_dir / "plots" / f"epoch_{epoch:03d}.png",
            epoch=epoch,
            color_values=plot_colors,
            subtitle=(
                f"SSL pretraining · fixed {len(plot_idx)}-window sample, online encoder · "
                f"latent_std={std:.4f} · eff_rank={rank:.1f}/{embed_dim}"
            ),
        )

    # A FIXED set of holdout masks, drawn once from a dedicated RNG. The eval used
    # to sample a fresh mask each epoch from the *training* rng, so epoch-to-epoch
    # holdout differences were dominated by which mask got drawn rather than by the
    # encoder — which made best-epoch selection a coin flip on a flat plateau.
    # Same masks every epoch → the number moves only when the model does.
    n_hold_batches = len(hold_loader) if hold_loader is not None else 0
    mask_rng = random.Random(seed + 1)
    hold_masks = [
        sample_block_mask(n_patches, n_targets, min_block, max_block, mask_rng)
        for _ in range(max(n_hold_batches, 1))
    ]

    step = 0
    best_hold = float("inf")
    best_rank = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        encoder.train()
        predictor.train()
        running, running_pred, running_var, n_batches = 0.0, 0.0, 0.0, 0
        last_std, last_rank = 0.0, 0.0

        for glucose in fit_loader:
            glucose = glucose.to(device, non_blocking=True)

            ctx, tgt = sample_block_mask(n_patches, n_targets, min_block, max_block, rng)
            ctx_idx = torch.tensor(ctx, device=device, dtype=torch.long)
            tgt_idx = torch.tensor(tgt, device=device, dtype=torch.long)

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, pred_loss, var_loss, full = _forward_loss(
                    encoder, target_encoder, predictor, glucose, ctx_idx, tgt_idx,
                    var_weight=var_reg_weight, var_target=var_reg_target,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()

            ema_update(target_encoder, encoder, momentum_at(step, total_steps, ema_base))

            running += loss.item()
            running_pred += pred_loss.item()
            running_var += var_loss.item()
            n_batches += 1
            step += 1
            if n_batches == len(fit_loader):  # diagnostics on the last batch of the epoch
                last_std, last_rank = collapse_metrics(full)

        train_loss = running / max(n_batches, 1)
        pred_term = running_pred / max(n_batches, 1)
        var_term = running_var / max(n_batches, 1)

        hold_loss = float("nan")
        if hold_loader is not None:
            encoder.eval()
            predictor.eval()
            tot, nb = 0.0, 0
            with torch.no_grad():
                for bi, glucose in enumerate(hold_loader):
                    glucose = glucose.to(device, non_blocking=True)
                    ctx, tgt = hold_masks[bi]  # same mask for this batch every epoch
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        _, pred_loss, _, _ = _forward_loss(
                            encoder, target_encoder, predictor, glucose,
                            torch.tensor(ctx, device=device, dtype=torch.long),
                            torch.tensor(tgt, device=device, dtype=torch.long),
                        )
                    # prediction term only — comparable across --var-reg-weight values
                    tot += pred_loss.item()
                    nb += 1
            hold_loss = tot / max(nb, 1)

        momentum = momentum_at(step, total_steps, ema_base)
        var_str = f" | var={var_term:.4f}" if var_reg_weight > 0 else ""
        echo_plain(
            f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.5f} | "
            f"pred={pred_term:.5f}{var_str} | "
            f"holdout={hold_loss:.5f} | latent_std={last_std:.4f} | "
            f"eff_rank={last_rank:.1f}/{embed_dim} | ema={momentum:.4f}"
        )
        metrics.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            # Split out, because a total loss that mixes the two hides which one moved.
            "pred_loss": round(pred_term, 6),
            "var_loss": round(var_term, 6),
            "holdout_loss": "" if math.isnan(hold_loss) else round(hold_loss, 6),
            # The two that decide whether the loss above means anything at all.
            "latent_std": round(last_std, 5),
            "eff_rank": round(last_rank, 3),
            "embed_dim": embed_dim,
            "ema_momentum": round(momentum, 5),
            "lr": round(lr_at(step - 1), 8),
            "epoch_seconds": round(time.time() - t_epoch, 2),
        })

        _plot_epoch(epoch)

        if last_std < 1e-3:
            echo_plain(
                "  WARNING: latent_std ~ 0 — the encoder has collapsed. This run is dead; "
                "the falling loss is an artifact, not learning."
            )
        elif var_reg_weight > 0 and var_term > 1e-3:
            # Only when the hinge is doing real work — it sits at ~0 on a healthy run,
            # and a note every epoch would train you to ignore it.
            echo_plain(
                f"  Note: the variance floor is engaged (var={var_term:.4f}) — dimensions "
                f"are being held above std={var_reg_target}. Without it this run would be "
                "drifting toward collapse."
            )

        torch.save(encoder.state_dict(), run_dir / "encoder.pt")

        # "Best" must be collapse-aware. Selecting purely on the lowest holdout
        # loss would preferentially save the MOST degenerate checkpoint the run
        # ever produced — a collapsing encoder makes masked prediction trivial and
        # scores a beautiful loss. So an epoch only qualifies if the representation
        # is healthy: dimensions above the variance floor, and effective rank that
        # has not fallen away from the best rank seen so far.
        best_rank = max(best_rank, last_rank)
        healthy = (
            last_std >= max(var_reg_target, 0.3)
            and last_rank >= 0.8 * best_rank
        )
        if hold_loader is not None and healthy and hold_loss < best_hold:
            best_hold = hold_loss
            best_epoch = epoch
            torch.save(encoder.state_dict(), run_dir / "encoder_best.pt")
            (run_dir / "best_info.json").write_text(json.dumps({
                "epoch": epoch,
                "holdout_pred_loss": hold_loss,
                "latent_std": last_std,
                "eff_rank": last_rank,
                "criterion": "lowest holdout prediction loss among epochs with a "
                             "healthy (non-collapsing) representation",
            }, indent=2))

    metrics.close()

    config.update({
        "end_time": datetime.now().isoformat(),
        "final_latent_std": last_std,
        "final_eff_rank": last_rank,
        "best_epoch": best_epoch or None,
        "best_holdout_loss": best_hold if best_hold < float("inf") else None,
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    healthy_end = last_std >= max(var_reg_target, 0.3) and last_rank >= 0.8 * best_rank
    recommended = "encoder.pt" if healthy_end else "encoder_best.pt"

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"  encoder.pt      epoch {epochs} (final) | "
               f"latent_std={last_std:.3f} eff_rank={last_rank:.1f}")
    if best_epoch:
        typer.echo(f"  encoder_best.pt epoch {best_epoch} | "
                   f"holdout={best_hold:.5f} (lowest among HEALTHY epochs)")
    if not healthy_end:
        typer.echo(
            "\n  WARNING: the final encoder looks degraded (latent_std/eff_rank fell away "
            "from this run's best). The run probably drifted toward collapse — prefer "
            "encoder_best.pt, and inspect pretrain_metrics.csv before trusting either."
        )
    typer.echo(f"\n  Use: {run_dir / recommended}")
    typer.echo(f"  Metrics: {metrics_csv}")
    typer.echo(f"{'=' * 60}")

    typer.echo(
        "\nFine-tune with:\n"
        f"  uv run python scripts/sugar_jepa/train_sugar_jepa.py --csv {csv} "
        f"--input-steps {input_steps} --jepa-patch-size {patch_size} "
        f"--jepa-embed-dim {embed_dim} --jepa-layers {n_layers} --jepa-heads {n_heads} "
        f"--jepa-init {run_dir / recommended}"
    )
    typer.echo(
        "\nA low SSL loss is NOT evidence the encoder is good — a collapsing encoder scores "
        "the best loss of all. Judge it by the fine-tuned forecast against the random-init "
        "control."
    )


if __name__ == "__main__":
    app()
