#!/usr/bin/env python3
"""
SugarOne architecture module — Insulin & Carb extension of GluMind.

Covariates: Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g).

Architecture:
  Same parallel cross-attention + multi-scale self-attention philosophy as
  GluMind (Farahmand et al., arXiv:2509.18457), extended from 2 auxiliaries
  to 3.  Cross-attention now merges basal, bolus, and carb streams with a
  learnable softmax mixing weight so the model can up-weight whichever
  auxiliary is most informative (bolus & carbs are sparse event signals;
  basal is a slower background rate — the network learns this distinction).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (identical to base GluMind)."""

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


class CrossAttentionSugarOneBlock(nn.Module):
    """
    Three-stream cross-attention: glucose (Q) attends to basal, bolus, and
    carbs (each as K/V) via separate attention heads.

    Outputs are merged with a learnable softmax weight instead of a fixed
    average, allowing the model to learn that bolus/carbs (sparse events)
    and basal (continuous background) contribute differently.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_basal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_bolus = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_carbs = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Learnable 3-way mixing: after softmax these become non-negative and sum to 1.
        self.mix_logits = nn.Parameter(torch.zeros(3))

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
        glucose: torch.Tensor,   # (seq, batch, d_model)
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
    ) -> torch.Tensor:
        out_basal, _ = self.attn_basal(glucose, basal, basal)
        out_bolus, _ = self.attn_bolus(glucose, bolus, bolus)
        out_carbs, _ = self.attn_carbs(glucose, carbs, carbs)

        res_basal = self.ln1(glucose + self.dropout(out_basal))
        res_bolus = self.ln1(glucose + self.dropout(out_bolus))
        res_carbs = self.ln1(glucose + self.dropout(out_carbs))

        w = F.softmax(self.mix_logits, dim=0)
        merged = w[0] * res_basal + w[1] * res_bolus + w[2] * res_carbs

        ff = self.ffn(merged)
        return self.ln2(merged + self.dropout(ff))


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale self-attention at 3 resolutions: DS=1, DS=2, DS=4.
    Low-resolution outputs are upsampled back and summed (identical to base GluMind).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low4 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, batch, d_model)."""
        seq_len = x.size(0)

        high_out, _ = self.attn_high(x, x, x)
        high = self.ln1(x + self.dropout(high_out))

        xt = high.permute(1, 2, 0)  # (batch, d_model, seq)

        low2 = self.pool2(xt).permute(2, 0, 1)
        low2_out, _ = self.attn_low2(low2, low2, low2)
        up2 = F.interpolate(
            low2_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        low4 = self.pool4(xt).permute(2, 0, 1)
        low4_out, _ = self.attn_low4(low4, low4, low4)
        up4 = F.interpolate(
            low4_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        fused = high + self.dropout(up2) + self.dropout(up4)
        ff = self.ffn(fused)
        return self.ln2(fused + self.dropout(ff))


class SugarOneParallelBlock(nn.Module):
    """
    One SugarOne block: cross-attention (3-aux) and multi-scale run IN PARALLEL,
    outputs summed — same philosophy as base GluMind.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_units: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = CrossAttentionSugarOneBlock(d_model, n_heads, ff_units, dropout)
        self.multiscale = MultiScaleAttentionBlock(d_model, n_heads, ff_units, dropout)
        self.ln_fuse = nn.LayerNorm(d_model)

    def forward(
        self,
        glucose: torch.Tensor,
        basal: torch.Tensor,
        bolus: torch.Tensor,
        carbs: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs: (seq_len, batch, d_model). Returns same shape."""
        cross_out = self.cross_attn(glucose, basal, bolus, carbs)
        ms_out = self.multiscale(glucose)
        return self.ln_fuse(cross_out + ms_out)


class SugarOneModel(nn.Module):
    """
    SugarOne: Multimodal Parallel-Attention Transformer with Insulin & Carb covariates.

    Input:  (batch, seq_len, 4)  — [glucose, basal_rate, bolus_insulin, carbohydrates]
    Output: (batch, horizon)     — predicted glucose values

    Differences vs base GluMind:
      - 4 input channels instead of 3 (basal/bolus/carbs replaces HR/steps).
      - CrossAttentionSugarOneBlock uses 3 auxiliaries with learnable mixing weights
        so the model can autonomously prioritise the pharmacologically more
        relevant channels at each training step.
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

        self.blocks = nn.ModuleList(
            [
                SugarOneParallelBlock(d_model, n_heads, ff_units, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 4) — glucose, basal, bolus, carbs."""
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

        out = g_e
        for block in self.blocks:
            out = block(out, b_e, bo_e, c_e)

        out = out.permute(1, 2, 0)          # (batch, d_model, seq)
        out = out.reshape(out.size(0), -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)             # (batch, horizon)
