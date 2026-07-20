#!/usr/bin/env python3
"""
GluMindUni architecture module — univariate version of GluMind.

Removes all covariate (HR / steps) cross-attention machinery and optimises
the architecture for pure glucose-only forecasting:

  * CrossAttentionBlock and GluMindParallelBlock are gone.
  * Each transformer block is now a standard self-attention + FFN block
    applied to the glucose embedding sequence.
  * Multi-scale self-attention (DS=1/2/4) is preserved because it captures
    temporal patterns at different resolutions, which is still useful with
    univariate data.

This file intentionally contains model-only code so checkpoints can be loaded
without pulling the full training pipeline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

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
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class MultiScaleSelfAttentionBlock(nn.Module):
    """
    Multi-scale self-attention at 3 resolutions: DS=1, DS=2, DS=4.
    Low-resolution outputs are upsampled back and summed with high-res,
    followed by an FFN + residual + LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, ff_units: int,
                 dropout: float = 0.1):
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

        # DS=1: full-resolution self-attention
        high_out, _ = self.attn_high(x, x, x)
        high = self.ln1(x + self.dropout(high_out))

        # Transpose for pooling: (batch, d_model, seq_len)
        xt = high.permute(1, 2, 0)

        # DS=2
        low2 = self.pool2(xt).permute(2, 0, 1)  # (seq/2, batch, d_model)
        low2_out, _ = self.attn_low2(low2, low2, low2)
        up2 = F.interpolate(
            low2_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        # DS=4
        low4 = self.pool4(xt).permute(2, 0, 1)  # (seq/4, batch, d_model)
        low4_out, _ = self.attn_low4(low4, low4, low4)
        up4 = F.interpolate(
            low4_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        # Fuse scales
        fused = high + self.dropout(up2) + self.dropout(up4)
        ff = self.ffn(fused)
        return self.ln2(fused + self.dropout(ff))


class GluMindUniModel(nn.Module):
    """
    GluMindUni: Univariate Multi-Scale Attention Transformer.

    Univariate variant of GluMind — accepts glucose-only input and drops the
    cross-attention branches that fused HR and step-count covariates.

    Input:  (batch, seq_len, 1)  — glucose values
    Output: (batch, horizon)     — predicted glucose values
    """

    def __init__(
        self,
        n_time_steps: int,
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

        # Single-channel glucose embedding
        self.embed_glucose = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_time_steps)

        # Stacked multi-scale self-attention blocks
        self.blocks = nn.ModuleList(
            [
                MultiScaleSelfAttentionBlock(d_model, n_heads, ff_units, dropout)
                for _ in range(n_blocks)
            ]
        )

        # Output head
        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 1)."""
        # 1. Embed + positional encoding
        g_e = self.pos_enc(self.embed_glucose(x))  # (batch, seq, d_model)

        # 2. Transpose to (seq, batch, d_model) for attention
        g_e = g_e.permute(1, 0, 2)

        # 3. Stacked multi-scale self-attention blocks
        out = g_e
        for block in self.blocks:
            out = block(out)

        # 4. Output head: flatten → FC → prediction
        out = out.permute(1, 2, 0)  # (batch, d_model, seq)
        batch_size = out.size(0)
        out = out.reshape(batch_size, -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)  # (batch, horizon)
