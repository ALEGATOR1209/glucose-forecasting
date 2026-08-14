"""The blocks support both tensor layouts; they must agree numerically.

`batch_first=False` is the legacy (seq, batch, d_model) contract SugarJepaModel
passes; `batch_first=True` is what SugarJepaModel2 uses. Feeding the wrong layout
to nn.MultiheadAttention does NOT raise — it silently attends across the batch
axis — so this equivalence is what guards the conversion.
"""
from __future__ import annotations

import pytest
import torch

from sugar_jepa.sugar_jepa_model import (
    CrossAttentionSugarJepaBlock,
    MultiScaleAttentionBlock,
    SugarJepaModel2,
    SugarJepaParallelBlock,
)

SEQ, BATCH, D_MODEL, N_HEADS, FF, JEPA_SEQ = 16, 3, 8, 2, 16, 4


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()


@pytest.mark.parametrize(
    "block_cls", [CrossAttentionSugarJepaBlock, SugarJepaParallelBlock]
)
def test_cross_attention_layouts_agree(block_cls):
    torch.manual_seed(0)
    seq_first = block_cls(D_MODEL, N_HEADS, FF, dropout=0.0, batch_first=False)
    batch_first = block_cls(D_MODEL, N_HEADS, FF, dropout=0.0, batch_first=True)
    _same_weights(seq_first, batch_first)

    args_bf = [torch.randn(BATCH, SEQ, D_MODEL) for _ in range(4)]
    args_bf.append(torch.randn(BATCH, JEPA_SEQ, D_MODEL))  # jepa: shorter K/V stream
    args_sf = [t.transpose(0, 1) for t in args_bf]

    out_bf = batch_first(*args_bf)
    out_sf = seq_first(*args_sf).transpose(0, 1)

    assert out_bf.shape == (BATCH, SEQ, D_MODEL)
    torch.testing.assert_close(out_bf, out_sf)


def test_multiscale_layouts_agree():
    torch.manual_seed(0)
    seq_first = MultiScaleAttentionBlock(D_MODEL, N_HEADS, FF, dropout=0.0, batch_first=False)
    batch_first = MultiScaleAttentionBlock(D_MODEL, N_HEADS, FF, dropout=0.0, batch_first=True)
    _same_weights(seq_first, batch_first)

    x_bf = torch.randn(BATCH, SEQ, D_MODEL)
    out_bf = batch_first(x_bf)
    out_sf = seq_first(x_bf.transpose(0, 1)).transpose(0, 1)

    assert out_bf.shape == (BATCH, SEQ, D_MODEL)
    torch.testing.assert_close(out_bf, out_sf)


def test_blocks_default_to_seq_first():
    """SugarJepaModel relies on the default; changing it would break it silently."""
    for cls in (CrossAttentionSugarJepaBlock, MultiScaleAttentionBlock, SugarJepaParallelBlock):
        assert cls(D_MODEL, N_HEADS, FF).batch_first is False


def test_sugar_jepa_model2_forward_shape():
    model = SugarJepaModel2(
        n_time_steps=128,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        ff_units=FF,
        n_blocks=2,
        prediction_horizon=12,
        jepa_patch_size=8,
        jepa_embed_dim=16,
        jepa_layers=1,
        jepa_heads=2,
    ).eval()

    assert model.jepa_encoder.n_patches == 16
    assert model(torch.randn(BATCH, 128, 4)).shape == (BATCH, 12)


def test_patch_size_must_divide_input_steps():
    with pytest.raises(ValueError, match="divisible"):
        SugarJepaModel2(n_time_steps=128, jepa_patch_size=12)  # the CGM-JEPA default
