"""Smoke tests for the SugarJepa proof-of-concept (scripts/sugar_jepa/).

Not full coverage (time-boxed POC per CLAUDE.md) — just enough to catch
shape regressions in the model, the sliding-window dataset, and the
checkpoint round-trip. Uses the vendored, locally-cached pretrained CGM-JEPA
encoder (scripts/sugar_jepa/pretrained/cgm_jepa/) — no network access needed.
"""
from __future__ import annotations

import torch

import pytest

from scripts.sugar_jepa.sugar_jepa_model import (
    JepaEncoder,
    SugarJepaModel,
    SugarJepaModel2,
)
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
)

JEPA_WEIGHTS_DIR = "scripts/sugar_jepa/pretrained/cgm_jepa"
JEPA_PATCH_SIZE = 12

BATCH = 3
JEPA_WINDOW = 24  # 2 patches — smallest sensible multiple of JEPA_PATCH_SIZE for a fast test


def _tiny_model(freeze: bool = True) -> SugarJepaModel:
    return SugarJepaModel(
        n_time_steps=TINY_INPUT_STEPS,
        n_features=4,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
        jepa_weights_dir=JEPA_WEIGHTS_DIR,
        jepa_patch_size=JEPA_PATCH_SIZE,
        jepa_freeze=freeze,
    )


# ---------------------------------------------------------------------------
# Model forward / shapes
# ---------------------------------------------------------------------------


def test_sugar_jepa_forward_shape_and_dtype() -> None:
    model = _tiny_model()
    x = torch.randn(BATCH, TINY_INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    out = model(x, jepa)
    assert out.shape == (BATCH, TINY_HORIZON)
    assert out.dtype == torch.float32


def test_sugar_jepa_state_dict_key_patterns_stable() -> None:
    model = _tiny_model()
    keys = set(model.state_dict().keys())
    expected_prefixes = [
        "embed_glucose.",
        "embed_basal.",
        "embed_bolus.",
        "embed_carbs.",
        "jepa_encoder.encoder.",
        "jepa_encoder.proj.",
        "blocks.0.cross_attn.mix_logits",
        "blocks.0.cross_attn.attn_jepa.",
        "blocks.0.multiscale.",
        "flatten_fc.",
        "out_fc.",
    ]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"
    # 4-way mix (basal, bolus, carbs, jepa) — the one deliberate architectural
    # difference from SugarOneModel's 3-way mix_logits.
    assert model.blocks[0].cross_attn.mix_logits.shape == (4,)


def test_sugar_jepa_frozen_encoder_has_no_gradient() -> None:
    model = _tiny_model(freeze=True)
    x = torch.randn(BATCH, TINY_INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    model(x, jepa).sum().backward()
    for p in model.jepa_encoder.encoder.parameters():
        assert not p.requires_grad
        assert p.grad is None
    # Everything else (incl. the new jepa_encoder.proj) should have gradients.
    assert model.jepa_encoder.proj.weight.grad is not None
    assert model.embed_glucose.weight.grad is not None


def test_sugar_jepa_finetune_encoder_has_gradient() -> None:
    model = _tiny_model(freeze=False)
    x = torch.randn(BATCH, TINY_INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    model(x, jepa).sum().backward()
    assert any(p.grad is not None for p in model.jepa_encoder.encoder.parameters())


# ---------------------------------------------------------------------------
# Two windows in one tensor (SugarJepaModel2)
#
# These replace the old SugarJepaWindowDataset tests: the second window is no
# longer a second tensor built by a bespoke dataset, it is a trailing slice the
# model takes for itself, so that slicing is what needs guarding.
# ---------------------------------------------------------------------------

JEPA_WINDOW_LONG = 24  # 3 patches at JEPA2_PATCH_SIZE, and > TINY_INPUT_STEPS
JEPA2_PATCH_SIZE = 8


def _tiny_model2(jepa_window: int | None = None) -> SugarJepaModel2:
    return SugarJepaModel2(
        n_time_steps=TINY_INPUT_STEPS,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
        jepa_window=jepa_window,
        jepa_patch_size=JEPA2_PATCH_SIZE,
        jepa_embed_dim=TINY_D_MODEL,
        jepa_layers=1,
        jepa_heads=TINY_N_HEADS,
    )


def test_lookback_defaults_to_the_backbone_window() -> None:
    """Back-compat: no jepa_window means one window, exactly as before."""
    model = _tiny_model2()
    assert model.jepa_window == TINY_INPUT_STEPS
    assert model.lookback == TINY_INPUT_STEPS
    assert model(torch.randn(BATCH, TINY_INPUT_STEPS, 4)).shape == (BATCH, TINY_HORIZON)


def test_longer_jepa_window_sets_the_lookback_and_forward_accepts_it() -> None:
    model = _tiny_model2(JEPA_WINDOW_LONG)
    assert model.lookback == JEPA_WINDOW_LONG
    out = model(torch.randn(BATCH, JEPA_WINDOW_LONG, 4))
    assert out.shape == (BATCH, TINY_HORIZON)


def test_forward_rejects_a_window_that_is_not_the_lookback() -> None:
    """A silently-accepted short window would mean the JEPA branch reads
    whatever happens to be there — better to fail at the shape."""
    model = _tiny_model2(JEPA_WINDOW_LONG)
    with pytest.raises(ValueError, match="expected 24 steps"):
        model(torch.randn(BATCH, TINY_INPUT_STEPS, 4))


def test_backbone_ignores_covariates_outside_its_own_window() -> None:
    """The two views are trailing slices ending at the same instant: the
    backbone must see only the last TINY_INPUT_STEPS, while the JEPA branch
    reads glucose across the whole lookback.

    Perturbing basal/bolus/carbs in the leading region touches nothing any
    branch reads, so the prediction must not move. Perturbing GLUCOSE there
    must move it — that is the extra history the long window exists for.
    """
    torch.manual_seed(0)
    model = _tiny_model2(JEPA_WINDOW_LONG).eval()
    x = torch.randn(BATCH, JEPA_WINDOW_LONG, 4)
    lead = JEPA_WINDOW_LONG - TINY_INPUT_STEPS  # steps only the JEPA branch sees

    with torch.no_grad():
        base = model(x)

        covariates_only = x.clone()
        covariates_only[:, :lead, 1:] += 5.0
        torch.testing.assert_close(model(covariates_only), base)

        with_glucose = x.clone()
        with_glucose[:, :lead, 0] += 5.0
        assert not torch.allclose(model(with_glucose), base), (
            "the JEPA branch is not reading the extra history"
        )


def test_jepa_encoder_matches_a_checkpoint_pretrained_at_the_same_window() -> None:
    """--jepa-init loads with strict=True, so the model's encoder must be
    parameter-for-parameter what jepa_pretrain.py produces at that window."""
    model = _tiny_model2(JEPA_WINDOW_LONG)
    pretrained = JepaEncoder(
        n_time_steps=JEPA_WINDOW_LONG,
        patch_size=JEPA2_PATCH_SIZE,
        embed_dim=TINY_D_MODEL,
        n_layers=1,
        n_heads=TINY_N_HEADS,
    )
    model.jepa_encoder.load_state_dict(pretrained.state_dict(), strict=True)

    # And an encoder pretrained at the WRONG window must not load silently.
    mismatched = JepaEncoder(
        n_time_steps=JEPA_WINDOW_LONG * 2,
        patch_size=JEPA2_PATCH_SIZE,
        embed_dim=TINY_D_MODEL,
        n_layers=1,
        n_heads=TINY_N_HEADS,
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.jepa_encoder.load_state_dict(mismatched.state_dict(), strict=True)
