"""Smoke tests for the SugarJepa proof-of-concept (scripts/sugar_jepa/).

Not full coverage (time-boxed POC per CLAUDE.md) — just enough to catch
shape regressions in the model, the sliding-window dataset, and the
checkpoint round-trip. Uses the vendored, locally-cached pretrained CGM-JEPA
encoder (scripts/sugar_jepa/pretrained/cgm_jepa/) — no network access needed.
"""
from __future__ import annotations

import torch

from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel
from scripts.sugar_jepa.train_sugar_jepa import SugarJepaWindowDataset
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
    window_frame,
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


def test_sugar_jepa_window_dataset_window_count() -> None:
    input_steps, horizon, jepa_window = 4, 2, 8
    lookback = max(input_steps, jepa_window)
    window_len = lookback + horizon
    n_rows = 20
    df = window_frame("sugar_one", {"a": n_rows})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    assert len(ds) == n_rows - window_len + 1


def test_sugar_jepa_window_dataset_skips_short_series() -> None:
    input_steps, horizon, jepa_window = 4, 2, 16
    lookback = max(input_steps, jepa_window)
    window_len = lookback + horizon
    df = window_frame("sugar_one", {"short": 5, "long": 30})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    assert len(ds) == 30 - window_len + 1
    assert "short" not in ds.series_ids


def test_sugar_jepa_window_dataset_shapes_and_branch_scaling() -> None:
    input_steps, horizon, jepa_window = 4, 2, 12
    df = window_frame("sugar_one", {"a": 20})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    x, jepa, y = ds[0]
    assert x.shape == (input_steps, 4)
    assert jepa.shape == (jepa_window,)
    assert y.shape == (horizon,)
    assert x.min() >= 0.0 - 1e-6
    assert x.max() <= 1.0 + 1e-6
    assert ds.scaler_glucose is not ds.scaler_glucose_jepa
    assert jepa.min() < 0.0


def test_sugar_jepa_window_dataset_reuses_scalers_not_refit() -> None:
    input_steps, horizon, jepa_window = 4, 2, 8
    train_df = window_frame("sugar_one", {"train": 20})
    val_df = window_frame("sugar_one", {"val": 16})

    train_ds = SugarJepaWindowDataset(train_df, input_steps, horizon, jepa_window, fit_scalers=True)
    val_ds = SugarJepaWindowDataset(
        val_df, input_steps, horizon, jepa_window,
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        scaler_glucose_jepa=train_ds.scaler_glucose_jepa,
        fit_scalers=False,
    )
    assert val_ds.scaler_glucose is train_ds.scaler_glucose
    assert val_ds.scaler_glucose_jepa is train_ds.scaler_glucose_jepa
