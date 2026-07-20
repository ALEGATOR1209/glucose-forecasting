"""Unit tests for sliding-window Dataset classes across model families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from scripts.glumind.train_glumind import GlucoseWindowDataset
from scripts.sugar_one.train_sugar_one import SugarOneWindowDataset
from tests.conftest import window_frame


@dataclass(frozen=True)
class WindowDatasetSpec:
    name: str
    frame_family: str
    cls: type
    n_features: int
    scaler_attrs: tuple[str, ...]


_SPECS = [
    WindowDatasetSpec(
        name="glumind",
        frame_family="glumind",
        cls=GlucoseWindowDataset,
        n_features=3,
        scaler_attrs=("scaler_glucose", "scaler_hr", "scaler_steps"),
    ),
    WindowDatasetSpec(
        name="sugar_one",
        frame_family="sugar_one",
        cls=SugarOneWindowDataset,
        n_features=4,
        scaler_attrs=("scaler_glucose", "scaler_basal", "scaler_bolus", "scaler_carbs"),
    ),
]


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_window_dataset_window_count(spec: WindowDatasetSpec) -> None:
    input_steps, horizon = 4, 2
    window_len = input_steps + horizon
    n_rows = 12
    ds = spec.cls(
        window_frame(spec.frame_family, {"a": n_rows}),
        input_steps,
        horizon,
        fit_scalers=True,
    )
    assert len(ds) == n_rows - window_len + 1


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_window_dataset_skips_short_series(spec: WindowDatasetSpec) -> None:
    input_steps, horizon = 8, 2
    window_len = input_steps + horizon
    short_n, long_n = 5, 15
    ds = spec.cls(
        window_frame(spec.frame_family, {"short": short_n, "long": long_n}),
        input_steps,
        horizon,
        fit_scalers=True,
    )
    assert len(ds) == long_n - window_len + 1
    assert "short" not in ds.series_ids


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_window_dataset_getitem_shapes_and_scaling(spec: WindowDatasetSpec) -> None:
    input_steps, horizon = 4, 2
    ds = spec.cls(
        window_frame(spec.frame_family, {"a": 10}),
        input_steps,
        horizon,
        fit_scalers=True,
    )
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (input_steps, spec.n_features)
    assert y.shape == (horizon,)
    assert x.min() >= 0.0 - 1e-6
    assert x.max() <= 1.0 + 1e-6
    # Monotonic glucose 100..109 maps first step to 0 under MinMaxScaler.
    assert x[0, 0].item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_window_dataset_reuses_scaler_not_refit(spec: WindowDatasetSpec) -> None:
    input_steps, horizon = 4, 2
    train_df = window_frame(spec.frame_family, {"train": 10})
    val_df = window_frame(spec.frame_family, {"val": 8})
    train_ds = spec.cls(train_df, input_steps, horizon, fit_scalers=True)
    reuse_kwargs: dict[str, Any] = {
        attr: getattr(train_ds, attr) for attr in spec.scaler_attrs
    }
    val_ds = spec.cls(val_df, input_steps, horizon, fit_scalers=False, **reuse_kwargs)
    assert val_ds.scaler_glucose is train_ds.scaler_glucose

    raw_val_glucose = val_df.sort(["unique_id", "ds"])["glucose"].to_numpy()
    expected = train_ds.scaler_glucose.transform(raw_val_glucose.reshape(-1, 1)).ravel()
    x0, _ = val_ds[0]
    assert x0[0, 0].item() == pytest.approx(float(expected[0]), abs=1e-5)
