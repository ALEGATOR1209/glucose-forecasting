"""Unit tests for scripts/common/metrics.py."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import MinMaxScaler

from scripts.common.metrics import mae_rmse_mard, per_study_group_breakdown


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_mae", "expected_rmse", "expected_mard"),
    [
        (
            np.array([100.0, 200.0, 50.0]),
            np.array([110.0, 180.0, 60.0]),
            (10 + 20 + 10) / 3,
            math.sqrt((100 + 400 + 100) / 3),
            (0.1 + 0.1 + 0.2) / 3 * 100,
        ),
        (
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
            1.5,
            math.sqrt((1 + 4) / 2),
            math.nan,
        ),
        (
            np.array([100.0]),
            np.array([90.0]),
            10.0,
            10.0,
            10.0,
        ),
        (
            # zero entries are excluded from MARD but included in MAE/RMSE
            np.array([0.0, 100.0]),
            np.array([5.0, 110.0]),
            (5 + 10) / 2,
            math.sqrt((25 + 100) / 2),
            10.0,
        ),
    ],
    ids=["hand_computed", "zero_glucose_mard_nan", "single_value", "mixed_zero_nonzero"],
)
def test_mae_rmse_mard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_mae: float,
    expected_rmse: float,
    expected_mard: float,
) -> None:
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    assert mae == pytest.approx(expected_mae)
    assert rmse == pytest.approx(expected_rmse)
    if math.isnan(expected_mard):
        assert math.isnan(mard)
    else:
        assert mard == pytest.approx(expected_mard)


def test_per_study_group_breakdown_correct_and_sorted() -> None:
    scaler = MinMaxScaler().fit(np.array([[0.0], [200.0]]))
    # Two groups: "A" has small error, "B" has large error -> A should sort first.
    true_scaled = scaler.transform(np.array([[100.0], [100.0], [100.0], [100.0]])).ravel()
    pred_scaled = scaler.transform(np.array([[101.0], [99.0], [150.0], [50.0]])).ravel()
    groups = ["B", "B", "A", "A"]

    out = per_study_group_breakdown(true_scaled, pred_scaled, scaler, groups)
    assert out is not None
    # Explicitly check sort order by mae ascending.
    maes = out["mae"].to_list()
    assert maes == sorted(maes)
    assert out.filter(pl.col("study_group") == "B")["n_windows"][0] == 2
    assert out.filter(pl.col("study_group") == "A")["n_windows"][0] == 2


def test_per_study_group_breakdown_length_mismatch_returns_none() -> None:
    scaler = MinMaxScaler().fit(np.array([[0.0], [200.0]]))
    true_scaled = np.array([0.5, 0.5, 0.5])
    pred_scaled = np.array([0.5, 0.5, 0.5])
    groups = ["A", "B"]  # mismatched length
    out = per_study_group_breakdown(true_scaled, pred_scaled, scaler, groups)
    assert out is None
