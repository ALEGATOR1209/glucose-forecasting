"""Unit tests for reusable NeuralForecast benchmark primitives."""
from __future__ import annotations

import math

import polars as pl
import pytest

from glucose_forecasting.backends.neuralforecast.benchmark import (
    calculate_metrics,
    to_neuralforecast_frame,
)


def test_to_neuralforecast_frame_selects_schema_and_sorts_rows() -> None:
    split = pl.DataFrame(
        {
            "unique_id": ["series-b", "series-a", "series-a"],
            "ds": [
                "2026-01-01T00:05:00",
                "2026-01-01T00:05:00",
                "2026-01-01T00:00:00",
            ],
            "y": [120.0, 110.0, 100.0],
            "hr": [70.0, 80.0, 75.0],
            "study_group": ["T2DM", "T1DM", "T1DM"],
        }
    ).with_columns(pl.col("ds").str.to_datetime())

    result = to_neuralforecast_frame(split, historical_exogenous=("hr",))

    assert result.columns == ["unique_id", "ds", "y", "hr"]
    assert result["unique_id"].to_list() == ["series-a", "series-a", "series-b"]
    assert result["y"].to_list() == [100.0, 110.0, 120.0]


def test_to_neuralforecast_frame_rejects_missing_columns() -> None:
    split = pl.DataFrame({"unique_id": ["series-a"], "ds": [1], "y": [100.0]})

    with pytest.raises(ValueError, match="hr"):
        to_neuralforecast_frame(split, historical_exogenous=("hr",))


def test_calculate_metrics_matches_legacy_overall_and_group_outputs() -> None:
    predictions = pl.DataFrame(
        {
            "y": [100.0, 0.0, 200.0, 100.0],
            "yhat": [110.0, 5.0, 180.0, 130.0],
            "study_group": ["B", "B", "A", "A"],
        }
    )

    result = calculate_metrics(predictions)

    assert result.overall.mae == pytest.approx(16.25)
    assert result.overall.rmse == pytest.approx(math.sqrt(356.25))
    assert result.overall.mard == pytest.approx((10.0 + 10.0 + 30.0) / 3)
    assert result.by_study_group.columns == [
        "study_group",
        "n_points",
        "mae",
        "rmse",
        "mard",
    ]
    assert result.by_study_group["study_group"].to_list() == ["B", "A"]
    assert result.by_study_group["n_points"].to_list() == [2, 2]
    assert result.by_study_group["mae"].to_list() == pytest.approx([7.5, 25.0])
