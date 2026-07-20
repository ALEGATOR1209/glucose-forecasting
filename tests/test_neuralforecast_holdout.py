"""Tests for SugarOne-compatible NeuralForecast fixed-split evaluation."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from glucose_forecasting.backends.neuralforecast.config import NeuralForecastRunConfig
from glucose_forecasting.backends.neuralforecast.evaluations.holdout import (
    _evaluate_dense_split,
    _validate_loaded_bundle,
    run_loaded_holdout,
)


class _DenseForecast:
    """Minimal loaded-bundle substitute returning overlapping forecast points."""

    def cross_validation(self, **kwargs: object) -> pl.DataFrame:
        assert kwargs["test_size"] == 2
        assert kwargs["step_size"] == 1
        assert kwargs["use_fitted"] is True
        assert kwargs["refit"] is False
        start = datetime(2026, 1, 1)
        return pl.DataFrame(
            {
                "unique_id": ["series-a", "series-a", "series-a"],
                "ds": [
                    start + timedelta(minutes=10),
                    start + timedelta(minutes=15),
                    start + timedelta(minutes=15),
                ],
                "cutoff": [
                    start + timedelta(minutes=5),
                    start + timedelta(minutes=5),
                    start + timedelta(minutes=10),
                ],
                "TFT": [102.0, 103.0, 104.0],
                "y": [102.0, 103.0, 103.0],
            }
        )


def test_dense_evaluation_keeps_overlapping_horizon_points() -> None:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "unique_id": ["series-a"] * 4,
            "ds": [start + timedelta(minutes=5 * offset) for offset in range(4)],
            "y": [100.0, 101.0, 102.0, 103.0],
            "basal": [1.0] * 4,
            "bolus": [0.0] * 4,
            "carbohydrates": [0.0] * 4,
            "study_group": ["T1DM"] * 4,
            "event_type": ["EGV"] * 4,
        }
    )

    result = _evaluate_dense_split(
        _DenseForecast(),
        frame=frame,
        profile_exogenous=("basal", "bolus", "carbohydrates"),
        input_size=2,
    )

    assert result.height == 3
    assert result["y"].to_list() == [102.0, 103.0, 103.0]
    assert result["yhat"].to_list() == [102.0, 103.0, 104.0]
    assert result["forecast_origin"].n_unique() == 2


def test_sugarone_protocol_requires_shared_geometry(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.touch()

    config = NeuralForecastRunConfig(csv=csv_path)

    assert config.input_hours == pytest.approx(128 * 5 / 60)
    assert config.step_size == 1
    with pytest.raises(ValueError, match="input_steps=128"):
        NeuralForecastRunConfig(csv=csv_path, input_hours=6)


def test_loaded_bundle_geometry_must_match_source_config(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.touch()
    config = NeuralForecastRunConfig(csv=csv_path)

    _validate_loaded_bundle(
        SimpleNamespace(models=[SimpleNamespace(h=12, input_size=128)]),
        config,
    )
    with pytest.raises(ValueError, match="geometry"):
        _validate_loaded_bundle(
            SimpleNamespace(models=[SimpleNamespace(h=12, input_size=72)]),
            config,
        )


def test_loaded_holdout_reuses_bundle_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "loop.csv"
    _write_loop_splits(csv_path, rows_per_split=84)
    loaded = _ReloadedForecast()
    from neuralforecast import NeuralForecast

    monkeypatch.setattr(NeuralForecast, "load", lambda _: loaded)
    config = NeuralForecastRunConfig(
        csv=csv_path,
        profile="loop",
        input_hours=6,
        step_size=12,
        holdout_protocol="dense",
        plot=False,
    )

    output_dir = run_loaded_holdout(
        config,
        bundle_dir=tmp_path / "bundle",
        run_dir=tmp_path / "evaluation",
    )

    assert loaded.cross_validation_calls == 2
    assert (output_dir / "test_metrics_overall.csv").is_file()
    assert (output_dir / "evaluation_metadata.json").read_text(encoding="utf-8").find(
        '"sugarone_comparable": false'
    ) > 0


class _ReloadedForecast:
    """Saved-model stand-in that exposes no fitting operation."""

    def __init__(self) -> None:
        self.models = [SimpleNamespace(h=12, input_size=72)]
        self.cross_validation_calls = 0
        self.freq = "5m"

    def cross_validation(self, *, df: pl.DataFrame, **_: object) -> pl.DataFrame:
        self.cross_validation_calls += 1
        target = df.tail(1)
        return target.select(
            "unique_id",
            "ds",
            pl.col("ds").alias("cutoff"),
            pl.col("y").alias("TFT"),
            "y",
        )


def _write_loop_splits(path: Path, *, rows_per_split: int) -> None:
    header = (
        "sequence_id,Timestamp,Event Type,User ID,Glucose (mg/dL),Basal Rate (U/h),"
        "Bolus Insulin (U),Carbohydrates (g),Recommended Split,Study Group"
    )
    rows = [header]
    start = datetime(2026, 1, 1)
    for split_index, split in enumerate(("train", "val", "test")):
        for offset in range(rows_per_split):
            timestamp = start + timedelta(minutes=5 * (split_index * rows_per_split + offset))
            rows.append(
                f"series-{split},{timestamp:%Y-%m-%dT%H:%M:%S},EGV,user-{split},"
                f"{100 + offset},1.0,0.0,0.0,{split},T1DM"
            )
    path.write_text("\n".join(rows), encoding="utf-8")
