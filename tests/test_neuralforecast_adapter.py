"""Tests for profile-aware NeuralForecast data preparation."""
from __future__ import annotations

from pathlib import Path

from glucose_forecasting.backends.neuralforecast.adapter import (
    AI_READI_PROFILE,
    LOOP_PROFILE,
    detect_profile,
    prepare_splits,
)


def _write_loop_csv(path: Path) -> None:
    rows = [
        "sequence_id,Timestamp,Event Type,User ID,Glucose (mg/dL),Basal Rate (U/h),Bolus Insulin (U),Carbohydrates (g),Recommended Split,Study Group",
        *[
            f"s1,2026-01-01T00:{minute:02d}:00,EGV,u1,{100 + minute},1.0,0.0,,{split},T1DM"
            for split, minute in (("train", 0), ("train", 5), ("val", 10), ("val", 15), ("test", 20), ("test", 25))
        ],
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_prepare_splits_normalizes_loop_columns_and_imputes_events(tmp_path: Path) -> None:
    path = tmp_path / "loop.csv"
    _write_loop_csv(path)

    splits = prepare_splits(path)

    assert splits.profile == LOOP_PROFILE
    assert {"unique_id", "ds", "y", "basal", "bolus", "carbohydrates"}.issubset(splits.train.columns)
    assert splits.train["carbohydrates"].to_list() == [0.0, 0.0]
    assert splits.validation.height == 2
    assert splits.test.height == 2


def test_explicit_profile_does_not_need_csv_schema(tmp_path: Path) -> None:
    path = tmp_path / "arbitrary.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    assert detect_profile(path, "ai-readi") == AI_READI_PROFILE
