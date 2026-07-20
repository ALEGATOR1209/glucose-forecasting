"""Shared fixtures for the glucose-forecasting test suite.

Generalizes the synthetic-CSV pattern from ``_write_loop_ic_csv()`` in
``tests/test_tune_sugar_one_smoke.py`` and adds tiny in-memory Polars
DataFrame builders so individual test files don't re-duplicate this setup.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl

# ---------------------------------------------------------------------------
# Tiny model dims shared across tests (fast forward/backward passes on CPU).
# ---------------------------------------------------------------------------
TINY_D_MODEL = 8
TINY_N_HEADS = 2
TINY_N_BLOCKS = 1
TINY_FF_UNITS = 16
TINY_INPUT_STEPS = 8
TINY_HORIZON = 2
SeriesSpec: TypeAlias = tuple[str, str, str, int, float]
TINY_TRAIN_SERIES: list[SeriesSpec] = [
    ("series-train", "train", "T1DM", 40, 100.0),
    ("series-val", "val", "T1DM", 30, 110.0),
    ("series-test", "test", "T1DM", 24, 105.0),
]

_TINY_TRAIN_FLAGS = {
    "mode": "global",
    "input_steps": str(TINY_INPUT_STEPS),
    "horizon": str(TINY_HORIZON),
    "d_model": str(TINY_D_MODEL),
    "n_heads": str(TINY_N_HEADS),
    "n_blocks": str(TINY_N_BLOCKS),
    "ff_units": str(TINY_FF_UNITS),
    "epochs": "1",
    "batch_size": "8",
    "patience": "0",
    "log_every": "1",
    "val_every_n_epochs": "1",
    "num_workers": "0",
    "device": "cpu",
}


def tiny_train_args(
    style: Literal["snake", "kebab"],
    out_dir: Path,
) -> list[str]:
    separator = "_" if style == "snake" else "-"
    args = [
        item
        for key, value in _TINY_TRAIN_FLAGS.items()
        for item in (f"--{key.replace('_', separator)}", value)
    ]
    out_name = "out_dir" if style == "snake" else "out-dir"
    return [*args, f"--{out_name}", str(out_dir)]

FeatureValue: TypeAlias = float | int | str
FeatureFactory: TypeAlias = FeatureValue | Callable[[int], FeatureValue]

WINDOW_FEATURES: dict[str, dict[str, FeatureFactory]] = {
    "glumind": {
        "glucose": lambda index: 100.0 + index,
        "hr": lambda index: 70.0 + index,
        "steps": float,
    },
    "sugar_one": {
        "glucose": lambda index: 100.0 + index,
        "basal": 1.0,
        "bolus": lambda index: 2.0 if index % 5 == 0 else 0.0,
        "carbs": lambda index: 10.0 if index % 7 == 0 else 0.0,
    },
}


def window_frame(family: str, rows_per_series: dict[str, int]) -> pl.DataFrame:
    """Build canonical sliding-window input without family-specific row loops."""
    features = WINDOW_FEATURES[family]
    return pl.DataFrame(
        {
            "unique_id": unique_id,
            "ds": index,
            **{
                name: value(index) if callable(value) else value
                for name, value in features.items()
            },
            "study_group": "T1DM",
        }
        for unique_id, row_count in rows_per_series.items()
        for index in range(row_count)
    )


RawRowFactory: TypeAlias = Callable[
    [str, str, str, int, float, datetime],
    dict[str, object],
]


def _write_series_csv(
    path: Path,
    series: list[SeriesSpec],
    row_factory: RawRowFactory,
) -> None:
    start = datetime(2020, 1, 1)
    pl.DataFrame(
        row_factory(
            unique_id,
            split,
            group,
            index,
            glucose0,
            start + timedelta(minutes=5 * index),
        )
        for unique_id, split, group, row_count, glucose0 in series
        for index in range(row_count)
    ).write_csv(path)


def _glumind_row(
    unique_id: str,
    split: str,
    group: str,
    index: int,
    glucose0: float,
    timestamp: datetime,
) -> dict[str, object]:
    return {
        "sequence_id": unique_id,
        "User ID": unique_id,
        "Timestamp (YYYY-MM-DDThh:mm:ss)": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "Event Type": "EGV",
        "Study Group": group,
        "Glucose Value (mg/dL)": glucose0 + index * 0.5,
        "Heart Rate": 70.0 + index * 0.1,
        "Step Count": 10.0 * index,
        "Recommended Split": split,
    }


def _sugar_one_row(
    unique_id: str,
    split: str,
    group: str,
    index: int,
    glucose0: float,
    timestamp: datetime,
) -> dict[str, object]:
    return {
        "sequence_id": unique_id,
        "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "Event Type": "EGV",
        "User ID": unique_id,
        "Glucose (mg/dL)": glucose0 + index * 0.5,
        "Basal Rate (U/h)": "1.0",
        "Bolus Insulin (U)": "2.0" if index % 7 == 0 else "",
        "Carbohydrates (g)": "15.0" if index % 9 == 0 else "",
        "Recommended Split": split,
        "Study Group": group,
    }


def write_glumind_csv(
    path: Path,
    *,
    series: list[SeriesSpec] | None = None,
) -> None:
    _write_series_csv(
        path,
        series
        or [
            ("g-train-a", "train", "T1DM", 40, 100.0),
            ("g-val-b", "val", "Healthy", 30, 110.0),
            ("g-test-c", "test", "T1DM", 25, 105.0),
        ],
        _glumind_row,
    )


def write_sugar_one_csv(
    path: Path,
    *,
    series: list[SeriesSpec] | None = None,
) -> None:
    _write_series_csv(
        path,
        series
        or [
            ("s-train-a", "train", "T1DM", 40, 100.0),
            ("s-val-b", "val", "Insulin-T2DM", 30, 110.0),
            ("s-test-c", "test", "T1DM", 25, 105.0),
        ],
        _sugar_one_row,
    )
