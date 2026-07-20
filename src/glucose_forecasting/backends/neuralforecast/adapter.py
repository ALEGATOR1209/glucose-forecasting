"""Profile-aware loading and preparation for NeuralForecast CSV data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from glucose_forecasting.common.data_loading import (
    apply_split_scheme,
    impute_and_sort,
    limit_series,
    load_splits_streaming,
)

DataProfile = Literal["auto", "ai-readi", "loop"]


@dataclass(frozen=True, slots=True)
class NeuralForecastProfile:
    """Source-schema mapping and historical covariates for a data family."""

    name: Literal["ai-readi", "loop"]
    timestamp_column: str
    glucose_column: str
    historical_exogenous: tuple[str, ...]
    value_columns: dict[str, str]
    forward_fill_columns: tuple[str, ...]
    zero_fill_columns: tuple[str, ...] = ()


AI_READI_PROFILE = NeuralForecastProfile(
    name="ai-readi",
    timestamp_column="Timestamp (YYYY-MM-DDThh:mm:ss)",
    glucose_column="Glucose Value (mg/dL)",
    historical_exogenous=("hr", "steps"),
    value_columns={
        "y": "Glucose Value (mg/dL)",
        "hr": "Heart Rate",
        "steps": "Step Count",
    },
    forward_fill_columns=("y", "hr", "steps"),
)
LOOP_PROFILE = NeuralForecastProfile(
    name="loop",
    timestamp_column="Timestamp",
    glucose_column="Glucose (mg/dL)",
    historical_exogenous=("basal", "bolus", "carbohydrates"),
    value_columns={
        "y": "Glucose (mg/dL)",
        "basal": "Basal Rate (U/h)",
        "bolus": "Bolus Insulin (U)",
        "carbohydrates": "Carbohydrates (g)",
    },
    forward_fill_columns=("y", "basal"),
    zero_fill_columns=("bolus", "carbohydrates"),
)


@dataclass(frozen=True, slots=True)
class PreparedSplits:
    """Normalized and imputed fixed train/validation/test splits."""

    profile: NeuralForecastProfile
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame


def detect_profile(csv_path: Path, requested: DataProfile = "auto") -> NeuralForecastProfile:
    """Resolve a profile explicitly or from CSV headers."""
    if requested == "ai-readi":
        return AI_READI_PROFILE
    if requested == "loop":
        return LOOP_PROFILE

    headers = set(pl.read_csv(csv_path, n_rows=0).columns)
    matches = [
        profile
        for profile in (AI_READI_PROFILE, LOOP_PROFILE)
        if set(profile.value_columns.values()).issubset(headers)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "could not detect NeuralForecast data profile; pass --profile ai-readi or --profile loop"
        )
    raise ValueError("multiple NeuralForecast data profiles match; pass --profile explicitly")


def prepare_splits(
    csv_path: Path,
    *,
    profile_name: DataProfile = "auto",
    unique_id_choice: Literal["sequence_id", "user_id"] = "sequence_id",
    split_scheme: Literal["classic", "trainval_test_as_val"] = "classic",
    drop_interpolated: bool = False,
    max_train_series: int = 0,
    max_points_per_series: int = 0,
) -> PreparedSplits:
    """Load, normalize, impute, and optionally limit all labeled data splits."""
    profile = detect_profile(csv_path, profile_name)
    train, validation, test = load_splits_streaming(
        csv_path,
        unique_id_choice,
        drop_interpolated,
        col_seq="sequence_id",
        col_user="User ID",
        col_ts=profile.timestamp_column,
        col_split="Recommended Split",
        col_group="Study Group",
        col_event="Event Type",
        value_columns=profile.value_columns,
        ts_format="%Y-%m-%dT%H:%M:%S",
    )
    train, validation, test = apply_split_scheme(train, validation, test, split_scheme)
    prepared = tuple(
        _prepare_frame(
            frame,
            profile,
            max_points_per_series=max_points_per_series,
        )
        for frame in (train, validation, test)
    )
    return PreparedSplits(
        profile=profile,
        train=limit_series(prepared[0], max_train_series),
        validation=prepared[1],
        test=prepared[2],
    )


def filter_minimum_length(frame: pl.DataFrame, minimum: int) -> pl.DataFrame:
    """Keep only series that can produce one input-plus-horizon window."""
    if frame.is_empty():
        return frame
    eligible = (
        frame.group_by("unique_id")
        .len()
        .filter(pl.col("len") >= minimum)
        .select("unique_id")
    )
    return frame.join(eligible, on="unique_id", how="semi")


def _prepare_frame(
    frame: pl.DataFrame,
    profile: NeuralForecastProfile,
    *,
    max_points_per_series: int,
) -> pl.DataFrame:
    prepared = impute_and_sort(
        frame,
        ffill_bfill_columns=list(profile.forward_fill_columns),
        zero_fill_columns=list(profile.zero_fill_columns),
    )
    if max_points_per_series > 0:
        prepared = (
            prepared.group_by("unique_id", maintain_order=True)
            .tail(max_points_per_series)
            .sort(["unique_id", "ds"])
        )
    return prepared
