#!/usr/bin/env python3
"""Shared, model-agnostic CSV loading / splitting / imputation utilities.

Extracted from ``scripts/glumind/train_glumind.py`` and
``scripts/sugar_one/train_sugar_one.py``, which reimplemented (near-)identical
logic with different hardcoded column sets. The functions here are
genericized to accept the source/value column names as parameters so both
call sites (and any future ones) can share a single implementation.
"""
from __future__ import annotations

import os
import re

import polars as pl
import torch

# ---------------------------------------------------------------------------
# Study-group label normalization (identical across GluMind / SugarOne /
# GluMindUni training scripts).
# ---------------------------------------------------------------------------

STUDY_GROUP_ORDER = ["Healthy", "Pre-T2DM", "Oral-T2DM", "Insulin-T2DM", "T1DM"]

STUDY_GROUP_ALIASES = {
    "healthy": "Healthy",
    "pre_t2dm": "Pre-T2DM",
    "prediabetes": "Pre-T2DM",
    "pre_diabetes": "Pre-T2DM",
    "pre_diabetes_lifestyle_controlled": "Pre-T2DM",
    "oral_t2dm": "Oral-T2DM",
    "oral_medication": "Oral-T2DM",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral-T2DM",
    "insulin_t2dm": "Insulin-T2DM",
    "insulin_dependent": "Insulin-T2DM",
}


def normalize_study_group_label(value: str) -> str:
    """Map raw dataset cohort labels to canonical study-group names."""
    raw = str(value).strip()
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return STUDY_GROUP_ALIASES.get(key, raw)


def normalize_study_groups_column(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize study_group labels in-place-safe form."""
    if df.is_empty():
        return df
    return df.with_columns(
        pl.col("study_group")
        .cast(pl.Utf8)
        .map_elements(normalize_study_group_label, return_dtype=pl.Utf8)
    )


def resolve_num_workers(num_workers: int, device: torch.device) -> int:
    """Resolve DataLoader workers with an auto mode tuned for GPU training."""
    if num_workers >= 0:
        return num_workers
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(8, max(2, cpu_count // 2))


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_splits_streaming(
    csv_path,
    unique_id_choice: str,
    drop_interpolated: bool,
    *,
    col_seq: str,
    col_user: str,
    col_ts: str,
    col_split: str,
    col_group: str,
    col_event: str,
    value_columns: dict[str, str],
    ts_format: str,
    utf8_value_columns: tuple[str, ...] = (),
    log_fn=print,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Lazy CSV scan via Polars — returns (train, val, test) DataFrames.

    ``value_columns`` maps canonical output column name -> source CSV column
    name (e.g. ``{"glucose": "Glucose Value (mg/dL)", "hr": "Heart Rate"}``).
    All value columns are cast to Float32 after renaming.

    ``utf8_value_columns`` lists canonical value-column names whose source CSV
    column should be scanned as Utf8 first (e.g. sparse numeric columns that
    may contain empty strings in the raw CSV) before being cast to Float32.

    ``log_fn``: output function (``print`` or ``typer.echo``) to preserve each
    caller's original console output style.
    """
    uid_col = col_seq if unique_id_choice == "sequence_id" else col_user
    log_fn("Loading train/val/test splits (streaming)...")

    schema_overrides: dict[str, pl.DataType] = {col_seq: pl.Utf8, col_user: pl.Utf8}
    for canonical in utf8_value_columns:
        src = value_columns.get(canonical)
        if src is not None:
            schema_overrides[src] = pl.Utf8

    select_cols = [uid_col, col_ts, col_split, col_group, col_event, *value_columns.values()]
    rename_map = {
        uid_col: "unique_id",
        col_ts: "ds",
        col_group: "study_group",
        col_split: "split",
        col_event: "event_type",
        **{src: canonical for canonical, src in value_columns.items()},
    }

    lf = (
        pl.scan_csv(
            csv_path,
            infer_schema_length=10_000,
            schema_overrides=schema_overrides,
        )
        .select(select_cols)
        .rename(rename_map)
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, ts_format, strict=False),
            *[pl.col(c).cast(pl.Float32, strict=False) for c in value_columns.keys()],
        ])
        .drop_nulls(subset=["unique_id", "ds", "split", "study_group"])
    )

    if drop_interpolated:
        lf = lf.filter(pl.col("event_type") != "Interpolated")

    df = lf.collect()
    log_fn(f"  ... loaded {len(df):,} rows total")

    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == "test")
    return train_df, val_df, test_df


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
    *,
    log_fn=print,
    applied_message: str = (
        "Applied split scheme: train <- train+val | val <- test | "
        "test disabled."
    ),
    note_message: str | None = (
        "Note: this mode is for tuning only and does not produce held-out "
        "test metrics."
    ),
    error_repr: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply optional split remapping while preserving classic defaults.

    ``log_fn``/``applied_message``/``note_message`` let callers preserve their
    exact original console output (``print`` vs ``typer.echo``, slightly
    different wording) after this logic was deduplicated.
    """
    if split_scheme == "classic":
        return train_df, val_df, test_df

    if split_scheme == "trainval_test_as_val":
        if test_df.is_empty():
            raise ValueError(
                "split_scheme=trainval_test_as_val requires a non-empty test split."
            )
        merged_train = pl.concat([train_df, val_df]) if not val_df.is_empty() else train_df
        remapped_val = test_df
        remapped_test = test_df.clear()
        log_fn(applied_message)
        if note_message is not None:
            log_fn(note_message)
        return merged_train, remapped_val, remapped_test

    if error_repr:
        raise ValueError(f"Unknown split_scheme: {split_scheme!r}")
    raise ValueError(f"Unknown split_scheme: {split_scheme}")


def impute_and_sort(
    df: pl.DataFrame,
    *,
    ffill_bfill_columns: list[str] = (),
    zero_fill_columns: list[str] = (),
) -> pl.DataFrame:
    """Sort by (unique_id, ds) and impute per-series.

    ``ffill_bfill_columns``: forward-fill then back-fill then fill_null(0.0)
        (continuous signals that persist until changed, e.g. glucose, HR,
        step count, basal rate).
    ``zero_fill_columns``: fill_null(0.0) directly, no carry-over (discrete
        event signals, e.g. bolus insulin, carbohydrates).
    """
    if df.is_empty():
        return df
    exprs = [
        pl.col(c)
        .forward_fill()
        .backward_fill()
        .fill_null(0.0)
        .cast(pl.Float32)
        .over("unique_id")
        for c in ffill_bfill_columns
    ] + [
        pl.col(c)
        .fill_null(0.0)
        .cast(pl.Float32)
        .over("unique_id")
        for c in zero_fill_columns
    ]
    return df.sort(["unique_id", "ds"]).with_columns(exprs)


def limit_series(df: pl.DataFrame, max_series: int) -> pl.DataFrame:
    if df.is_empty() or max_series <= 0:
        return df
    keep = df["unique_id"].unique(maintain_order=True).head(max_series)
    return df.filter(pl.col("unique_id").is_in(keep))
