#!/usr/bin/env python3
"""Shared, model-agnostic evaluation library.

Extracted from ``scripts/sugar_one/evaluate_model.py`` (the more
feature-complete of the two eval scripts): flexible CSV loading with
covariate aliasing/ablation, and the inference loop with ETA logging.
Generalized only enough to keep both evaluate_glumind.py and
evaluate_model.py call sites working.
"""
from __future__ import annotations

import csv
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

ModelKind = Literal["glumind", "sugar_one"]

COL_EVENT = "Event Type"

GLUMIND_COVARIATES: dict[str, list[str]] = {
    "glucose": ["Glucose Value (mg/dL)", "Glucose (mg/dL)"],
    "hr": ["Heart Rate"],
    "steps": ["Step Count"],
}

SUGAR_ONE_COVARIATES: dict[str, list[str]] = {
    "glucose": ["Glucose Value (mg/dL)", "Glucose (mg/dL)"],
    "basal": ["Basal Rate (U/h)"],
    "bolus": ["Bolus Insulin (U)"],
    "carbs": ["Carbohydrates (g)"],
}

# User-facing names accepted by --include-cov / --exclude-cov (case-insensitive).
COVARIATE_NAME_ALIASES: dict[str, list[str]] = {
    "hr": ["hr", "heart_rate", "heart rate", "heartrate"],
    "steps": ["steps", "step", "step_count", "step count", "stepcount"],
    "basal": ["basal", "basal_rate", "basal rate", "basalrate"],
    "bolus": ["bolus", "bolus_insulin", "bolus insulin", "insulin", "bolusinsulin"],
    "carbs": ["carbs", "carb", "carbohydrates", "carbohydrate", "carbohydrate_g"],
}


def _covariate_map(model_kind: ModelKind) -> dict[str, list[str]]:
    return GLUMIND_COVARIATES if model_kind == "glumind" else SUGAR_ONE_COVARIATES


def _canonical_feature_cols(model_kind: ModelKind) -> list[str]:
    if model_kind == "glumind":
        return ["glucose", "hr", "steps"]
    return ["glucose", "basal", "bolus", "carbs"]


def _non_glucose_covariate_cols(model_kind: ModelKind) -> list[str]:
    return [c for c in _canonical_feature_cols(model_kind) if c != "glucose"]


def _normalize_covariate_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def _alias_to_canonical(name: str, model_kind: ModelKind) -> str:
    """Map a user token to a canonical covariate name for the model kind."""
    normalized = _normalize_covariate_token(name)
    valid = set(_non_glucose_covariate_cols(model_kind))
    if normalized in valid:
        return normalized
    for canonical, aliases in COVARIATE_NAME_ALIASES.items():
        alias_norms = {_normalize_covariate_token(a) for a in aliases}
        if normalized in alias_norms or normalized == canonical:
            if canonical in valid:
                return canonical
    valid_list = ", ".join(sorted(valid))
    raise ValueError(
        f"Unknown covariate {name!r} for model {model_kind!r}. "
        f"Valid names: {valid_list}"
    )


def _split_cov_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_covariate_names(raw: str | None, model_kind: ModelKind) -> list[str]:
    tokens = _split_cov_arg(raw)
    if not tokens:
        return []
    canonical: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        name = _alias_to_canonical(token, model_kind)
        if name not in seen:
            canonical.append(name)
            seen.add(name)
    return canonical


def _resolve_covariate_zeroing(
    model_kind: ModelKind,
    *,
    zero_cov: bool,
    include_cov: str | None,
    exclude_cov: str | None,
) -> tuple[list[str], list[str]]:
    """Return (active_non_glucose_covariates, zeroed_non_glucose_covariates)."""
    all_cov = _non_glucose_covariate_cols(model_kind)
    include_tokens = _split_cov_arg(include_cov)
    exclude_tokens = _split_cov_arg(exclude_cov)

    if zero_cov and (include_tokens or exclude_tokens):
        raise ValueError(
            "Use either --zero-cov or --include-cov / --exclude-cov, not both."
        )
    if include_tokens and exclude_tokens:
        raise ValueError("Use either --include-cov or --exclude-cov, not both.")

    if zero_cov:
        return [], list(all_cov)

    if include_tokens:
        included = _parse_covariate_names(include_cov, model_kind)
        if not included:
            raise ValueError("--include-cov requires at least one covariate name.")
        included_set = set(included)
        zeroed = [c for c in all_cov if c not in included_set]
        return included, zeroed

    if exclude_tokens:
        excluded = _parse_covariate_names(exclude_cov, model_kind)
        if not excluded:
            raise ValueError("--exclude-cov requires at least one covariate name.")
        excluded_set = set(excluded)
        active = [c for c in all_cov if c not in excluded_set]
        return active, excluded

    return list(all_cov), []


def _zero_covariates(df: pl.DataFrame, covariates: list[str]) -> pl.DataFrame:
    """Replace named covariates with 0.0 (applied after imputation)."""
    if not covariates:
        return df
    present = [c for c in covariates if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in present])


def _pick_header_column(header: list[str], aliases: list[str]) -> str | None:
    header_set = set(header)
    for alias in aliases:
        if alias in header_set:
            return alias
    return None


def _load_csv_flexible(
    csv_path: Path,
    model_kind: ModelKind,
    unique_id_choice: str,
    drop_interpolated: bool,
    eval_split: str | None,
    train_only: bool,
    *,
    col_seq: str,
    col_user: str,
    col_split: str,
    col_group: str,
    ts_aliases: list[str],
    ts_format: str,
) -> pl.DataFrame:
    """Load CSV with canonical columns; missing covariates become 0.0."""
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))

    uid_aliases = [col_seq] if unique_id_choice == "sequence_id" else [col_user]
    uid_col = _pick_header_column(header, uid_aliases)
    if uid_col is None:
        typer.echo(
            f"Error: Could not find unique id column ({uid_aliases}) in {csv_path.name}.",
            err=True,
        )
        raise typer.Exit(1)

    ts_col = _pick_header_column(header, ts_aliases)
    if ts_col is None:
        typer.echo(f"Error: Could not find timestamp column in {csv_path.name}.", err=True)
        raise typer.Exit(1)

    cov_map = _covariate_map(model_kind)
    glucose_col = _pick_header_column(header, cov_map["glucose"])
    if glucose_col is None:
        typer.echo(f"Error: Could not find glucose column in {csv_path.name}.", err=True)
        raise typer.Exit(1)

    has_split_col = col_split in header
    has_group_col = col_group in header
    has_event_col = COL_EVENT in header

    schema_overrides: dict[str, pl.DataType] = {col_seq: pl.Utf8, col_user: pl.Utf8}
    for aliases in cov_map.values():
        for alias in aliases:
            schema_overrides[alias] = pl.Utf8

    select_exprs: list[pl.Expr] = [
        pl.col(uid_col).alias("unique_id"),
        pl.col(ts_col).alias("ds"),
        pl.col(glucose_col).alias("glucose"),
    ]

    missing_covariates: list[str] = []
    for canonical, aliases in cov_map.items():
        if canonical == "glucose":
            continue
        source_col = _pick_header_column(header, aliases)
        if source_col is None:
            select_exprs.append(pl.lit(0.0).alias(canonical))
            missing_covariates.append(canonical)
        else:
            select_exprs.append(pl.col(source_col).alias(canonical))

    if has_group_col:
        select_exprs.append(pl.col(col_group).alias("study_group"))
    if has_split_col:
        select_exprs.append(pl.col(col_split).alias("split"))
    if has_event_col:
        select_exprs.append(pl.col(COL_EVENT).alias("event_type"))

    lf = (
        pl.scan_csv(csv_path, infer_schema_length=10_000, schema_overrides=schema_overrides)
        .select(select_exprs)
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, ts_format, strict=False),
            pl.col("glucose").cast(pl.Float32, strict=False),
            *[
                pl.col(c).cast(pl.Float32, strict=False)
                for c in _canonical_feature_cols(model_kind)
                if c != "glucose"
            ],
        ])
        .drop_nulls(subset=["unique_id", "ds"])
    )

    if drop_interpolated and has_event_col:
        lf = lf.filter(pl.col("event_type") != "Interpolated")

    if train_only and has_split_col:
        lf = lf.filter(pl.col("split") == "train")
    elif eval_split is not None and has_split_col:
        lf = lf.filter(pl.col("split") == eval_split)
    elif eval_split is not None and not has_split_col:
        typer.echo(
            f"Warning: --test-split='{eval_split}' requested but CSV has no "
            f"'{col_split}' column — using all rows.",
            err=True,
        )

    df = lf.collect()

    if "study_group" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("study_group"))

    if missing_covariates:
        typer.echo(
            f"  Missing covariates filled with 0.0: {', '.join(missing_covariates)}"
        )

    split_label = "train" if train_only else (eval_split or "all")
    typer.echo(f"  Loaded {len(df):,} rows ({split_label}) from {csv_path.name}")
    return df


DEFAULT_INFERENCE_LOG_INTERVAL_S = 10.0


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


@torch.no_grad()
def _run_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_windows: int,
    log_interval_s: float = DEFAULT_INFERENCE_LOG_INTERVAL_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with periodic progress logs and ETA."""
    model.eval()
    device_t = torch.device(device)
    n_batches_total = len(loader)
    batch_size = loader.batch_size or 1

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_last_log = 0.0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device_t), y.to(device_t)
        pred = model(x)
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

        now = time.perf_counter()
        elapsed = now - t_start
        should_log = (
            batch_idx == 1
            or batch_idx == n_batches_total
            or (elapsed - t_last_log) >= log_interval_s
        )
        if should_log:
            pct = 100.0 * batch_idx / n_batches_total
            batches_per_s = batch_idx / elapsed if elapsed > 0 else 0.0
            remaining_batches = n_batches_total - batch_idx
            eta_s = remaining_batches / batches_per_s if batches_per_s > 0 else 0.0
            windows_done = min(batch_idx * batch_size, n_windows)
            typer.echo(
                f"  inference {batch_idx:,}/{n_batches_total:,} batches "
                f"({pct:.1f}%) | ~{windows_done:,}/{n_windows:,} windows | "
                f"elapsed {_format_duration(elapsed)} | "
                f"ETA {_format_duration(eta_s)}"
            )
            t_last_log = elapsed

    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return true_arr, pred_arr
