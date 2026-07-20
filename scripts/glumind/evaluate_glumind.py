#!/usr/bin/env python3
"""
evaluate_glumind.py — Evaluate a trained GluMind checkpoint on arbitrary test data.

Three ways to specify the model:
  1. --registry-dir  path to a marked_runs/.../.../ folder containing
                     _analysis_registry.csv — auto-picks the run with the
                     lowest val_mae.
  2. --run-dir       path to a specific run step directory (e.g.
                     marked_runs/glumind/.../step_05_T1DM_...) that holds
                     tuning_meta.json and best_model.pt.
  3. --checkpoint    path to a raw .pt weights file; requires --run-dir for
                     the architecture meta (or --meta-override).

Test data is fully decoupled from the training CSV stored in the metadata:
  --test-csv         CSV to evaluate on (required).  All rows are used unless
                     --test-split is given.
  --train-csv        Override the training CSV used for scaler fitting
                     (default: the CSV referenced inside tuning_meta.json).

Example:
  uv run scripts/glumind/evaluate_glumind.py \\
      --registry-dir marked_runs/glumind/ai_ready_plus_type1 \\
      --test-csv data/livia/livia_glumind_ready.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
import typer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from glucose_forecasting.data.glumind import (
    COL_GLU,
    COL_GROUP,
    COL_HR,
    COL_SEQ,
    COL_SPLIT,
    COL_STEPS,
    COL_TS,
    COL_USER,
    TS_FORMAT,
    GlucoseWindowDataset,
    apply_split_scheme,
    impute_and_sort,
    load_splits_streaming,
)
from glucose_forecasting.models.glumind import GluMindModel
from glucose_forecasting.training.glumind import (
    evaluate,
    mae_rmse_mard,
)
from scripts.common.checkpoint import strip_compile_prefix
from scripts.common.registry import (
    find_best_run_dir as _common_find_best_run_dir,
    load_run_meta as _load_meta,
    resolve_checkpoint as _common_resolve_checkpoint,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Registry helpers (re-exported under original private names via
# scripts.common.registry; thin wrappers below bind project_root).
# ---------------------------------------------------------------------------

def _find_best_run_dir(registry_dir: Path) -> tuple[Path, dict]:
    """Parse _analysis_registry.csv and return (step_dir, row) for lowest val_mae."""
    return _common_find_best_run_dir(registry_dir, project_root)


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    return _common_resolve_checkpoint(run_dir, checkpoint)


# ---------------------------------------------------------------------------
# Loading test CSV without requiring a split column
# ---------------------------------------------------------------------------

def _load_test_csv(
    csv_path: Path,
    unique_id_col: str,
    drop_interpolated: bool,
    test_split: str | None,
) -> pl.DataFrame:
    """
    Load a CSV as a test-only dataframe.

    Columns Study Group, Recommended Split, and Event Type are all optional.
    If test_split is given, filter rows where 'Recommended Split' == test_split.
    Otherwise all rows are used.
    """
    schema_overrides = {COL_SEQ: pl.Utf8, COL_USER: pl.Utf8}

    # Peek at header to detect optional columns
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))

    has_split_col = COL_SPLIT in header
    has_event_col = "Event Type" in header
    has_group_col = COL_GROUP in header

    uid_col = COL_SEQ if unique_id_col == "sequence_id" else COL_USER

    select_cols = [uid_col, COL_TS, COL_GLU, COL_HR, COL_STEPS]
    rename_map = {
        uid_col: "unique_id",
        COL_TS: "ds",
        COL_GLU: "glucose",
        COL_HR: "hr",
        COL_STEPS: "steps",
    }

    if has_group_col:
        select_cols.insert(2, COL_GROUP)
        rename_map[COL_GROUP] = "study_group"
    if has_split_col:
        select_cols.insert(2, COL_SPLIT)
        rename_map[COL_SPLIT] = "split"
    if has_event_col:
        select_cols.append("Event Type")
        rename_map["Event Type"] = "event_type"

    lf = (
        pl.scan_csv(csv_path, infer_schema_length=10_000, schema_overrides=schema_overrides)
        .select(select_cols)
        .rename(rename_map)
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, TS_FORMAT, strict=False),
            pl.col("glucose").cast(pl.Float32, strict=False),
            pl.col("hr").cast(pl.Float32, strict=False),
            pl.col("steps").cast(pl.Float32, strict=False),
        ])
        .drop_nulls(subset=["unique_id", "ds"])
    )

    if drop_interpolated and has_event_col:
        lf = lf.filter(pl.col("event_type") != "Interpolated")

    if test_split and has_split_col:
        lf = lf.filter(pl.col("split") == test_split)
    elif test_split and not has_split_col:
        typer.echo(
            f"Warning: --test-split='{test_split}' requested but the test CSV has no "
            f"'{COL_SPLIT}' column — using all rows.",
            err=True,
        )

    df = lf.collect()

    # GlucoseWindowDataset expects a 'study_group' column
    if "study_group" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("study_group"))

    typer.echo(f"  Loaded {len(df):,} test rows from {csv_path.name}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    registry_dir: Optional[Path] = typer.Option(
        None,
        "--registry-dir",
        help=(
            "Directory containing _analysis_registry.csv (e.g. "
            "marked_runs/glumind/ai_ready_plus_type1). "
            "Auto-selects the run with the lowest val_mae."
        ),
    ),
    run_dir: Optional[Path] = typer.Option(
        None,
        "--run-dir",
        help=(
            "Path to a specific run step directory holding tuning_meta.json and "
            "best_model.pt. Takes precedence over --registry-dir."
        ),
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help=(
            "Path to a specific .pt weights file. Requires --run-dir for the "
            "architecture metadata."
        ),
    ),
    test_csv: Path = typer.Option(
        ...,
        "--test-csv",
        help="CSV file to evaluate on. All rows are used unless --test-split is given.",
    ),
    train_csv: Optional[Path] = typer.Option(
        None,
        "--train-csv",
        help=(
            "CSV used to fit the scalers (glucose / HR / steps MinMaxScaler). "
            "Defaults to the CSV stored in tuning_meta.json."
        ),
    ),
    test_split: Optional[str] = typer.Option(
        None,
        "--test-split",
        help=(
            "If set, filter test CSV by 'Recommended Split' == this value before "
            "evaluating (e.g. 'test' or 'val')."
        ),
    ),
    glucose_only: bool = typer.Option(
        False,
        "--glucose-only",
        help="Zero out HR and step-count features before inference.",
    ),
    default_value: str = typer.Option(
        "zero",
        help=(
            "Replacement strategy for non-glucose features when --glucose-only is set. "
            "Choices: zero, mean, median."
        ),
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="DataLoader batch size (default: from tuning_meta.json).",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    ),
) -> None:
    # -----------------------------------------------------------------------
    # 1. Resolve run directory and metadata
    # -----------------------------------------------------------------------
    if run_dir is not None:
        resolved_run_dir = run_dir
        typer.echo(f"Using run directory: {resolved_run_dir}")
    elif registry_dir is not None:
        resolved_run_dir, _ = _find_best_run_dir(registry_dir)
        typer.echo(f"Resolved run directory: {resolved_run_dir}")
    else:
        typer.echo(
            "Error: Provide at least one of --registry-dir or --run-dir.", err=True
        )
        raise typer.Exit(1)

    if not resolved_run_dir.exists():
        typer.echo(f"Error: Run directory does not exist: {resolved_run_dir}", err=True)
        raise typer.Exit(1)

    meta = _load_meta(resolved_run_dir)

    # -----------------------------------------------------------------------
    # 2. Resolve checkpoint path
    # -----------------------------------------------------------------------
    ckpt_path = _resolve_checkpoint(resolved_run_dir, checkpoint)
    typer.echo(f"Checkpoint : {ckpt_path}")

    # -----------------------------------------------------------------------
    # 3. Load training data for scaler fitting
    # -----------------------------------------------------------------------
    scaler_csv = train_csv
    if scaler_csv is None:
        scaler_csv = Path(meta["csv"])
        if not scaler_csv.exists():
            scaler_csv = project_root / meta["csv"]

    typer.echo(f"Fitting scalers from: {scaler_csv}")
    train_df_raw, val_df_raw, test_df_raw = load_splits_streaming(
        scaler_csv,
        unique_id_choice=meta.get("unique_id", "sequence_id"),
        drop_interpolated=meta.get("drop_interpolated", False),
    )

    split_scheme = meta.get("split_scheme", "classic")
    train_df_raw, val_df_raw, _ = apply_split_scheme(
        train_df_raw, val_df_raw, test_df_raw, split_scheme
    )
    train_df_for_scalers = impute_and_sort(train_df_raw)

    train_ds = GlucoseWindowDataset(
        train_df_for_scalers,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        fit_scalers=True,
    )
    typer.echo(
        f"Scalers fitted on {len(train_df_for_scalers):,} training rows, "
        f"{len(train_ds):,} windows."
    )

    # -----------------------------------------------------------------------
    # 4. Load test data
    # -----------------------------------------------------------------------
    if not test_csv.exists():
        typer.echo(f"Error: Test CSV not found: {test_csv}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading test data from: {test_csv}")
    test_df = _load_test_csv(
        test_csv,
        unique_id_col=meta.get("unique_id", "sequence_id"),
        drop_interpolated=meta.get("drop_interpolated", False),
        test_split=test_split,
    )
    test_df = impute_and_sort(test_df)

    if test_df.is_empty():
        typer.echo("Error: Test dataframe is empty after loading/filtering.", err=True)
        raise typer.Exit(1)

    eval_ds = GlucoseWindowDataset(
        test_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        scaler_glucose=train_ds.scaler_glucose,
        scaler_hr=train_ds.scaler_hr,
        scaler_steps=train_ds.scaler_steps,
        fit_scalers=False,
    )

    if len(eval_ds) == 0:
        typer.echo(
            f"Error: No windows could be built from the test data. "
            f"Each series needs at least {meta['input_steps'] + meta['horizon']} rows.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Test windows: {len(eval_ds):,}")

    resolved_batch_size = batch_size or meta.get("batch_size", 4096)
    eval_loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)

    # -----------------------------------------------------------------------
    # 5. Apply glucose-only mode
    # -----------------------------------------------------------------------
    if glucose_only:
        if default_value not in ("zero", "mean", "median"):
            typer.echo(
                f"Error: --default-value must be one of zero, mean, median (got '{default_value}').",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"Glucose-only mode (default-value={default_value})...")
        hr_replace = 0.0
        steps_replace = 0.0

        if default_value == "mean":
            hr_replace = float(train_ds.scaler_hr.transform([[train_df_for_scalers["hr"].mean()]])[0, 0])
            steps_replace = float(train_ds.scaler_steps.transform([[train_df_for_scalers["steps"].mean()]])[0, 0])
        elif default_value == "median":
            hr_replace = float(train_ds.scaler_hr.transform([[train_df_for_scalers["hr"].median()]])[0, 0])
            steps_replace = float(train_ds.scaler_steps.transform([[train_df_for_scalers["steps"].median()]])[0, 0])

        typer.echo(f"  HR={hr_replace:.4f}, Steps={steps_replace:.4f} (scaled)")
        for i in range(len(eval_ds._series_h)):
            eval_ds._series_h[i][:] = hr_replace
            eval_ds._series_s[i][:] = steps_replace

    # -----------------------------------------------------------------------
    # 6. Load model
    # -----------------------------------------------------------------------
    typer.echo(f"Loading model from {ckpt_path}...")
    model = GluMindModel(
        n_time_steps=meta["input_steps"],
        n_features=3,
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        ff_units=meta["ff_units"],
        n_blocks=meta["n_blocks"],
        prediction_horizon=meta["horizon"],
        dropout=meta.get("dropout", 0.1),
    )

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compile_prefix(state)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # -----------------------------------------------------------------------
    # 7. Run inference
    # -----------------------------------------------------------------------
    typer.echo("Running inference...")
    loss_fn = torch.nn.MSELoss()
    _, y_true_scaled, y_pred_scaled = evaluate(
        model, eval_loader, loss_fn, torch.device(device)
    )

    y_true = train_ds.scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = train_ds.scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()

    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)

    typer.echo("\n" + "=" * 50)
    typer.echo(f"EVALUATION RESULTS")
    typer.echo(f"  Test CSV  : {test_csv}")
    typer.echo(f"  Checkpoint: {ckpt_path}")
    typer.echo(f"  Windows   : {len(eval_ds):,}")
    typer.echo("-" * 50)
    typer.echo(f"  MAE : {mae:.4f}")
    typer.echo(f"  RMSE: {rmse:.4f}")
    typer.echo(f"  MARD: {mard:.4f}%")
    typer.echo("=" * 50)


if __name__ == "__main__":
    app()
