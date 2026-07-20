#!/usr/bin/env python3
"""
Inference script to reproduce results from a GluMind run directory.

Mode is auto-detected from tuning_meta.json split_scheme when --mode=auto (default).
Saved metrics (test_metrics_overall.csv / val_metrics_overall.csv) are compared
directly from the run directory — no _analysis_registry.csv required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import csv

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from glucose_forecasting.data.glumind import (
    GlucoseWindowDataset,
    apply_split_scheme,
    impute_and_sort,
    load_splits_streaming,
)
from glucose_forecasting.models.glumind import GluMindModel
from glucose_forecasting.training.glumind import (
    mae_rmse_mard,
    evaluate,
)

_SPLIT_SCHEME_TO_MODE: dict[str, str] = {
    "classic": "test",
    "trainval_test_as_val": "val_as_test",
}

app = typer.Typer(add_completion=False)


def _resolve_mode(run_dir: Path, mode: str, meta: dict) -> str:
    """Resolve 'auto' mode from split_scheme stored in tuning_meta.json."""
    if mode != "auto":
        return mode
    split_scheme = meta.get("split_scheme", "classic")
    resolved = _SPLIT_SCHEME_TO_MODE.get(split_scheme)
    if resolved is None:
        typer.echo(
            f"Warning: unknown split_scheme '{split_scheme}' in tuning_meta.json, "
            "falling back to 'test'.",
            err=True,
        )
        return "test"
    typer.echo(f"Auto-detected mode='{resolved}' from split_scheme='{split_scheme}'")
    return resolved


def _load_saved_metrics(run_dir: Path, mode: str) -> dict[str, float] | None:
    """Load pre-saved metrics from the run directory based on mode."""
    csv_name = "test_metrics_overall.csv" if mode == "test" else "val_metrics_overall.csv"
    metrics_path = run_dir / csv_name
    if not metrics_path.exists():
        return None
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        return None
    return {"mae": float(row["mae"]), "rmse": float(row["rmse"]), "mard": float(row["mard"])}


@app.command()
def main(
    run_dir: Path = typer.Option(..., help="Path to the run directory in marked_runs"),
    mode: str = typer.Option(
        "auto",
        help=(
            "Which split to evaluate on. 'auto' derives the mode from split_scheme "
            "in tuning_meta.json ('classic' -> 'test', 'trainval_test_as_val' -> "
            "'val_as_test'). Explicit values: 'test' or 'val_as_test'."
        ),
    ),
    glucose_only: bool = typer.Option(False, "--glucose-only", help="Use only glucose values as input, zeroing other features."),
    default_value: str = typer.Option(
        "zero",
        help="Replacement strategy for non-glucose features when --glucose-only is set. Choices: zero, mean, median.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on.",
    ),
) -> None:
    if not run_dir.exists():
        typer.echo(f"Error: Run directory {run_dir} does not exist.", err=True)
        raise typer.Exit(1)

    if mode not in ("auto", "test", "val_as_test"):
        typer.echo(f"Error: --mode must be one of 'auto', 'test', 'val_as_test'.", err=True)
        raise typer.Exit(1)

    # 1. Load metadata
    meta_path = run_dir / "tuning_meta.json"
    if not meta_path.exists():
        meta_path = run_dir / "config.json"
    if not meta_path.exists():
        typer.echo(f"Error: No metadata file found in {run_dir}", err=True)
        raise typer.Exit(1)

    with open(meta_path) as f:
        meta: dict = json.load(f)

    # 2. Resolve mode
    resolved_mode = _resolve_mode(run_dir, mode, meta)

    # 3. Load dataset
    csv_path = Path(meta["csv"])
    if not csv_path.exists():
        csv_path = project_root / meta["csv"]

    typer.echo(f"Loading dataset from {csv_path}...")
    train_df, val_df, test_df = load_splits_streaming(
        csv_path,
        unique_id_choice=meta.get("unique_id", "sequence_id"),
        drop_interpolated=meta.get("drop_interpolated", False),
    )

    # 4. Apply split scheme
    split_scheme = meta.get("split_scheme", "classic")
    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)

    # 5. Impute and sort
    typer.echo("Preprocessing splits...")
    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    # 6. Build datasets and fit scalers on train set
    typer.echo("Building datasets and fitting scalers on train set...")
    train_ds = GlucoseWindowDataset(
        train_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        fit_scalers=True,
    )

    # 7. Select evaluation set
    eval_df = test_df if resolved_mode == "test" else val_df
    if eval_df.is_empty():
        typer.echo(f"Error: Resolved mode '{resolved_mode}' produced an empty dataset.", err=True)
        raise typer.Exit(1)

    eval_ds = GlucoseWindowDataset(
        eval_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        scaler_glucose=train_ds.scaler_glucose,
        scaler_hr=train_ds.scaler_hr,
        scaler_steps=train_ds.scaler_steps,
        fit_scalers=False,
    )

    eval_loader = DataLoader(eval_ds, batch_size=meta.get("batch_size", 4096), shuffle=False)

    # 8. Apply glucose-only mode if requested
    if glucose_only:
        typer.echo(f"Applying glucose-only mode with default-value={default_value}...")
        hr_replace = 0.0
        steps_replace = 0.0

        if default_value == "mean":
            hr_replace = train_ds.scaler_hr.transform([[train_df["hr"].mean()]])[0, 0]
            steps_replace = train_ds.scaler_steps.transform([[train_df["steps"].mean()]])[0, 0]
        elif default_value == "median":
            hr_replace = train_ds.scaler_hr.transform([[train_df["hr"].median()]])[0, 0]
            steps_replace = train_ds.scaler_steps.transform([[train_df["steps"].median()]])[0, 0]

        typer.echo(f"  HR={hr_replace:.4f}, Steps={steps_replace:.4f} (scaled)")
        for i in range(len(eval_ds.windows_x)):
            eval_ds.windows_x[i][:, 1] = hr_replace
            eval_ds.windows_x[i][:, 2] = steps_replace

    # 9. Load model
    best_model_path = run_dir / "best_model.pt"
    if not best_model_path.exists():
        best_model_path = run_dir / "last_model.pt"
    if not best_model_path.exists():
        typer.echo(f"Error: No model weights found in {run_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading model from {best_model_path}...")
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

    state = torch.load(best_model_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 10. Inference
    typer.echo("Running inference...")
    loss_fn = torch.nn.MSELoss()
    _, y_true_scaled, y_pred_scaled = evaluate(
        model, eval_loader, loss_fn, torch.device(device)
    )

    # 11. Inverse transform and compute metrics
    y_true = train_ds.scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = train_ds.scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()

    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)

    typer.echo("\n" + "=" * 40)
    typer.echo(f"REPRODUCED METRICS ({resolved_mode})")
    typer.echo(f"MAE : {mae:.4f}")
    typer.echo(f"RMSE: {rmse:.4f}")
    typer.echo(f"MARD: {mard:.4f}%")
    typer.echo("=" * 40)

    # 12. Compare with saved metrics from the run directory
    saved = _load_saved_metrics(run_dir, resolved_mode)
    if saved is not None:
        typer.echo("\nCOMPARISON WITH SAVED METRICS (from run directory):")
        typer.echo(f"Saved  MAE : {saved['mae']:.4f}")
        typer.echo(f"Saved  RMSE: {saved['rmse']:.4f}")
        typer.echo(f"Saved  MARD: {saved['mard']:.4f}%")
        typer.echo(f"\nDifferences (Reproduced - Saved):")
        typer.echo(f"  dMAE : {mae - saved['mae']:.6f}")
        typer.echo(f"  dRMSE: {rmse - saved['rmse']:.6f}")
        typer.echo(f"  dMARD: {mard - saved['mard']:.6f}%")
    else:
        csv_name = "test_metrics_overall.csv" if resolved_mode == "test" else "val_metrics_overall.csv"
        typer.echo(f"\nNote: No saved metrics found ({csv_name} missing from {run_dir})")


if __name__ == "__main__":
    app()
