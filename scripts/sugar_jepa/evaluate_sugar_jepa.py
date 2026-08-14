#!/usr/bin/env python3
"""
Standalone evaluation CLI for SugarJepa run directories.

Not wired into the unified `evaluate-model` CLI (scripts/sugar_one/evaluate_model.py)
— this is a deliberately separate script so the SugarJepa work doesn't touch any
existing, shared evaluation code. Reuses the model-reconstruction pattern from
evaluate_model.py (scripts/common/registry.py) and the training script's
dataset/eval/metrics functions.

Evaluates `SugarJepaModel2`, whose JEPA branch reads its glucose from the
trailing `jepa_window` steps of `x[..., 0]`. The dataset contract is SugarOne's
plain `(x, y)` — one window of `max(input_steps, jepa_window)` steps, no fifth
scaler, no extra tensor in the eval loop — so the only thing the two windows
change here is how long that window is.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import typer

from scripts.common.data_loading import impute_and_sort as _common_impute_and_sort
from scripts.common.registry import load_run_meta, resolve_checkpoint, resolve_csv_path
from scripts.common.checkpoint import strip_compile_prefix
from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel2
from scripts.sugar_jepa.train_sugar_jepa import (
    SugarJepaWindowDataset,
    _mix_weights,
    compute_and_print_metrics,
    evaluate,
    load_splits_streaming,
)

import torch.nn as nn
from torch.utils.data import DataLoader

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir", help="Run directory with tuning_meta.json/config.json + best_model.pt."),
    test_csv: Path = typer.Option(..., "--test-csv", help="CSV to evaluate on."),
    train_csv: Path | None = typer.Option(
        None, "--train-csv", help="CSV to fit MinMax/z-score scalers on (default: csv from the run's config)."
    ),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Explicit .pt weights (default: best_model.pt)."),
    test_split: str = typer.Option("test", "--test-split", help="Recommended Split value to filter to; '' disables filtering."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Default: from the run's config."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device for inference."),
) -> None:
    run_dir = run_dir.resolve()
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_run_meta(run_dir)
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)

    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Checkpoint   : {ckpt_path}")

    train_path = resolve_csv_path(train_csv or cfg["csv"], project_root)
    test_path = resolve_csv_path(test_csv, project_root)
    typer.echo(f"Train CSV (scaler fit): {train_path}")
    typer.echo(f"Test CSV  (evaluation): {test_path}")

    dev = torch.device(device)
    resolved_batch_size = batch_size or cfg.get("batch_size", 256)

    # The JEPA hyperparameters must reproduce the checkpoint's architecture;
    # strict=True below is what turns a mismatch into an error instead of
    # silently-wrong weights.
    model = SugarJepaModel2(
        n_time_steps=cfg["input_steps"],
        n_features=4,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_units=cfg["ff_units"],
        n_blocks=cfg["n_blocks"],
        prediction_horizon=cfg["horizon"],
        dropout=cfg["dropout"],
        # Older runs predate the separate window and used input_steps for both.
        jepa_window=cfg.get("jepa_window", cfg["input_steps"]),
        jepa_patch_size=cfg.get("jepa_patch_size", 8),
        jepa_embed_dim=cfg.get("jepa_embed_dim", 96),
        jepa_layers=cfg.get("jepa_layers", 3),
        jepa_heads=cfg.get("jepa_heads", 6),
        jepa_norm=cfg.get("jepa_norm", "instance"),
    ).to(dev)

    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    state = strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    typer.echo(f"Mix weights (mean over blocks): {_mix_weights(model)}")

    # Fit scalers on the training CSV's train split (matches train_sugar_jepa.py).
    train_df, _, _ = load_splits_streaming(train_path, cfg["unique_id"], cfg["drop_interpolated"])
    train_df = _common_impute_and_sort(
        train_df, ffill_bfill_columns=["glucose", "basal"], zero_fill_columns=["bolus", "carbs"],
    )
    # The window the dataset must emit is the model's, not the backbone's.
    lookback = max(cfg["input_steps"], cfg.get("jepa_window", cfg["input_steps"]))
    train_ds = SugarJepaWindowDataset(
        train_df, lookback, cfg["horizon"], fit_scalers=True,
    )

    test_train_df, test_val_df, test_test_df = load_splits_streaming(
        test_path, cfg["unique_id"], cfg["drop_interpolated"]
    )
    if test_split == "":
        eval_df = pl.concat([test_train_df, test_val_df, test_test_df])
    elif test_split == "train":
        eval_df = test_train_df
    elif test_split == "val":
        eval_df = test_val_df
    else:
        eval_df = test_test_df
    if eval_df is None or eval_df.is_empty():
        typer.echo(f"Error: no rows for --test-split={test_split!r} in {test_path}", err=True)
        raise typer.Exit(1)
    eval_df = _common_impute_and_sort(
        eval_df, ffill_bfill_columns=["glucose", "basal"], zero_fill_columns=["bolus", "carbs"],
    )

    eval_ds = SugarJepaWindowDataset(
        eval_df, lookback, cfg["horizon"],
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
    )
    typer.echo(f"Evaluation windows: {len(eval_ds):,}")

    loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    _, true_arr, pred_arr = evaluate(model, loader, loss_fn, dev, split_label=test_split or "all")
    compute_and_print_metrics(true_arr, pred_arr, train_ds.scaler_glucose, test_split or "all", run_dir, eval_ds)


if __name__ == "__main__":
    app()
