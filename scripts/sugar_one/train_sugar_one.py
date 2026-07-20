#!/usr/bin/env python3
"""Legacy SugarOne Typer CLI and compatibility re-exports.

Reusable training logic is implemented in :mod:`glucose_forecasting.training.sugar_one`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer

from glucose_forecasting.data.sugar_one import (
    COL_BASAL,
    COL_BOLUS,
    COL_CARB,
    COL_EVENT,
    COL_GLU,
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS,
    COL_USER,
    N_FEATURES,
    TS_FORMAT,
    SugarOneWindowDataset,
    apply_split_scheme,
    build_datasets,
    impute_and_sort,
    load_splits_streaming,
)
from glucose_forecasting.training.sugar_one import (
    STUDY_GROUP_ALIASES,
    STUDY_GROUP_ORDER,
    _mode_cohort_wise,
    _mode_continual,
    _mode_global,
    _mode_per_group,
    _model_kwargs,
    compute_and_print_metrics,
    evaluate,
    limit_series,
    load_full_checkpoint,
    mae_rmse_mard,
    make_model,
    make_optimizer_and_scheduler,
    normalize_study_group_label,
    normalize_study_groups_column,
    read_checkpoint_meta,
    resolve_num_workers,
    run_train_and_eval,
    save_full_checkpoint,
    train_loop,
    train_one_epoch,
    update_latest_symlink,
)
from glucose_forecasting.models.sugar_one import SugarOneModel

app = typer.Typer(
    name="train_sugar_one",
    add_completion=False,
    help="SugarOne: Parallel-Attention Transformer with Insulin & Carb covariates.",
)


@app.command()
def main(
    csv: Path = typer.Option(..., help="Path to loop_ai_ready_joined_loop_columns.csv."),
    unique_id: str = typer.Option("sequence_id", help="sequence_id or user_id."),
    max_train_series: int = typer.Option(0, help="Limit training series (0 = all)."),
    max_eval_series: int = typer.Option(0, help="Limit evaluation series (0 = all)."),
    drop_interpolated: bool = typer.Option(False, help="Drop Interpolated rows."),
    study_groups: str = typer.Option("", help="Comma-separated Study Group filter (empty = all)."),
    split_scheme: str = typer.Option("classic", help="classic | trainval_test_as_val."),
    mode: str = typer.Option("global", help="global | per_group | cohort_wise | continual."),
    horizon: int = typer.Option(12, help="Prediction horizon steps (12 = 60 min at 5-min freq)."),
    input_steps: int = typer.Option(80, help="Input window steps (80 = 400 min)."),
    d_model: int = typer.Option(32, help="Embedding dimension."),
    n_heads: int = typer.Option(4, help="Attention heads."),
    n_blocks: int = typer.Option(3, help="Parallel transformer blocks."),
    ff_units: int = typer.Option(128, help="FFN hidden units."),
    dropout: float = typer.Option(0.1, help="Dropout rate."),
    epochs: int = typer.Option(200, help="Training epochs."),
    batch_size: int = typer.Option(64, help="Batch size."),
    precision: str = typer.Option("bf16", help="fp32 | bf16 | fp16."),
    compile_mode: str = typer.Option("none", help="none | default | reduce-overhead | max-autotune."),
    disable_tf32: bool = typer.Option(False, help="Disable TF32 on CUDA."),
    num_workers: int = typer.Option(-1, help="DataLoader workers (-1 = auto)."),
    prefetch_factor: int = typer.Option(4, help="DataLoader prefetch factor."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    patience: int = typer.Option(20, help="Early stopping patience (0 = disabled)."),
    log_every: int = typer.Option(10, help="Print every N epochs."),
    ckpt_every_n_epochs: int = typer.Option(0, help="Save checkpoint every N epochs (0 = off)."),
    val_every_n_epochs: int = typer.Option(1, help="Run validation every N epochs."),
    resume_from: str = typer.Option("", help="Path to checkpoint.pt to resume from."),
    lwf_lambda: float = typer.Option(0.5, help="LwF distillation weight (continual mode)."),
    continual_order: str = typer.Option("default", help="default | reverse."),
    continual_val_scope: str = typer.Option("current_group", help="current_group | all_groups."),
    device_name: str = typer.Option("cuda", "--device", help="cpu | mps | cuda."),
    seed: int = typer.Option(42, help="Random seed."),
    out_dir: Path = typer.Option(Path("data/output/runs/sugar_one"), help="Output directory."),
) -> None:
    """Train SugarOne on insulin + carb covariate data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            typer.echo("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    typer.echo(f"Device: {device}")

    train_df, val_df, test_df = load_splits_streaming(csv, unique_id, drop_interpolated)
    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)
    if study_groups:
        selected = [normalize_study_group_label(group.strip()) for group in study_groups.split(",") if group.strip()]
        train_df = train_df.filter(pl.col("study_group").is_in(selected))
        val_df = val_df.filter(pl.col("study_group").is_in(selected))
        test_df = test_df.filter(pl.col("study_group").is_in(selected))
    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)
    if max_train_series > 0:
        train_df = limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df, test_df = limit_series(val_df, max_eval_series), limit_series(test_df, max_eval_series)
    train_df, val_df, test_df = (
        impute_and_sort(train_df),
        impute_and_sort(val_df),
        impute_and_sort(test_df),
    )
    cfg = {
        "csv": str(csv), "unique_id": unique_id, "drop_interpolated": drop_interpolated,
        "study_groups": study_groups, "split_scheme": split_scheme, "mode": mode,
        "horizon": horizon, "input_steps": input_steps, "d_model": d_model, "n_heads": n_heads,
        "n_blocks": n_blocks, "ff_units": ff_units, "dropout": dropout, "epochs": epochs,
        "batch_size": batch_size, "precision": precision, "compile_mode": compile_mode,
        "disable_tf32": disable_tf32, "num_workers": num_workers, "prefetch_factor": prefetch_factor,
        "lr": lr, "weight_decay": weight_decay, "patience": patience, "log_every": log_every,
        "ckpt_every_n_epochs": ckpt_every_n_epochs, "val_every_n_epochs": val_every_n_epochs,
        "resume_from": resume_from, "batch_log_every": 0, "eval_batch_log_every": 0,
        "lwf_lambda": lwf_lambda, "continual_order": continual_order,
        "continual_val_scope": continual_val_scope, "device": device_name, "seed": seed,
        "out_dir": str(out_dir),
    }
    mode_functions = {
        "global": _mode_global, "per_group": _mode_per_group,
        "cohort_wise": _mode_cohort_wise, "continual": _mode_continual,
    }
    if mode not in mode_functions:
        raise typer.BadParameter(f"Unknown mode: {mode!r}. Choose from: {list(mode_functions)}")
    mode_functions[mode](train_df, val_df, test_df, cfg, device, out_dir)
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
