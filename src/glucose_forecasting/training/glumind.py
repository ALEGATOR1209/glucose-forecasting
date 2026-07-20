"""Reusable GluMind training and evaluation core.

This module intentionally contains no command-line parsing. The legacy
``scripts.glumind.train_glumind`` module owns argparse orchestration and
re-exports these functions for existing callers.
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from glucose_forecasting.common.checkpoint import (
    load_full_checkpoint as _common_load_full_checkpoint,
    read_checkpoint_meta,
    save_full_checkpoint as _common_save_full_checkpoint,
    update_latest_symlink,
)
from glucose_forecasting.common.data_loading import (
    STUDY_GROUP_ALIASES,
    STUDY_GROUP_ORDER,
    limit_series,
    normalize_study_group_label,
    normalize_study_groups_column,
    resolve_num_workers,
)
from glucose_forecasting.common.metrics import mae_rmse_mard
from glucose_forecasting.data.glumind import (
    COL_EVENT,
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
    build_datasets,
    impute_and_sort,
    load_splits_streaming,
)
from glucose_forecasting.models.glumind import GluMindModel


def train_one_epoch(
    model: GluMindModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            task_loss = loss_fn(pred, y)
            if teacher is not None and lwf_lambda > 0:
                with torch.no_grad():
                    teacher_pred = teacher(x)
                distill_loss = loss_fn(pred, teacher_pred)
                loss = (1 - lwf_lambda) * task_loss + lwf_lambda * distill_loss
            else:
                loss = task_loss
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: GluMindModel,
    loader: DataLoader[Any],
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Evaluate a model and return average loss, targets, and predictions."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true: list[np.ndarray[Any, Any]] = []
    all_pred: list[np.ndarray[Any, Any]] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return total_loss / max(n_batches, 1), true_arr, pred_arr


def compute_and_print_metrics(
    true_arr: np.ndarray[Any, Any],
    pred_arr: np.ndarray[Any, Any],
    scaler_glucose: MinMaxScaler,
    split_name: str,
    run_dir: Path,
    dataset: GlucoseWindowDataset | None = None,
) -> tuple[float, float, float]:
    """Inverse-transform, report, and persist GluMind metrics."""
    true_values = scaler_glucose.inverse_transform(true_arr.ravel().reshape(-1, 1)).ravel()
    pred_values = scaler_glucose.inverse_transform(pred_arr.ravel().reshape(-1, 1)).ravel()
    mae, rmse, mard = mae_rmse_mard(true_values, pred_values)
    print(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MARD: {mard:.2f}%")
    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )
    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups = np.array(dataset.study_groups)
        rows: list[dict[str, Any]] = []
        for group in sorted(set(groups)):
            mask = groups == group
            if not mask.any():
                continue
            group_true = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            group_pred = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            group_mae, group_rmse, group_mard = mae_rmse_mard(group_true, group_pred)
            rows.append(
                {
                    "study_group": group,
                    "n_windows": int(mask.sum()),
                    "mae": group_mae,
                    "rmse": group_rmse,
                    "mard": group_mard,
                }
            )
        by_group = pl.DataFrame(rows).sort("mae")
        print(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
        print(by_group)
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")
    return mae, rmse, mard


def save_full_checkpoint(
    path: Path,
    model: GluMindModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    args: Any,
) -> None:
    """Save model, optimizer, scheduler, and CLI-derived configuration."""
    _common_save_full_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch,
        best_val_loss,
        vars(args),
        config_key="args",
        stringify_paths=True,
    )


def load_full_checkpoint(
    path: Path,
    model: GluMindModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """Load a full checkpoint and return its epoch and best validation loss."""
    return _common_load_full_checkpoint(path, model, optimizer, scheduler, device)


def train_loop(
    model: GluMindModel,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any] | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
    verbose_every: int = 10,
    ckpt_every_n_epochs: int = 0,
    ckpt_eval_callback: Callable[[GluMindModel, int, Path], None] | None = None,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    args: Any = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    val_every_n_epochs: int = 1,
) -> GluMindModel:
    """Run training with validation, checkpoints, and early stopping."""
    wait = 0
    best_epoch = start_epoch - 1
    total_time = 0.0
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, teacher, lwf_lambda, use_amp, amp_dtype, scaler
        )
        val_loss_str = "SKIP"
        should_eval_val = val_loader is not None and (
            epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs
        )
        if should_eval_val:
            val_loss, _, _ = evaluate(model, val_loader, loss_fn, device, use_amp, amp_dtype)
            val_loss_str = f"{val_loss:.6f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with (run_dir / "best_info.json").open("w") as file:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss}, file)
                print(f"  ★ New best at epoch {epoch} (val_loss={val_loss:.6f})")
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    print(f"  Early stopping at epoch {epoch} (patience={patience})")
                    break
        elif val_loader is None:
            val_loss_str = "N/A"
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with (run_dir / "best_info.json").open("w") as file:
                    json.dump({"epoch": epoch, "train_loss": float(train_loss)}, file)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - started
        total_time += elapsed
        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            eta = timedelta(seconds=int(total_time / (epoch - start_epoch + 1) * (epochs - epoch)))
            print(
                f"  Epoch {epoch:4d}/{epochs} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss_str} | {elapsed:.1f}s/epoch | ETA: {eta}"
            )
        if ckpt_every_n_epochs > 0 and epoch % ckpt_every_n_epochs == 0:
            checkpoint_dir = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_full_checkpoint(
                checkpoint_dir / "checkpoint.pt", model, optimizer, scheduler, epoch, best_val_loss, args
            )
            print(f"  [Checkpoint] Saved at epoch {epoch} → {checkpoint_dir}")
            if ckpt_eval_callback is not None:
                ckpt_eval_callback(model, epoch, checkpoint_dir)
    save_full_checkpoint(run_dir / "last_checkpoint.pt", model, optimizer, scheduler, epoch, best_val_loss, args)
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    print(f"\n  Summary: best_model.pt = epoch {best_epoch} | last_model.pt = epoch {epoch}")
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model


def make_model(args: Any, device: torch.device) -> GluMindModel:
    """Construct a GluMind model without changing checkpoint key names."""
    model = GluMindModel(
        n_time_steps=args.input_steps,
        n_features=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_units=args.ff_units,
        n_blocks=args.n_blocks,
        prediction_horizon=args.horizon,
        dropout=args.dropout,
    ).to(device)
    if device.type == "cuda" and args.compile_mode != "none":
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as error:
            print(f"Warning: torch.compile failed, using eager mode ({error})")
    return model


def make_optimizer_and_scheduler(
    model: GluMindModel, args: Any
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build the optimizer and schedule used by legacy GluMind training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    return optimizer, scheduler


def run_train_and_eval(
    model: GluMindModel,
    train_ds: GlucoseWindowDataset,
    val_ds: GlucoseWindowDataset | None,
    test_ds: GlucoseWindowDataset | None,
    args: Any,
    device: torch.device,
    run_name: str,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
) -> GluMindModel:
    """Train a prepared dataset triplet and save all run artifacts."""
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    meta = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    meta.update(
        {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds) if val_ds else 0,
            "test_samples": len(test_ds) if test_ds else 0,
            "start_time": datetime.now().isoformat(),
        }
    )
    with (run_dir / "tuning_meta.json").open("w") as file:
        json.dump(meta, file, indent=2)
    update_latest_symlink(run_dir, args.out_dir)
    num_workers = resolve_num_workers(args.num_workers, device)
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if test_ds is not None and len(test_ds) > 0
        else None
    )
    optimizer, scheduler = make_optimizer_and_scheduler(model, args)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and args.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_loss = load_full_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch += 1
    print(f"\n{'=' * 60}")
    print(
        f"Training: {len(train_ds):,} windows | Val: {len(val_ds) if val_ds else 0:,} | "
        f"Test: {len(test_ds) if test_ds else 0:,} | Params: {sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    if start_epoch > 1:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'=' * 60}")

    def checkpoint_evaluation(current_model: GluMindModel, epoch: int, checkpoint_dir: Path) -> None:
        if val_loader is not None:
            _, actual, predicted = evaluate(current_model, val_loader, loss_fn, device, use_amp, amp_dtype)
            compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose, f"val_epoch{epoch:04d}", checkpoint_dir, val_ds)
        if test_loader is not None:
            _, actual, predicted = evaluate(current_model, test_loader, loss_fn, device, use_amp, amp_dtype)
            compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose, f"test_epoch{epoch:04d}", checkpoint_dir, test_ds)

    model = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args.epochs,
        args.patience,
        run_dir,
        teacher,
        lwf_lambda,
        args.log_every,
        args.ckpt_every_n_epochs,
        checkpoint_evaluation,
        start_epoch,
        best_val_loss,
        args,
        use_amp,
        amp_dtype,
        scaler,
        args.val_every_n_epochs,
    )
    if val_loader is not None:
        _, actual, predicted = evaluate(model, val_loader, loss_fn, device, use_amp, amp_dtype)
        compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose, "val", run_dir, val_ds)
    if test_loader is not None:
        _, actual, predicted = evaluate(model, test_loader, loss_fn, device, use_amp, amp_dtype)
        compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose, "test", run_dir, test_ds)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)
    return model


def mode_global(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, args: Any, device: torch.device
) -> None:
    """Train one model on all data."""
    print("\n=== MODE: GLOBAL ===")
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"glumind_global_h{args.horizon}_{timestamp}"
    run_dir = args.out_dir / run_name
    update_latest_symlink(run_dir, args.out_dir)
    print(f"--> Training global model (dir={run_dir})")
    run_train_and_eval(make_model(args, device), train_ds, val_ds, test_ds, args, device, run_name)


def mode_per_group(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, args: Any, device: torch.device
) -> None:
    """Train a distinct model per study group."""
    print("\n=== MODE: PER_GROUP ===")
    present = set(train_df["study_group"].unique().to_list())
    for group in (group for group in STUDY_GROUP_ORDER if group in present):
        print(f"\n--- Group: {group} ---")
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        test = test_df.filter(pl.col("study_group") == group) if not test_df.is_empty() else test_df
        if train.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(train, val, test, args)
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_group_{safe_name}_h{args.horizon}_{datetime.now():%Y%m%d_%H%M%S}"
        run_train_and_eval(make_model(args, device), train_ds, val_ds, test_ds, args, device, run_name)


def mode_cohort_wise(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, args: Any, device: torch.device
) -> None:
    """Train independently on each cohort and evaluate against all test data."""
    print("\n=== MODE: COHORT_WISE ===")
    present = set(train_df["study_group"].unique().to_list())
    for group in (group for group in STUDY_GROUP_ORDER if group in present):
        print(f"\n--- Cohort: {group} ---")
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        if train.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(train, val, test_df, args)
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_cohort_{safe_name}_h{args.horizon}_{datetime.now():%Y%m%d_%H%M%S}"
        run_train_and_eval(make_model(args, device), train_ds, val_ds, test_ds, args, device, run_name)


def mode_continual(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, args: Any, device: torch.device
) -> None:
    """Train sequentially across cohorts using Learning without Forgetting."""
    print("\n=== MODE: CONTINUAL (LwF) ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [group for group in STUDY_GROUP_ORDER if group in present]
    if args.continual_order == "reverse":
        groups = list(reversed(groups))
    print(f"Continual group order: {groups}")
    parent = f"glumind_continual_h{args.horizon}_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Continual parent run dir: {args.out_dir / parent}")
    model = make_model(args, device)
    teacher: GluMindModel | None = None
    global_ds = GlucoseWindowDataset(train_df, args.input_steps, args.horizon, fit_scalers=True)
    scalers = {
        "scaler_glucose": global_ds.scaler_glucose,
        "scaler_hr": global_ds.scaler_hr,
        "scaler_steps": global_ds.scaler_steps,
    }
    for index, group in enumerate(groups):
        print(f"\n--- Continual step {index + 1}/{len(groups)}: {group} ---")
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df if args.continual_val_scope == "all_groups" else val_df.filter(pl.col("study_group") == group)
        if train.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue
        train_ds = GlucoseWindowDataset(train, args.input_steps, args.horizon, **scalers)
        val_ds = GlucoseWindowDataset(val, args.input_steps, args.horizon, **scalers) if not val.is_empty() else None
        test_ds = GlucoseWindowDataset(test_df, args.input_steps, args.horizon, **scalers) if not test_df.is_empty() else None
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"{parent}/step_{index + 1:02d}_{safe_name}_{datetime.now():%Y%m%d_%H%M%S}"
        model = run_train_and_eval(
            model, train_ds, val_ds, test_ds, args, device, run_name, teacher, args.lwf_lambda if teacher else 0.0
        )
        teacher = copy.deepcopy(model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"  Saved teacher snapshot after {group}")


__all__ = [
    "COL_EVENT", "COL_GLU", "COL_GROUP", "COL_HR", "COL_SEQ", "COL_SPLIT", "COL_STEPS", "COL_TS",
    "COL_USER", "STUDY_GROUP_ALIASES", "STUDY_GROUP_ORDER", "TS_FORMAT", "GluMindModel",
    "GlucoseWindowDataset", "apply_split_scheme", "build_datasets", "compute_and_print_metrics",
    "evaluate", "impute_and_sort", "limit_series", "load_full_checkpoint", "load_splits_streaming",
    "mae_rmse_mard", "make_model", "make_optimizer_and_scheduler", "mode_cohort_wise",
    "mode_continual", "mode_global", "mode_per_group", "normalize_study_group_label",
    "normalize_study_groups_column", "read_checkpoint_meta", "resolve_num_workers", "run_train_and_eval",
    "save_full_checkpoint", "train_loop", "train_one_epoch", "update_latest_symlink",
]
