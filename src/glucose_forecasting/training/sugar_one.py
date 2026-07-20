"""Reusable SugarOne training and evaluation core.

This module intentionally contains no command-line parsing.  The legacy
``scripts.sugar_one.train_sugar_one`` module owns the Typer interface and
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
    update_latest_symlink as _common_update_latest_symlink,
)
from glucose_forecasting.common.console import safe_echo
from glucose_forecasting.common.data_loading import (
    STUDY_GROUP_ALIASES,
    STUDY_GROUP_ORDER,
    limit_series,
    normalize_study_group_label,
    normalize_study_groups_column,
    resolve_num_workers,
)
from glucose_forecasting.common.metrics import mae_rmse_mard
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
from glucose_forecasting.models.sugar_one import SugarOneModel
from scripts.common.metrics_log import EpochMetricsWriter


def echo_plain(message: str) -> None:
    """Emit plain console output without CLI ownership."""
    safe_echo(message)


def train_one_epoch(
    model: SugarOneModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    teacher: SugarOneModel | None = None,
    lwf_lambda: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    batch_log_every: int = 0,
    log_interval_s: float = 0.0,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    n_batches_total = len(loader)
    started = last_log = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            task_loss = loss_fn(pred, y)
            if teacher is not None and lwf_lambda > 0:
                with torch.no_grad():
                    teacher_pred = teacher(x)
                loss = (1 - lwf_lambda) * task_loss + lwf_lambda * loss_fn(pred, teacher_pred)
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
        n_batches += 1
        total_loss += loss.item()
        now = time.perf_counter()
        should_log = (
            log_interval_s > 0
            and (n_batches == 1 or n_batches == n_batches_total or now - last_log >= log_interval_s)
        ) or (
            log_interval_s <= 0
            and batch_log_every > 0
            and (n_batches == 1 or n_batches % batch_log_every == 0 or n_batches == n_batches_total)
        )
        if should_log:
            last_log = now
            elapsed = now - started
            rate = n_batches / elapsed if elapsed > 0 else 0.0
            eta = (n_batches_total - n_batches) / rate if rate > 0 else 0.0
            echo_plain(
                f"  train epoch {epoch:3d} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{rate:.2f} batch/s | loss={total_loss / n_batches:.6f} | "
                f"epoch ETA {timedelta(seconds=int(eta))}"
            )
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: SugarOneModel,
    loader: DataLoader[Any],
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    batch_log_every: int = 0,
    log_interval_s: float = 0.0,
    split_label: str = "eval",
) -> tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    true_batches: list[np.ndarray[Any, Any]] = []
    pred_batches: list[np.ndarray[Any, Any]] = []
    n_batches_total = len(loader)
    started = last_log = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        true_batches.append(y.float().cpu().numpy())
        pred_batches.append(pred.float().cpu().numpy())
        now = time.perf_counter()
        should_log = (
            log_interval_s > 0
            and (n_batches == 1 or n_batches == n_batches_total or now - last_log >= log_interval_s)
        ) or (
            log_interval_s <= 0
            and batch_log_every > 0
            and (n_batches == 1 or n_batches % batch_log_every == 0 or n_batches == n_batches_total)
        )
        if should_log:
            last_log = now
            elapsed = now - started
            rate = n_batches / elapsed if elapsed > 0 else 0.0
            eta = (n_batches_total - n_batches) / rate if rate > 0 else 0.0
            echo_plain(
                f"  {split_label} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{rate:.2f} batch/s | ETA {timedelta(seconds=int(eta))}"
            )
    true_arr = np.concatenate(true_batches, axis=0) if true_batches else np.array([])
    pred_arr = np.concatenate(pred_batches, axis=0) if pred_batches else np.array([])
    return total_loss / max(n_batches, 1), true_arr, pred_arr


def compute_and_print_metrics(
    true_arr: np.ndarray[Any, Any],
    pred_arr: np.ndarray[Any, Any],
    scaler_glucose: MinMaxScaler,
    split_name: str,
    run_dir: Path,
    dataset: SugarOneWindowDataset | None = None,
) -> tuple[float, float, float]:
    true_values = scaler_glucose.inverse_transform(true_arr.ravel().reshape(-1, 1)).ravel()
    pred_values = scaler_glucose.inverse_transform(pred_arr.ravel().reshape(-1, 1)).ravel()
    mae, rmse, mard = mae_rmse_mard(true_values, pred_values)
    echo_plain(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    echo_plain(f"  MAE : {mae:.4f}\n  RMSE: {rmse:.4f}\n  MARD: {mard:.2f}%")
    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )
    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups = np.array(dataset.study_groups)
        rows: list[dict[str, Any]] = []
        for group in sorted(set(groups)):
            mask = groups == group
            group_true = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            group_pred = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            group_mae, group_rmse, group_mard = mae_rmse_mard(group_true, group_pred)
            rows.append({"study_group": group, "n_windows": int(mask.sum()), "mae": group_mae,
                         "rmse": group_rmse, "mard": group_mard})
        by_group = pl.DataFrame(rows).sort("mae")
        for row in by_group.iter_rows(named=True):
            echo_plain(f"  {row['study_group']}: n={row['n_windows']} mae={row['mae']:.4f} "
                       f"rmse={row['rmse']:.4f} mard={row['mard']:.2f}%")
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")
    return mae, rmse, mard


def save_full_checkpoint(
    path: Path, model: SugarOneModel, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None, epoch: int,
    best_val_loss: float, cfg: dict[str, Any], *, wait: int = 0, best_epoch: int = 0,
) -> None:
    _common_save_full_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, cfg,
                                 config_key="config", wait=wait, best_epoch=best_epoch, atomic=True)


def load_full_checkpoint(
    path: Path, model: SugarOneModel, optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float, int, int]:
    """Return last completed epoch, validation loss, wait count, and best epoch."""
    return _common_load_full_checkpoint(path, model, optimizer, scheduler, device,
                                        return_wait_and_best_epoch=True, log_fn=echo_plain)


def update_latest_symlink(run_dir: Path, out_dir: Path) -> None:
    _common_update_latest_symlink(run_dir, out_dir, log_fn=echo_plain)


def make_optimizer_and_scheduler(
    model: SugarOneModel, lr: float, weight_decay: float, epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )


def train_loop(
    model: SugarOneModel, train_loader: DataLoader[Any], val_loader: DataLoader[Any] | None,
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module, device: torch.device, epochs: int, patience: int, run_dir: Path,
    cfg: dict[str, Any], teacher: SugarOneModel | None = None, lwf_lambda: float = 0.0,
    verbose_every: int = 10, ckpt_every_n_epochs: int = 0,
    ckpt_eval_callback: Callable[[SugarOneModel, int, Path], None] | None = None,
    start_epoch: int = 1, best_val_loss: float = float("inf"), start_wait: int = 0,
    start_best_epoch: int = 0, use_amp: bool = False, amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None, val_every_n_epochs: int = 1,
    batch_log_every: int = 0, eval_batch_log_every: int = 0, log_interval_s: float = 0.0,
    metrics_csv: Path | None = None,
    extra_metrics_fn=None,
) -> SugarOneModel:
    wait, best_epoch = start_wait, start_best_epoch if start_best_epoch > 0 else max(0, start_epoch - 1)
    last_completed_epoch, total_time = max(0, start_epoch - 1), 0.0

    # Opt-in: None keeps every existing caller's behaviour byte-for-byte.
    metrics_writer = EpochMetricsWriter(metrics_csv) if metrics_csv is not None else None

    def _log_metrics_row(epoch: int, train_loss: float, val_loss: float | None, dt: float) -> None:
        if metrics_writer is None:
            return
        row: dict = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            # blank, not 0.0 — a skipped validation is missing data, not a score
            "val_loss": "" if val_loss is None else float(val_loss),
            "lr": optimizer.param_groups[0]["lr"],
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else "",
            "epoch_seconds": round(dt, 2),
        }
        for i, group in enumerate(optimizer.param_groups[1:], start=1):
            row[f"lr_group{i}"] = group["lr"]
        if extra_metrics_fn is not None:
            row.update(extra_metrics_fn())
        metrics_writer.log(row)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, teacher,
                                     lwf_lambda, use_amp, amp_dtype, scaler, batch_log_every,
                                     log_interval_s, epoch)
        val_loss_str = "SKIP"
        epoch_val_loss: float | None = None
        should_eval = val_loader is not None and (
            epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs
        )
        if should_eval:
            val_loss, _, _ = evaluate(model, val_loader, loss_fn, device, use_amp, amp_dtype,
                                      eval_batch_log_every, log_interval_s, "val")
            val_loss_str = f"{val_loss:.6f}"
            epoch_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss, best_epoch, wait = val_loss, epoch, 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                (run_dir / "best_info.json").write_text(
                    json.dumps({"epoch": epoch, "val_loss": best_val_loss}), encoding="utf-8"
                )
                echo_plain(f"  New best at epoch {epoch} (val_loss={val_loss:.6f})")
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    echo_plain(f"  Early stopping at epoch {epoch} (patience={patience})")
                    # Record the stopping epoch before leaving, or the curve is
                    # missing its final point.
                    _log_metrics_row(epoch, train_loss, epoch_val_loss, time.time() - started)
                    break
        elif val_loader is None:
            val_loss_str = "N/A"
            if train_loss < best_val_loss:
                best_val_loss, best_epoch = train_loss, epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                (run_dir / "best_info.json").write_text(
                    json.dumps({"epoch": epoch, "train_loss": float(train_loss)}), encoding="utf-8"
                )
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - started
        total_time += elapsed
        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            eta = timedelta(seconds=int(total_time / (epoch - start_epoch + 1) * (epochs - epoch)))
            echo_plain(f"  Epoch {epoch:4d}/{epochs} | train_loss={train_loss:.6f} | "
                       f"val_loss={val_loss_str} | {elapsed:.1f}s/epoch | ETA: {eta}")
        save_full_checkpoint(run_dir / "last_checkpoint.pt", model, optimizer, scheduler, epoch,
                             best_val_loss, cfg, wait=wait, best_epoch=best_epoch)
        if ckpt_every_n_epochs > 0 and epoch % ckpt_every_n_epochs == 0:
            ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _log_metrics_row(epoch, train_loss, epoch_val_loss, elapsed)
            save_full_checkpoint(ckpt_dir / "checkpoint.pt", model, optimizer, scheduler, epoch,
                                 best_val_loss, cfg, wait=wait, best_epoch=best_epoch)
            if ckpt_eval_callback is not None:
                ckpt_eval_callback(model, epoch, ckpt_dir)
        last_completed_epoch = epoch

    if metrics_writer is not None:
        metrics_writer.close()
        echo_plain(f"  Per-epoch metrics: {metrics_csv}")

    save_full_checkpoint(run_dir / "last_checkpoint.pt", model, optimizer, scheduler,
                         last_completed_epoch, best_val_loss, cfg, wait=wait, best_epoch=best_epoch)
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model


def make_model(
    input_steps: int, d_model: int, n_heads: int, ff_units: int, n_blocks: int,
    horizon: int, dropout: float, compile_mode: str, device: torch.device,
) -> SugarOneModel:
    model = SugarOneModel(input_steps, N_FEATURES, d_model, n_heads, ff_units, n_blocks, horizon, dropout).to(device)
    if device.type == "cuda" and compile_mode != "none":
        model = torch.compile(model, mode=compile_mode)
        echo_plain(f"torch.compile enabled (mode={compile_mode})")
    return model


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: cfg[key] for key in ("input_steps", "d_model", "n_heads", "ff_units",
                                       "n_blocks", "horizon", "dropout", "compile_mode")}


def run_train_and_eval(
    model: SugarOneModel, train_ds: SugarOneWindowDataset, val_ds: SugarOneWindowDataset | None,
    test_ds: SugarOneWindowDataset | None, cfg: dict[str, Any], device: torch.device,
    run_name: str, out_dir: Path, teacher: SugarOneModel | None = None, lwf_lambda: float = 0.0,
) -> SugarOneModel:
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(cfg) | {"train_samples": len(train_ds), "val_samples": len(val_ds) if val_ds else 0,
                        "test_samples": len(test_ds) if test_ds else 0, "start_time": datetime.now().isoformat()}
    (run_dir / "tuning_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    update_latest_symlink(run_dir, out_dir)
    workers = resolve_num_workers(cfg["num_workers"], device)
    loader_kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": device.type == "cuda",
                                     "persistent_workers": workers > 0}
    if workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs) if val_ds and len(val_ds) else None
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs) if test_ds and len(test_ds) else None
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg["lr"], cfg["weight_decay"], cfg["epochs"])
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg["precision"] == "fp16"))
    start_epoch, best_val_loss, start_wait, start_best_epoch = 1, float("inf"), 0, 0
    if cfg.get("resume_from"):
        last_done, best_val_loss, start_wait, start_best_epoch = load_full_checkpoint(
            Path(cfg["resume_from"]), model, optimizer, scheduler, device
        )
        start_epoch = last_done + 1
        if start_epoch > cfg["epochs"]:
            start_epoch = cfg["epochs"] + 1
    def checkpoint_evaluation(current_model: SugarOneModel, epoch: int, checkpoint_dir: Path) -> None:
        for label, loader, ds in (("val", val_loader, val_ds), ("test", test_loader, test_ds)):
            if loader is not None and ds is not None:
                _, actual, predicted = evaluate(current_model, loader, loss_fn, device, use_amp, amp_dtype,
                                                int(cfg.get("eval_batch_log_every", 0)), 0.0, label)
                compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose,
                                          f"{label}_epoch{epoch:04d}", checkpoint_dir, ds)
    model = train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device,
                       cfg["epochs"], cfg["patience"], run_dir, cfg, teacher, lwf_lambda,
                       cfg["log_every"], cfg["ckpt_every_n_epochs"], checkpoint_evaluation,
                       start_epoch, best_val_loss, start_wait, start_best_epoch, use_amp, amp_dtype,
                       grad_scaler, cfg["val_every_n_epochs"], int(cfg.get("batch_log_every", 0)),
                       int(cfg.get("eval_batch_log_every", 0)))
    for label, loader, ds in (("val", val_loader, val_ds), ("test", test_loader, test_ds)):
        if loader is not None and ds is not None:
            _, actual, predicted = evaluate(model, loader, loss_fn, device, use_amp, amp_dtype,
                                            int(cfg.get("eval_batch_log_every", 0)), 0.0, label)
            compute_and_print_metrics(actual, predicted, train_ds.scaler_glucose, label, run_dir, ds)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return model


def _mode_global(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                 cfg: dict[str, Any], device: torch.device, out_dir: Path) -> None:
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, cfg["input_steps"], cfg["horizon"])
    name = f"sugar_one_global_h{cfg['horizon']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_train_and_eval(make_model(**_model_kwargs(cfg), device=device), train_ds, val_ds, test_ds, cfg, device, name, out_dir)


def _mode_per_group(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                    cfg: dict[str, Any], device: torch.device, out_dir: Path) -> None:
    for group in (g for g in STUDY_GROUP_ORDER if g in set(train_df["study_group"].unique().to_list())):
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        test = test_df.filter(pl.col("study_group") == group) if not test_df.is_empty() else test_df
        train_ds, val_ds, test_ds = build_datasets(train, val, test, cfg["input_steps"], cfg["horizon"])
        safe = group.replace(" ", "_").replace("-", "_")
        name = f"sugar_one_group_{safe}_h{cfg['horizon']}_{datetime.now():%Y%m%d_%H%M%S}"
        run_train_and_eval(make_model(**_model_kwargs(cfg), device=device), train_ds, val_ds, test_ds, cfg, device, name, out_dir)


def _mode_cohort_wise(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                      cfg: dict[str, Any], device: torch.device, out_dir: Path) -> None:
    for group in (g for g in STUDY_GROUP_ORDER if g in set(train_df["study_group"].unique().to_list())):
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        train_ds, val_ds, test_ds = build_datasets(train, val, test_df, cfg["input_steps"], cfg["horizon"])
        safe = group.replace(" ", "_").replace("-", "_")
        name = f"sugar_one_cohort_{safe}_h{cfg['horizon']}_{datetime.now():%Y%m%d_%H%M%S}"
        run_train_and_eval(make_model(**_model_kwargs(cfg), device=device), train_ds, val_ds, test_ds, cfg, device, name, out_dir)


def _mode_continual(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame,
                    cfg: dict[str, Any], device: torch.device, out_dir: Path) -> None:
    groups = [g for g in STUDY_GROUP_ORDER if g in set(train_df["study_group"].unique().to_list())]
    if cfg.get("continual_order") == "reverse":
        groups.reverse()
    global_ds = SugarOneWindowDataset(train_df, cfg["input_steps"], cfg["horizon"], fit_scalers=True)
    scalers = {"scaler_glucose": global_ds.scaler_glucose, "scaler_basal": global_ds.scaler_basal,
               "scaler_bolus": global_ds.scaler_bolus, "scaler_carbs": global_ds.scaler_carbs}
    model, teacher = make_model(**_model_kwargs(cfg), device=device), None
    parent = f"sugar_one_continual_h{cfg['horizon']}_{datetime.now():%Y%m%d_%H%M%S}"
    for index, group in enumerate(groups, start=1):
        train = train_df.filter(pl.col("study_group") == group)
        val = val_df if cfg.get("continual_val_scope") == "all_groups" else val_df.filter(pl.col("study_group") == group)
        train_ds = SugarOneWindowDataset(train, cfg["input_steps"], cfg["horizon"], **scalers)
        val_ds = SugarOneWindowDataset(val, cfg["input_steps"], cfg["horizon"], **scalers) if not val.is_empty() else None
        test_ds = SugarOneWindowDataset(test_df, cfg["input_steps"], cfg["horizon"], **scalers) if not test_df.is_empty() else None
        safe = group.replace(" ", "_").replace("-", "_")
        model = run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device,
                                   f"{parent}/step_{index:02d}_{safe}_{datetime.now():%Y%m%d_%H%M%S}", out_dir,
                                   teacher, cfg["lwf_lambda"] if teacher is not None else 0.0)
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)


__all__ = [
    "COL_BASAL", "COL_BOLUS", "COL_CARB", "COL_EVENT", "COL_GLU", "COL_GROUP", "COL_SEQ",
    "COL_SPLIT", "COL_TS", "COL_USER", "N_FEATURES", "STUDY_GROUP_ALIASES", "STUDY_GROUP_ORDER",
    "TS_FORMAT", "SugarOneModel", "SugarOneWindowDataset", "_mode_cohort_wise", "_mode_continual",
    "_mode_global", "_mode_per_group", "_model_kwargs", "apply_split_scheme", "build_datasets",
    "compute_and_print_metrics", "evaluate", "impute_and_sort", "limit_series", "load_full_checkpoint",
    "load_splits_streaming", "make_model", "make_optimizer_and_scheduler", "mae_rmse_mard",
    "normalize_study_group_label", "normalize_study_groups_column", "read_checkpoint_meta",
    "resolve_num_workers", "run_train_and_eval", "save_full_checkpoint", "train_loop",
    "train_one_epoch", "update_latest_symlink",
]
