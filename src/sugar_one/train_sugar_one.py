#!/usr/bin/env python3
"""
SugarOne — Insulin & Carb Parallel-Attention Transformer for Blood Glucose Forecasting.

Covariates: Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g).
Dataset:    data/input/loop_and_ai_ready/loop_ai_ready_joined_loop_columns.csv

Architecture: SugarOneModel (see sugar_one_model.py).
  - Identical parallel cross-attention + multi-scale self-attention structure
    as base GluMind, extended to 3 auxiliaries with learnable mixing weights.

Imputation policy (physiologically motivated):
  - Basal Rate: forward-fill then back-fill (continuous background rate that
    persists until changed), then fill_null(0.0).
  - Bolus Insulin: fill_null(0.0) directly — discrete dosing event, no carry-over.
  - Carbohydrates: fill_null(0.0) directly — discrete meal event, no carry-over.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sugar_one.console_log import echo_plain
from sugar_one.sugar_one_model import SugarOneModel

from common.data.columns import (
    COL_BASAL,
    COL_BOLUS,
    COL_CARB,
    COL_EVENT,
    COL_GLU,
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS_SHORT as COL_TS,
    COL_USER,
    TS_FORMAT,
)
from common.data.loading import (
    STUDY_GROUP_ALIASES as STUDY_GROUP_ALIASES,
    STUDY_GROUP_ORDER as STUDY_GROUP_ORDER,
    apply_split_scheme as _common_apply_split_scheme,
    impute_and_sort as _common_impute_and_sort,
    limit_series as limit_series,
    load_splits_streaming as _common_load_splits_streaming,
    normalize_study_group_label as normalize_study_group_label,
    normalize_study_groups_column as normalize_study_groups_column,
    resolve_num_workers as resolve_num_workers,
)
from common.metrics import mae_rmse_mard as mae_rmse_mard
from common.metrics_log import EpochMetricsWriter
from common.checkpoint import (
    read_checkpoint_meta as read_checkpoint_meta,
    update_latest_symlink as _common_update_latest_symlink,
)
from common.checkpoint import load_full_checkpoint as _common_load_full_checkpoint
from common.checkpoint import save_full_checkpoint as _common_save_full_checkpoint
from common.paths import DEFAULT_RUNS_ROOT
from common.scalers import SCALERS_FILENAME, save_scalers_for_run
from sugar_one.sugar_one_spec import SUGAR_ONE_SPEC

app = typer.Typer(
    name="train_sugar_one",
    add_completion=False,
    help="SugarOne: Parallel-Attention Transformer with Insulin & Carb covariates.",
)

N_FEATURES = 4  # glucose, basal, bolus, carbs

# ============================================================================
#  DATA LOADING
# ============================================================================

def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
    keep_columns: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Lazy CSV scan — returns (train, val, test) DataFrames."""
    return _common_load_splits_streaming(
        csv_path,
        unique_id_choice,
        drop_interpolated,
        col_seq=COL_SEQ,
        col_user=COL_USER,
        col_ts=COL_TS,
        col_split=COL_SPLIT,
        col_group=COL_GROUP,
        col_event=COL_EVENT,
        value_columns={
            "glucose": COL_GLU, "basal": COL_BASAL,
            "bolus": COL_BOLUS, "carbs": COL_CARB,
        },
        ts_format=TS_FORMAT,
        utf8_value_columns=("basal", "bolus", "carbs"),
        keep_columns=keep_columns,
        log_fn=typer.echo,
    )


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return _common_apply_split_scheme(
        train_df, val_df, test_df, split_scheme,
        log_fn=typer.echo,
        applied_message="Applied split scheme: train <- train+val | val <- test | test disabled.",
        note_message="Note: tuning-only mode; no held-out test metrics.",
        error_repr=True,
    )


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (unique_id, ds), then impute per-series.

    Basal Rate: forward-fill → back-fill (continuous rate; persists until changed).
    Bolus / Carbs: fill_null(0.0) only (discrete events; must not carry over).
    Glucose: forward-fill → back-fill → 0.0 fallback.
    """
    return _common_impute_and_sort(
        df,
        ffill_bfill_columns=["glucose", "basal"],
        zero_fill_columns=["bolus", "carbs"],
    )


# ============================================================================
#  SLIDING-WINDOW DATASET
# ============================================================================

from common.data import SugarOneWindowDataset  # noqa: E402  — re-export for train/eval callers



# ============================================================================
#  TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(
    model: SugarOneModel,
    loader: DataLoader,
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
    total_loss = 0.0
    n_batches = 0
    n_batches_total = len(loader)
    t_epoch = time.perf_counter()
    last_log_t = t_epoch

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

        n_batches += 1
        total_loss += loss.item()

        should_log = False
        if log_interval_s > 0:
            now = time.perf_counter()
            if (
                n_batches == 1
                or n_batches == n_batches_total
                or now - last_log_t >= log_interval_s
            ):
                should_log = True
                last_log_t = now
        elif batch_log_every > 0 and (
            n_batches == 1
            or n_batches % batch_log_every == 0
            or n_batches == n_batches_total
        ):
            should_log = True

        if should_log:
            elapsed = time.perf_counter() - t_epoch
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0.0
            remaining = n_batches_total - n_batches
            eta_s = remaining / batches_per_sec if batches_per_sec > 0 else 0.0
            avg_loss = total_loss / n_batches
            echo_plain(
                f"  train epoch {epoch:3d} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{batches_per_sec:.2f} batch/s | loss={avg_loss:.6f} | "
                f"epoch ETA {timedelta(seconds=int(eta_s))}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: SugarOneModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    batch_log_every: int = 0,
    log_interval_s: float = 0.0,
    split_label: str = "eval",
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []
    n_batches_total = len(loader)
    t_eval = time.perf_counter()
    last_log_t = t_eval

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

        should_log = False
        if log_interval_s > 0:
            now = time.perf_counter()
            if (
                n_batches == 1
                or n_batches == n_batches_total
                or now - last_log_t >= log_interval_s
            ):
                should_log = True
                last_log_t = now
        elif batch_log_every > 0 and (
            n_batches == 1
            or n_batches % batch_log_every == 0
            or n_batches == n_batches_total
        ):
            should_log = True

        if should_log:
            elapsed = time.perf_counter() - t_eval
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0.0
            remaining = n_batches_total - n_batches
            eta_s = remaining / batches_per_sec if batches_per_sec > 0 else 0.0
            echo_plain(
                f"  {split_label} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{batches_per_sec:.2f} batch/s | "
                f"ETA {timedelta(seconds=int(eta_s))}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return avg_loss, true_arr, pred_arr


def compute_and_print_metrics(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    scaler_glucose: MinMaxScaler,
    split_name: str,
    run_dir: Path,
    dataset: SugarOneWindowDataset | None = None,
) -> tuple[float, float, float]:
    t_inv = scaler_glucose.inverse_transform(true_arr.ravel().reshape(-1, 1)).ravel()
    p_inv = scaler_glucose.inverse_transform(pred_arr.ravel().reshape(-1, 1)).ravel()
    mae, rmse, mard = mae_rmse_mard(t_inv, p_inv)

    typer.echo(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    typer.echo(f"  MAE : {mae:.4f}")
    typer.echo(f"  RMSE: {rmse:.4f}")
    typer.echo(f"  MARD: {mard:.2f}%")

    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )

    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups_arr = np.array(dataset.study_groups)
        rows = []
        for g in sorted(set(groups_arr)):
            mask = groups_arr == g
            if not mask.any():
                continue
            tg = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            pg = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            m, r, md = mae_rmse_mard(tg, pg)
            rows.append({"study_group": g, "n_windows": int(mask.sum()),
                         "mae": m, "rmse": r, "mard": md})
        by_group = pl.DataFrame(rows).sort("mae")
        typer.echo(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
        for row in by_group.iter_rows(named=True):
            echo_plain(
                f"  {row['study_group']}: n={row['n_windows']} "
                f"mae={row['mae']:.4f} rmse={row['rmse']:.4f} mard={row['mard']:.2f}%"
            )
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")

    return mae, rmse, mard


def save_full_checkpoint(
    path: Path,
    model: SugarOneModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    cfg: dict,
    *,
    wait: int = 0,
    best_epoch: int = 0,
) -> None:
    _common_save_full_checkpoint(
        path, model, optimizer, scheduler, epoch, best_val_loss, cfg,
        config_key="config", wait=wait, best_epoch=best_epoch, atomic=True,
    )


def load_full_checkpoint(
    path: Path,
    model: SugarOneModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float, int, int]:
    """Return (last_completed_epoch, best_val_loss, patience_wait, best_epoch)."""
    return _common_load_full_checkpoint(
        path, model, optimizer, scheduler, device,
        return_wait_and_best_epoch=True, log_fn=echo_plain,
    )


def update_latest_symlink(run_dir: Path, out_dir: Path) -> None:
    _common_update_latest_symlink(run_dir, out_dir, log_fn=typer.echo)


def make_optimizer_and_scheduler(
    model: SugarOneModel,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    return optimizer, scheduler


def train_loop(
    model: SugarOneModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
    cfg: dict,
    teacher: SugarOneModel | None = None,
    lwf_lambda: float = 0.0,
    verbose_every: int = 10,
    ckpt_every_n_epochs: int = 0,
    ckpt_eval_callback=None,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    start_wait: int = 0,
    start_best_epoch: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    val_every_n_epochs: int = 1,
    batch_log_every: int = 0,
    eval_batch_log_every: int = 0,
    log_interval_s: float = 0.0,
    metrics_csv: Path | None = None,
    extra_metrics_fn=None,
) -> SugarOneModel:
    wait = start_wait
    best_epoch = start_best_epoch if start_best_epoch > 0 else max(0, start_epoch - 1)
    last_completed_epoch = max(0, start_epoch - 1)
    total_time = 0.0
    train_batches = len(train_loader)

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
        t0 = time.time()
        if epoch == start_epoch and (batch_log_every > 0 or log_interval_s > 0):
            echo_plain(
                f"  Epoch {epoch}/{epochs}: {train_batches:,} train batches "
                f"(batch_size={train_loader.batch_size})"
            )
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            teacher=teacher, lwf_lambda=lwf_lambda,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            batch_log_every=batch_log_every,
            log_interval_s=log_interval_s,
            epoch=epoch,
        )

        val_loss_str = "SKIP"
        epoch_val_loss: float | None = None
        should_eval = (
            val_loader is not None
            and (epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs)
        )
        if should_eval:
            val_loss, _, _ = evaluate(
                model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                batch_log_every=eval_batch_log_every,
                log_interval_s=log_interval_s,
                split_label="val",
            )
            val_loss_str = f"{val_loss:.6f}"
            epoch_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss}, f)
                echo_plain(f"  New best at epoch {epoch} (val_loss={val_loss:.6f})")
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    typer.echo(f"  Early stopping at epoch {epoch} (patience={patience})")
                    # Record the stopping epoch before leaving, or the curve is
                    # missing its final point.
                    _log_metrics_row(epoch, train_loss, epoch_val_loss, time.time() - t0)
                    break
        elif val_loader is None:
            val_loss_str = "N/A"
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "train_loss": float(train_loss)}, f)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        total_time += dt
        avg_time = total_time / (epoch - start_epoch + 1)
        eta_str = str(timedelta(seconds=int(avg_time * (epochs - epoch))))

        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            typer.echo(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss_str} | "
                f"{dt:.1f}s/epoch | ETA: {eta_str}"
            )

        _log_metrics_row(epoch, train_loss, epoch_val_loss, dt)

        save_full_checkpoint(
            run_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            cfg,
            wait=wait,
            best_epoch=best_epoch,
        )

        if ckpt_every_n_epochs > 0 and epoch % ckpt_every_n_epochs == 0:
            ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_full_checkpoint(
                ckpt_dir / "checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                cfg,
                wait=wait,
                best_epoch=best_epoch,
            )
            echo_plain(f"  Checkpoint saved at epoch {epoch} -> {ckpt_dir}")
            if ckpt_eval_callback is not None:
                ckpt_eval_callback(model, epoch, ckpt_dir)

        last_completed_epoch = epoch

    if metrics_writer is not None:
        metrics_writer.close()
        echo_plain(f"  Per-epoch metrics: {metrics_csv}")

    save_full_checkpoint(
        run_dir / "last_checkpoint.pt",
        model,
        optimizer,
        scheduler,
        last_completed_epoch,
        best_val_loss,
        cfg,
        wait=wait,
        best_epoch=best_epoch,
    )
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    echo_plain(
        f"\n  Summary: best_model.pt = epoch {best_epoch} | "
        f"last_checkpoint.pt = epoch {last_completed_epoch}"
    )

    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model


# ============================================================================
#  TRAINING MODES
# ============================================================================

def make_model(
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    horizon: int,
    dropout: float,
    compile_mode: str,
    device: torch.device,
) -> SugarOneModel:
    model = SugarOneModel(
        n_time_steps=input_steps,
        n_features=N_FEATURES,
        d_model=d_model,
        n_heads=n_heads,
        ff_units=ff_units,
        n_blocks=n_blocks,
        prediction_horizon=horizon,
        dropout=dropout,
    ).to(device)
    if device.type == "cuda" and compile_mode != "none":
        model = torch.compile(model, mode=compile_mode)
        typer.echo(f"torch.compile enabled (mode={compile_mode})")
    return model


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    input_steps: int,
    horizon: int,
) -> tuple[
    SugarOneWindowDataset,
    SugarOneWindowDataset | None,
    SugarOneWindowDataset | None,
]:
    train_ds = SugarOneWindowDataset(
        train_df, input_steps, horizon, fit_scalers=True,
    )
    val_ds = (
        SugarOneWindowDataset(
            val_df, input_steps, horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_basal=train_ds.scaler_basal,
            scaler_bolus=train_ds.scaler_bolus,
            scaler_carbs=train_ds.scaler_carbs,
        )
        if not val_df.is_empty()
        else None
    )
    test_ds = (
        SugarOneWindowDataset(
            test_df, input_steps, horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_basal=train_ds.scaler_basal,
            scaler_bolus=train_ds.scaler_bolus,
            scaler_carbs=train_ds.scaler_carbs,
        )
        if not test_df.is_empty()
        else None
    )
    return train_ds, val_ds, test_ds


def run_train_and_eval(
    model: SugarOneModel,
    train_ds: SugarOneWindowDataset,
    val_ds: SugarOneWindowDataset | None,
    test_ds: SugarOneWindowDataset | None,
    cfg: dict,
    device: torch.device,
    run_name: str,
    out_dir: Path,
    teacher: SugarOneModel | None = None,
    lwf_lambda: float = 0.0,
) -> SugarOneModel:
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Run directory: {run_dir}")

    save_scalers_for_run(
        run_dir,
        kind=SUGAR_ONE_SPEC.kind,
        scalers=SUGAR_ONE_SPEC.extract_scalers(train_ds),
        provenance={
            "csv": str(cfg.get("csv", "")),
            "split_scheme": cfg.get("split_scheme", "classic"),
            "unique_id": cfg.get("unique_id", "sequence_id"),
            "mode": cfg.get("mode", ""),
            "train_windows": len(train_ds),
        },
    )

    meta = dict(cfg)
    meta.update({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "start_time": datetime.now().isoformat(),
        "scalers": SCALERS_FILENAME,
    })
    with open(run_dir / "tuning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    update_latest_symlink(run_dir, out_dir)

    num_workers = resolve_num_workers(cfg["num_workers"], device)
    pin_memory = device.type == "cuda"
    loader_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    echo_plain(
        f"  DataLoader: train_batches/epoch={len(train_loader):,} | "
        f"val_batches={len(val_loader) if val_loader else 0:,} | "
        f"test_batches={len(test_loader) if test_loader else 0:,} | "
        f"batch_size={cfg['batch_size']} | num_workers={num_workers}"
    )

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, cfg["lr"], cfg["weight_decay"], cfg["epochs"]
    )
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and cfg["precision"] == "fp16")
    )

    start_epoch = 1
    best_val_loss = float("inf")
    start_wait = 0
    start_best_epoch = 0
    if cfg.get("resume_from"):
        resume_path = Path(cfg["resume_from"])
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        last_done, best_val_loss, start_wait, start_best_epoch = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = last_done + 1
        if start_epoch > cfg["epochs"]:
            echo_plain(
                f"  Checkpoint already at epoch {last_done} >= max {cfg['epochs']}; "
                "skipping training loop."
            )
            start_epoch = cfg["epochs"] + 1

    typer.echo(f"\n{'=' * 60}")
    typer.echo(
        f"Training: {len(train_ds):,} windows | "
        f"Val: {len(val_ds) if val_ds else 0:,} | "
        f"Test: {len(test_ds) if test_ds else 0:,} | "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )
    typer.echo(f"{'=' * 60}")

    batch_log_every = int(cfg.get("batch_log_every", 0))
    eval_batch_log_every = int(cfg.get("eval_batch_log_every", 0))

    def ckpt_eval_callback(mdl: SugarOneModel, epoch: int, ckpt_dir: Path) -> None:
        if val_loader is not None:
            _, vt, vp = evaluate(
                mdl, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
                batch_log_every=eval_batch_log_every, split_label="val",
            )
            compute_and_print_metrics(vt, vp, train_ds.scaler_glucose, f"val_epoch{epoch:04d}", ckpt_dir, val_ds)
        if test_loader is not None:
            _, tt, tp = evaluate(
                mdl, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
                batch_log_every=eval_batch_log_every, split_label="test",
            )
            compute_and_print_metrics(tt, tp, train_ds.scaler_glucose, f"test_epoch{epoch:04d}", ckpt_dir, test_ds)

    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        loss_fn, device, cfg["epochs"], cfg["patience"], run_dir, cfg,
        teacher=teacher, lwf_lambda=lwf_lambda,
        verbose_every=cfg["log_every"],
        ckpt_every_n_epochs=cfg["ckpt_every_n_epochs"],
        ckpt_eval_callback=ckpt_eval_callback,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        start_wait=start_wait,
        start_best_epoch=start_best_epoch,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=grad_scaler,
        val_every_n_epochs=cfg["val_every_n_epochs"],
        batch_log_every=batch_log_every,
        eval_batch_log_every=eval_batch_log_every,
    )

    if val_loader is not None:
        _, vt, vp = evaluate(
            model, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=eval_batch_log_every, split_label="val",
        )
        compute_and_print_metrics(vt, vp, train_ds.scaler_glucose, "val", run_dir, val_ds)

    if test_loader is not None:
        _, tt, tp = evaluate(
            model, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=eval_batch_log_every, split_label="test",
        )
        compute_and_print_metrics(tt, tp, train_ds.scaler_glucose, "test", run_dir, test_ds)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return model


# ============================================================================
#  TRAINING MODES
# ============================================================================

def _mode_global(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: GLOBAL ===")
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, cfg["input_steps"], cfg["horizon"]
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sugar_one_global_h{cfg['horizon']}_{ts}"
    update_latest_symlink(out_dir / run_name, out_dir)

    model = make_model(**_model_kwargs(cfg), device=device)
    run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir)


def _mode_per_group(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: PER_GROUP ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]

    for group in groups:
        typer.echo(f"\n--- Group: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        te = test_df.filter(pl.col("study_group") == group) if not test_df.is_empty() else test_df
        if tr.is_empty():
            typer.echo(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(tr, va, te, cfg["input_steps"], cfg["horizon"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = group.replace(" ", "_").replace("-", "_")
        run_name = f"sugar_one_group_{safe}_h{cfg['horizon']}_{ts}"
        model = make_model(**_model_kwargs(cfg), device=device)
        run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir)


def _mode_cohort_wise(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: COHORT_WISE ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]

    for group in groups:
        typer.echo(f"\n--- Cohort: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        if tr.is_empty():
            typer.echo(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(tr, va, test_df, cfg["input_steps"], cfg["horizon"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = group.replace(" ", "_").replace("-", "_")
        run_name = f"sugar_one_cohort_{safe}_h{cfg['horizon']}_{ts}"
        model = make_model(**_model_kwargs(cfg), device=device)
        run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir)


def _mode_continual(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: CONTINUAL (LwF) ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]
    if cfg.get("continual_order") == "reverse":
        groups = list(reversed(groups))
    typer.echo(f"Continual group order: {groups}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_parent = f"sugar_one_continual_h{cfg['horizon']}_{run_ts}"

    all_train_ds = SugarOneWindowDataset(
        train_df, cfg["input_steps"], cfg["horizon"], fit_scalers=True
    )
    global_sg = all_train_ds.scaler_glucose
    global_sb = all_train_ds.scaler_basal
    global_sbo = all_train_ds.scaler_bolus
    global_sc = all_train_ds.scaler_carbs

    model = make_model(**_model_kwargs(cfg), device=device)
    teacher: SugarOneModel | None = None

    for i, group in enumerate(groups):
        typer.echo(f"\n--- Continual step {i + 1}/{len(groups)}: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        if tr.is_empty():
            typer.echo(f"  No training data for {group}, skipping.")
            continue

        if not val_df.is_empty():
            va = val_df if cfg.get("continual_val_scope") == "all_groups" else \
                val_df.filter(pl.col("study_group") == group)
        else:
            va = val_df

        train_ds = SugarOneWindowDataset(
            tr, cfg["input_steps"], cfg["horizon"],
            scaler_glucose=global_sg, scaler_basal=global_sb,
            scaler_bolus=global_sbo, scaler_carbs=global_sc,
        )
        val_ds = (
            SugarOneWindowDataset(
                va, cfg["input_steps"], cfg["horizon"],
                scaler_glucose=global_sg, scaler_basal=global_sb,
                scaler_bolus=global_sbo, scaler_carbs=global_sc,
            )
            if not va.is_empty()
            else None
        )
        test_ds = (
            SugarOneWindowDataset(
                test_df, cfg["input_steps"], cfg["horizon"],
                scaler_glucose=global_sg, scaler_basal=global_sb,
                scaler_bolus=global_sbo, scaler_carbs=global_sc,
            )
            if not test_df.is_empty()
            else None
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = group.replace(" ", "_").replace("-", "_")
        run_name = f"{run_parent}/step_{i + 1:02d}_{safe}_{ts}"
        lwf = cfg["lwf_lambda"] if teacher is not None else 0.0

        model = run_train_and_eval(
            model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir,
            teacher=teacher, lwf_lambda=lwf,
        )

        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        typer.echo(f"  Saved teacher snapshot after {group}")


def _model_kwargs(cfg: dict) -> dict:
    return dict(
        input_steps=cfg["input_steps"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_units=cfg["ff_units"],
        n_blocks=cfg["n_blocks"],
        horizon=cfg["horizon"],
        dropout=cfg["dropout"],
        compile_mode=cfg["compile_mode"],
    )


# ============================================================================
#  CLI  (single root command — run: python train_sugar_one.py --csv ...  no "train" word)
# ============================================================================

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
    out_dir: Path = typer.Option(DEFAULT_RUNS_ROOT / "sugar_one", help="Output directory."),
) -> None:
    """Train SugarOne on insulin + carb covariate data (root CLI; do not pass a subcommand)."""
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
    typer.echo(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)

    if study_groups:
        group_list = [
            normalize_study_group_label(g.strip())
            for g in study_groups.split(",") if g.strip()
        ]
        train_df = train_df.filter(pl.col("study_group").is_in(group_list))
        val_df = val_df.filter(pl.col("study_group").is_in(group_list))
        test_df = test_df.filter(pl.col("study_group").is_in(group_list))
        typer.echo(f"Filtered to {group_list}: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)

    # Limit BEFORE impute so we only sort/fill the series we actually need.
    if max_train_series > 0:
        train_df = limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = limit_series(val_df, max_eval_series)
        test_df = limit_series(test_df, max_eval_series)

    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    typer.echo(f"After limits: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    typer.echo(f"Study groups in train: {sorted(train_df['study_group'].unique().to_list())}")

    cfg = {
        "csv": str(csv), "unique_id": unique_id, "drop_interpolated": drop_interpolated,
        "study_groups": study_groups, "split_scheme": split_scheme, "mode": mode,
        "horizon": horizon, "input_steps": input_steps,
        "d_model": d_model, "n_heads": n_heads, "n_blocks": n_blocks,
        "ff_units": ff_units, "dropout": dropout,
        "epochs": epochs, "batch_size": batch_size, "precision": precision,
        "compile_mode": compile_mode, "disable_tf32": disable_tf32,
        "num_workers": num_workers, "prefetch_factor": prefetch_factor,
        "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "log_every": log_every, "ckpt_every_n_epochs": ckpt_every_n_epochs,
        "val_every_n_epochs": val_every_n_epochs, "resume_from": resume_from,
        "batch_log_every": 0, "eval_batch_log_every": 0,
        "lwf_lambda": lwf_lambda, "continual_order": continual_order,
        "continual_val_scope": continual_val_scope,
        "device": device_name, "seed": seed, "out_dir": str(out_dir),
    }

    mode_fn = {
        "global": _mode_global,
        "per_group": _mode_per_group,
        "cohort_wise": _mode_cohort_wise,
        "continual": _mode_continual,
    }
    if mode not in mode_fn:
        raise typer.BadParameter(f"Unknown mode: {mode!r}. Choose from: {list(mode_fn)}")

    mode_fn[mode](train_df, val_df, test_df, cfg, device, out_dir)
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
