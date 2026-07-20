#!/usr/bin/env python3
"""
GluMindUni — Univariate Multi-Scale Attention Transformer for Blood Glucose Forecasting.

Univariate variant of GluMind: cross-attention covariate branches (HR, step count)
are removed. Only glucose values are used for input.

Works with the same CSV format as train_glumind.py.
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

from glumind_uni_model import GluMindUniModel

from scripts.common.data_loading import (
    STUDY_GROUP_ALIASES as STUDY_GROUP_ALIASES,
    STUDY_GROUP_ORDER as STUDY_GROUP_ORDER,
    limit_series as limit_series,
    normalize_study_group_label as normalize_study_group_label,
    normalize_study_groups_column as normalize_study_groups_column,
    resolve_num_workers as resolve_num_workers,
)
from scripts.common.data_loading import apply_split_scheme as _common_apply_split_scheme
from scripts.common.data_loading import impute_and_sort as _common_impute_and_sort
from scripts.common.data_loading import load_splits_streaming as _common_load_splits_streaming
from scripts.common.metrics import mae_rmse_mard as mae_rmse_mard
from scripts.common.checkpoint import (
    load_full_checkpoint as _common_load_full_checkpoint,
    save_full_checkpoint as _common_save_full_checkpoint,
    update_latest_symlink as update_latest_symlink,
)

app = typer.Typer(help="GluMindUni: Univariate glucose transformer trainer.")

# ---------------------------------------------------------------------------
# Source CSV columns
# ---------------------------------------------------------------------------
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"
COL_GLU = "Glucose Value (mg/dL)"

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ============================================================================
#  DATA LOADING
# ============================================================================

def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Lazy CSV scan via Polars — returns (train, val, test) DataFrames."""
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
        value_columns={"glucose": COL_GLU},
        ts_format=TS_FORMAT,
    )


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply optional split remapping while preserving classic defaults."""
    return _common_apply_split_scheme(
        train_df, val_df, test_df, split_scheme,
        applied_message=(
            "Applied split scheme: train <- train+val | val <- test | test disabled."
        ),
        note_message=None,
    )


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (unique_id, ds), forward-fill then back-fill glucose per series."""
    return _common_impute_and_sort(df, ffill_bfill_columns=["glucose"])


# ============================================================================
#  SLIDING-WINDOW DATASET
# ============================================================================

class GlucoseUniWindowDataset(Dataset):
    """Lazy sliding-window dataset for univariate glucose forecasting.

    Stores only the scaled per-series arrays; windows are sliced on-the-fly in
    __getitem__ so peak RAM is O(n_rows) instead of O(n_windows × input_steps).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        fit_scalers: bool = False,
    ):
        self.input_steps = input_steps
        self.horizon = horizon
        window_len = input_steps + horizon

        # Gather raw glucose arrays per series using fast Polars group iteration
        raw_glucose: list[np.ndarray] = []
        uids: list = []
        sgroups: list[str] = []
        for (uid_val,), grp in df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())

        # Fit or reuse scaler on concatenated data
        if fit_scalers or scaler_glucose is None:
            all_g = np.concatenate(raw_glucose).reshape(-1, 1)
            self.scaler_glucose = MinMaxScaler().fit(all_g)
        else:
            self.scaler_glucose = scaler_glucose

        # Scale each series and build a flat index: (series_idx, window_start)
        self._series: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        self.series_ids: list = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, raw) in enumerate(zip(uids, sgroups, raw_glucose)):
            g = self.scaler_glucose.transform(raw.reshape(-1, 1)).ravel().astype(np.float32)
            self._series.append(g)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(n_windows):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            print(f"  Note: Skipped {n_skipped} series/segments shorter than "
                  f"{window_len} steps.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_idx, start = self._index[idx]
        g = self._series[series_idx]
        x = g[start : start + self.input_steps].reshape(-1, 1)
        y = g[start + self.input_steps : start + self.input_steps + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================================
#  TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(
    model: GluMindUniModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    teacher: GluMindUniModel | None = None,
    lwf_lambda: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
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
    model: GluMindUniModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (avg_loss, all_true, all_pred)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

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
    dataset: GlucoseUniWindowDataset | None = None,
) -> tuple[float, float, float]:
    """Inverse-transform, compute MAE/RMSE/MARD, save CSVs."""
    t_flat = true_arr.ravel().reshape(-1, 1)
    p_flat = pred_arr.ravel().reshape(-1, 1)
    t_inv = scaler_glucose.inverse_transform(t_flat).ravel()
    p_inv = scaler_glucose.inverse_transform(p_flat).ravel()

    mae, rmse, mard = mae_rmse_mard(t_inv, p_inv)
    print(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MARD: {mard:.2f}%")

    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )

    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups = np.array(dataset.study_groups)
        rows = []
        for g in sorted(set(groups)):
            mask = groups == g
            if not mask.any():
                continue
            tg_inv = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            pg_inv = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            m, r, md = mae_rmse_mard(tg_inv, pg_inv)
            rows.append({"study_group": g, "n_windows": int(mask.sum()),
                          "mae": m, "rmse": r, "mard": md})

        by_group = pl.DataFrame(rows).sort("mae")
        print(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
        print(by_group)
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")

    return mae, rmse, mard


def save_full_checkpoint(
    path: Path,
    model: GluMindUniModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    cfg: dict,
) -> None:
    """Save a full checkpoint: model + optimizer + scheduler + metadata."""
    _common_save_full_checkpoint(
        path, model, optimizer, scheduler, epoch, best_val_loss, cfg,
        config_key="cfg",
    )


def load_full_checkpoint(
    path: Path,
    model: GluMindUniModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """Load a full checkpoint. Returns (epoch, best_val_loss)."""
    return _common_load_full_checkpoint(path, model, optimizer, scheduler, device)


def train_loop(
    model: GluMindUniModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
    teacher: GluMindUniModel | None = None,
    lwf_lambda: float = 0.0,
    verbose_every: int = 10,
    ckpt_every_n_epochs: int = 0,
    ckpt_eval_callback=None,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    cfg: dict | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    val_every_n_epochs: int = 1,
) -> GluMindUniModel:
    """Full training loop with early stopping on validation loss."""
    wait = 0
    best_epoch = start_epoch - 1
    total_time = 0.0

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            teacher=teacher, lwf_lambda=lwf_lambda,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
        )

        val_loss_str = "SKIP"
        should_eval_val = (
            val_loader is not None
            and (epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs)
        )
        if should_eval_val:
            val_loss, _, _ = evaluate(
                model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            val_loss_str = f"{val_loss:.6f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss}, f)
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
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "train_loss": float(train_loss)}, f)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        total_time += dt
        avg_time = total_time / (epoch - start_epoch + 1)
        remaining_epochs = epochs - epoch
        eta_str = str(timedelta(seconds=int(avg_time * remaining_epochs)))

        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss_str} | "
                  f"{dt:.1f}s/epoch | ETA: {eta_str}")

        if ckpt_every_n_epochs > 0 and epoch % ckpt_every_n_epochs == 0:
            ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_full_checkpoint(
                ckpt_dir / "checkpoint.pt",
                model, optimizer, scheduler, epoch, best_val_loss, cfg or {},
            )
            print(f"  [Checkpoint] Saved at epoch {epoch} → {ckpt_dir}")
            if ckpt_eval_callback is not None:
                ckpt_eval_callback(model, epoch, ckpt_dir)

    save_full_checkpoint(
        run_dir / "last_checkpoint.pt",
        model, optimizer, scheduler, epoch, best_val_loss, cfg or {},
    )
    torch.save(model.state_dict(), run_dir / "last_model.pt")

    print(f"\n  Summary: best_model.pt = epoch {best_epoch} | "
          f"last_model.pt = epoch {epoch}")

    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=True))
    return model


# ============================================================================
#  HELPERS
# ============================================================================

def make_model(
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    horizon: int,
    dropout: float,
    device: torch.device,
    compile_mode: str,
) -> GluMindUniModel:
    model = GluMindUniModel(
        n_time_steps=input_steps,
        d_model=d_model,
        n_heads=n_heads,
        ff_units=ff_units,
        n_blocks=n_blocks,
        prediction_horizon=horizon,
        dropout=dropout,
    ).to(device)
    if device.type == "cuda" and compile_mode != "none":
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"Warning: torch.compile failed, running in eager mode. ({e})")
    return model


def make_optimizer_and_scheduler(
    model: GluMindUniModel,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    return optimizer, scheduler


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    input_steps: int,
    horizon: int,
) -> tuple[GlucoseUniWindowDataset, GlucoseUniWindowDataset | None, GlucoseUniWindowDataset | None]:
    """Build window datasets, fitting scaler on train."""
    train_ds = GlucoseUniWindowDataset(
        train_df, input_steps, horizon, fit_scalers=True,
    )
    val_ds: GlucoseUniWindowDataset | None = None
    if not val_df.is_empty():
        val_ds = GlucoseUniWindowDataset(
            val_df, input_steps, horizon,
            scaler_glucose=train_ds.scaler_glucose,
        )
    test_ds: GlucoseUniWindowDataset | None = None
    if not test_df.is_empty():
        test_ds = GlucoseUniWindowDataset(
            test_df, input_steps, horizon,
            scaler_glucose=train_ds.scaler_glucose,
        )
    return train_ds, val_ds, test_ds


def run_train_and_eval(
    model: GluMindUniModel,
    train_ds: GlucoseUniWindowDataset,
    val_ds: GlucoseUniWindowDataset | None,
    test_ds: GlucoseUniWindowDataset | None,
    run_name: str,
    out_dir: Path,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    prefetch_factor: int,
    precision: str,
    log_every: int,
    ckpt_every_n_epochs: int,
    val_every_n_epochs: int,
    resume_from: str,
    lwf_lambda: float,
    device: torch.device,
    teacher: GluMindUniModel | None = None,
    cfg: dict | None = None,
) -> GluMindUniModel:
    """Train, evaluate, and save results."""
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    meta = dict(cfg or {})
    meta.update({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "start_time": datetime.now().isoformat(),
    })
    with open(run_dir / "tuning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    update_latest_symlink(run_dir, out_dir)

    resolved_workers = resolve_num_workers(num_workers, device)
    pin_memory = device.type == "cuda"
    loader_kwargs: dict = dict(
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        persistent_workers=resolved_workers > 0,
    )
    if resolved_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader: DataLoader | None = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader: DataLoader | None = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    optimizer, scheduler = make_optimizer_and_scheduler(model, lr, weight_decay, epochs)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and precision == "fp16"),
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_loss = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, device,
        )
        start_epoch += 1

    print(f"\n{'=' * 60}")
    print(f"Training: {len(train_ds):,} windows | "
          f"Val: {len(val_ds) if val_ds else 0:,} | "
          f"Test: {len(test_ds) if test_ds else 0:,} | "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    if start_epoch > 1:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'=' * 60}")

    def ckpt_eval_callback(mdl: GluMindUniModel, epoch: int, ckpt_dir: Path) -> None:
        if val_loader is not None:
            _, vt, vp = evaluate(mdl, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
            compute_and_print_metrics(vt, vp, train_ds.scaler_glucose, f"val_epoch{epoch:04d}", ckpt_dir, val_ds)
        if test_loader is not None:
            _, tt, tp = evaluate(mdl, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
            compute_and_print_metrics(tt, tp, train_ds.scaler_glucose, f"test_epoch{epoch:04d}", ckpt_dir, test_ds)

    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        loss_fn, device, epochs, patience, run_dir,
        teacher=teacher, lwf_lambda=lwf_lambda,
        verbose_every=log_every,
        ckpt_every_n_epochs=ckpt_every_n_epochs,
        ckpt_eval_callback=ckpt_eval_callback,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        cfg=cfg,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=grad_scaler,
        val_every_n_epochs=val_every_n_epochs,
    )

    if val_loader is not None:
        _, val_true, val_pred = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
        compute_and_print_metrics(val_true, val_pred, train_ds.scaler_glucose, "val", run_dir, val_ds)

    if test_loader is not None:
        _, test_true, test_pred = evaluate(model, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
        compute_and_print_metrics(test_true, test_pred, train_ds.scaler_glucose, "test", run_dir, test_ds)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg or {}, f, indent=2)

    return model


# ============================================================================
#  TRAINING MODES
# ============================================================================

def mode_global(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    out_dir: Path,
    horizon: int,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    dropout: float,
    compile_mode: str,
    device: torch.device,
    **train_kwargs,
) -> None:
    """Train one model on all data."""
    print("\n=== MODE: GLOBAL ===")
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, input_steps, horizon)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"glumind_uni_global_h{horizon}_{ts}"
    run_dir = out_dir / run_name
    update_latest_symlink(run_dir, out_dir)
    print(f"--> Training global model (dir={run_dir})")
    model = make_model(input_steps, d_model, n_heads, ff_units, n_blocks, horizon, dropout, device, compile_mode)
    run_train_and_eval(model, train_ds, val_ds, test_ds, run_name, out_dir, device=device, **train_kwargs)


def mode_per_group(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    out_dir: Path,
    horizon: int,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    dropout: float,
    compile_mode: str,
    device: torch.device,
    **train_kwargs,
) -> None:
    """Train separate model per study group."""
    print("\n=== MODE: PER_GROUP ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]
    for group in groups:
        print(f"\n--- Group: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        te = test_df.filter(pl.col("study_group") == group) if not test_df.is_empty() else test_df
        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(tr, va, te, input_steps, horizon)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_uni_group_{safe_name}_h{horizon}_{ts}"
        model = make_model(input_steps, d_model, n_heads, ff_units, n_blocks, horizon, dropout, device, compile_mode)
        run_train_and_eval(model, train_ds, val_ds, test_ds, run_name, out_dir, device=device, **train_kwargs)


def mode_cohort_wise(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    out_dir: Path,
    horizon: int,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    dropout: float,
    compile_mode: str,
    device: torch.device,
    **train_kwargs,
) -> None:
    """Sequential training across study groups, fresh model each time."""
    print("\n=== MODE: COHORT_WISE ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]
    for group in groups:
        print(f"\n--- Cohort: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue
        train_ds, val_ds, test_ds = build_datasets(tr, va, test_df, input_steps, horizon)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_uni_cohort_{safe_name}_h{horizon}_{ts}"
        model = make_model(input_steps, d_model, n_heads, ff_units, n_blocks, horizon, dropout, device, compile_mode)
        run_train_and_eval(model, train_ds, val_ds, test_ds, run_name, out_dir, device=device, **train_kwargs)


def mode_continual(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    out_dir: Path,
    horizon: int,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    dropout: float,
    compile_mode: str,
    device: torch.device,
    lwf_lambda: float,
    continual_order: str,
    continual_val_scope: str,
    **train_kwargs,
) -> None:
    """Continual learning across cohorts with LwF knowledge retention."""
    print("\n=== MODE: CONTINUAL (LwF) ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]
    if continual_order == "reverse":
        groups = list(reversed(groups))
    print(f"Continual group order: {groups}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_parent = f"glumind_uni_continual_h{horizon}_{run_ts}"
    print(f"Continual parent run dir: {out_dir / run_parent}")

    model = make_model(input_steps, d_model, n_heads, ff_units, n_blocks, horizon, dropout, device, compile_mode)
    teacher: GluMindUniModel | None = None

    all_train_ds = GlucoseUniWindowDataset(train_df, input_steps, horizon, fit_scalers=True)
    global_scaler_g = all_train_ds.scaler_glucose

    for i, group in enumerate(groups):
        print(f"\n--- Continual step {i + 1}/{len(groups)}: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        if not val_df.is_empty():
            va = val_df if continual_val_scope == "all_groups" else val_df.filter(pl.col("study_group") == group)
        else:
            va = val_df
        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue

        train_ds = GlucoseUniWindowDataset(tr, input_steps, horizon, scaler_glucose=global_scaler_g)
        val_ds: GlucoseUniWindowDataset | None = None
        if not va.is_empty():
            val_ds = GlucoseUniWindowDataset(va, input_steps, horizon, scaler_glucose=global_scaler_g)
        test_ds: GlucoseUniWindowDataset | None = None
        if not test_df.is_empty():
            test_ds = GlucoseUniWindowDataset(test_df, input_steps, horizon, scaler_glucose=global_scaler_g)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"{run_parent}/step_{i + 1:02d}_{safe_name}_{ts}"

        lwf = lwf_lambda if teacher is not None else 0.0
        model = run_train_and_eval(
            model, train_ds, val_ds, test_ds, run_name, out_dir, device=device,
            teacher=teacher, lwf_lambda=lwf, **train_kwargs,
        )

        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"  Saved teacher snapshot after {group}")


# ============================================================================
#  CLI
# ============================================================================

@app.command()
def train(
    csv: Path = typer.Option(..., help="Path to processed dataset CSV."),
    unique_id: str = typer.Option("sequence_id", help="ID column: sequence_id or user_id."),
    max_train_series: int = typer.Option(0, help="Limit training series (0 = all)."),
    max_eval_series: int = typer.Option(0, help="Limit evaluation series (0 = all)."),
    drop_interpolated: bool = typer.Option(False, help="Drop interpolated rows."),
    study_groups: str = typer.Option("", help="Comma-separated study groups (empty = all)."),
    split_scheme: str = typer.Option("classic", help="classic or trainval_test_as_val."),
    mode: str = typer.Option("global", help="global | per_group | cohort_wise | continual."),
    horizon: int = typer.Option(12, help="Prediction horizon in steps (12=60min)."),
    input_steps: int = typer.Option(80, help="Input window steps (80=400min at 5-min freq)."),
    d_model: int = typer.Option(32, help="Model embedding dimension."),
    n_heads: int = typer.Option(4, help="Number of attention heads."),
    n_blocks: int = typer.Option(3, help="Number of transformer blocks."),
    ff_units: int = typer.Option(128, help="Feed-forward hidden units."),
    dropout: float = typer.Option(0.1, help="Dropout rate."),
    epochs: int = typer.Option(200, help="Training epochs."),
    batch_size: int = typer.Option(64, help="Batch size."),
    precision: str = typer.Option("bf16", help="Mixed precision: fp32 | bf16 | fp16."),
    compile_mode: str = typer.Option("none", help="torch.compile mode: none | default | reduce-overhead | max-autotune."),
    disable_tf32: bool = typer.Option(False, help="Disable TF32 on CUDA."),
    num_workers: int = typer.Option(-1, help="DataLoader workers (-1 = auto)."),
    prefetch_factor: int = typer.Option(4, help="DataLoader prefetch factor."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    patience: int = typer.Option(20, help="Early stopping patience (0 = disabled)."),
    log_every: int = typer.Option(10, help="Print loss every N epochs."),
    ckpt_every_n_epochs: int = typer.Option(0, help="Save checkpoint every N epochs (0=off)."),
    val_every_n_epochs: int = typer.Option(1, help="Run validation every N epochs."),
    resume_from: str = typer.Option("", help="Path to checkpoint.pt to resume from."),
    lwf_lambda: float = typer.Option(0.5, help="LwF distillation weight for continual mode."),
    continual_order: str = typer.Option("default", help="Continual mode group order: default | reverse."),
    continual_val_scope: str = typer.Option("current_group", help="current_group | all_groups."),
    device_name: str = typer.Option("cuda", "--device", help="Device: cpu | mps | cuda."),
    seed: int = typer.Option(42, help="Random seed."),
    out_dir: Path = typer.Option(Path("data/output/runs/glumind_uni"), help="Output directory."),
) -> None:
    """Train GluMindUni on glucose-only data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    train_df, val_df, test_df = load_splits_streaming(csv, unique_id, drop_interpolated)
    print(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)

    if study_groups:
        groups_list = [normalize_study_group_label(g.strip()) for g in study_groups.split(",") if g.strip()]
        train_df = train_df.filter(pl.col("study_group").is_in(groups_list))
        val_df = val_df.filter(pl.col("study_group").is_in(groups_list))
        test_df = test_df.filter(pl.col("study_group").is_in(groups_list))
        print(f"Filtered to groups {groups_list}: train={len(train_df):,} | "
              f"val={len(val_df):,} | test={len(test_df):,}")

    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)

    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    if max_train_series > 0:
        train_df = limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = limit_series(val_df, max_eval_series)
        test_df = limit_series(test_df, max_eval_series)

    print(f"After limits: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    print(f"Study groups in train: {sorted(train_df['study_group'].unique().to_list())}")

    cfg = {
        "csv": str(csv), "mode": mode, "horizon": horizon, "input_steps": input_steps,
        "d_model": d_model, "n_heads": n_heads, "n_blocks": n_blocks, "ff_units": ff_units,
        "dropout": dropout, "epochs": epochs, "batch_size": batch_size, "precision": precision,
        "lr": lr, "weight_decay": weight_decay, "patience": patience, "seed": seed,
        "out_dir": str(out_dir),
    }

    common_train_kwargs = dict(
        out_dir=out_dir,
        horizon=horizon,
        input_steps=input_steps,
        d_model=d_model,
        n_heads=n_heads,
        ff_units=ff_units,
        n_blocks=n_blocks,
        dropout=dropout,
        compile_mode=compile_mode,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        precision=precision,
        log_every=log_every,
        ckpt_every_n_epochs=ckpt_every_n_epochs,
        val_every_n_epochs=val_every_n_epochs,
        resume_from=resume_from,
        lwf_lambda=lwf_lambda,
        cfg=cfg,
    )

    mode_fns = {
        "global": mode_global,
        "per_group": mode_per_group,
        "cohort_wise": mode_cohort_wise,
    }

    if mode == "continual":
        mode_continual(
            train_df, val_df, test_df,
            device=device,
            lwf_lambda=lwf_lambda,
            continual_order=continual_order,
            continual_val_scope=continual_val_scope,
            **common_train_kwargs,
        )
    elif mode in mode_fns:
        mode_fns[mode](train_df, val_df, test_df, device=device, **common_train_kwargs)
    else:
        raise typer.BadParameter(f"Unknown mode: {mode}")

    print("\nDone.")


if __name__ == "__main__":
    app()
