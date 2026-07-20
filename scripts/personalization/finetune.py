#!/usr/bin/env python3
"""Fine-tune a SugarOne-family global checkpoint on one person's data.

Personal-only fine-tuning with optional Learning without Forgetting (LwF):
the frozen global model acts as a teacher to reduce catastrophic forgetting
while adapting to one person's CGM/pump timeline.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from scripts.common.console import init_cli_console, safe_echo
from scripts.common.data_loading import resolve_num_workers
from scripts.common.metrics import mae_rmse_mard, overall_metrics_to_csv
from scripts.personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_FT_PATIENCE,
    DEFAULT_PROGRESS_LOG_INTERVAL_S,
    DEFAULT_SEED,
    DEFAULT_TRAIN_WINDOW_STRIDE,
    DEFAULT_VAL_EVERY_N_EPOCHS,
    DENSE_WINDOW_STRIDE,
    SUGAR_ONE_VALUE_COLUMNS,
)
from scripts.personalization.registry import build_model_from_meta, load_base_checkpoint
from scripts.sugar_one.train_sugar_one import (
    SugarOneWindowDataset,
    evaluate,
    impute_and_sort,
    load_full_checkpoint,
    load_splits_streaming,
    make_optimizer_and_scheduler,
    train_loop,
)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Fine-tune SugarOne-family models on personal CGM/pump data.",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split_frames(csv_path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, val_df, test_df = load_splits_streaming(
        csv_path, unique_id_choice="sequence_id", drop_interpolated=False
    )
    return impute_and_sort(train_df), impute_and_sort(val_df), impute_and_sort(test_df)


def _make_lwf_teacher(model: nn.Module) -> nn.Module:
    """Frozen copy of the global model for LwF distillation."""
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _dataloader_kwargs(num_workers: int, device: torch.device, prefetch_factor: int) -> dict[str, Any]:
    """Match SugarOne training DataLoader settings."""
    workers = resolve_num_workers(num_workers, device)
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _metrics_dict(mae: float, rmse: float, mard: float) -> dict[str, float]:
    return {"mae": float(mae), "rmse": float(rmse), "mard": float(mard)}


def _load_saved_run_config(run_dir: Path) -> dict[str, Any]:
    for fname in ("config.json", "tuning_meta.json"):
        path = run_dir / fname
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    raise ValueError(f"no config.json or tuning_meta.json in {run_dir}")


def _compute_quiet_metrics(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    scaler_ds: SugarOneWindowDataset,
    split_name: str,
    run_dir: Path,
    dataset: SugarOneWindowDataset | None = None,
) -> tuple[float, float, float]:
    """Compute metrics, save CSVs, emit a single summary line."""
    t_inv = scaler_ds.scaler_glucose.inverse_transform(true_arr.ravel().reshape(-1, 1)).ravel()
    p_inv = scaler_ds.scaler_glucose.inverse_transform(pred_arr.ravel().reshape(-1, 1)).ravel()
    mae, rmse, mard = mae_rmse_mard(t_inv, p_inv)
    overall_metrics_to_csv(mae, rmse, mard, run_dir, split_name)
    safe_echo(
        f"  {split_name}: MAE={mae:.2f} RMSE={rmse:.2f} MARD={mard:.1f}%"
    )

    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups_arr = np.array(dataset.study_groups)
        unique_groups = sorted(set(groups_arr))
        if len(unique_groups) > 1:
            rows = []
            for g in unique_groups:
                mask = groups_arr == g
                if not mask.any():
                    continue
                tg = scaler_ds.scaler_glucose.inverse_transform(
                    true_arr[mask].ravel().reshape(-1, 1)
                ).ravel()
                pg = scaler_ds.scaler_glucose.inverse_transform(
                    pred_arr[mask].ravel().reshape(-1, 1)
                ).ravel()
                m, r, md = mae_rmse_mard(tg, pg)
                rows.append(
                    {
                        "study_group": g,
                        "n_windows": int(mask.sum()),
                        "mae": m,
                        "rmse": r,
                        "mard": md,
                    }
                )
            pl.DataFrame(rows).sort("mae").write_csv(
                run_dir / f"{split_name}_metrics_by_study_group.csv"
            )
    return mae, rmse, mard


def _eval_split(
    model: nn.Module,
    ds: SugarOneWindowDataset | None,
    scaler_ds: SugarOneWindowDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    run_dir: Path,
    split_name: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    log_interval_s: float = 0.0,
) -> dict[str, float] | None:
    if ds is None or len(ds) == 0:
        return None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    loss_fn = nn.MSELoss()
    _, true_arr, pred_arr = evaluate(
        model,
        loader,
        loss_fn,
        device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        log_interval_s=log_interval_s,
        split_label=split_name,
    )
    mae, rmse, mard = _compute_quiet_metrics(
        true_arr, pred_arr, scaler_ds, split_name, run_dir, ds
    )
    return _metrics_dict(mae, rmse, mard)


def run_finetune(
    *,
    base_run_dir: Path,
    personal_csv: Path,
    out_dir: Path,
    run_name: str | None = None,
    personal_days: int | None = None,
    model_type: str | None = None,
    lwf_lambda: float = 0.5,
    epochs: int = 40,
    lr: float | None = None,
    weight_decay: float | None = None,
    patience: int | None = None,
    val_every_n_epochs: int | None = None,
    progress_log_interval_s: float = DEFAULT_PROGRESS_LOG_INTERVAL_S,
    batch_size: int = 256,
    train_window_stride: int = DEFAULT_TRAIN_WINDOW_STRIDE,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    precision: str = "fp32",
    num_workers: int = 0,
    eval_zero_shot: bool = True,
    from_scratch: bool = False,
    resume_from: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fine-tune on personal train split; return ``(run_dir, results)``."""
    init_cli_console()
    resume_path = Path(resume_from).resolve() if resume_from is not None else None
    if resume_path is not None and not resume_path.is_file():
        raise ValueError(f"resume checkpoint not found: {resume_path}")

    base_run_dir = Path(base_run_dir)
    personal_csv = Path(personal_csv)
    out_dir = Path(out_dir)
    saved_cfg: dict[str, Any] | None = None
    prior_wall_time_s = 0.0
    if resume_path is not None:
        run_dir = resume_path.parent
        saved_cfg = _load_saved_run_config(run_dir)
        prior_wall_time_s = float(saved_cfg.get("wall_time_s", 0) or 0)
        base_run_dir = Path(saved_cfg.get("base_run_dir", base_run_dir))
        personal_csv = Path(saved_cfg.get("personal_csv", personal_csv))
        lwf_lambda = float(saved_cfg.get("lwf_lambda", lwf_lambda))
        epochs = int(saved_cfg.get("epochs", epochs))
        lr = float(saved_cfg["lr"]) if saved_cfg.get("lr") is not None else lr
        weight_decay = (
            float(saved_cfg["weight_decay"])
            if saved_cfg.get("weight_decay") is not None
            else weight_decay
        )
        patience = int(saved_cfg["patience"]) if saved_cfg.get("patience") is not None else patience
        val_every_n_epochs = (
            int(saved_cfg["val_every_n_epochs"])
            if saved_cfg.get("val_every_n_epochs") is not None
            else val_every_n_epochs
        )
        batch_size = int(saved_cfg.get("batch_size", batch_size))
        train_window_stride = int(saved_cfg.get("train_window_stride", train_window_stride))
        seed = int(saved_cfg.get("seed", seed))
        device = str(saved_cfg.get("device", device))
        precision = str(saved_cfg.get("precision", precision))
        num_workers = int(saved_cfg.get("num_workers", num_workers))
        personal_days = saved_cfg.get("personal_days", personal_days)
        from_scratch = bool(saved_cfg.get("from_scratch", from_scratch))
        eval_zero_shot = False

    if not base_run_dir.exists():
        raise ValueError(f"base run dir not found: {base_run_dir}")
    if not personal_csv.exists():
        raise ValueError(f"personal CSV not found: {personal_csv}")
    if not 0.0 <= lwf_lambda <= 1.0:
        raise ValueError(f"lwf_lambda must be in [0, 1], got {lwf_lambda}")
    if train_window_stride < 1:
        raise ValueError(f"train_window_stride must be >= 1, got {train_window_stride}")

    _set_seed(seed)
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch_device.type != "cuda":
        safe_echo("Warning: CUDA requested but unavailable; using CPU.", err=True)

    model, base_meta, resolved_type, ckpt_path = load_base_checkpoint(
        base_run_dir, model_type=model_type, device=torch_device
    )
    if from_scratch:
        model = build_model_from_meta(resolved_type, base_meta, torch_device)
        safe_echo("Training from scratch (base weights discarded).")

    resolved_lr = float(lr if lr is not None else base_meta.get("lr", 4e-4))
    resolved_wd = float(weight_decay if weight_decay is not None else base_meta.get("weight_decay", 3e-5))
    resolved_patience = int(patience if patience is not None else DEFAULT_FT_PATIENCE)
    resolved_val_every = int(
        val_every_n_epochs if val_every_n_epochs is not None else DEFAULT_VAL_EVERY_N_EPOCHS
    )
    t_run_start = time.perf_counter()

    input_steps = int(base_meta.get("input_steps", 128))
    horizon = int(base_meta.get("horizon", 12))

    p_train, p_val, p_test = _load_split_frames(personal_csv)
    if personal_days is not None:
        if p_train.is_empty():
            raise ValueError("personal train split is empty")
        t0 = p_train.select(pl.col("ds").min()).item()
        t_end = t0 + timedelta(days=personal_days)
        p_train = p_train.filter(pl.col("ds") < t_end)
        if p_train.is_empty():
            raise ValueError(f"personal_days={personal_days} left no train rows")

    scaler_ds = SugarOneWindowDataset(
        p_train, input_steps, horizon, fit_scalers=True, window_stride=1
    )

    def _make_ds(
        df: pl.DataFrame,
        *,
        window_stride: int = 1,
    ) -> SugarOneWindowDataset | None:
        if df.is_empty():
            return None
        return SugarOneWindowDataset(
            df,
            input_steps,
            horizon,
            scaler_glucose=scaler_ds.scaler_glucose,
            scaler_basal=scaler_ds.scaler_basal,
            scaler_bolus=scaler_ds.scaler_bolus,
            scaler_carbs=scaler_ds.scaler_carbs,
            window_stride=window_stride,
        )

    personal_train_ds = _make_ds(p_train, window_stride=train_window_stride)
    personal_val_ds = _make_ds(p_val, window_stride=1)
    personal_test_ds = _make_ds(p_test, window_stride=1)
    if personal_train_ds is None or len(personal_train_ds) == 0:
        raise ValueError("no personal train windows (check days / input_steps)")

    if resume_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_run_name = run_name or (
            f"ft_{resolved_type}_d{personal_days or 'all'}_lwf{lwf_lambda:g}_{stamp}"
        )
        run_dir = out_dir / resolved_run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        safe_echo(f"Resuming from {resume_path}")

    cfg: dict[str, Any] = {
        "personalization": True,
        "model_type": resolved_type,
        "base_run_dir": str(base_run_dir),
        "base_checkpoint": str(ckpt_path),
        "personal_csv": str(personal_csv),
        "personal_days": personal_days,
        "lwf_lambda": lwf_lambda,
        "from_scratch": from_scratch,
        "value_columns": dict(SUGAR_ONE_VALUE_COLUMNS),
        "input_steps": input_steps,
        "horizon": horizon,
        "d_model": int(base_meta.get("d_model", 32)),
        "n_heads": int(base_meta.get("n_heads", 8)),
        "ff_units": int(base_meta.get("ff_units", 128)),
        "n_blocks": int(base_meta.get("n_blocks", 5)),
        "dropout": float(base_meta.get("dropout", 0.1)),
        "epochs": epochs,
        "lr": resolved_lr,
        "weight_decay": resolved_wd,
        "patience": resolved_patience,
        "val_every_n_epochs": resolved_val_every,
        "progress_log_interval_s": progress_log_interval_s,
        "batch_size": batch_size,
        "train_window_stride": train_window_stride,
        "eval_window_stride": 1,
        "seed": seed,
        "device": str(torch_device),
        "precision": precision,
        "num_workers": num_workers,
        "prefetch_factor": 4,
        "log_every": 1,
        "ckpt_every_n_epochs": 0,
        "val_every_n_epochs": resolved_val_every,
        "batch_log_every": 0,
        "eval_batch_log_every": 0,
        "log_interval_s": progress_log_interval_s,
        "eval_zero_shot": eval_zero_shot,
        "resume_from": str(resume_path) if resume_path is not None else "",
        "train_windows": len(personal_train_ds),
        "val_windows": len(personal_val_ds) if personal_val_ds else 0,
        "test_windows": len(personal_test_ds) if personal_test_ds else 0,
        "start_time": (
            str(saved_cfg.get("start_time"))
            if saved_cfg is not None and saved_cfg.get("start_time")
            else datetime.now().isoformat()
        ),
    }
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    workers = resolve_num_workers(num_workers, torch_device)
    loader_kwargs = _dataloader_kwargs(num_workers, torch_device, int(cfg["prefetch_factor"]))
    use_amp = torch_device.type == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    results: dict[str, Any] = {"config": cfg}

    if eval_zero_shot and not from_scratch:
        safe_echo("Zero-shot baseline (frozen base weights):")
        zs_model, _, _, _ = load_base_checkpoint(
            base_run_dir, model_type=resolved_type, device=torch_device
        )
        results["zero_shot_test"] = _eval_split(
            zs_model,
            personal_test_ds,
            scaler_ds,
            torch_device,
            batch_size,
            workers,
            run_dir,
            "zero_shot_test",
            use_amp,
            amp_dtype,
            log_interval_s=progress_log_interval_s,
        )

    train_loader = DataLoader(
        personal_train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            personal_val_ds,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if personal_val_ds is not None and len(personal_val_ds) > 0
        else None
    )

    teacher: nn.Module | None = None
    if lwf_lambda > 0.0 and not from_scratch:
        teacher = _make_lwf_teacher(model)

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, resolved_lr, resolved_wd, epochs
    )
    loss_fn = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(torch_device.type == "cuda" and precision == "fp16")
    )

    start_epoch = 1
    best_val_loss = float("inf")
    start_wait = 0
    start_best_epoch = 0
    if resume_path is not None:
        last_done, best_val_loss, start_wait, start_best_epoch = load_full_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            torch_device,
        )
        start_epoch = last_done + 1
        if start_epoch > epochs:
            safe_echo(
                f"Checkpoint already at epoch {last_done} >= max {epochs}; "
                "skipping training loop."
            )
            start_epoch = epochs + 1

    lwf_note = f" | lwf={lwf_lambda}" if teacher is not None else ""
    stride_note = (
        f" | train_stride={train_window_stride}"
        if train_window_stride != DENSE_WINDOW_STRIDE
        else ""
    )
    safe_echo(
        f"Fine-tuning: train={len(personal_train_ds):,} val={len(personal_val_ds) if personal_val_ds else 0:,} "
        f"test={len(personal_test_ds) if personal_test_ds else 0:,} | days={personal_days or 'all'} | "
        f"lr={resolved_lr:g} wd={resolved_wd:g} patience={resolved_patience} "
        f"val_every={resolved_val_every}{lwf_note}{stride_note}"
    )

    model = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        torch_device,
        epochs,
        resolved_patience,
        run_dir,
        cfg,
        teacher=teacher,
        lwf_lambda=lwf_lambda if teacher is not None else 0.0,
        verbose_every=1,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=grad_scaler,
        val_every_n_epochs=resolved_val_every,
        log_interval_s=progress_log_interval_s,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        start_wait=start_wait,
        start_best_epoch=start_best_epoch,
    )

    results["finetuned_val"] = _eval_split(
        model, personal_val_ds, scaler_ds, torch_device, batch_size, workers,
        run_dir, "val", use_amp, amp_dtype,
    )
    results["finetuned_test"] = _eval_split(
        model, personal_test_ds, scaler_ds, torch_device, batch_size, workers,
        run_dir, "test", use_amp, amp_dtype,
    )

    wall_time_s = prior_wall_time_s + (time.perf_counter() - t_run_start)
    results["wall_time_s"] = wall_time_s
    cfg["wall_time_s"] = wall_time_s
    cfg["end_time"] = datetime.now().isoformat()
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with (run_dir / "personalization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    safe_echo(f"\nDone in {timedelta(seconds=int(wall_time_s))}. Run dir: {run_dir}")
    zs = results.get("zero_shot_test")
    ft_test = results.get("finetuned_test")
    if isinstance(zs, dict) and isinstance(ft_test, dict):
        safe_echo(
            f"Comparison test MAE: zero-shot={zs['mae']:.4f} -> fine-tuned={ft_test['mae']:.4f}"
        )
    return run_dir, results


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    personal_csv: Path = typer.Option(..., "--personal-csv"),
    out_dir: Path = typer.Option(Path("data/output/runs/personalization"), "--out-dir"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    personal_days: Optional[int] = typer.Option(
        None,
        "--personal-days",
        help="Limit personal train to first N days. Default: all train rows.",
    ),
    lwf_lambda: float = typer.Option(0.5, "--lwf-lambda"),
    model_type: Optional[str] = typer.Option(None, "--model-type"),
    epochs: int = typer.Option(40, "--epochs"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Default: base model meta lr."),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    patience: Optional[int] = typer.Option(
        None,
        "--patience",
        help=f"Early stopping patience (default: {DEFAULT_FT_PATIENCE}).",
    ),
    val_every_n_epochs: Optional[int] = typer.Option(
        None,
        "--val-every-n-epochs",
        help=f"Validate every N epochs (default: {DEFAULT_VAL_EVERY_N_EPOCHS}).",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    train_window_stride: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_STRIDE,
        "--train-window-stride",
        help="Sliding-window start stride for train split only (default: 6 = 30 min at 5-min sampling).",
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    precision: str = typer.Option("fp32", "--precision"),
    num_workers: int = typer.Option(-1, "--num-workers", help="DataLoader workers (-1 = auto)."),
    eval_zero_shot: bool = typer.Option(True, "--eval-zero-shot/--no-eval-zero-shot"),
    from_scratch: bool = typer.Option(False, "--from-scratch/--no-from-scratch"),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help="Resume from last_checkpoint.pt (reuses run dir, skips zero-shot).",
    ),
) -> None:
    """Fine-tune a global checkpoint on one person's chronological splits."""
    init_cli_console()
    try:
        run_finetune(
            base_run_dir=base_run_dir,
            personal_csv=personal_csv,
            out_dir=out_dir,
            run_name=run_name,
            personal_days=personal_days,
            lwf_lambda=lwf_lambda,
            model_type=model_type,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            val_every_n_epochs=val_every_n_epochs,
            batch_size=batch_size,
            train_window_stride=train_window_stride,
            seed=seed,
            device=device,
            precision=precision,
            num_workers=num_workers,
            eval_zero_shot=eval_zero_shot,
            from_scratch=from_scratch,
            resume_from=resume_from,
        )
    except ValueError as exc:
        safe_echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
