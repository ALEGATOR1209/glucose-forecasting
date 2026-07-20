"""Live inference adapter for custom PyTorch model run directories.

Supports GluMind, SugarOne, and SugarJEPA checkpoints.  Loads weights,
fits scalers on training data, runs inference on evaluation data, and
returns a ``SingleModelResult`` with metrics in the original glucose
scale (mg/dL).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glucose_forecasting.backends.neuralforecast.benchmark import RegressionMetrics
from glucose_forecasting.common.checkpoint import strip_compile_prefix
from glucose_forecasting.common.evaluation import (
    _load_csv_flexible as _common_load_csv_flexible,
)
from glucose_forecasting.common.metrics import mae_rmse_mard, per_study_group_breakdown
from glucose_forecasting.evaluation.types import (
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)

ModelKind = Literal["glumind", "sugar_one", "sugar_jepa"]

_COL_SEQ = "sequence_id"
_COL_USER = "User ID"
_COL_SPLIT = "Recommended Split"
_COL_GROUP = "Study Group"
_TS_ALIASES = [
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
    "Timestamp",
]
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def evaluate_pytorch_run_dir(
    run_dir: Path,
    model_name: str,
    *,
    data: Path,
    train_data: Path | None = None,
    device: str = "auto",
    output_dir: Path | None = None,
) -> SingleModelResult:
    """Load a PyTorch checkpoint and run inference on *data*.

    When *train_data* is ``None`` the training CSV is resolved from the
    run directory metadata (``tuning_meta.json`` or ``config.json``).
    """
    meta = _load_meta(run_dir)
    ckpt_path = _resolve_checkpoint(run_dir)
    resolved_device = _resolve_device(device)
    kind = _detect_model_kind(meta, ckpt_path, resolved_device)

    train_df = _load_train_data(data, kind, meta, train_data)
    train_ds = _build_train_dataset(train_df, kind, meta)

    write_dir = output_dir or run_dir
    split_results: dict[str, SplitMetrics] = {}

    splits_to_try = [("test", "test"), ("val", "val")]
    for split_name, eval_split in splits_to_try:
        eval_df = _load_eval_data(data, kind, eval_split)
        if eval_df is None or eval_df.is_empty():
            continue
        split_results[split_name] = _eval_one_split(
            eval_df, train_ds, kind, meta, ckpt_path,
            resolved_device, model_name, split_name, write_dir,
        )

    if not split_results:
        all_df = _load_eval_data(data, kind, None)
        if all_df is not None and not all_df.is_empty():
            print(
                f"  no test/val split found — evaluating all {len(all_df):,} rows as 'test'",
                file=sys.stderr,
            )
            split_results["test"] = _eval_one_split(
                all_df, train_ds, kind, meta, ckpt_path,
                resolved_device, model_name, "test", write_dir,
            )

    if not split_results:
        raise ValueError(f"no evaluation data found for any split in {data}")

    return SingleModelResult(
        model_name=model_name,
        run_dir=run_dir,
        kind=RunDirKind.CUSTOM_PYTORCH,
        split_results=split_results,
    )


def _eval_one_split(
    eval_df: pl.DataFrame,
    train_ds,
    kind: ModelKind,
    meta: dict,
    ckpt_path: Path,
    device: str,
    model_name: str,
    split_name: str,
    write_dir: Path,
) -> SplitMetrics:
    eval_ds = _build_eval_dataset(eval_df, train_ds, kind, meta)
    if len(eval_ds) == 0:
        raise ValueError(f"eval dataset for {split_name} has 0 windows")
    batch_size = meta.get("batch_size", 4096)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    model = _build_model(kind, meta)
    _load_model_weights(model, ckpt_path, device)

    y_true_scaled, y_pred_scaled = _run_inference(model, loader, device, kind)

    y_true = train_ds.scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = train_ds.scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    overall = RegressionMetrics(mae=mae, rmse=rmse, mard=mard)

    by_group_frame = per_study_group_breakdown(
        y_true_scaled, y_pred_scaled, train_ds.scaler_glucose,
        getattr(eval_ds, "study_groups", []),
    )
    if by_group_frame is not None and "n_windows" in by_group_frame.columns:
        by_group_frame = by_group_frame.rename({"n_windows": "n_points"})
    if by_group_frame is None:
        by_group_frame = pl.DataFrame(
            schema={"study_group": pl.String, "n_points": pl.UInt32,
                    "mae": pl.Float64, "rmse": pl.Float64, "mard": pl.Float64}
        )

    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        write_dir / f"{split_name}_metrics_overall.csv"
    )
    by_group_frame.write_csv(write_dir / f"{split_name}_metrics_by_study_group.csv")

    print(
        f"  {model_name} {split_name}: MAE={mae:.3f}  RMSE={rmse:.3f}  MARD={mard:.3f}%",
        file=sys.stderr,
    )
    return SplitMetrics(overall=overall, by_study_group=by_group_frame)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_meta(run_dir: Path) -> dict:
    for name in ("tuning_meta.json", "config.json"):
        path = run_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"no tuning_meta.json or config.json in {run_dir}")


def _resolve_checkpoint(run_dir: Path) -> Path:
    for name in ("best_model.pt", "last_model.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise ValueError(f"no best_model.pt or last_model.pt in {run_dir}")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_model_kind(meta: dict, ckpt_path: Path, device: str) -> ModelKind:
    for key in ("model_type", "model"):
        value = meta.get(key)
        if isinstance(value, str):
            normalized = value.lower().replace("_", "").replace("-", "")
            if normalized in ("sugarjepa", "sugar_jepa"):
                return "sugar_jepa"
            if normalized in ("sugarone", "glumindic"):
                return "sugar_one"
            if normalized == "glumind":
                return "glumind"

    if "jepa_weights_dir" in meta:
        return "sugar_jepa"

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    keys = {k.removeprefix("_orig_mod.") for k in state}
    if "embed_basal.weight" in keys and "embed_bolus.weight" in keys:
        if any("jepa" in k.lower() for k in keys):
            return "sugar_jepa"
        return "sugar_one"
    if "embed_hr.weight" in keys and "embed_steps.weight" in keys:
        return "glumind"
    raise ValueError(f"cannot detect model kind from {ckpt_path}")


def _load_train_data(
    eval_csv: Path, kind: ModelKind, meta: dict, train_csv: Path | None
) -> pl.DataFrame:
    """Load training data for scaler fitting."""
    if train_csv is not None:
        csv_path = train_csv
    else:
        raw = meta.get("csv", "")
        csv_path = _resolve_csv_path(raw, eval_csv)

    model_kind_2 = "sugar_one" if kind == "sugar_jepa" else kind
    df = _common_load_csv_flexible(
        csv_path, model_kind_2, "sequence_id", False, None, True,
        col_seq=_COL_SEQ, col_user=_COL_USER, col_split=_COL_SPLIT,
        col_group=_COL_GROUP, ts_aliases=_TS_ALIASES, ts_format=_TS_FORMAT,
    )
    if df.is_empty():
        df = _common_load_csv_flexible(
            csv_path, model_kind_2, "sequence_id", False, None, False,
            col_seq=_COL_SEQ, col_user=_COL_USER, col_split=_COL_SPLIT,
            col_group=_COL_GROUP, ts_aliases=_TS_ALIASES, ts_format=_TS_FORMAT,
        )
    return _impute(df, kind)


def _load_eval_data(
    csv_path: Path, kind: ModelKind, eval_split: str | None
) -> pl.DataFrame | None:
    model_kind_2 = "sugar_one" if kind == "sugar_jepa" else kind
    try:
        df = _common_load_csv_flexible(
            csv_path, model_kind_2, "sequence_id", False, eval_split, False,
            col_seq=_COL_SEQ, col_user=_COL_USER, col_split=_COL_SPLIT,
            col_group=_COL_GROUP, ts_aliases=_TS_ALIASES, ts_format=_TS_FORMAT,
        )
    except (ValueError, KeyError):
        return None
    if df.is_empty():
        return None
    return _impute(df, kind)


def _impute(df: pl.DataFrame, kind: ModelKind) -> pl.DataFrame:
    from glucose_forecasting.common.data_loading import impute_and_sort as _impute_sort

    if kind in ("sugar_one", "sugar_jepa"):
        return _impute_sort(
            df, ffill_bfill_columns=["glucose", "basal"],
            zero_fill_columns=["bolus", "carbs"],
        )
    return _impute_sort(
        df, ffill_bfill_columns=["glucose", "hr"],
        zero_fill_columns=["steps"],
    )


def _resolve_csv_path(raw: str, fallback: Path) -> Path:
    """Resolve a CSV path from metadata, falling back to the eval CSV."""
    if not raw:
        return fallback
    normalized = raw.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_file():
        return candidate
    if fallback.parent != Path("."):
        relative = fallback.parent / candidate.name
        if relative.is_file():
            return relative
    return fallback


def _build_train_dataset(df: pl.DataFrame, kind: ModelKind, meta: dict):
    """Build a dataset with ``fit_scalers=True`` to train scalers."""
    input_steps = meta.get("input_steps", 128)
    horizon = meta.get("horizon", 12)

    if kind == "sugar_jepa":
        from scripts.sugar_jepa.train_sugar_jepa import SugarJepaWindowDataset
        return SugarJepaWindowDataset(
            df, input_steps=input_steps, horizon=horizon,
            jepa_window=meta.get("jepa_window", 288),
            fit_scalers=True,
        )
    if kind == "sugar_one":
        from glucose_forecasting.data.sugar_one import SugarOneWindowDataset
        return SugarOneWindowDataset(
            df, input_steps=input_steps, horizon=horizon, fit_scalers=True,
        )
    from glucose_forecasting.data.glumind import GlucoseWindowDataset
    return GlucoseWindowDataset(
        df, input_steps=input_steps, horizon=horizon, fit_scalers=True,
    )


def _build_eval_dataset(df: pl.DataFrame, train_ds, kind: ModelKind, meta: dict):
    """Build a dataset with pre-fitted scalers from the training dataset."""
    input_steps = meta.get("input_steps", 128)
    horizon = meta.get("horizon", 12)

    if kind == "sugar_jepa":
        from scripts.sugar_jepa.train_sugar_jepa import SugarJepaWindowDataset
        return SugarJepaWindowDataset(
            df, input_steps=input_steps, horizon=horizon,
            jepa_window=meta.get("jepa_window", 288),
            scaler_glucose=train_ds.scaler_glucose,
            scaler_basal=train_ds.scaler_basal,
            scaler_bolus=train_ds.scaler_bolus,
            scaler_carbs=train_ds.scaler_carbs,
            scaler_glucose_jepa=train_ds.scaler_glucose_jepa,
            fit_scalers=False,
        )
    if kind == "sugar_one":
        from glucose_forecasting.data.sugar_one import SugarOneWindowDataset
        return SugarOneWindowDataset(
            df, input_steps=input_steps, horizon=horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_basal=train_ds.scaler_basal,
            scaler_bolus=train_ds.scaler_bolus,
            scaler_carbs=train_ds.scaler_carbs,
            fit_scalers=False,
        )
    from glucose_forecasting.data.glumind import GlucoseWindowDataset
    return GlucoseWindowDataset(
        df, input_steps=input_steps, horizon=horizon,
        scaler_glucose=train_ds.scaler_glucose,
        scaler_hr=train_ds.scaler_hr,
        scaler_steps=train_ds.scaler_steps,
        fit_scalers=False,
    )


def _build_model(kind: ModelKind, meta: dict) -> nn.Module:
    common = dict(
        n_time_steps=meta.get("input_steps", 128),
        d_model=meta.get("d_model", 32),
        n_heads=meta.get("n_heads", 8),
        ff_units=meta.get("ff_units", 128),
        n_blocks=meta.get("n_blocks", 5),
        prediction_horizon=meta.get("horizon", 12),
        dropout=meta.get("dropout", 0.1),
    )
    if kind == "sugar_jepa":
        from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel
        return SugarJepaModel(
            **common,
            n_features=4,
            jepa_weights_dir=meta.get("jepa_weights_dir", "scripts/sugar_jepa/pretrained/cgm_jepa"),
            jepa_patch_size=meta.get("jepa_patch_size", 12),
            jepa_freeze=not meta.get("finetune_jepa", False),
        )
    if kind == "sugar_one":
        from glucose_forecasting.models.sugar_one import SugarOneModel
        return SugarOneModel(n_features=4, **common)
    from glucose_forecasting.models.glumind import GluMindModel
    return GluMindModel(n_features=3, **common)


def _load_model_weights(model: nn.Module, ckpt_path: Path, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compile_prefix(state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


@torch.no_grad()
def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    kind: ModelKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, handling both 2-tensor and 3-tensor (SugarJEPA) batches."""
    model.eval()
    device_t = torch.device(device)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    t_start = time.perf_counter()
    n_total = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        if kind == "sugar_jepa":
            x, jepa, y = batch
            x, jepa, y = x.to(device_t), jepa.to(device_t), y.to(device_t)
            pred = model(x, jepa)
        else:
            x, y = batch
            x, y = x.to(device_t), y.to(device_t)
            pred = model(x)
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

        if batch_idx == 1 or batch_idx == n_total or batch_idx % 50 == 0:
            elapsed = time.perf_counter() - t_start
            pct = 100.0 * batch_idx / n_total
            eta = (elapsed / batch_idx) * (n_total - batch_idx) if batch_idx > 0 else 0
            print(
                f"  inference {batch_idx:,}/{n_total:,} ({pct:.0f}%) "
                f"elapsed {timedelta(seconds=int(elapsed))} "
                f"ETA {timedelta(seconds=int(eta))}",
                file=sys.stderr,
            )

    return (
        np.concatenate(all_true) if all_true else np.array([]),
        np.concatenate(all_pred) if all_pred else np.array([]),
    )
