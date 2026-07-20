#!/usr/bin/env python3
"""Shared regression metrics for glucose forecasting models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler


def mae_rmse_mard(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Compute MAE, RMSE, MARD (same formula across all training scripts)."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    nonzero = y_true != 0
    if nonzero.any():
        mard = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100)
    else:
        mard = float("nan")
    return mae, rmse, mard


def per_study_group_breakdown(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    scaler_glucose: MinMaxScaler,
    study_groups: list[str],
) -> pl.DataFrame | None:
    """Compute per-study-group MAE/RMSE/MARD breakdown, sorted by MAE.

    Returns None if ``study_groups`` length doesn't match ``true_arr``.
    """
    if len(study_groups) != len(true_arr):
        return None
    groups = np.array(study_groups)
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
    return pl.DataFrame(rows).sort("mae")


def overall_metrics_to_csv(
    mae: float, rmse: float, mard: float, run_dir: Path, split_name: str
) -> None:
    """Save overall MAE/RMSE/MARD to ``{split_name}_metrics_overall.csv``."""
    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )
