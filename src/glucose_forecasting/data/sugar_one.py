"""SugarOne-specific data preparation and sliding-window utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import typer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from glucose_forecasting.common.data_loading import (
    apply_split_scheme as _common_apply_split_scheme,
    impute_and_sort as _common_impute_and_sort,
    load_splits_streaming as _common_load_splits_streaming,
)

COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"
COL_GLU = "Glucose (mg/dL)"
COL_BASAL = "Basal Rate (U/h)"
COL_BOLUS = "Bolus Insulin (U)"
COL_CARB = "Carbohydrates (g)"
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"
N_FEATURES = 4


def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load SugarOne CSV splits into the canonical frame schema."""
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
            "glucose": COL_GLU,
            "basal": COL_BASAL,
            "bolus": COL_BOLUS,
            "carbs": COL_CARB,
        },
        ts_format=TS_FORMAT,
        utf8_value_columns=("basal", "bolus", "carbs"),
        log_fn=typer.echo,
    )


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply SugarOne's logged tuning split scheme."""
    return _common_apply_split_scheme(
        train_df,
        val_df,
        test_df,
        split_scheme,
        log_fn=typer.echo,
        applied_message="Applied split scheme: train <- train+val | val <- test | test disabled.",
        note_message="Note: tuning-only mode; no held-out test metrics.",
        error_repr=True,
    )


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort and impute continuous basal/glucose and discrete event covariates."""
    return _common_impute_and_sort(
        df,
        ffill_bfill_columns=["glucose", "basal"],
        zero_fill_columns=["bolus", "carbs"],
    )


class SugarOneWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy sliding-window dataset with glucose, insulin, and carbohydrate features."""

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_basal: MinMaxScaler | None = None,
        scaler_bolus: MinMaxScaler | None = None,
        scaler_carbs: MinMaxScaler | None = None,
        fit_scalers: bool = False,
        window_stride: int = 1,
    ) -> None:
        self.input_steps = input_steps
        self.horizon = horizon
        if window_stride < 1:
            raise ValueError(f"window_stride must be >= 1, got {window_stride}")
        self.window_stride = window_stride
        window_len = input_steps + horizon

        raw_glucose: list[np.ndarray[Any, Any]] = []
        raw_basal: list[np.ndarray[Any, Any]] = []
        raw_bolus: list[np.ndarray[Any, Any]] = []
        raw_carbs: list[np.ndarray[Any, Any]] = []
        uids: list[Any] = []
        sgroups: list[str] = []
        for (uid_val,), grp in df.sort(["unique_id", "ds"]).group_by(
            ["unique_id"], maintain_order=True
        ):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())
            raw_basal.append(grp["basal"].to_numpy())
            raw_bolus.append(grp["bolus"].to_numpy())
            raw_carbs.append(grp["carbs"].to_numpy())

        if fit_scalers or scaler_glucose is None:
            self.scaler_glucose = MinMaxScaler().fit(np.concatenate(raw_glucose).reshape(-1, 1))
            self.scaler_basal = MinMaxScaler().fit(np.concatenate(raw_basal).reshape(-1, 1))
            self.scaler_bolus = MinMaxScaler().fit(np.concatenate(raw_bolus).reshape(-1, 1))
            self.scaler_carbs = MinMaxScaler().fit(np.concatenate(raw_carbs).reshape(-1, 1))
        else:
            self.scaler_glucose = scaler_glucose
            self.scaler_basal = scaler_basal
            self.scaler_bolus = scaler_bolus
            self.scaler_carbs = scaler_carbs

        self._series_g: list[np.ndarray[Any, Any]] = []
        self._series_b: list[np.ndarray[Any, Any]] = []
        self._series_bo: list[np.ndarray[Any, Any]] = []
        self._series_c: list[np.ndarray[Any, Any]] = []
        self._index: list[tuple[int, int]] = []
        self.series_ids: list[Any] = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, rg, rb, rbo, rc) in enumerate(
            zip(uids, sgroups, raw_glucose, raw_basal, raw_bolus, raw_carbs)
        ):
            g = self.scaler_glucose.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            b = self.scaler_basal.transform(rb.reshape(-1, 1)).ravel().astype(np.float32)
            bo = self.scaler_bolus.transform(rbo.reshape(-1, 1)).ravel().astype(np.float32)
            c = self.scaler_carbs.transform(rc.reshape(-1, 1)).ravel().astype(np.float32)
            self._series_g.append(g)
            self._series_b.append(b)
            self._series_bo.append(bo)
            self._series_c.append(c)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(0, n_windows, window_stride):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            typer.echo(f"  Note: Skipped {n_skipped} series shorter than {window_len} steps.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_idx, start = self._index[idx]
        end = start + self.input_steps
        x = np.stack(
            [
                self._series_g[series_idx][start:end],
                self._series_b[series_idx][start:end],
                self._series_bo[series_idx][start:end],
                self._series_c[series_idx][start:end],
            ],
            axis=-1,
        )
        y = self._series_g[series_idx][end : end + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    input_steps: int,
    horizon: int,
) -> tuple[SugarOneWindowDataset, SugarOneWindowDataset | None, SugarOneWindowDataset | None]:
    """Build SugarOne datasets, fitting scalers exclusively on training data."""
    train_ds = SugarOneWindowDataset(train_df, input_steps, horizon, fit_scalers=True)
    scaler_kwargs = {
        "scaler_glucose": train_ds.scaler_glucose,
        "scaler_basal": train_ds.scaler_basal,
        "scaler_bolus": train_ds.scaler_bolus,
        "scaler_carbs": train_ds.scaler_carbs,
    }
    val_ds = (
        SugarOneWindowDataset(val_df, input_steps, horizon, **scaler_kwargs)
        if not val_df.is_empty()
        else None
    )
    test_ds = (
        SugarOneWindowDataset(test_df, input_steps, horizon, **scaler_kwargs)
        if not test_df.is_empty()
        else None
    )
    return train_ds, val_ds, test_ds


__all__ = [
    "COL_BASAL", "COL_BOLUS", "COL_CARB", "COL_EVENT", "COL_GLU", "COL_GROUP",
    "COL_SEQ", "COL_SPLIT", "COL_TS", "COL_USER", "N_FEATURES", "TS_FORMAT",
    "SugarOneWindowDataset", "apply_split_scheme", "build_datasets",
    "impute_and_sort", "load_splits_streaming",
]
