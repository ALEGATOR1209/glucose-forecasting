"""GluMind-specific data preparation and sliding-window utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from glucose_forecasting.common.data_loading import (
    apply_split_scheme,
    impute_and_sort as _common_impute_and_sort,
    load_splits_streaming as _common_load_splits_streaming,
)

COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"
COL_GLU = "Glucose Value (mg/dL)"
COL_HR = "Heart Rate"
COL_STEPS = "Step Count"
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load GluMind CSV splits into the canonical frame schema."""
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
        value_columns={"glucose": COL_GLU, "hr": COL_HR, "steps": COL_STEPS},
        ts_format=TS_FORMAT,
    )


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort series and forward/back-fill GluMind continuous signals."""
    return _common_impute_and_sort(df, ffill_bfill_columns=["glucose", "hr", "steps"])


class GlucoseWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy multimodal GluMind sliding-window dataset."""

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_hr: MinMaxScaler | None = None,
        scaler_steps: MinMaxScaler | None = None,
        fit_scalers: bool = False,
    ) -> None:
        self.input_steps = input_steps
        self.horizon = horizon
        window_len = input_steps + horizon

        raw_glucose: list[np.ndarray[Any, Any]] = []
        raw_hr: list[np.ndarray[Any, Any]] = []
        raw_steps: list[np.ndarray[Any, Any]] = []
        uids: list[Any] = []
        sgroups: list[str] = []
        for (uid_val,), grp in df.sort(["unique_id", "ds"]).group_by(
            ["unique_id"], maintain_order=True
        ):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())
            raw_hr.append(grp["hr"].to_numpy())
            raw_steps.append(grp["steps"].to_numpy())

        if fit_scalers or scaler_glucose is None:
            all_g = np.concatenate(raw_glucose).reshape(-1, 1)
            all_h = np.concatenate(raw_hr).reshape(-1, 1)
            all_s = np.concatenate(raw_steps).reshape(-1, 1)
            self.scaler_glucose = MinMaxScaler().fit(all_g)
            self.scaler_hr = MinMaxScaler().fit(all_h)
            self.scaler_steps = MinMaxScaler().fit(all_s)
        else:
            self.scaler_glucose = scaler_glucose
            self.scaler_hr = scaler_hr
            self.scaler_steps = scaler_steps

        self._series_g: list[np.ndarray[Any, Any]] = []
        self._series_h: list[np.ndarray[Any, Any]] = []
        self._series_s: list[np.ndarray[Any, Any]] = []
        self._index: list[tuple[int, int]] = []
        self.series_ids: list[Any] = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, rg, rh, rs) in enumerate(
            zip(uids, sgroups, raw_glucose, raw_hr, raw_steps)
        ):
            g = self.scaler_glucose.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            h = self.scaler_hr.transform(rh.reshape(-1, 1)).ravel().astype(np.float32)
            s = self.scaler_steps.transform(rs.reshape(-1, 1)).ravel().astype(np.float32)
            self._series_g.append(g)
            self._series_h.append(h)
            self._series_s.append(s)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(n_windows):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            print(f"  Note: Skipped {n_skipped} series/segments shorter than {window_len} steps.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_idx, start = self._index[idx]
        g = self._series_g[series_idx]
        h = self._series_h[series_idx]
        s = self._series_s[series_idx]
        x = np.stack(
            [
                g[start : start + self.input_steps],
                h[start : start + self.input_steps],
                s[start : start + self.input_steps],
            ],
            axis=-1,
        )
        y = g[start + self.input_steps : start + self.input_steps + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    args: Any,
) -> tuple[GlucoseWindowDataset, GlucoseWindowDataset | None, GlucoseWindowDataset | None]:
    """Build GluMind datasets, fitting scalers exclusively on training data."""
    train_ds = GlucoseWindowDataset(train_df, args.input_steps, args.horizon, fit_scalers=True)
    val_ds = (
        GlucoseWindowDataset(
            val_df,
            args.input_steps,
            args.horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
        )
        if not val_df.is_empty()
        else None
    )
    test_ds = (
        GlucoseWindowDataset(
            test_df,
            args.input_steps,
            args.horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
        )
        if not test_df.is_empty()
        else None
    )
    return train_ds, val_ds, test_ds


__all__ = [
    "COL_EVENT", "COL_GLU", "COL_GROUP", "COL_HR", "COL_SEQ", "COL_SPLIT",
    "COL_STEPS", "COL_TS", "COL_USER", "TS_FORMAT", "GlucoseWindowDataset",
    "apply_split_scheme", "build_datasets", "impute_and_sort", "load_splits_streaming",
]
