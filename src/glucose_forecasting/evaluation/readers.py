"""Read precomputed metrics CSVs from any backend's run directory."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from glucose_forecasting.backends.neuralforecast.benchmark import RegressionMetrics
from glucose_forecasting.evaluation.types import (
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)

_SPLITS = ("val", "test")
_CANONICAL_COUNT_COLUMN = "n_points"
_LEGACY_COUNT_COLUMN = "n_windows"


def read_precomputed_result(
    run_dir: Path,
    model_name: str,
    *,
    kind: RunDirKind = RunDirKind.PRECOMPUTED,
) -> SingleModelResult:
    """Build a ``SingleModelResult`` from on-disk metrics CSVs.

    Normalises the per-study-group count column to ``n_points`` regardless
    of whether the source uses ``n_windows`` (custom PyTorch) or ``n_points``
    (NeuralForecast).
    """
    split_results: dict[str, SplitMetrics] = {}
    for split_name in _SPLITS:
        overall_path = run_dir / f"{split_name}_metrics_overall.csv"
        if not overall_path.is_file():
            continue
        overall_frame = pl.read_csv(overall_path)
        overall = RegressionMetrics(
            mae=overall_frame["mae"].item(),
            rmse=overall_frame["rmse"].item(),
            mard=overall_frame["mard"].item(),
        )
        group_path = run_dir / f"{split_name}_metrics_by_study_group.csv"
        if group_path.is_file():
            by_group = _normalize_group_frame(pl.read_csv(group_path))
        else:
            by_group = _empty_group_frame()
        split_results[split_name] = SplitMetrics(overall=overall, by_study_group=by_group)
    if not split_results:
        raise ValueError(f"no precomputed metrics found in {run_dir}")
    return SingleModelResult(
        model_name=model_name,
        run_dir=run_dir,
        kind=kind,
        split_results=split_results,
    )


def _normalize_group_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Rename ``n_windows`` to ``n_points`` and enforce column order."""
    if _LEGACY_COUNT_COLUMN in frame.columns and _CANONICAL_COUNT_COLUMN not in frame.columns:
        frame = frame.rename({_LEGACY_COUNT_COLUMN: _CANONICAL_COUNT_COLUMN})
    expected = ["study_group", _CANONICAL_COUNT_COLUMN, "mae", "rmse", "mard"]
    present = [col for col in expected if col in frame.columns]
    return frame.select(present)


def _empty_group_frame() -> pl.DataFrame:
    """Return an empty frame with the canonical study-group schema."""
    return pl.DataFrame(
        schema={
            "study_group": pl.String,
            _CANONICAL_COUNT_COLUMN: pl.UInt32,
            "mae": pl.Float64,
            "rmse": pl.Float64,
            "mard": pl.Float64,
        }
    )
