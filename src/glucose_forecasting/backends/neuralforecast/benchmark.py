"""Reusable primitives for fixed-split NeuralForecast holdout evaluation.

Loading, orchestration, model fitting, and command-line concerns live in the
profile-aware adapter and evaluation modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import polars as pl


CatalogModelName = Literal["tft", "nhits", "nbeatsx"]

_BASE_COLUMNS = ("unique_id", "ds", "y")
_METRIC_COLUMNS: dict[str, pl.DataType] = {
    "study_group": pl.String,
    "n_points": pl.UInt32,
    "mae": pl.Float64,
    "rmse": pl.Float64,
    "mard": pl.Float64,
}


@dataclass(frozen=True)
class BenchmarkModelConfig:
    """Training configuration retained for direct fixed-split model construction."""

    model: CatalogModelName
    learning_rate: float
    max_steps: int
    val_check_steps: int
    batch_size: int
    valid_batch_size: int
    windows_batch_size: int
    inference_windows_batch_size: int
    step_size: int


@dataclass(frozen=True)
class RegressionMetrics:
    """MAE, RMSE, and MARD values in the original glucose scale."""

    mae: float
    rmse: float
    mard: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Overall and per-study-group metrics for one evaluated split."""

    overall: RegressionMetrics
    by_study_group: pl.DataFrame


def to_neuralforecast_frame(
    split: pl.DataFrame,
    historical_exogenous: Sequence[str] = (),
) -> pl.DataFrame:
    """Select and sort normalized split columns for ``NeuralForecast.fit/predict``.

    The input must already use the normalized names ``unique_id``, ``ds``, and
    ``y``.  Metadata columns such as ``study_group`` are intentionally omitted.
    Pandas is imported only when this conversion is requested because
    NeuralForecast natively accepts Polars frames.
    """
    columns = (*_BASE_COLUMNS, *historical_exogenous)
    _require_columns(split, columns)

    if len(set(columns)) != len(columns):
        raise ValueError("historical_exogenous must not repeat NeuralForecast columns.")

    return split.select(columns).sort(["unique_id", "ds"])


def build_catalog_model(
    config: BenchmarkModelConfig,
    *,
    horizon: int,
    input_size: int,
    historical_exogenous: Sequence[str],
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Construct one configured legacy benchmark model using lazy imports."""
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NBEATSx, NHITS, TFT

    model_classes: Mapping[CatalogModelName, type[Any]] = {
        "tft": TFT,
        "nhits": NHITS,
        "nbeatsx": NBEATSx,
    }
    kwargs = {
        "h": horizon,
        "input_size": input_size,
        "loss": MAE(),
        "valid_loss": MAE(),
        "max_steps": config.max_steps,
        "val_check_steps": min(config.val_check_steps, config.max_steps),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "valid_batch_size": config.valid_batch_size,
        "windows_batch_size": config.windows_batch_size,
        "inference_windows_batch_size": config.inference_windows_batch_size,
        "step_size": config.step_size,
        "hist_exog_list": list(historical_exogenous),
        **dict(trainer_kwargs or {}),
    }
    return model_classes[config.model](**kwargs)


def calculate_metrics(
    predictions: pl.DataFrame,
    *,
    target_column: str = "y",
    prediction_column: str = "yhat",
    study_group_column: str = "study_group",
) -> BenchmarkMetrics:
    """Calculate holdout overall and per-study-group MAE/RMSE/MARD."""
    _require_columns(
        predictions, (target_column, prediction_column, study_group_column)
    )
    if predictions.is_empty():
        raise ValueError("Cannot calculate metrics from an empty prediction frame.")

    overall = _calculate_regression_metrics(
        predictions[target_column].to_numpy(), predictions[prediction_column].to_numpy()
    )
    groups = predictions.partition_by(study_group_column, as_dict=True)
    rows = [
        {
            "study_group": str(group[0] if isinstance(group, tuple) else group),
            "n_points": group_frame.height,
            **_calculate_regression_metrics(
                group_frame[target_column].to_numpy(),
                group_frame[prediction_column].to_numpy(),
            ).__dict__,
        }
        for group, group_frame in groups.items()
    ]
    by_study_group = (
        pl.DataFrame(rows, schema=_METRIC_COLUMNS).sort("mae")
        if rows
        else pl.DataFrame(schema=_METRIC_COLUMNS)
    )
    return BenchmarkMetrics(overall=overall, by_study_group=by_study_group)


def _calculate_regression_metrics(
    y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]
) -> RegressionMetrics:
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    error = true - predicted
    nonzero = true != 0
    return RegressionMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error * error))),
        mard=(
            float(np.mean(np.abs(error[nonzero]) / np.abs(true[nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        ),
    )


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(f"Frame is missing required columns: {formatted}")
