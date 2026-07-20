"""Cross-model comparison reports from unified evaluation results."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from glucose_forecasting.backends.neuralforecast.reporting import (
    write_metrics_figure,
    write_study_group_figure,
)
from glucose_forecasting.evaluation.types import SingleModelResult

_METRICS = ("mae", "rmse", "mard")


def write_comparison_report(
    results: list[SingleModelResult],
    output_dir: Path,
    *,
    plot: bool = True,
) -> Path:
    """Combine multiple ``SingleModelResult`` objects into a comparison report.

    Produces the same directory layout as the NeuralForecast holdout summary
    so downstream tooling can consume either interchangeably.
    """
    if not results:
        raise ValueError("at least one result is required for a comparison report")
    output_dir.mkdir(parents=True, exist_ok=True)

    val_summary = _build_metrics_summary(results, "val")
    test_summary = _build_metrics_summary(results, "test")
    study_group_metrics = _build_study_group_metrics(results)

    if test_summary is not None:
        test_summary.write_csv(output_dir / "test_metrics_summary.csv")
    if val_summary is not None:
        val_summary.write_csv(output_dir / "val_metrics_summary.csv")
    if study_group_metrics is not None:
        study_group_metrics.write_csv(output_dir / "study_group_metrics.csv")

    manifest = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "report_type": "cross_model_comparison",
        "models": [
            {
                "model_name": r.model_name,
                "run_dir": str(r.run_dir),
                "kind": r.kind.value,
            }
            for r in results
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if plot and test_summary is not None and len(results) > 1:
        effective_val = val_summary if val_summary is not None else _empty_metrics_summary()
        write_metrics_figure(
            output_dir,
            val_metrics=effective_val,
            test_metrics=test_summary,
            config={},
        )
        if study_group_metrics is not None and not study_group_metrics.is_empty():
            write_study_group_figure(output_dir, study_group_metrics)

    return output_dir


def _build_metrics_summary(
    results: list[SingleModelResult], split: str
) -> pl.DataFrame | None:
    """One row per model with overall metrics, sorted by MAE."""
    rows = []
    for result in results:
        split_metrics = result.split_results.get(split)
        if split_metrics is None:
            continue
        rows.append(
            {
                "model": result.model_name,
                "mae": split_metrics.overall.mae,
                "rmse": split_metrics.overall.rmse,
                "mard": split_metrics.overall.mard,
            }
        )
    if not rows:
        return None
    return pl.DataFrame(rows).sort("mae")


def _build_study_group_metrics(
    results: list[SingleModelResult],
) -> pl.DataFrame | None:
    """Long-form study-group metrics across all models and splits."""
    parts: list[pl.DataFrame] = []
    for result in results:
        for split_name, split_metrics in result.split_results.items():
            frame = split_metrics.by_study_group
            if frame.is_empty():
                continue
            parts.append(
                frame.with_columns(
                    pl.lit(result.model_name).alias("model"),
                    pl.lit(split_name).alias("split"),
                )
            )
    if not parts:
        return None
    combined = pl.concat(parts, how="vertical")
    expected = ["split", "model", "study_group", "n_points", "mae", "rmse", "mard"]
    present = [col for col in expected if col in combined.columns]
    return combined.select(present)


def _empty_metrics_summary() -> pl.DataFrame:
    return pl.DataFrame(schema={"model": pl.String, "mae": pl.Float64, "rmse": pl.Float64, "mard": pl.Float64})
