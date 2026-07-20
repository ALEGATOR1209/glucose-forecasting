"""Plan registry-backed evaluations and write immutable result records.

This module deliberately separates compatibility and artifact resolution from
model-specific inference.  Legacy run directories and validated release bundles
are recognized, but they are not assigned synthetic metrics when their required
preprocessing adapter is unavailable.

The ``evaluate_run_dir`` and ``evaluate_and_compare`` entry points provide
the unified run-directory evaluation path (``glucose evaluate --run-dir``).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from glucose_forecasting.config import DatasetSpec, ModelSelection
from glucose_forecasting.evaluation.comparison import write_comparison_report
from glucose_forecasting.evaluation.detect import detect_run_dir, infer_model_name
from glucose_forecasting.evaluation.readers import read_precomputed_result
from glucose_forecasting.evaluation.types import RunDirKind, SingleModelResult
from glucose_forecasting.models.registry import ModelArtifact, ModelRegistry, resolve_data_path
from glucose_forecasting.release import validate_inference_bundle

MetricName = Literal["mae", "rmse", "mard"]
_METRICS: tuple[MetricName, ...] = ("mae", "rmse", "mard")
_LOOP_COLUMNS = {
    "Basal Rate (U/h)": "basal",
    "Bolus Insulin (U)": "bolus",
    "Carbohydrates (g)": "carbohydrates",
}
_AI_READI_COLUMNS = {"Heart Rate": "heart_rate", "Step Count": "steps"}


@dataclass(frozen=True)
class EvaluationRun:
    """Locations and records produced by one evaluation invocation."""

    output_dir: Path
    manifest_path: Path
    metrics_csv_path: Path
    metrics_json_path: Path
    records: tuple[dict[str, object], ...]


def expand_values(values: list[str]) -> list[str]:
    """Expand repeatable comma-separated command-line values deterministically."""
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def infer_dataset_spec(data_path: Path) -> DatasetSpec:
    """Infer the supported input contract from a CSV header without coercing data."""
    if not data_path.is_file():
        raise FileNotFoundError(f"data file not found: {data_path}")

    with data_path.open(encoding="utf-8", newline="") as data_file:
        header = set(next(csv.reader(data_file), []))

    loop_covariates = tuple(
        canonical for source, canonical in _LOOP_COLUMNS.items() if source in header
    )
    ai_readi_covariates = tuple(
        canonical for source, canonical in _AI_READI_COLUMNS.items() if source in header
    )
    if loop_covariates and ai_readi_covariates:
        raise ValueError(
            "CSV mixes Loop and AI-READI covariate columns; choose a dataset with one schema."
        )
    if loop_covariates:
        schema, covariates = "loop-v1", loop_covariates
    elif ai_readi_covariates:
        schema, covariates = "ai-readi-v1", ai_readi_covariates
    else:
        raise ValueError(
            "could not infer dataset schema from CSV columns; expected Loop or AI-READI covariates"
        )

    return DatasetSpec(
        name=data_path.stem,
        path=data_path,
        data_schema=schema,
        covariates=covariates,
        cadence_minutes=5,
        horizon_steps=12,
    )


def parse_model_selection(value: str) -> ModelSelection:
    """Parse ``NAME`` or ``NAME@VERSION`` while keeping version selection explicit."""
    name, separator, version = value.partition("@")
    if not name or (separator and not version):
        raise ValueError(f"invalid model selector {value!r}; use NAME or NAME@VERSION")
    return ModelSelection(name=name, version=version or None)


def _resolve_artifact_path(
    artifact: ModelArtifact, *, project_root: Path, registry_path: Path
) -> Path:
    """Resolve registry artifact paths for project-local and colocated registries."""
    if artifact.artifact_path.is_absolute():
        return artifact.artifact_path

    project_candidate = project_root / artifact.artifact_path
    registry_candidate = registry_path.parent / artifact.artifact_path
    if project_candidate.exists() or not registry_candidate.exists():
        return project_candidate
    return registry_candidate


def _artifact_readiness(artifact_path: Path) -> tuple[str, str | None]:
    """Classify an artifact without loading model weights or emitting metrics."""
    if not artifact_path.exists():
        return "skipped_artifact_missing", f"artifact not found: {artifact_path}"
    if artifact_path.is_dir() and (artifact_path / "manifest.json").is_file():
        try:
            validate_inference_bundle(artifact_path)
        except (OSError, ValueError) as error:
            return "skipped_artifact_invalid", f"release bundle validation failed: {error}"
        return (
            "not_evaluated",
            "release bundle is valid, but a model-specific inference adapter is not available",
        )
    if artifact_path.is_dir() and (
        (artifact_path / "best_model.pt").is_file() or (artifact_path / "last_model.pt").is_file()
    ) and (
        (artifact_path / "config.json").is_file() or (artifact_path / "tuning_meta.json").is_file()
    ):
        return (
            "not_evaluated",
            "legacy run directory recognized; model-specific preprocessing is not available",
        )
    return (
        "skipped_artifact_invalid",
        "artifact must be a validated release bundle or a legacy run directory "
        "with checkpoint and config metadata",
    )


def _base_record(
    *,
    dataset: DatasetSpec,
    selector: str,
    artifact: ModelArtifact | None,
    status: str,
    reason: str | None,
    artifact_path: Path | None = None,
) -> dict[str, object]:
    """Build the model/dataset fields shared by every long-form metric record."""
    return {
        "dataset": dataset.name,
        "dataset_path": str(dataset.path),
        "data_schema": dataset.data_schema,
        "dataset_covariates": list(dataset.covariates),
        "requested_model": selector,
        "model_name": artifact.name if artifact is not None else None,
        "model_version": artifact.version if artifact is not None else None,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "status": status,
        "reason": reason,
    }


def _metric_records(base: dict[str, object]) -> list[dict[str, object]]:
    """Emit one null-valued long-form record per standard metric."""
    return [{**base, "metric": metric, "value": None} for metric in _METRICS]


def _records_for_dataset(
    *,
    dataset: DatasetSpec,
    registry: ModelRegistry,
    registry_path: Path,
    project_root: Path,
    selectors: list[str],
) -> list[dict[str, object]]:
    """Resolve requested models for one dataset and preserve every outcome."""
    requested = selectors or ["default"]
    records: list[dict[str, object]] = []
    for selector in requested:
        selection = ModelSelection() if selector == "default" else parse_model_selection(selector)
        candidates = registry.models
        if selection.name is not None:
            candidates = tuple(
                artifact for artifact in candidates if artifact.name == selection.name
            )
        elif selection.version is not None:
            candidates = ()
        elif selector == "default":
            candidates = tuple(artifact for artifact in candidates if artifact.stable)
        if selection.version is not None:
            candidates = tuple(
                artifact for artifact in candidates if artifact.version == selection.version
            )

        if not candidates:
            records.extend(
                _metric_records(
                    _base_record(
                        dataset=dataset,
                        selector=selector,
                        artifact=None,
                        status="skipped_model_not_found",
                        reason="no registered model matches the requested selector",
                    )
                )
            )
            continue

        compatible = tuple(
            artifact for artifact in candidates if artifact.is_compatible_with(dataset)
        )
        if not compatible:
            artifact = min(
                candidates,
                key=lambda item: (item.validation_metric, item.name, item.version),
            )
            records.extend(
                _metric_records(
                    _base_record(
                        dataset=dataset,
                        selector=selector,
                        artifact=artifact,
                        status="skipped_incompatible",
                        reason="; ".join(artifact.compatibility_errors(dataset)),
                    )
                )
            )
            continue

        # ModelRegistry.resolve uses only validation_metric. Keep that invariant
        # explicit here, including for user-supplied selectors.
        artifact = min(
            compatible,
            key=lambda item: (item.validation_metric, item.name, item.version),
        )
        artifact_path = _resolve_artifact_path(
            artifact, project_root=project_root, registry_path=registry_path
        )
        status, reason = _artifact_readiness(artifact_path)
        records.extend(
            _metric_records(
                _base_record(
                    dataset=dataset,
                    selector=selector,
                    artifact=artifact,
                    status=status,
                    reason=reason,
                    artifact_path=artifact_path,
                )
            )
        )
    return records


def _default_output_dir(project_root: Path) -> Path:
    """Return the documented UTC timestamped output location."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "data" / "output" / "runs" / timestamp


def run_evaluation(
    *,
    data: list[str],
    models: list[str],
    registry: ModelRegistry,
    registry_path: Path,
    project_root: Path,
    output_dir: Path | None = None,
) -> EvaluationRun:
    """Plan all dataset/model combinations and write immutable result files."""
    data_values = expand_values(data)
    model_values = expand_values(models)
    if not data_values:
        raise ValueError("at least one --data value is required")

    resolved_output = output_dir or _default_output_dir(project_root)
    if resolved_output.exists():
        raise FileExistsError(f"evaluation output already exists: {resolved_output}")

    datasets = [
        infer_dataset_spec(resolve_data_path(value, project_root))
        for value in data_values
    ]
    for selector in model_values:
        parse_model_selection(selector)
    records = [
        record
        for dataset in datasets
        for record in _records_for_dataset(
            dataset=dataset,
            registry=registry,
            registry_path=registry_path,
            project_root=project_root,
            selectors=model_values,
        )
    ]
    resolved_output.mkdir(parents=True)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path),
        "requested_data": data_values,
        "requested_models": model_values or ["default"],
        "datasets": [dataset.model_dump(mode="json", by_alias=True) for dataset in datasets],
        "record_count": len(records),
        "metrics": list(_METRICS),
    }
    manifest_path = resolved_output / "run.json"
    metrics_csv_path = resolved_output / "metrics.csv"
    metrics_json_path = resolved_output / "metrics.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_records = [
        {
            **record,
            "dataset_covariates": json.dumps(record["dataset_covariates"]),
        }
        for record in records
    ]
    pl.DataFrame(csv_records).write_csv(metrics_csv_path)
    return EvaluationRun(
        output_dir=resolved_output,
        manifest_path=manifest_path,
        metrics_csv_path=metrics_csv_path,
        metrics_json_path=metrics_json_path,
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Unified run-directory evaluation (``glucose evaluate --run-dir``)
# ---------------------------------------------------------------------------


def evaluate_run_dir(
    run_dir: Path,
    *,
    data: Path | None = None,
    train_data: Path | None = None,
    label: str | None = None,
    device: str = "auto",
    output_dir: Path | None = None,
) -> SingleModelResult:
    """Evaluate a single run directory, auto-detecting its backend.

    When *data* is ``None`` the function reads precomputed metrics CSVs
    already present in the run directory and prints a warning.  Set the
    ``GLUCOSE_EVAL_RERUN=1`` environment variable to force re-inference
    even when precomputed results exist.

    When *data* is provided, inference is dispatched to the appropriate
    backend adapter.  *train_data* supplies the training CSV for scaler
    fitting (custom PyTorch models); when omitted the adapter resolves it
    from the run directory metadata or falls back to *data* itself.
    """
    kind = detect_run_dir(run_dir)
    model_name = infer_model_name(run_dir, kind, label=label)
    force_rerun = os.environ.get("GLUCOSE_EVAL_RERUN", "").strip() == "1"

    if data is not None and not data.is_file():
        raise FileNotFoundError(f"data file not found: {data}")

    if data is not None and (force_rerun or kind != RunDirKind.PRECOMPUTED):
        return _evaluate_with_inference(
            run_dir, kind, model_name,
            data=data, train_data=train_data, device=device, output_dir=output_dir,
        )

    has_precomputed = (
        (run_dir / "test_metrics_overall.csv").is_file()
        or (run_dir / "val_metrics_overall.csv").is_file()
    )
    if has_precomputed:
        if data is None:
            print(
                f"[glucose evaluate] reading precomputed metrics from {run_dir} "
                "(pass --data to re-run inference)",
                file=sys.stderr,
            )
        return read_precomputed_result(run_dir, model_name, kind=kind)

    raise ValueError(
        f"no precomputed metrics in {run_dir} and no --data provided for inference"
    )


def _evaluate_with_inference(
    run_dir: Path,
    kind: RunDirKind,
    model_name: str,
    *,
    data: Path,
    train_data: Path | None = None,
    device: str,
    output_dir: Path | None = None,
) -> SingleModelResult:
    """Dispatch to a backend-specific inference adapter."""
    if kind == RunDirKind.NEURALFORECAST:
        from glucose_forecasting.evaluation.nf_adapter import evaluate_nf_run_dir

        return evaluate_nf_run_dir(
            run_dir, model_name, data=data, output_dir=output_dir,
        )
    if kind == RunDirKind.CUSTOM_PYTORCH:
        from glucose_forecasting.evaluation.pytorch_adapter import evaluate_pytorch_run_dir

        return evaluate_pytorch_run_dir(
            run_dir, model_name,
            data=data, train_data=train_data, device=device, output_dir=output_dir,
        )
    return read_precomputed_result(run_dir, model_name, kind=kind)


def evaluate_and_compare(
    run_dirs: list[Path],
    *,
    data: Path | None = None,
    train_data: Path | None = None,
    labels: list[str] | None = None,
    output_dir: Path | None = None,
    device: str = "auto",
    plot: bool = True,
) -> Path:
    """Evaluate one or more run directories and produce a comparison report.

    Returns the output directory path.
    """
    effective_labels = labels or []
    results: list[SingleModelResult] = []
    for index, run_dir in enumerate(run_dirs):
        label = effective_labels[index] if index < len(effective_labels) else None
        result = evaluate_run_dir(
            run_dir, data=data, train_data=train_data, label=label, device=device,
        )
        results.append(result)

    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("data/output/comparisons") / timestamp

    return write_comparison_report(results, output_dir, plot=plot)
