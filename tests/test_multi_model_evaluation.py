"""Tests for modern multi-dataset model evaluation planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from glucose_forecasting.cli import app
from glucose_forecasting.models.registry import ModelArtifact, ModelRegistry, save_registry

runner = CliRunner()


def _legacy_run(path: Path) -> None:
    """Create the minimum recognized legacy-run layout without model execution."""
    path.mkdir(parents=True)
    (path / "best_model.pt").write_bytes(b"checkpoint")
    (path / "config.json").write_text("{}", encoding="utf-8")


def _write_csv(path: Path, covariate_columns: list[str]) -> None:
    """Write a small header-valid CSV; planning intentionally does not infer rows."""
    path.write_text(
        ",".join(["sequence_id", "Glucose (mg/dL)", *covariate_columns]) + "\n"
        + ",".join(["series-1", "110", *(["1"] * len(covariate_columns))])
        + "\n",
        encoding="utf-8",
    )


def _registry(path: Path) -> Path:
    """Create compatible Loop and AI-READI artifacts with legacy run directories."""
    return save_registry(
        ModelRegistry(
            models=(
                ModelArtifact(
                    name="sugar-one",
                    version="1.0.0",
                    artifact_path="runs/sugar-one-v1",
                    schema="loop-v1",
                    covariates=("basal", "bolus", "carbohydrates"),
                    validation_metric=9.0,
                ),
                ModelArtifact(
                    name="sugar-one",
                    version="2.0.0",
                    artifact_path="runs/sugar-one-v2",
                    schema="loop-v1",
                    covariates=("basal", "bolus", "carbohydrates"),
                    validation_metric=3.0,
                ),
                ModelArtifact(
                    name="glumind",
                    version="1.0.0",
                    artifact_path="runs/glumind-v1",
                    schema="ai-readi-v1",
                    covariates=("heart_rate", "steps"),
                    validation_metric=4.0,
                ),
            )
        ),
        path,
    )


def test_evaluate_cli_writes_immutable_multi_dataset_long_form_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each model/dataset pair is resolved or explicitly skipped in one run."""
    _legacy_run(tmp_path / "runs" / "sugar-one-v1")
    _legacy_run(tmp_path / "runs" / "sugar-one-v2")
    _legacy_run(tmp_path / "runs" / "glumind-v1")
    data_dir = tmp_path / "data" / "input"
    data_dir.mkdir(parents=True)
    _write_csv(
        data_dir / "loop.csv",
        ["Basal Rate (U/h)", "Bolus Insulin (U)", "Carbohydrates (g)"],
    )
    _write_csv(data_dir / "ai_readi.csv", ["Heart Rate", "Step Count"])
    registry_path = _registry(tmp_path / "registry.json")
    output_dir = tmp_path / "data" / "output" / "runs" / "evaluation"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--data",
            "loop.csv",
            "--data",
            "ai_readi.csv",
            "--models",
            "sugar-one,glumind",
            "--registry",
            str(registry_path),
            "--out",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "run.json").is_file()
    assert (output_dir / "metrics.csv").is_file()
    records = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(records) == 12
    assert {
        (record["dataset"], record["requested_model"], record["status"])
        for record in records
    } == {
        ("loop", "sugar-one", "not_evaluated"),
        ("loop", "glumind", "skipped_incompatible"),
        ("ai_readi", "sugar-one", "skipped_incompatible"),
        ("ai_readi", "glumind", "not_evaluated"),
    }
    sugar_records = [
        record
        for record in records
        if record["dataset"] == "loop" and record["requested_model"] == "sugar-one"
    ]
    assert {record["model_version"] for record in sugar_records} == {"2.0.0"}
    assert {record["value"] for record in records} == {None}

    repeated = runner.invoke(
        app,
        [
            "evaluate",
            "--data",
            "loop.csv",
            "--registry",
            str(registry_path),
            "--out",
            str(output_dir),
        ],
    )
    assert repeated.exit_code != 0
    assert "output already exists" in repeated.output
