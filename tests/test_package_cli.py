"""Smoke tests for the installable src-package command line interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from glucose_forecasting.cli import app
from glucose_forecasting.config import DatasetSpec, EvaluationConfig, save_evaluation_config
from glucose_forecasting.models.registry import (
    ModelArtifact,
    ModelRegistry,
    save_registry,
)
from tests.release_fixtures import release_manifest as _release_manifest

runner = CliRunner()


def test_neuralforecast_train_lists_packaged_yaml_suites(tmp_path: Path) -> None:
    """The package CLI exposes the YAML-defined NeuralForecast auto suite."""
    data_path = tmp_path / "placeholder.csv"
    data_path.write_text("unused\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "train",
            "--backend",
            "neuralforecast",
            "--data",
            str(data_path),
            "--list-models",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "auto:" in result.output
    assert "xLSTM" in result.output


def _registry(path: Path) -> Path:
    """Create a minimal real registry for CLI tests."""
    return save_registry(
        ModelRegistry(
            models=(
                ModelArtifact(
                    name="sugar-one",
                    version="1.0.0",
                    artifact_path="models/sugar-one.pt",
                    schema="loop-v1",
                    covariates=("basal", "bolus", "carbohydrates"),
                    validation_metric=8.5,
                ),
            )
        ),
        path,
    )


def test_models_commands_read_registry_and_resolve_csv(tmp_path: Path) -> None:
    """Registry-backed inspection commands return actual registered metadata."""
    registry_path = _registry(tmp_path / "registry.json")
    data_path = tmp_path / "loop.csv"
    data_path.write_text(
        "Glucose (mg/dL),Basal Rate (U/h),Bolus Insulin (U),Carbohydrates (g)\n"
        "120,1.0,0.0,30\n",
        encoding="utf-8",
    )

    listed = runner.invoke(app, ["models", "list", "--registry", str(registry_path)])
    shown = runner.invoke(
        app, ["models", "show", "sugar-one", "--registry", str(registry_path)]
    )
    resolved = runner.invoke(
        app,
        [
            "models",
            "resolve",
            "--data",
            str(data_path),
            "--registry",
            str(registry_path),
        ],
    )

    assert listed.exit_code == 0, listed.output
    assert shown.exit_code == 0, shown.output
    assert resolved.exit_code == 0, resolved.output
    assert '"name": "sugar-one"' in listed.output
    assert '"version": "1.0.0"' in shown.output
    assert '"artifact_path": "models/sugar-one.pt"' in resolved.output


def test_config_check_and_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config validation and bare data-name resolution use their real inputs."""
    config_path = tmp_path / "evaluation.json"
    save_evaluation_config(
        EvaluationConfig(
            dataset=DatasetSpec(
                name="loop",
                path="loop.csv",
                data_schema="loop-v1",
                covariates=("basal", "bolus", "carbohydrates"),
                cadence_minutes=5,
                horizon_steps=12,
            )
        ),
        config_path,
    )
    monkeypatch.chdir(tmp_path)

    checked = runner.invoke(app, ["config", "check", "--config", str(config_path)])
    resolved = runner.invoke(app, ["data", "path", "loop.csv"])

    assert checked.exit_code == 0, checked.output
    assert "Configuration valid: loop" in checked.output
    assert resolved.exit_code == 0, resolved.output
    assert resolved.output.strip() == str(tmp_path / "data" / "input" / "loop.csv")


def test_models_commands_explain_missing_registry(tmp_path: Path) -> None:
    """A missing default registry is reported as a useful CLI error."""
    result = runner.invoke(app, ["models", "list"], catch_exceptions=False)

    assert result.exit_code != 0
    assert "registry not found" in result.output


def test_release_publish_calls_release_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release publish command forwards the bundle and repository options."""
    publish = Mock(return_value=_release_manifest())
    monkeypatch.setattr("glucose_forecasting.cli.publish_inference_bundle", publish)
    bundle_dir = tmp_path / "bundle"

    result = runner.invoke(
        app,
        ["release", "publish", str(bundle_dir), "--repo", "GlucoseDAO/sugar-one", "--private"],
    )

    assert result.exit_code == 0, result.output
    publish.assert_called_once_with(
        bundle_dir,
        repo_id="GlucoseDAO/sugar-one",
        private=True,
    )
    assert result.output.strip() == (
        "Published inference release sugar-one-2026-07 to GlucoseDAO/sugar-one"
    )


def test_release_pull_calls_release_api_with_default_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release pull command downloads main unless a revision is supplied."""
    download = Mock(return_value=_release_manifest())
    monkeypatch.setattr("glucose_forecasting.cli.download_inference_bundle", download)
    output_dir = tmp_path / "bundle"

    result = runner.invoke(
        app,
        ["release", "pull", "--repo", "GlucoseDAO/sugar-one", "--out", str(output_dir)],
    )

    assert result.exit_code == 0, result.output
    download.assert_called_once_with(
        "GlucoseDAO/sugar-one",
        revision="main",
        target_dir=output_dir,
    )
    assert result.output.strip() == (
        f"Downloaded inference release sugar-one-2026-07 to {output_dir}"
    )
