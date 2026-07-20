"""Unit tests for the typed JSON model registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from glucose_forecasting.config import DatasetSpec, EvaluationConfig, ModelSelection
from glucose_forecasting.models.registry import (
    ModelArtifact,
    ModelRegistry,
    ModelResolutionError,
    load_registry,
    resolve_data_path,
    save_registry,
)


def _dataset(**overrides: object) -> DatasetSpec:
    values: dict[str, object] = {
        "name": "loop",
        "path": "loop.csv",
        "schema": "loop-v1",
        "covariates": ("basal", "bolus", "carbohydrates"),
        "cadence_minutes": 5,
        "horizon_steps": 12,
    }
    values.update(overrides)
    return DatasetSpec.model_validate(values)


def _artifact(
    name: str,
    version: str,
    validation_metric: float,
    **overrides: object,
) -> ModelArtifact:
    values: dict[str, object] = {
        "name": name,
        "version": version,
        "artifact_path": f"models/{name}-{version}.pt",
        "schema": "loop-v1",
        "covariates": ("basal", "bolus"),
        "cadence_minutes": 5,
        "horizon_steps": 12,
        "validation_metric": validation_metric,
    }
    values.update(overrides)
    return ModelArtifact.model_validate(values)


def test_config_models_are_json_compatible() -> None:
    config = EvaluationConfig(
        dataset=_dataset(path=Path("data.csv")),
        model_selection=ModelSelection(name="sugar-one", version="1.0.0"),
    )

    encoded = config.model_dump_json()
    restored = EvaluationConfig.model_validate_json(encoded)

    assert restored == config


def test_config_rejects_invalid_selection_and_duplicates() -> None:
    with pytest.raises(ValidationError, match="requires a model name"):
        ModelSelection(version="1.0.0")
    with pytest.raises(ValidationError, match="covariates must be unique"):
        _dataset(covariates=("basal", "basal"))
    with pytest.raises(ValidationError, match="metrics must be unique"):
        EvaluationConfig(dataset=_dataset(), metrics=("mae", "mae"))


def test_registry_default_uses_lowest_validation_metric_among_stable_models() -> None:
    registry = ModelRegistry(
        models=(
            _artifact("baseline", "1.0.0", 10.0),
            _artifact("preferred", "1.0.0", 7.5),
            _artifact("experimental", "2.0.0", 0.1, stable=False),
        )
    )

    resolved = registry.resolve(_dataset())

    assert (resolved.name, resolved.version) == ("preferred", "1.0.0")


def test_registry_explicit_name_and_version_can_select_nonstable_artifact() -> None:
    registry = ModelRegistry(
        models=(
            _artifact("baseline", "1.0.0", 8.0),
            _artifact("experimental", "2.0.0", 2.0, stable=False),
        )
    )

    resolved = registry.resolve(
        _dataset(),
        ModelSelection(name="experimental", version="2.0.0"),
    )

    assert resolved.name == "experimental"


def test_registry_explicit_name_uses_lowest_validation_metric() -> None:
    registry = ModelRegistry(
        models=(
            _artifact("sugar-one", "1.0.0", 8.0),
            _artifact("sugar-one", "2.0.0", 6.0),
            _artifact("glumind", "1.0.0", 1.0),
        )
    )

    resolved = registry.resolve(_dataset(), ModelSelection(name="sugar-one"))

    assert resolved.version == "2.0.0"


@pytest.mark.parametrize(
    ("artifact_overrides", "expected_error"),
    [
        ({"schema": "ai-readi-v1"}, "schema mismatch"),
        ({"covariates": ("heart_rate",)}, "missing required covariates"),
        ({"cadence_minutes": 15}, "cadence mismatch"),
        ({"horizon_steps": 6}, "horizon mismatch"),
    ],
)
def test_registry_rejects_incompatible_models(
    artifact_overrides: dict[str, object],
    expected_error: str,
) -> None:
    artifact = _artifact("incompatible", "1.0.0", 1.0, **artifact_overrides)
    registry = ModelRegistry(models=(artifact,))

    with pytest.raises(ModelResolutionError, match=expected_error):
        registry.resolve(_dataset())


def test_registry_rejects_duplicate_versions_and_nonfinite_metrics() -> None:
    artifact = _artifact("sugar-one", "1.0.0", 1.0)
    with pytest.raises(ValidationError, match="duplicate name/version"):
        ModelRegistry(models=(artifact, artifact))
    with pytest.raises(ValidationError):
        _artifact("sugar-one", "2.0.0", float("inf"))


def test_registry_json_round_trip(tmp_path: Path) -> None:
    registry = ModelRegistry(models=(_artifact("sugar-one", "1.0.0", 4.2),))
    registry_path = save_registry(registry, tmp_path / "nested" / "registry.json")

    assert load_registry(registry_path) == registry


def test_resolve_data_path_uses_data_input_for_bare_names(tmp_path: Path) -> None:
    absolute_path = tmp_path / "external.csv"

    assert resolve_data_path("dataset.csv", tmp_path) == tmp_path / "data" / "input" / "dataset.csv"
    assert resolve_data_path("data/custom.csv", tmp_path) == tmp_path / "data" / "custom.csv"
    assert resolve_data_path(absolute_path, tmp_path) == absolute_path
