"""Unit tests for strict versioned YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from glucose_forecasting.config import (
    DatasetSpec,
    EvaluationConfig,
    ModelSelection,
    load_dataset_spec,
    load_evaluation_config,
    save_dataset_spec,
    save_evaluation_config,
)


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        name="loop",
        path="data/loop.csv",
        schema="loop-v1",
        covariates=("basal", "bolus", "carbohydrates"),
    )


def test_dataset_spec_yaml_round_trip(tmp_path: Path) -> None:
    spec = _dataset()
    path = save_dataset_spec(spec, tmp_path / "dataset.yaml")

    assert load_dataset_spec(path) == spec
    assert path.read_text().startswith("version: 1\n")


def test_evaluation_config_yaml_round_trip(tmp_path: Path) -> None:
    config = EvaluationConfig(
        dataset=_dataset(),
        model_selection=ModelSelection(name="sugar-one", version="1.0.0"),
        split="validation",
        metrics=("mae", "mard"),
    )
    path = save_evaluation_config(config, tmp_path / "nested" / "evaluation.yaml")

    assert load_evaluation_config(path) == config
    assert path.read_text().startswith("version: 1\n")


@pytest.mark.parametrize("yaml_text", ["", "[]\n", "loop\n"])
def test_yaml_load_rejects_non_mapping_shapes(tmp_path: Path, yaml_text: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValueError, match="top-level mapping"):
        load_dataset_spec(path)


def test_yaml_load_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "unknown-key.yaml"
    path.write_text(
        "version: 1\n"
        "name: loop\n"
        "path: data/loop.csv\n"
        "schema: loop-v1\n"
        "unknown_option: true\n"
    )

    with pytest.raises(ValidationError, match="unknown_option"):
        load_dataset_spec(path)


def test_yaml_load_requires_current_version_field(tmp_path: Path) -> None:
    missing_version = tmp_path / "missing-version.yaml"
    missing_version.write_text("name: loop\npath: data/loop.csv\nschema: loop-v1\n")
    with pytest.raises(ValueError, match="declare a top-level version"):
        load_dataset_spec(missing_version)

    unsupported_version = tmp_path / "unsupported-version.yaml"
    unsupported_version.write_text(
        "version: 2\nname: loop\npath: data/loop.csv\nschema: loop-v1\n"
    )
    with pytest.raises(ValidationError, match="version"):
        load_dataset_spec(unsupported_version)
