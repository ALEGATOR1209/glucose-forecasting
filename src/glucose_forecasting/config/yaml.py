"""Safe YAML serialization for versioned configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from glucose_forecasting.config.models import DatasetSpec, EvaluationConfig

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def _load_yaml_model(path: str | Path, model_type: type[ConfigModel]) -> ConfigModel:
    """Safely parse a YAML mapping and validate it with a Pydantic model."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Configuration YAML must contain a top-level mapping, got {type(payload).__name__}."
        )
    if "version" not in payload:
        raise ValueError("Configuration YAML must declare a top-level version.")
    if "config_version" in payload:
        raise ValueError("Configuration YAML must use 'version', not 'config_version'.")
    return model_type.model_validate(payload)


def _save_yaml_model(model: BaseModel, path: str | Path) -> Path:
    """Safely serialize a Pydantic model to versioned YAML."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            model.model_dump(mode="json", by_alias=True),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load a versioned :class:`DatasetSpec` YAML file."""
    return _load_yaml_model(path, DatasetSpec)


def save_dataset_spec(spec: DatasetSpec, path: str | Path) -> Path:
    """Write a versioned :class:`DatasetSpec` YAML file."""
    return _save_yaml_model(spec, path)


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load a versioned :class:`EvaluationConfig` YAML file."""
    return _load_yaml_model(path, EvaluationConfig)


def save_evaluation_config(config: EvaluationConfig, path: str | Path) -> Path:
    """Write a versioned :class:`EvaluationConfig` YAML file."""
    return _save_yaml_model(config, path)
