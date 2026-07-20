"""Detect the kind of a model run directory and infer its model name."""
from __future__ import annotations

import json
from pathlib import Path

from glucose_forecasting.evaluation.types import RunDirKind


def detect_run_dir(path: Path) -> RunDirKind:
    """Classify a run directory by its backend origin.

    Raises ``ValueError`` when the directory does not match any known layout.
    """
    if not path.is_dir():
        raise ValueError(f"run directory does not exist: {path}")
    if (path / "neuralforecast").is_dir() and (path / "run_config.json").is_file():
        return RunDirKind.NEURALFORECAST
    has_checkpoint = (path / "best_model.pt").is_file() or (path / "last_model.pt").is_file()
    has_config = (path / "tuning_meta.json").is_file() or (path / "config.json").is_file()
    if has_checkpoint and has_config:
        return RunDirKind.CUSTOM_PYTORCH
    has_metrics = (
        (path / "test_metrics_overall.csv").is_file()
        or (path / "val_metrics_overall.csv").is_file()
    )
    if has_metrics:
        return RunDirKind.PRECOMPUTED
    raise ValueError(
        f"cannot detect run directory kind: {path} — expected a NeuralForecast bundle, "
        "a PyTorch checkpoint with config, or precomputed metrics CSVs"
    )


def infer_model_name(run_dir: Path, kind: RunDirKind, *, label: str | None = None) -> str:
    """Derive a human-readable model name for a run directory.

    When *label* is provided it takes precedence.  Otherwise the name is
    inferred from metadata files or the directory naming convention.
    """
    if label:
        return label
    if kind == RunDirKind.NEURALFORECAST:
        return _nf_model_name(run_dir)
    if kind == RunDirKind.CUSTOM_PYTORCH:
        return _pytorch_model_name(run_dir)
    return run_dir.name


def _nf_model_name(run_dir: Path) -> str:
    """Parse NeuralForecast model name from directory convention ``Model_timestamp``."""
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            models = config.get("selected_models")
            if isinstance(models, list) and len(models) == 1:
                return models[0]
        except (json.JSONDecodeError, OSError):
            pass
    name, sep, _ = run_dir.name.rpartition("_")
    if sep and name:
        return name
    return run_dir.name


def _pytorch_model_name(run_dir: Path) -> str:
    """Read model type from ``tuning_meta.json`` or ``config.json``."""
    for config_name in ("tuning_meta.json", "config.json"):
        config_path = run_dir / config_name
        if not config_path.is_file():
            continue
        try:
            meta = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("model_type", "model"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    return run_dir.name
