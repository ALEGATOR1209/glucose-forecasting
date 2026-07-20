"""Unified evaluation result types shared across all model backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import polars as pl

from glucose_forecasting.backends.neuralforecast.benchmark import RegressionMetrics


class RunDirKind(StrEnum):
    """Classification of a run directory by its backend origin."""

    NEURALFORECAST = "neuralforecast"
    CUSTOM_PYTORCH = "custom_pytorch"
    PRECOMPUTED = "precomputed"


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics for one evaluation split (val or test)."""

    overall: RegressionMetrics
    by_study_group: pl.DataFrame


@dataclass(frozen=True)
class SingleModelResult:
    """Evaluation result for one model, potentially across multiple splits."""

    model_name: str
    run_dir: Path
    kind: RunDirKind
    split_results: dict[str, SplitMetrics] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
