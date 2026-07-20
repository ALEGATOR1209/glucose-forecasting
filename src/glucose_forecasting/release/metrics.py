"""Evaluation contract for an inference release."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glucose_forecasting.release.base import ReleaseModel


class SelectionMetric(ReleaseModel):
    """Metric used to choose the release checkpoint."""

    name: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]


class EvaluationProtocol(ReleaseModel):
    """Dataset split and evaluation settings used for reported metrics."""

    name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MetricsSpec(ReleaseModel):
    """Versioned validation and test metrics for a release."""

    format_version: Literal["1.0"] = "1.0"
    selection_metric: SelectionMetric
    validation: dict[str, float] = Field(min_length=1)
    test: dict[str, float] = Field(min_length=1)
    protocol: EvaluationProtocol
