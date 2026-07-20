"""Pydantic models shared by dataset and evaluation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]


class DatasetSpec(BaseModel):
    """Versioned data contract a model can be evaluated against."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    config_version: Literal[1] = Field(default=1, alias="version")
    name: str = Field(min_length=1)
    path: Path
    data_schema: str = Field(alias="schema", min_length=1)
    covariates: tuple[str, ...] = ()
    cadence_minutes: PositiveInt = 5
    horizon_steps: PositiveInt = 12

    @field_validator("covariates")
    @classmethod
    def require_unique_covariates(cls, covariates: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate or empty covariate names."""
        if any(not covariate.strip() for covariate in covariates):
            raise ValueError("covariates must not contain empty names")
        if len(covariates) != len(set(covariates)):
            raise ValueError("covariates must be unique")
        return covariates


class ModelSelection(BaseModel):
    """Optional constraints for selecting an artifact from a model registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def version_requires_name(self) -> ModelSelection:
        """A version has no meaning without a model name."""
        if self.version is not None and self.name is None:
            raise ValueError("model selection version requires a model name")
        return self


class EvaluationConfig(BaseModel):
    """Versioned configuration for a future evaluation invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    config_version: Literal[1] = Field(default=1, alias="version")
    dataset: DatasetSpec
    model_selection: ModelSelection = Field(default_factory=ModelSelection)
    split: Literal["validation", "test"] = "test"
    metrics: tuple[Literal["mae", "rmse", "mard"], ...] = ("mae", "rmse", "mard")

    @field_validator("metrics")
    @classmethod
    def require_unique_metrics(
        cls, metrics: tuple[Literal["mae", "rmse", "mard"], ...]
    ) -> tuple[Literal["mae", "rmse", "mard"], ...]:
        """Require at least one, non-duplicated metric."""
        if not metrics:
            raise ValueError("metrics must contain at least one metric")
        if len(metrics) != len(set(metrics)):
            raise ValueError("metrics must be unique")
        return metrics
