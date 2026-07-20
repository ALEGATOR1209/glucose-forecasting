"""Typed NeuralForecast run and YAML model-suite configuration."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from glucose_forecasting.backends.neuralforecast.catalog import NeuralForecastModel


class ModelSuite(BaseModel):
    """One named curated suite; membership is defined only in YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profiles: tuple[Literal["ai-readi", "loop"], ...]
    models: tuple[NeuralForecastModel, ...]
    covariate_mode: Literal["historical-exogenous", "univariate"] = "historical-exogenous"

    @field_validator("models")
    @classmethod
    def require_models(cls, models: tuple[NeuralForecastModel, ...]) -> tuple[NeuralForecastModel, ...]:
        if not models:
            raise ValueError("model suite must contain at least one model")
        if len(models) != len(set(models)):
            raise ValueError("model suite must not contain duplicate models")
        return models


class ModelSuiteConfig(BaseModel):
    """Versioned YAML model-suite document."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    version: Literal[1] = 1
    suites: dict[str, ModelSuite]

    @field_validator("suites")
    @classmethod
    def require_auto_suite(cls, suites: dict[str, ModelSuite]) -> dict[str, ModelSuite]:
        if "auto" not in suites:
            raise ValueError("model suite configuration must define an 'auto' suite")
        return suites


class NeuralForecastRunConfig(BaseModel):
    """Shared options for holdout and rolling-CV NeuralForecast runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    csv: Path
    profile: Literal["auto", "ai-readi", "loop"] = "auto"
    models: str = "auto"
    model_config_path: Path | None = None
    evaluation: Literal["holdout", "cross-val"] = "holdout"
    holdout_protocol: Literal["sugarone-compatible", "dense", "tail"] = "sugarone-compatible"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    unique_id: Literal["sequence_id", "user_id"] = "sequence_id"
    split_scheme: Literal["classic", "trainval_test_as_val"] = "classic"
    global_model: bool = False
    study_groups: tuple[str, ...] = ()
    out_dir: Path = Path("data/output/runs")
    h_minutes: int = Field(default=60, gt=0)
    freq: str = "5min"
    input_hours: float = Field(default=10.666666666666666, gt=0)
    train_tail_val_hours: float = Field(default=24.0, gt=0)
    max_steps: int = Field(default=2000, gt=0)
    val_check_steps: int = Field(default=400, gt=0)
    batch_size: int = Field(default=8, gt=0)
    valid_batch_size: int = Field(default=8, gt=0)
    windows_batch_size: int = Field(default=256, gt=0)
    inference_windows_batch_size: int = Field(default=256, gt=0)
    step_size: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    max_train_series: int = Field(default=0, ge=0)
    max_eval_series: int = Field(default=0, ge=0)
    max_points_per_series: int = Field(default=0, ge=0)
    drop_interpolated: bool = False
    mask_interpolated_targets: bool = False
    save_predictions: bool = False
    plot: bool = True
    max_plot_series: int = Field(default=3, gt=0)
    n_windows: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def require_existing_csv(self) -> NeuralForecastRunConfig:
        if not self.csv.is_file():
            raise ValueError(f"CSV data file not found: {self.csv}")
        return self

    @model_validator(mode="after")
    def require_sugarone_geometry(self) -> NeuralForecastRunConfig:
        """Keep comparable holdout runs on the shared 128/12/stride-1 geometry."""
        if self.evaluation != "holdout" or self.holdout_protocol != "sugarone-compatible":
            return self
        step_minutes = frequency_minutes(self.freq)
        input_size = round(self.input_hours * 60 / step_minutes)
        horizon = self.h_minutes // step_minutes
        if self.h_minutes % step_minutes or input_size != 128 or horizon != 12 or self.step_size != 1:
            raise ValueError(
                "sugarone-compatible holdout requires input_steps=128, horizon=12, "
                "and step_size=1; use --holdout-protocol dense or tail for an experimental geometry"
            )
        return self


def frequency_minutes(freq: str) -> int:
    """Parse the minute-based sampling frequencies supported by this benchmark."""
    if not freq.endswith("min"):
        raise ValueError(f"only minute-based frequencies are supported, got {freq!r}")
    try:
        minutes = int(freq.removesuffix("min"))
    except ValueError as error:
        raise ValueError(f"invalid minute-based frequency {freq!r}") from error
    if minutes <= 0:
        raise ValueError(f"frequency must be positive, got {freq!r}")
    return minutes


def neuralforecast_frequency(freq: str) -> str:
    """Return the frequency spelling expected by NeuralForecast's Polars path."""
    return f"{frequency_minutes(freq)}m"


def load_model_suites(path: Path | None = None) -> tuple[ModelSuiteConfig, str]:
    """Load a custom suite YAML or the package default and return its source text."""
    if path is None:
        resource = files("glucose_forecasting.backends.neuralforecast").joinpath("model_suites.yaml")
        text = resource.read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("model suite YAML must contain a top-level mapping")
    return ModelSuiteConfig.model_validate(payload), text
