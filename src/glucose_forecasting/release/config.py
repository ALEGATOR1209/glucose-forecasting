"""Model configuration contract for an inference release."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, PositiveInt

from glucose_forecasting.release.base import ReleaseModel


class InferenceConfig(ReleaseModel):
    """Versioned model and input-shape configuration."""

    format_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    architecture: dict[str, JsonValue]
    feature_order: tuple[str, ...] = Field(min_length=1)
    horizon: PositiveInt
    cadence: PositiveInt
