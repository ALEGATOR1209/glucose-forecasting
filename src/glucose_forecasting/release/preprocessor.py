"""Input preprocessing contract for an inference release."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt

from glucose_forecasting.release.base import ReleaseModel


class ScalerSpec(ReleaseModel):
    """Parameters for scaling one input feature."""

    kind: Literal["standard", "minmax", "none", "custom"]
    parameters: dict[str, float] = Field(default_factory=dict)


class ImputationSpec(ReleaseModel):
    """Imputation policy for one input feature."""

    method: Literal["forward_fill", "backward_fill", "zero", "constant", "none"]
    value: float | None = None


class WindowSpec(ReleaseModel):
    """Windowing parameters expressed in sample steps."""

    input_steps: PositiveInt
    stride_steps: PositiveInt = 1
    target_offset_steps: int = 0


class PreprocessorSpec(ReleaseModel):
    """Versioned preprocessing behavior required before inference."""

    format_version: Literal["1.0"] = "1.0"
    scalers: dict[str, ScalerSpec] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    imputation: dict[str, ImputationSpec] = Field(default_factory=dict)
    window: WindowSpec
    units: dict[str, str] = Field(default_factory=dict)
