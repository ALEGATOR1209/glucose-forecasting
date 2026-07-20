"""Top-level inference-release manifest contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glucose_forecasting.release.base import ReleaseModel
from glucose_forecasting.release.config import InferenceConfig
from glucose_forecasting.release.metrics import MetricsSpec
from glucose_forecasting.release.preprocessor import PreprocessorSpec
from glucose_forecasting.release.provenance import ProvenanceSpec


class ReleaseManifest(ReleaseModel):
    """Single versioned document describing an inference-ready release."""

    format_version: Literal["1.0"] = "1.0"
    release_id: str = Field(min_length=1)
    config: InferenceConfig
    preprocessor: PreprocessorSpec
    metrics: MetricsSpec
    provenance: ProvenanceSpec
