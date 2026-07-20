"""Training provenance contract for an inference release."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from glucose_forecasting.release.base import ReleaseModel


class ProvenanceSpec(ReleaseModel):
    """Versioned, reproducibility-relevant release provenance."""

    format_version: Literal["1.0"] = "1.0"
    git_sha: str = Field(min_length=1)
    lock_hash: str = Field(min_length=1)
    env: dict[str, str] = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    seed: int
