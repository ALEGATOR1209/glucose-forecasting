"""Shared primitives for versioned release-contract models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReleaseModel(BaseModel):
    """Immutable, strict base class for release-contract documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)
