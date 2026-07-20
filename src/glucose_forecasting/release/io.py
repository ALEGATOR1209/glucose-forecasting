"""Deterministic JSON serialization for release contracts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from glucose_forecasting.release.manifest import ReleaseManifest

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json_bytes(model: BaseModel) -> bytes:
    """Serialize a Pydantic model as stable UTF-8 JSON with a trailing newline."""
    payload = json.dumps(
        model.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{payload}\n".encode("utf-8")


def write_json(path: Path, model: BaseModel) -> None:
    """Atomically write a model in the deterministic release JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary_file:
        temporary_file.write(canonical_json_bytes(model))
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def read_json(path: Path, model_type: type[ModelT]) -> ModelT:
    """Read and validate a JSON document as the requested Pydantic model."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return model_type.model_validate(raw)


def write_manifest(path: Path, manifest: ReleaseManifest) -> None:
    """Write an inference release manifest."""
    write_json(path, manifest)


def read_manifest(path: Path) -> ReleaseManifest:
    """Read and validate an inference release manifest."""
    return read_json(path, ReleaseManifest)
