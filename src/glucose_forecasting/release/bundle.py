"""Local, integrity-checked inference bundle persistence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from glucose_forecasting.release.checksums import sha256_file, verify_sha256
from glucose_forecasting.release.config import InferenceConfig
from glucose_forecasting.release.io import read_json, read_manifest, write_json, write_manifest
from glucose_forecasting.release.manifest import ReleaseManifest
from glucose_forecasting.release.metrics import MetricsSpec
from glucose_forecasting.release.preprocessor import PreprocessorSpec
from glucose_forecasting.release.provenance import ProvenanceSpec

MODEL_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.json"
PREPROCESSOR_FILENAME = "preprocessor.json"
METRICS_FILENAME = "metrics.json"
PROVENANCE_FILENAME = "provenance.json"
CHECKSUMS_FILENAME = "checksums.sha256"

_ARTIFACT_FILENAMES = (
    MODEL_FILENAME,
    MANIFEST_FILENAME,
    CONFIG_FILENAME,
    PREPROCESSOR_FILENAME,
    METRICS_FILENAME,
    PROVENANCE_FILENAME,
)

ModelFactory = Callable[[InferenceConfig], torch.nn.Module]


@dataclass(frozen=True)
class LoadedInferenceBundle:
    """An integrity-checked manifest paired with its reconstructed model."""

    manifest: ReleaseManifest
    model: torch.nn.Module


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a safetensors-compatible, CPU-only state dictionary."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        normalized_name = name.removeprefix("_orig_mod.")
        if normalized_name in state_dict:
            message = f"State dictionary contains duplicate key after normalization: {normalized_name}"
            raise ValueError(message)
        state_dict[normalized_name] = tensor.detach().cpu().contiguous()
    return state_dict


def _write_model(path: Path, model: torch.nn.Module) -> None:
    """Atomically save a model's normalized state dictionary as safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        save_file(_cpu_state_dict(model), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_checksums(bundle_dir: Path) -> None:
    """Write deterministic SHA256 entries for every bundle artifact."""
    content = "".join(
        f"{sha256_file(bundle_dir / filename)}  {filename}\n"
        for filename in _ARTIFACT_FILENAMES
    )
    checksum_path = bundle_dir / CHECKSUMS_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=bundle_dir,
        prefix=f".{CHECKSUMS_FILENAME}.",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    try:
        os.replace(temporary_path, checksum_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_checksums(bundle_dir: Path) -> dict[str, str]:
    """Read and validate the checksums file's fixed local-bundle format."""
    checksum_path = bundle_dir / CHECKSUMS_FILENAME
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Bundle checksum file is missing: {checksum_path}")

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            checksum, filename = line.split("  ", maxsplit=1)
        except ValueError as error:
            message = f"Malformed checksum entry at line {line_number}"
            raise ValueError(message) from error
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            message = f"Invalid SHA256 digest at line {line_number}"
            raise ValueError(message)
        if filename not in _ARTIFACT_FILENAMES or filename in checksums:
            message = f"Unexpected or duplicate checksum artifact at line {line_number}: {filename}"
            raise ValueError(message)
        checksums[filename] = checksum

    missing = set(_ARTIFACT_FILENAMES).difference(checksums)
    if missing:
        message = f"Missing checksums for bundle artifacts: {', '.join(sorted(missing))}"
        raise ValueError(message)
    return checksums


def validate_bundle_checksums(bundle_dir: Path) -> None:
    """Ensure every required local bundle artifact exists and matches its digest."""
    checksums = _read_checksums(bundle_dir)
    for filename in _ARTIFACT_FILENAMES:
        artifact_path = bundle_dir / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Bundle artifact is missing: {artifact_path}")
        if not verify_sha256(artifact_path, checksums[filename]):
            raise ValueError(f"Checksum mismatch for bundle artifact: {filename}")


def write_inference_bundle(
    bundle_dir: Path,
    *,
    manifest: ReleaseManifest,
    model: torch.nn.Module,
) -> None:
    """Write a local inference bundle without serializing training state."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_model(bundle_dir / MODEL_FILENAME, model)
    write_manifest(bundle_dir / MANIFEST_FILENAME, manifest)
    write_json(bundle_dir / CONFIG_FILENAME, manifest.config)
    write_json(bundle_dir / PREPROCESSOR_FILENAME, manifest.preprocessor)
    write_json(bundle_dir / METRICS_FILENAME, manifest.metrics)
    write_json(bundle_dir / PROVENANCE_FILENAME, manifest.provenance)
    _write_checksums(bundle_dir)


def _read_and_validate_contract(bundle_dir: Path) -> ReleaseManifest:
    """Load the manifest and ensure component documents cannot disagree."""
    manifest = read_manifest(bundle_dir / MANIFEST_FILENAME)
    components: Mapping[str, object] = {
        CONFIG_FILENAME: read_json(bundle_dir / CONFIG_FILENAME, InferenceConfig),
        PREPROCESSOR_FILENAME: read_json(bundle_dir / PREPROCESSOR_FILENAME, PreprocessorSpec),
        METRICS_FILENAME: read_json(bundle_dir / METRICS_FILENAME, MetricsSpec),
        PROVENANCE_FILENAME: read_json(bundle_dir / PROVENANCE_FILENAME, ProvenanceSpec),
    }
    expected_components: Mapping[str, object] = {
        CONFIG_FILENAME: manifest.config,
        PREPROCESSOR_FILENAME: manifest.preprocessor,
        METRICS_FILENAME: manifest.metrics,
        PROVENANCE_FILENAME: manifest.provenance,
    }
    for filename, component in components.items():
        if component != expected_components[filename]:
            raise ValueError(f"Bundle {filename} does not match {MANIFEST_FILENAME}")
    return manifest


def validate_inference_bundle(bundle_dir: Path) -> ReleaseManifest:
    """Validate a bundle's checksums and internally consistent release contract."""
    validate_bundle_checksums(bundle_dir)
    return _read_and_validate_contract(bundle_dir)


def load_inference_bundle(
    bundle_dir: Path,
    *,
    model_factory: ModelFactory,
) -> LoadedInferenceBundle:
    """Validate and load a local bundle into a caller-declared model architecture."""
    manifest = validate_inference_bundle(bundle_dir)
    model = model_factory(manifest.config)
    model.load_state_dict(load_file(bundle_dir / MODEL_FILENAME, device="cpu"), strict=True)
    model.cpu()
    model.eval()
    return LoadedInferenceBundle(manifest=manifest, model=model)
