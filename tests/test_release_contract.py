"""Focused tests for the inference release contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from glucose_forecasting.release import (
    load_inference_bundle,
    read_manifest,
    sha256_file,
    validate_bundle_checksums,
    verify_sha256,
    write_inference_bundle,
    write_manifest,
)
from tests.release_fixtures import TinyLinearModel as _TinyModel
from tests.release_fixtures import release_manifest as _manifest


class _PrefixedTinyModel(_TinyModel):
    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return {
            f"_orig_mod.{name}": value
            for name, value in super().state_dict(*args, **kwargs).items()
        }


def test_release_manifest_json_round_trip_is_deterministic(tmp_path: Path) -> None:
    manifest_path = tmp_path / "release.json"
    manifest = _manifest()

    write_manifest(manifest_path, manifest)
    first_content = manifest_path.read_bytes()
    restored = read_manifest(manifest_path)
    write_manifest(manifest_path, restored)

    assert restored == manifest
    assert manifest_path.read_bytes() == first_content


def test_release_manifest_checksum_verification(tmp_path: Path) -> None:
    manifest_path = tmp_path / "release.json"
    write_manifest(manifest_path, _manifest())
    checksum = sha256_file(manifest_path)

    assert verify_sha256(manifest_path, checksum)
    assert not verify_sha256(manifest_path, "0" * 64)


def test_inference_bundle_round_trip_loads_strict_cpu_weights(tmp_path: Path) -> None:
    torch.manual_seed(42)
    source_model = _TinyModel()
    bundle_dir = tmp_path / "bundle"

    write_inference_bundle(bundle_dir, manifest=_manifest(), model=source_model)
    restored = load_inference_bundle(bundle_dir, model_factory=lambda config: _TinyModel())

    assert restored.manifest == _manifest()
    assert not restored.model.training
    assert all(parameter.device.type == "cpu" for parameter in restored.model.parameters())
    assert restored.model.state_dict().keys() == source_model.state_dict().keys()
    for name, parameter in restored.model.state_dict().items():
        assert torch.equal(parameter, source_model.state_dict()[name])


def test_inference_bundle_strips_compiled_state_dict_prefix(tmp_path: Path) -> None:
    torch.manual_seed(42)
    source_model = _PrefixedTinyModel()
    bundle_dir = tmp_path / "bundle"

    write_inference_bundle(bundle_dir, manifest=_manifest(), model=source_model)
    restored = load_inference_bundle(bundle_dir, model_factory=lambda config: _TinyModel())

    assert set(restored.model.state_dict()) == {"linear.weight", "linear.bias"}
    assert torch.equal(restored.model.linear.weight, source_model.linear.weight)
    assert torch.equal(restored.model.linear.bias, source_model.linear.bias)


def test_inference_bundle_detects_tampered_artifact(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_inference_bundle(bundle_dir, manifest=_manifest(), model=_TinyModel())
    model_path = bundle_dir / "model.safetensors"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="Checksum mismatch.*model.safetensors"):
        validate_bundle_checksums(bundle_dir)
    with pytest.raises(ValueError, match="Checksum mismatch.*model.safetensors"):
        load_inference_bundle(bundle_dir, model_factory=lambda config: _TinyModel())
