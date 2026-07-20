"""Unit tests for Hub release packaging without network requests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from glucose_forecasting.release import (
    download_inference_bundle,
    package_bundle_for_hub,
    publish_inference_bundle,
    write_inference_bundle,
)
from tests.release_fixtures import TinyLinearModel as _TinyModel
from tests.release_fixtures import release_manifest as _manifest


class _FakeHubApi:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.info_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []
        self.uploaded_files: set[str] = set()

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        self.create_calls.append({"repo_id": repo_id, **kwargs})
        return object()

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.info_calls.append({"repo_id": repo_id, **kwargs})
        return object()

    def upload_folder(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        folder_path = Path(str(kwargs["folder_path"]))
        self.uploaded_files = {path.name for path in folder_path.iterdir()}
        return object()


def test_publish_packages_valid_bundle_and_uses_one_atomic_upload(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_inference_bundle(bundle_dir, manifest=_manifest(), model=_TinyModel())
    api = _FakeHubApi()

    published = publish_inference_bundle(
        bundle_dir,
        repo_id="GlucoseDAO/sugar-one",
        private=True,
        revision="release",
        api=api,
    )

    assert published == _manifest()
    assert api.create_calls == [
        {
            "repo_id": "GlucoseDAO/sugar-one",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.info_calls == [{"repo_id": "GlucoseDAO/sugar-one", "repo_type": "model"}]
    assert len(api.upload_calls) == 1
    assert api.upload_calls[0]["repo_id"] == "GlucoseDAO/sugar-one"
    assert api.upload_calls[0]["repo_type"] == "model"
    assert api.upload_calls[0]["revision"] == "release"
    assert api.upload_calls[0]["commit_message"] == "Publish inference release sugar-one-2026-07"
    assert api.uploaded_files == {
        "README.md",
        "checksums.sha256",
        "config.json",
        "manifest.json",
        "metrics.json",
        "model.safetensors",
        "preprocessor.json",
        "provenance.json",
    }


def test_model_card_contains_contract_metadata(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    package_dir = tmp_path / "package"
    write_inference_bundle(bundle_dir, manifest=_manifest(), model=_TinyModel())

    package_bundle_for_hub(bundle_dir, package_dir, repo_id="GlucoseDAO/sugar-one")
    card = (package_dir / "README.md").read_text(encoding="utf-8")

    assert card.startswith("---\nlibrary_name: pytorch\n")
    assert "### Validation metrics" in card
    assert "`mae`: 18.2" in card
    assert "### Test metrics" in card
    assert "`mae`: 19.1" in card
    assert "Source revision: `abc1234`" in card
    assert "## Limitations" in card
    assert "snapshot_download" in card


def test_download_validates_bundle_after_pinned_snapshot(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    write_inference_bundle(source_dir, manifest=_manifest(), model=_TinyModel())
    calls: list[dict[str, object]] = []

    def download_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        shutil.copytree(source_dir, Path(str(kwargs["local_dir"])))
        return str(kwargs["local_dir"])

    manifest = download_inference_bundle(
        "GlucoseDAO/sugar-one",
        revision="a1b2c3d4",
        target_dir=target_dir,
        downloader=download_snapshot,
    )

    assert manifest == _manifest()
    assert calls == [
        {
            "repo_id": "GlucoseDAO/sugar-one",
            "repo_type": "model",
            "revision": "a1b2c3d4",
            "local_dir": target_dir,
        }
    ]


def test_download_rejects_checksum_tampering(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    write_inference_bundle(source_dir, manifest=_manifest(), model=_TinyModel())

    def download_tampered_snapshot(**kwargs: object) -> str:
        shutil.copytree(source_dir, Path(str(kwargs["local_dir"])))
        model_path = Path(str(kwargs["local_dir"])) / "model.safetensors"
        model_path.write_bytes(model_path.read_bytes() + b"tampered")
        return str(kwargs["local_dir"])

    with pytest.raises(ValueError, match="Checksum mismatch.*model.safetensors"):
        download_inference_bundle(
            "GlucoseDAO/sugar-one",
            revision="a1b2c3d4",
            target_dir=target_dir,
            downloader=download_tampered_snapshot,
        )
