"""Hugging Face Hub publication and retrieval for inference bundles."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, snapshot_download

from glucose_forecasting.release.bundle import (
    CHECKSUMS_FILENAME,
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    PROVENANCE_FILENAME,
    validate_inference_bundle,
)
from glucose_forecasting.release.manifest import ReleaseManifest
from glucose_forecasting.release.model_card import generate_model_card

README_FILENAME = "README.md"
_BUNDLE_FILENAMES = (
    MODEL_FILENAME,
    MANIFEST_FILENAME,
    CONFIG_FILENAME,
    PREPROCESSOR_FILENAME,
    METRICS_FILENAME,
    PROVENANCE_FILENAME,
    CHECKSUMS_FILENAME,
)


class HubApi(Protocol):
    """Subset of the Hub client used by publication."""

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool | None = None,
        repo_type: str | None = None,
        exist_ok: bool = False,
    ) -> object: ...

    def repo_info(self, repo_id: str, *, repo_type: str | None = None) -> object: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        commit_message: str | None = None,
        repo_type: str | None = None,
        revision: str | None = None,
    ) -> object: ...


SnapshotDownloader = Callable[..., str]


def _require_text(value: str, *, name: str) -> None:
    """Reject blank identifiers before requesting the Hub."""
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def package_bundle_for_hub(
    bundle_dir: Path,
    package_dir: Path,
    *,
    repo_id: str,
) -> ReleaseManifest:
    """Validate and copy a local bundle plus generated card into a package directory."""
    _require_text(repo_id, name="repo_id")
    manifest = validate_inference_bundle(bundle_dir)
    if package_dir.exists() and any(package_dir.iterdir()):
        raise ValueError(f"Package directory must be empty: {package_dir}")
    package_dir.mkdir(parents=True, exist_ok=True)
    for filename in _BUNDLE_FILENAMES:
        shutil.copy2(bundle_dir / filename, package_dir / filename)
    (package_dir / README_FILENAME).write_text(
        generate_model_card(manifest, repo_id=repo_id),
        encoding="utf-8",
    )
    return manifest


def ensure_model_repo(
    repo_id: str,
    *,
    private: bool,
    api: HubApi | None = None,
) -> None:
    """Create a model repository when needed and verify it is reachable."""
    _require_text(repo_id, name="repo_id")
    hub_api = api if api is not None else HfApi()
    hub_api.create_repo(
        repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    hub_api.repo_info(repo_id, repo_type="model")


def publish_inference_bundle(
    bundle_dir: Path,
    *,
    repo_id: str,
    private: bool = False,
    revision: str | None = None,
    api: HubApi | None = None,
) -> ReleaseManifest:
    """Publish a validated bundle and model card in one Hub commit."""
    _require_text(repo_id, name="repo_id")
    hub_api = api if api is not None else HfApi()
    ensure_model_repo(repo_id, private=private, api=hub_api)
    with tempfile.TemporaryDirectory(prefix="glucose-release-") as temporary_dir:
        manifest = package_bundle_for_hub(bundle_dir, Path(temporary_dir), repo_id=repo_id)
        hub_api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=temporary_dir,
            commit_message=f"Publish inference release {manifest.release_id}",
            revision=revision,
        )
    return manifest


def download_inference_bundle(
    repo_id: str,
    *,
    revision: str,
    target_dir: Path,
    downloader: SnapshotDownloader = snapshot_download,
) -> ReleaseManifest:
    """Download a pinned release revision and validate its local bundle contract."""
    _require_text(repo_id, name="repo_id")
    _require_text(revision, name="revision")
    downloader(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=target_dir,
    )
    return validate_inference_bundle(target_dir)
