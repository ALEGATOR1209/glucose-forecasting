"""Typed contracts and utilities for inference release manifests."""

from glucose_forecasting.release.bundle import (
    CHECKSUMS_FILENAME,
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    PROVENANCE_FILENAME,
    LoadedInferenceBundle,
    load_inference_bundle,
    validate_bundle_checksums,
    validate_inference_bundle,
    write_inference_bundle,
)
from glucose_forecasting.release.checksums import (
    sha256_bytes,
    sha256_file,
    verify_sha256,
)
from glucose_forecasting.release.config import InferenceConfig
from glucose_forecasting.release.io import (
    canonical_json_bytes,
    read_json,
    read_manifest,
    write_json,
    write_manifest,
)
from glucose_forecasting.release.manifest import ReleaseManifest
from glucose_forecasting.release.hub import (
    download_inference_bundle,
    ensure_model_repo,
    package_bundle_for_hub,
    publish_inference_bundle,
)
from glucose_forecasting.release.model_card import generate_model_card
from glucose_forecasting.release.metrics import (
    EvaluationProtocol,
    MetricsSpec,
    SelectionMetric,
)
from glucose_forecasting.release.preprocessor import (
    ImputationSpec,
    PreprocessorSpec,
    ScalerSpec,
    WindowSpec,
)
from glucose_forecasting.release.provenance import ProvenanceSpec

__all__ = [
    "CHECKSUMS_FILENAME",
    "CONFIG_FILENAME",
    "EvaluationProtocol",
    "ImputationSpec",
    "InferenceConfig",
    "LoadedInferenceBundle",
    "MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "MetricsSpec",
    "MODEL_FILENAME",
    "PreprocessorSpec",
    "PREPROCESSOR_FILENAME",
    "PROVENANCE_FILENAME",
    "ProvenanceSpec",
    "ReleaseManifest",
    "ScalerSpec",
    "SelectionMetric",
    "WindowSpec",
    "canonical_json_bytes",
    "download_inference_bundle",
    "ensure_model_repo",
    "generate_model_card",
    "load_inference_bundle",
    "package_bundle_for_hub",
    "publish_inference_bundle",
    "read_json",
    "read_manifest",
    "sha256_bytes",
    "sha256_file",
    "validate_bundle_checksums",
    "validate_inference_bundle",
    "verify_sha256",
    "write_inference_bundle",
    "write_json",
    "write_manifest",
]
