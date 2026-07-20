"""Model-card rendering for validated inference releases."""

from __future__ import annotations

import json

from glucose_forecasting.release.manifest import ReleaseManifest


def _metric_lines(metrics: dict[str, float]) -> list[str]:
    """Render named metric values in deterministic order."""
    return [f"- `{name}`: {value:g}" for name, value in sorted(metrics.items())]


def generate_model_card(manifest: ReleaseManifest, *, repo_id: str) -> str:
    """Create a Hub-compatible README from release-contract metadata."""
    config = manifest.config
    provenance = manifest.provenance
    protocol = manifest.metrics.protocol
    tags = ["glucose-forecasting", "time-series-forecasting", config.model_type]
    lines = [
        "---",
        "library_name: pytorch",
        "tags:",
        *(f"- {tag}" for tag in tags),
        "---",
        "",
        f"# {repo_id}",
        "",
        "This repository contains a checksum-validated, inference-only glucose "
        "forecasting bundle.",
        "",
        "## Release",
        "",
        f"- Release ID: `{manifest.release_id}`",
        f"- Model ID: `{config.model_id}`",
        f"- Model type: `{config.model_type}`",
        f"- Input features: {', '.join(f'`{feature}`' for feature in config.feature_order)}",
        f"- Forecast horizon: {config.horizon} steps at {config.cadence}-minute cadence",
        f"- Input window: {manifest.preprocessor.window.input_steps} steps",
        "",
        "## Evaluation",
        "",
        f"Selection metric: `{manifest.metrics.selection_metric.name}` "
        f"({manifest.metrics.selection_metric.direction}).",
        "",
        "### Validation metrics",
        "",
        *_metric_lines(manifest.metrics.validation),
        "",
        "### Test metrics",
        "",
        *_metric_lines(manifest.metrics.test),
        "",
        f"Protocol: {protocol.name} (`{protocol.split}` split).",
    ]
    if protocol.details:
        lines.extend(["", "Protocol details:"])
        lines.extend(f"- `{name}`: {value}" for name, value in sorted(protocol.details.items()))

    lines.extend(
        [
            "",
            "## Source and provenance",
            "",
            f"- Source revision: `{provenance.git_sha}`",
            f"- Dependency lock hash: `{provenance.lock_hash}`",
            f"- Dataset fingerprint: `{provenance.dataset_fingerprint}`",
            f"- Random seed: `{provenance.seed}`",
            f"- Environment: {', '.join(f'`{name}={value}`' for name, value in sorted(provenance.env.items()))}",
            "",
            "## Loading",
            "",
            "```python",
            "from huggingface_hub import snapshot_download",
            "from glucose_forecasting.release import load_inference_bundle",
            "",
            f'bundle_dir = snapshot_download(repo_id={json.dumps(repo_id)}, revision="COMMIT_SHA")',
            "# Supply the architecture-specific factory for this model type.",
            "bundle = load_inference_bundle(bundle_dir, model_factory=build_model)",
            "```",
            "",
            "The downloader must use a pinned Hub revision. "
            "`load_inference_bundle` verifies SHA256 checksums and ensures the "
            "component documents match `manifest.json` before weights are loaded.",
            "",
            "## Limitations",
            "",
            "- This release is intended for research use and is not a medical device.",
            "- Performance metrics apply only to the documented evaluation protocol and dataset fingerprint.",
            "- Inputs must use the feature order, cadence, units, scaling, and imputation policy in `preprocessor.json`.",
        ]
    )
    return "\n".join(lines) + "\n"
