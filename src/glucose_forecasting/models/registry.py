"""JSON-backed model artifact registry and compatibility resolution."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from glucose_forecasting.config import DatasetSpec, ModelSelection


class ModelResolutionError(ValueError):
    """Raised when no registered artifact satisfies a selection request."""


class ModelArtifact(BaseModel):
    """A versioned, deployable model artifact and its input contract.

    ``validation_metric`` is intentionally the only ranking signal. Test
    metrics and artifact timestamps do not participate in default selection.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        populate_by_name=True,
    )

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    artifact_path: Path
    data_schema: str = Field(alias="schema", min_length=1)
    covariates: tuple[str, ...] = ()
    cadence_minutes: int = Field(default=5, gt=0)
    horizon_steps: int = Field(default=12, gt=0)
    validation_metric: float = Field(ge=0)
    stable: bool = True

    @field_validator("covariates")
    @classmethod
    def require_unique_covariates(cls, covariates: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate or empty covariate names."""
        if any(not covariate.strip() for covariate in covariates):
            raise ValueError("covariates must not contain empty names")
        if len(covariates) != len(set(covariates)):
            raise ValueError("covariates must be unique")
        return covariates

    def compatibility_errors(self, dataset: DatasetSpec) -> tuple[str, ...]:
        """Return all input-contract mismatches for ``dataset``."""
        errors: list[str] = []
        if self.data_schema != dataset.data_schema:
            errors.append(
                f"schema mismatch: model requires {self.data_schema!r}, "
                f"dataset provides {dataset.data_schema!r}"
            )

        missing_covariates = set(self.covariates) - set(dataset.covariates)
        if missing_covariates:
            errors.append(
                "missing required covariates: "
                + ", ".join(sorted(missing_covariates))
            )

        if self.cadence_minutes != dataset.cadence_minutes:
            errors.append(
                f"cadence mismatch: model requires {self.cadence_minutes} minutes, "
                f"dataset provides {dataset.cadence_minutes} minutes"
            )
        if self.horizon_steps != dataset.horizon_steps:
            errors.append(
                f"horizon mismatch: model requires {self.horizon_steps} steps, "
                f"dataset provides {dataset.horizon_steps} steps"
            )
        return tuple(errors)

    def is_compatible_with(self, dataset: DatasetSpec) -> bool:
        """Return whether the dataset satisfies this artifact's input contract."""
        return not self.compatibility_errors(dataset)


class ModelRegistry(BaseModel):
    """Collection of versioned model artifacts with deterministic resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = Field(default=1, ge=1)
    models: tuple[ModelArtifact, ...] = ()

    @model_validator(mode="after")
    def require_unique_model_versions(self) -> ModelRegistry:
        """Reject ambiguous duplicate name/version entries."""
        keys = [(model.name, model.version) for model in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("model registry contains duplicate name/version entries")
        return self

    def resolve(
        self,
        dataset: DatasetSpec,
        selection: ModelSelection | None = None,
    ) -> ModelArtifact:
        """Resolve an explicit artifact or the lowest-metric compatible stable one.

        An explicit name may still identify several versions, in which case the
        lowest validation metric decides. A version requires an exact match.
        Default resolution only considers stable artifacts and never uses test
        metrics or creation/recency metadata.
        """
        requested = selection or ModelSelection()
        candidates = self.models
        if requested.name is not None:
            candidates = tuple(model for model in candidates if model.name == requested.name)
        else:
            candidates = tuple(model for model in candidates if model.stable)
        if requested.version is not None:
            candidates = tuple(
                model for model in candidates if model.version == requested.version
            )

        if not candidates:
            raise ModelResolutionError(
                f"No registered model matches name={requested.name!r}, "
                f"version={requested.version!r}."
            )

        compatible = tuple(
            model for model in candidates if model.is_compatible_with(dataset)
        )
        if not compatible:
            details = "; ".join(
                f"{model.name}@{model.version}: {', '.join(model.compatibility_errors(dataset))}"
                for model in candidates
            )
            raise ModelResolutionError(
                f"No selected model is compatible with dataset {dataset.name!r}: {details}"
            )

        return min(
            compatible,
            key=lambda model: (model.validation_metric, model.name, model.version),
        )


def load_registry(path: str | Path) -> ModelRegistry:
    """Load and validate a JSON model registry."""
    registry_path = Path(path)
    with registry_path.open(encoding="utf-8") as registry_file:
        payload = json.load(registry_file)
    return ModelRegistry.model_validate(payload)


def save_registry(registry: ModelRegistry, path: str | Path) -> Path:
    """Write a JSON model registry and return its path."""
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            registry.model_dump(mode="json", by_alias=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry_path


def resolve_data_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve bare data filenames under ``project_root/data/input``.

    Absolute paths remain unchanged. Relative paths containing directories are
    interpreted relative to ``project_root``.
    """
    data_path = Path(path)
    if data_path.is_absolute():
        return data_path
    root = Path(project_root)
    if data_path.parent == Path("."):
        return root / "data" / "input" / data_path
    return root / data_path
