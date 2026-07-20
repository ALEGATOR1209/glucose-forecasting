"""Typed, lazy NeuralForecast model catalog.

The catalog is intentionally dependency-light: importing it only exposes static
metadata. NeuralForecast is imported when a selected model is constructed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from importlib.util import find_spec
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class NeuralForecastModel(StrEnum):
    """Common model names exported by NeuralForecast."""

    NBEATS = "NBEATS"
    NBEATSX = "NBEATSx"
    NHITS = "NHITS"
    MLP = "MLP"
    LSTM = "LSTM"
    GRU = "GRU"
    RNN = "RNN"
    DILATED_RNN = "DilatedRNN"
    TCN = "TCN"
    BITCN = "BiTCN"
    DLINEAR = "DLinear"
    NLINEAR = "NLinear"
    TFT = "TFT"
    TIDE = "TiDE"
    DEEPAR = "DeepAR"
    PATCHTST = "PatchTST"
    TIMEXER = "TimeXer"
    TSMIXERX = "TSMixerx"
    HINT = "HINT"
    XLSTM = "xLSTM"


class ModelProfile(StrEnum):
    """Curated, directly constructible model groups."""

    BASELINE = "baseline"
    HISTORICAL_EXOGENOUS = "historical-exogenous"
    COMMON = "common"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """NeuralForecast model requirements relevant to glucose forecasting."""

    supports_historical_exogenous: bool
    requires_special_initialization: bool = False
    requires_n_series: bool = False


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Static metadata and lazy import target for one model."""

    name: NeuralForecastModel
    capabilities: ModelCapabilities
    import_name: str
    optional_dependency: str | None = None


def _definition(
    name: NeuralForecastModel,
    *,
    supports_historical_exogenous: bool,
    requires_special_initialization: bool = False,
    requires_n_series: bool = False,
    optional_dependency: str | None = None,
) -> ModelDefinition:
    return ModelDefinition(
        name=name,
        capabilities=ModelCapabilities(
            supports_historical_exogenous=supports_historical_exogenous,
            requires_special_initialization=requires_special_initialization,
            requires_n_series=requires_n_series,
        ),
        import_name=name.value,
        optional_dependency=optional_dependency,
    )


MODEL_CATALOG: Mapping[NeuralForecastModel, ModelDefinition] = MappingProxyType(
    {
        NeuralForecastModel.NBEATS: _definition(
            NeuralForecastModel.NBEATS, supports_historical_exogenous=False
        ),
        NeuralForecastModel.NBEATSX: _definition(
            NeuralForecastModel.NBEATSX, supports_historical_exogenous=True
        ),
        NeuralForecastModel.NHITS: _definition(
            NeuralForecastModel.NHITS, supports_historical_exogenous=True
        ),
        NeuralForecastModel.MLP: _definition(
            NeuralForecastModel.MLP, supports_historical_exogenous=True
        ),
        NeuralForecastModel.LSTM: _definition(
            NeuralForecastModel.LSTM, supports_historical_exogenous=True
        ),
        NeuralForecastModel.GRU: _definition(
            NeuralForecastModel.GRU, supports_historical_exogenous=True
        ),
        NeuralForecastModel.RNN: _definition(
            NeuralForecastModel.RNN, supports_historical_exogenous=True
        ),
        NeuralForecastModel.DILATED_RNN: _definition(
            NeuralForecastModel.DILATED_RNN, supports_historical_exogenous=True
        ),
        NeuralForecastModel.TCN: _definition(
            NeuralForecastModel.TCN, supports_historical_exogenous=True
        ),
        NeuralForecastModel.BITCN: _definition(
            NeuralForecastModel.BITCN, supports_historical_exogenous=True
        ),
        NeuralForecastModel.DLINEAR: _definition(
            NeuralForecastModel.DLINEAR, supports_historical_exogenous=True
        ),
        NeuralForecastModel.NLINEAR: _definition(
            NeuralForecastModel.NLINEAR, supports_historical_exogenous=True
        ),
        NeuralForecastModel.TFT: _definition(
            NeuralForecastModel.TFT, supports_historical_exogenous=True
        ),
        NeuralForecastModel.TIDE: _definition(
            NeuralForecastModel.TIDE, supports_historical_exogenous=True
        ),
        NeuralForecastModel.DEEPAR: _definition(
            NeuralForecastModel.DEEPAR, supports_historical_exogenous=False
        ),
        NeuralForecastModel.PATCHTST: _definition(
            NeuralForecastModel.PATCHTST, supports_historical_exogenous=False
        ),
        NeuralForecastModel.TIMEXER: _definition(
            NeuralForecastModel.TIMEXER,
            supports_historical_exogenous=True,
            requires_n_series=True,
        ),
        NeuralForecastModel.TSMIXERX: _definition(
            NeuralForecastModel.TSMIXERX,
            supports_historical_exogenous=True,
            requires_n_series=True,
        ),
        NeuralForecastModel.HINT: _definition(
            NeuralForecastModel.HINT,
            supports_historical_exogenous=False,
            requires_special_initialization=True,
        ),
        NeuralForecastModel.XLSTM: _definition(
            NeuralForecastModel.XLSTM,
            supports_historical_exogenous=True,
            optional_dependency="xlstm",
        ),
    }
)

_PROFILE_MODELS: Mapping[ModelProfile, tuple[NeuralForecastModel, ...]] = MappingProxyType(
    {
        ModelProfile.BASELINE: (
            NeuralForecastModel.NBEATS,
            NeuralForecastModel.NHITS,
            NeuralForecastModel.NBEATSX,
        ),
        ModelProfile.HISTORICAL_EXOGENOUS: (
            NeuralForecastModel.NHITS,
            NeuralForecastModel.NBEATSX,
            NeuralForecastModel.LSTM,
            NeuralForecastModel.TFT,
            NeuralForecastModel.TIDE,
        ),
        ModelProfile.COMMON: (
            NeuralForecastModel.NBEATS,
            NeuralForecastModel.NHITS,
            NeuralForecastModel.NBEATSX,
            NeuralForecastModel.LSTM,
            NeuralForecastModel.GRU,
            NeuralForecastModel.TCN,
            NeuralForecastModel.DLINEAR,
            NeuralForecastModel.TFT,
        ),
    }
)


def select_models(
    profile: ModelProfile,
    *,
    requires_historical_exogenous: bool | None = None,
    include_special_initialization: bool = False,
    include_multivariate: bool = False,
) -> tuple[ModelDefinition, ...]:
    """Return profile models filtered by their declared capabilities."""
    definitions = (MODEL_CATALOG[name] for name in _PROFILE_MODELS[profile])
    return tuple(
        definition
        for definition in definitions
        if (
            (requires_historical_exogenous is None)
            or (
                definition.capabilities.supports_historical_exogenous
                is requires_historical_exogenous
            )
        )
        and (
            include_special_initialization
            or not definition.capabilities.requires_special_initialization
        )
        and (include_multivariate or not definition.capabilities.requires_n_series)
    )


def iter_models(
    *,
    requires_historical_exogenous: bool | None = None,
    include_special_initialization: bool = False,
    include_multivariate: bool = False,
) -> Iterator[ModelDefinition]:
    """Iterate over all catalog models filtered by capabilities."""
    for definition in MODEL_CATALOG.values():
        if (
            (requires_historical_exogenous is None)
            or (
                definition.capabilities.supports_historical_exogenous
                is requires_historical_exogenous
            )
        ) and (
            include_special_initialization
            or not definition.capabilities.requires_special_initialization
        ) and (include_multivariate or not definition.capabilities.requires_n_series):
            yield definition


def create_models(
    horizon: int,
    input_size: int,
    max_steps: int,
    profile: ModelProfile,
) -> tuple[object, ...]:
    """Construct the models in a profile, importing NeuralForecast only here."""
    if horizon <= 0 or input_size <= 0 or max_steps <= 0:
        raise ValueError("horizon, input_size, and max_steps must be positive")

    models_module = import_module("neuralforecast.models")
    return tuple(
        create_model(
            definition,
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
        )
        for definition in select_models(profile)
    )


def resolve_models(
    selection: str,
    *,
    suite_models: Mapping[str, Sequence[NeuralForecastModel]],
) -> tuple[ModelDefinition, ...]:
    """Resolve a suite name or comma-separated concrete model names."""
    if selection in suite_models:
        names = tuple(suite_models[selection])
    else:
        lookup = {model.value.lower(): model for model in NeuralForecastModel}
        names = tuple(
            lookup.get(item.strip().lower())
            for item in selection.split(",")
            if item.strip()
        )
        if not names or any(name is None for name in names):
            known = ", ".join(sorted((*suite_models, *(model.value for model in NeuralForecastModel))))
            raise ValueError(f"unknown NeuralForecast model or suite {selection!r}; choose from {known}")
    return tuple(MODEL_CATALOG[name] for name in names)


def create_model(
    definition: ModelDefinition,
    *,
    horizon: int,
    input_size: int,
    max_steps: int,
    historical_exogenous: Sequence[str] = (),
    trainer_kwargs: Mapping[str, Any] | None = None,
    learning_rate: float = 1e-3,
    val_check_steps: int = 400,
    batch_size: int = 8,
    valid_batch_size: int = 8,
    windows_batch_size: int = 256,
    inference_windows_batch_size: int = 256,
    step_size: int = 12,
) -> Any:
    """Instantiate one validated model using common glucose-forecast settings."""
    if definition.optional_dependency and find_spec(definition.optional_dependency) is None:
        raise RuntimeError(
            f"{definition.name.value} requires optional dependency {definition.optional_dependency!r}; "
            f"install it with `uv add {definition.optional_dependency}`."
        )
    models_module = import_module("neuralforecast.models")
    try:
        model_class = getattr(models_module, definition.import_name)
    except AttributeError as error:
        raise RuntimeError(
            f"installed neuralforecast does not export {definition.import_name}; "
            "upgrade neuralforecast or remove the model from the YAML suite."
        ) from error
    from neuralforecast.losses.pytorch import MAE

    return model_class(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=max_steps,
        val_check_steps=min(val_check_steps, max_steps),
        learning_rate=learning_rate,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        windows_batch_size=windows_batch_size,
        inference_windows_batch_size=inference_windows_batch_size,
        step_size=step_size,
        hist_exog_list=list(historical_exogenous)
        if definition.capabilities.supports_historical_exogenous
        else None,
        **dict(trainer_kwargs or {}),
    )
