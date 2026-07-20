"""Tests for the bounded, lazy NeuralForecast backend catalog."""

from __future__ import annotations

from glucose_forecasting.backends.neuralforecast.catalog import (
    MODEL_CATALOG,
    ModelProfile,
    NeuralForecastModel,
    create_models,
    iter_models,
    select_models,
)


def test_historical_exogenous_profile_contains_only_supported_models() -> None:
    models = select_models(
        ModelProfile.HISTORICAL_EXOGENOUS,
        requires_historical_exogenous=True,
    )

    assert {model.name for model in models} == {
        NeuralForecastModel.NHITS,
        NeuralForecastModel.NBEATSX,
        NeuralForecastModel.LSTM,
        NeuralForecastModel.TFT,
        NeuralForecastModel.TIDE,
    }
    assert all(
        model.capabilities.supports_historical_exogenous for model in models
    )


def test_capability_filtering_excludes_special_and_multivariate_models() -> None:
    standard_models = tuple(iter_models())
    extended_models = tuple(
        iter_models(include_special_initialization=True, include_multivariate=True)
    )

    assert NeuralForecastModel.HINT not in {model.name for model in standard_models}
    assert NeuralForecastModel.HINT in {model.name for model in extended_models}
    assert NeuralForecastModel.TIMEXER not in {model.name for model in standard_models}
    assert NeuralForecastModel.TIMEXER in {model.name for model in extended_models}
    assert MODEL_CATALOG[NeuralForecastModel.HINT].capabilities.requires_special_initialization


def test_create_models_uses_installed_neuralforecast_lazily() -> None:
    models = create_models(
        horizon=12,
        input_size=24,
        max_steps=1,
        profile=ModelProfile.BASELINE,
    )

    assert [model.__class__.__name__ for model in models] == [
        "NBEATS",
        "NHITS",
        "NBEATSx",
    ]
    assert all(model.h == 12 and model.input_size == 24 for model in models)
