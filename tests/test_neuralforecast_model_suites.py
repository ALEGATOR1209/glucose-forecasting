"""Tests for YAML-defined NeuralForecast model suites."""
from __future__ import annotations

import pytest

from glucose_forecasting.backends.neuralforecast.catalog import (
    NeuralForecastModel,
    create_model,
    resolve_models,
)
from glucose_forecasting.backends.neuralforecast.config import load_model_suites


def test_packaged_auto_suite_resolves_constructible_xlstm() -> None:
    suites, _ = load_model_suites()
    definitions = resolve_models(
        "auto",
        suite_models={name: suite.models for name, suite in suites.suites.items()},
    )
    xlstm = next(
        definition
        for definition in definitions
        if definition.name is NeuralForecastModel.XLSTM
    )
    model = create_model(
        xlstm,
        horizon=2,
        input_size=4,
        max_steps=1,
        historical_exogenous=("basal", "bolus", "carbohydrates"),
    )

    assert model.__class__.__name__ == "xLSTM"
    assert model.hist_exog_list == ["basal", "bolus", "carbohydrates"]


def test_model_suite_rejects_unknown_model_name() -> None:
    suites, _ = load_model_suites()

    with pytest.raises(ValueError, match="unknown NeuralForecast"):
        resolve_models("not-a-model", suite_models={"auto": suites.suites["auto"].models})
