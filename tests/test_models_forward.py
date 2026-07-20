"""Cheap regression guard for the frozen model architectures.

GluMindModel, SugarOneModel, GluMindUniModel are explicitly "don't touch
without asking" files. A shape/dtype/state_dict-key regression test is the
cheapest possible guard against an accidental architecture change.
"""
from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from scripts.glumind.glumind_model import GluMindModel
from scripts.glumind_uni.glumind_uni_model import GluMindUniModel
from scripts.sugar_one.sugar_one_model import SugarOneModel
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
)

BATCH = 3

_COMMON_KWARGS: dict[str, Any] = {
    "n_time_steps": TINY_INPUT_STEPS,
    "d_model": TINY_D_MODEL,
    "n_heads": TINY_N_HEADS,
    "ff_units": TINY_FF_UNITS,
    "n_blocks": TINY_N_BLOCKS,
    "prediction_horizon": TINY_HORIZON,
    "dropout": 0.0,
}


def _make_glumind() -> GluMindModel:
    return GluMindModel(n_features=3, **_COMMON_KWARGS)


def _make_sugar_one() -> SugarOneModel:
    return SugarOneModel(n_features=4, **_COMMON_KWARGS)


def _make_glumind_uni() -> GluMindUniModel:
    return GluMindUniModel(**_COMMON_KWARGS)


# (id, factory, n_features, required key prefixes, forbidden substrings)
_MODEL_SPECS: list[tuple[str, Callable[[], nn.Module], int, list[str], list[str]]] = [
    (
        "glumind",
        _make_glumind,
        3,
        [
            "embed_glucose.",
            "embed_hr.",
            "embed_steps.",
            "pos_enc.",
            "blocks.0.cross_attn.",
            "blocks.0.multiscale.",
            "flatten_fc.",
            "out_fc.",
        ],
        [],
    ),
    (
        "sugar_one",
        _make_sugar_one,
        4,
        [
            "embed_glucose.",
            "embed_basal.",
            "embed_bolus.",
            "embed_carbs.",
            "blocks.0.cross_attn.mix_logits",
            "blocks.0.multiscale.",
            "flatten_fc.",
            "out_fc.",
        ],
        [],
    ),
    (
        "glumind_uni",
        _make_glumind_uni,
        1,
        [
            "embed_glucose.",
            "pos_enc.",
            "blocks.0.attn_high.",
            "blocks.0.attn_low2.",
            "blocks.0.attn_low4.",
            "flatten_fc.",
            "out_fc.",
        ],
        ["cross_attn", "embed_hr", "embed_steps"],
    ),
]


@pytest.mark.parametrize(
    ("factory", "n_features"),
    [(factory, n_features) for _, factory, n_features, _, _ in _MODEL_SPECS],
    ids=[name for name, *_ in _MODEL_SPECS],
)
def test_forward_shape_and_dtype(factory: Callable[[], nn.Module], n_features: int) -> None:
    model = factory()
    out = model(torch.randn(BATCH, TINY_INPUT_STEPS, n_features))
    assert out.shape == (BATCH, TINY_HORIZON)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    ("factory", "expected_prefixes", "forbidden"),
    [
        (factory, prefixes, forbidden)
        for _, factory, _, prefixes, forbidden in _MODEL_SPECS
    ],
    ids=[name for name, *_ in _MODEL_SPECS],
)
def test_state_dict_key_patterns_stable(
    factory: Callable[[], nn.Module],
    expected_prefixes: list[str],
    forbidden: list[str],
) -> None:
    keys = set(factory().state_dict().keys())
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"
    for needle in forbidden:
        assert not any(needle in k for k in keys), f"unexpected key containing {needle!r}"
