"""Tests for evaluate-model covariate selection helpers."""

from __future__ import annotations

import pytest

from scripts.sugar_one.evaluate_model import (
    _alias_to_canonical,
    _resolve_covariate_zeroing,
)


def test_alias_to_canonical_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown covariate"):
        _alias_to_canonical("basal", "glumind")


@pytest.mark.parametrize(
    ("zero_cov", "include_cov", "exclude_cov", "expected"),
    [
        (True, None, None, ([], ["basal", "bolus", "carbs"])),
        (False, "basal,bolus", None, (["basal", "bolus"], ["carbs"])),
        (False, None, "carbs", (["basal", "bolus"], ["carbs"])),
    ],
)
def test_resolve_covariate_zeroing(
    zero_cov: bool,
    include_cov: str | None,
    exclude_cov: str | None,
    expected: tuple[list[str], list[str]],
) -> None:
    active, zeroed = _resolve_covariate_zeroing(
        "sugar_one",
        zero_cov=zero_cov,
        include_cov=include_cov,
        exclude_cov=exclude_cov,
    )
    assert (active, zeroed) == expected


@pytest.mark.parametrize(
    ("zero_cov", "include_cov", "exclude_cov", "message"),
    [
        (True, "basal", None, "either --zero-cov or --include-cov"),
        (False, "basal", "carbs", "either --include-cov or --exclude-cov"),
    ],
)
def test_resolve_covariate_zeroing_rejects_conflicts(
    zero_cov: bool,
    include_cov: str | None,
    exclude_cov: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_covariate_zeroing(
            "sugar_one",
            zero_cov=zero_cov,
            include_cov=include_cov,
            exclude_cov=exclude_cov,
        )
