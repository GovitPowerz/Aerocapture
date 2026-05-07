"""Regression guard: NN scaffolding param ordering must match the deploy chromosome layout."""
from __future__ import annotations

from aerocapture.training.param_spaces import (
    _EXIT_PARAMS,
    _LATERAL_PARAMS,
    _NAV_PARAMS,
    _SHAPING_PARAMS,
    _THERMAL_LIMITER_PARAMS,
    _NN_SCAFFOLDING_PARAMS,
)


def test_nn_scaffolding_is_concatenation_in_documented_order():
    expected = [*_NAV_PARAMS, *_LATERAL_PARAMS, *_EXIT_PARAMS, *_THERMAL_LIMITER_PARAMS, *_SHAPING_PARAMS]
    assert list(_NN_SCAFFOLDING_PARAMS) == expected


def test_nn_scaffolding_has_seventeen_params():
    assert len(_NN_SCAFFOLDING_PARAMS) == 17


def test_nn_scaffolding_names_are_unique_and_prefixed():
    names = [s.name for s in _NN_SCAFFOLDING_PARAMS]
    assert len(set(names)) == len(names), "duplicate names"
    valid = ("nav.", "lateral.", "exit.", "thermal.", "shaping.")
    for name in names:
        assert name.startswith(valid), f"unexpected prefix in {name!r}"
