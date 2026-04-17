"""Tests for nn_param_specs_from_v2: v2 architecture list -> PSO bounds."""

from __future__ import annotations

from aerocapture.training.encoding import (
    nn_param_specs_from_architecture,
    nn_param_specs_from_v2,
)
from aerocapture.training.rl.schemas import DenseSpec


def test_v2_all_dense_matches_v1() -> None:
    layer_sizes = [16, 24, 2]
    activations = ["tanh", "asinh"]
    v1_specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=1.0)

    architecture = [
        DenseSpec(type="dense", input_size=16, output_size=24, activation="tanh"),
        DenseSpec(type="dense", input_size=24, output_size=2, activation="asinh"),
    ]
    v2_specs = nn_param_specs_from_v2(architecture, bound_multiplier=1.0)

    assert len(v1_specs) == len(v2_specs)
    for s1, s2 in zip(v1_specs, v2_specs, strict=True):
        assert s1.p_min == s2.p_min
        assert s1.p_max == s2.p_max
        assert s1.log_scale == s2.log_scale


def test_v2_empty_architecture() -> None:
    assert nn_param_specs_from_v2([], bound_multiplier=1.0) == []
