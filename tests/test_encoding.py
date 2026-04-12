"""Tests for real-valued encoding/decoding."""

from __future__ import annotations

import math

import numpy as np
import pytest
from aerocapture.training.encoding import decode_normalized, encode_to_normalized, nn_param_specs_from_architecture
from aerocapture.training.param_spaces import ParamSpec


class TestLinearRoundtrip:
    """Normalized [0,1] <-> physical value roundtrip for linear params."""

    def test_midpoint(self):
        specs = [ParamSpec("x", 10.0, 20.0, 15.0)]
        physical = decode_normalized(np.array([0.5]), specs)
        assert physical["x"] == pytest.approx(15.0)

    def test_boundaries(self):
        specs = [ParamSpec("x", -5.0, 5.0, 0.0)]
        lo = decode_normalized(np.array([0.0]), specs)
        hi = decode_normalized(np.array([1.0]), specs)
        assert lo["x"] == pytest.approx(-5.0)
        assert hi["x"] == pytest.approx(5.0)

    def test_roundtrip(self):
        specs = [ParamSpec("a", 1.0, 100.0, 50.0), ParamSpec("b", -10.0, 10.0, 0.0)]
        original = {"a": 73.5, "b": -3.2}
        normalized = encode_to_normalized(original, specs)
        recovered = decode_normalized(normalized, specs)
        assert recovered["a"] == pytest.approx(73.5)
        assert recovered["b"] == pytest.approx(-3.2)


class TestLogScaleRoundtrip:
    """Normalized [0,1] <-> physical value roundtrip for log-scale params."""

    def test_midpoint_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        physical = decode_normalized(np.array([0.5]), specs)
        # Midpoint in log10 space: 10^((-8 + -5) / 2) = 10^-6.5
        assert physical["g"] == pytest.approx(10**-6.5)

    def test_boundaries_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        lo = decode_normalized(np.array([0.0]), specs)
        hi = decode_normalized(np.array([1.0]), specs)
        assert lo["g"] == pytest.approx(1e-8)
        assert hi["g"] == pytest.approx(1e-5)

    def test_roundtrip_log(self):
        specs = [ParamSpec("g", 1e-8, 1e-5, 1e-6, log_scale=True)]
        original = {"g": 3.7e-7}
        normalized = encode_to_normalized(original, specs)
        recovered = decode_normalized(normalized, specs)
        assert recovered["g"] == pytest.approx(3.7e-7, rel=1e-10)


class TestMixedParams:
    """Mixed linear + log-scale parameter vectors."""

    def test_multi_param_decode(self):
        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
            ParamSpec("angle", -180.0, 180.0, 0.0),
        ]
        x = np.array([0.0, 1.0, 0.5])
        result = decode_normalized(x, specs)
        assert result["tau"] == pytest.approx(2.0)
        assert result["gain"] == pytest.approx(1e-5)
        assert result["angle"] == pytest.approx(0.0)

    def test_encode_defaults(self):
        specs = [
            ParamSpec("tau", 2.0, 60.0, 30.0),
            ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
        ]
        defaults = {s.name: s.default for s in specs}
        normalized = encode_to_normalized(defaults, specs)
        assert 0.0 <= normalized[0] <= 1.0
        assert 0.0 <= normalized[1] <= 1.0
        recovered = decode_normalized(normalized, specs)
        assert recovered["tau"] == pytest.approx(30.0)
        assert recovered["gain"] == pytest.approx(1e-6, rel=1e-10)


class TestNNParamSpecs:
    """NN weight bound computation from architecture."""

    def test_layer_count(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        # Layer 0: 16*24 weights + 24 biases = 408
        # Layer 1: 24*2 weights + 2 biases = 50
        assert len(specs) == 458

    def test_weight_bounds_symmetric(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        for s in specs:
            assert s.p_min == pytest.approx(-s.p_max)
            assert s.p_max > 0.0

    def test_xavier_bound_layer0(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=1.0)
        # Xavier for tanh: sqrt(6 / (16 + 24)) = sqrt(6/40)
        expected_bound = math.sqrt(6.0 / 40.0)
        # First spec is a weight for layer 0
        assert specs[0].p_max == pytest.approx(expected_bound)

    def test_bias_bounds(self):
        layer_sizes = [16, 24, 2]
        activations = ["tanh", "tanh"]
        specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=2.0)
        # Biases for layer 0 are at indices 384..407 (after 16*24=384 weights)
        bias_spec = specs[384]
        assert "bias" in bias_spec.name
        # Bias bound = multiplier * xavier_bound
        expected = 2.0 * math.sqrt(6.0 / 40.0)
        assert bias_spec.p_max == pytest.approx(expected)
