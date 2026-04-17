"""Tests for NN weight initialization functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from aerocapture.training.encoding import nn_param_specs_from_architecture
from aerocapture.training.initialization import compute_layer_bound, generate_initialized_weights
from aerocapture.training.population import create_nn_initial_population
from aerocapture.training.weight_stats import compute_weight_stats
from hypothesis import given, settings
from hypothesis import strategies as st


class TestComputeLayerBound:
    def test_xavier_tanh(self) -> None:
        assert compute_layer_bound(6, 12, "tanh") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_sigmoid(self) -> None:
        assert compute_layer_bound(6, 12, "sigmoid") == pytest.approx(math.sqrt(6 / 18))

    def test_xavier_asinh(self) -> None:
        assert compute_layer_bound(12, 2, "asinh") == pytest.approx(math.sqrt(6 / 14))

    def test_he_relu(self) -> None:
        assert compute_layer_bound(6, 64, "relu") == pytest.approx(math.sqrt(6 / 6))

    def test_lecun_linear(self) -> None:
        assert compute_layer_bound(32, 2, "linear") == pytest.approx(math.sqrt(3 / 32))

    def test_unknown_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            compute_layer_bound(6, 12, "gelu")


class TestGenerateInitializedWeights:
    def test_shape_default_arch(self) -> None:
        """Output length matches n_base_coef for [6, 12, 2]."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        # 6*12 + 12 + 12*2 + 2 = 110
        assert len(weights) == 110

    def test_shape_deep_arch(self) -> None:
        """Output length matches n_base_coef for [6, 64, 32, 2]."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 64, 32, 2], ["relu", "tanh", "asinh"], rng)
        # 6*64 + 64 + 64*32 + 32 + 32*2 + 2 = 384 + 64 + 2048 + 32 + 64 + 2 = 2594
        assert len(weights) == 2594

    def test_weights_within_xavier_bounds(self) -> None:
        """Layer 0 weights (tanh) fall within Xavier limits."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        limit = math.sqrt(6 / 18)
        layer0_w = weights[:72]
        assert np.all(np.abs(layer0_w) <= limit + 1e-15)

    def test_weights_within_he_bounds(self) -> None:
        """Layer 0 weights (relu) fall within He limits."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 64, 32, 2], ["relu", "tanh", "asinh"], rng)
        limit = math.sqrt(6 / 6)
        layer0_w = weights[: 6 * 64]
        assert np.all(np.abs(layer0_w) <= limit + 1e-15)

    def test_biases_are_zero(self) -> None:
        """All biases initialized to zero."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        assert np.all(weights[72:84] == 0.0)
        assert np.all(weights[108:110] == 0.0)

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical weights."""
        w1 = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], np.random.default_rng(99))
        w2 = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], np.random.default_rng(99))
        np.testing.assert_array_equal(w1, w2)

    @given(data=st.data())
    @settings(max_examples=20)
    def test_property_weights_respect_bounds(self, data: st.DataObject) -> None:
        """For random architectures, all weights respect per-layer bounds."""
        n_layers = data.draw(st.integers(2, 5))
        layer_sizes = [data.draw(st.integers(2, 32)) for _ in range(n_layers)]
        activations_pool = ["tanh", "sigmoid", "asinh", "relu", "linear"]
        activations = [data.draw(st.sampled_from(activations_pool)) for _ in range(n_layers - 1)]
        rng = np.random.default_rng(42)

        weights = generate_initialized_weights(layer_sizes, activations, rng)

        expected_len = sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] for i in range(n_layers - 1))
        assert len(weights) == expected_len

        idx = 0
        for i in range(n_layers - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = compute_layer_bound(fan_in, fan_out, activations[i])
            n_weights = fan_in * fan_out
            layer_w = weights[idx : idx + n_weights]
            assert np.all(np.abs(layer_w) <= limit + 1e-15), f"Layer {i} weights exceed bound {limit}"
            idx += n_weights
            layer_b = weights[idx : idx + fan_out]
            assert np.all(layer_b == 0.0), f"Layer {i} biases not zero"
            idx += fan_out


class TestComputeWeightStats:
    def test_stats_keys(self) -> None:
        """Returns per-layer weight and bias stats."""
        weights = np.zeros(110)
        stats = compute_weight_stats(weights, [6, 12, 2])
        assert "layer_0_w" in stats
        assert "layer_0_b" in stats
        assert "layer_1_w" in stats
        assert "layer_1_b" in stats

    def test_stats_values(self) -> None:
        """Stats are computed correctly for known values."""
        rng = np.random.default_rng(42)
        weights = generate_initialized_weights([6, 12, 2], ["tanh", "asinh"], rng)
        stats = compute_weight_stats(weights, [6, 12, 2])
        layer0_w = weights[:72]
        assert stats["layer_0_w"]["min"] == pytest.approx(float(layer0_w.min()))
        assert stats["layer_0_w"]["max"] == pytest.approx(float(layer0_w.max()))
        assert stats["layer_0_w"]["mean"] == pytest.approx(float(layer0_w.mean()))
        assert stats["layer_0_w"]["std"] == pytest.approx(float(layer0_w.std()))

    def test_zero_biases(self) -> None:
        """Zero biases produce zero stats."""
        weights = np.zeros(110)
        stats = compute_weight_stats(weights, [6, 12, 2])
        assert stats["layer_0_b"]["std"] == 0.0


class TestNNPopulationInit:
    def test_nn_population_shape_and_bounds(self) -> None:
        """create_nn_initial_population returns correct shape in [0, 1]."""
        layer_sizes = [6, 12, 2]
        activations = ["tanh", "asinh"]
        n_pop = 10
        rng = np.random.default_rng(42)

        pop = create_nn_initial_population(layer_sizes, activations, n_pop, rng)

        specs = nn_param_specs_from_architecture(layer_sizes, activations)
        assert pop.shape == (n_pop, len(specs))
        assert np.all(pop >= 0.0)
        assert np.all(pop <= 1.0)

    def test_nn_population_with_seed_weights(self) -> None:
        """Seed weights are used for the first individual."""
        layer_sizes = [6, 12, 2]
        activations = ["tanh", "asinh"]
        rng = np.random.default_rng(42)
        seed_w = generate_initialized_weights(layer_sizes, activations, rng)

        pop = create_nn_initial_population(
            layer_sizes,
            activations,
            5,
            np.random.default_rng(0),
            seed_weights=seed_w,
        )
        # First individual should be close to seed_weights when decoded
        specs = nn_param_specs_from_architecture(layer_sizes, activations)
        for j, s in enumerate(specs):
            decoded = s.p_min + pop[0, j] * (s.p_max - s.p_min)
            if j < len(seed_w):
                assert abs(decoded - seed_w[j]) < 1e-10, f"param {j}: decoded={decoded}, seed={seed_w[j]}"
