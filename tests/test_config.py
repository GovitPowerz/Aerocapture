"""Tests for TrainingConfig invariants.

Verifies that n_params, chrom_length, and other derived properties
stay consistent with the underlying PARAM_SPACES definitions.
"""

from __future__ import annotations

import pytest

from aerocapture.training.config import GAConfig, NetworkConfig, TrainingConfig
from aerocapture.training.param_spaces import PARAM_SPACES
from tests.fixtures.factories import make_training_config

ALL_NON_NN_SCHEMES = list(PARAM_SPACES.keys())


class TestNParamsConsistency:
    @pytest.mark.parametrize("scheme", ALL_NON_NN_SCHEMES)
    def test_n_params_matches_param_space(self, scheme: str) -> None:
        """config.n_params must equal len(PARAM_SPACES[scheme]) for non-NN schemes."""
        config = make_training_config(scheme)
        assert config.n_params == len(PARAM_SPACES[scheme]), (
            f"scheme={scheme}: config.n_params={config.n_params}, "
            f"len(PARAM_SPACES)={len(PARAM_SPACES[scheme])}"
        )

    def test_nn_n_params_uses_network_config(self) -> None:
        """For neural_network, n_params == network.n_base_coef."""
        config = make_training_config("neural_network") if "neural_network" in PARAM_SPACES else TrainingConfig()
        # Build explicitly for NN
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=[6, 12, 2]),
            guidance_type="neural_network",
        )
        # n_base_coef = (6*12 + 12) + (12*2 + 2) = 84 + 26 = 110
        expected = config.network.n_base_coef
        assert config.n_params == expected, (
            f"NN n_params={config.n_params} != n_base_coef={expected}"
        )

    @pytest.mark.parametrize("layer_sizes", [[6, 12, 2], [4, 8, 8, 2], [6, 6, 2]])
    def test_nn_n_params_varies_with_architecture(self, layer_sizes: list[int]) -> None:
        """n_params reflects the actual NN architecture."""
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=layer_sizes),
            guidance_type="neural_network",
        )
        # Manual calculation
        expected = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        assert config.n_params == expected


class TestChromLengthConsistency:
    @pytest.mark.parametrize("scheme", ALL_NON_NN_SCHEMES)
    def test_chrom_length_is_n_params_times_n_bit(self, scheme: str) -> None:
        """chrom_length == n_params * n_bit for all schemes."""
        config = make_training_config(scheme)
        expected = config.n_params * config.ga.n_bit
        assert config.chrom_length == expected, (
            f"scheme={scheme}: chrom_length={config.chrom_length}, expected={expected}"
        )

    @pytest.mark.parametrize("n_bit", [8, 12, 16, 24])
    def test_chrom_length_scales_with_n_bit(self, n_bit: int) -> None:
        """chrom_length scales linearly with n_bit."""
        config = TrainingConfig(
            ga=GAConfig(n_bit=n_bit),
            guidance_type="equilibrium_glide",
        )
        expected = len(PARAM_SPACES["equilibrium_glide"]) * n_bit
        assert config.chrom_length == expected

    def test_nn_chrom_length_is_n_base_coef_times_n_bit(self) -> None:
        """For NN, chrom_length == n_base_coef * n_bit (direct encoding)."""
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=[6, 12, 2]),
            ga=GAConfig(n_bit=16),
            guidance_type="neural_network",
        )
        expected = config.network.n_base_coef * 16
        assert config.chrom_length == expected


class TestNetworkConfigProperties:
    def test_n_base_coef_matches_layer_computation(self) -> None:
        """n_base_coef: sum of (n_in * n_out + n_out) for each layer transition."""
        net = NetworkConfig(layer_sizes=[6, 12, 2])
        # Layer 0→1: 6*12 + 12 = 84; Layer 1→2: 12*2 + 2 = 26
        assert net.n_base_coef == 110

    def test_n_coef_is_double_n_base_coef(self) -> None:
        """n_coef includes sign bits — it is 2 * n_base_coef."""
        net = NetworkConfig(layer_sizes=[6, 12, 2])
        assert net.n_coef == 2 * net.n_base_coef

    def test_input_output_properties(self) -> None:
        """n_input and n_output reflect the first and last layer sizes."""
        net = NetworkConfig(layer_sizes=[4, 8, 3])
        assert net.n_input == 4
        assert net.n_output == 3
