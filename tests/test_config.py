"""Tests for TrainingConfig invariants.

Verifies that n_params and other derived properties
stay consistent with the underlying PARAM_SPACES definitions.
"""

from __future__ import annotations

import pytest
from aerocapture.training.config import NetworkConfig, TrainingConfig
from aerocapture.training.param_spaces import PARAM_SPACES

from tests.fixtures.factories import make_training_config

ALL_NON_NN_SCHEMES = list(PARAM_SPACES.keys())


class TestNParamsConsistency:
    @pytest.mark.parametrize("scheme", ALL_NON_NN_SCHEMES)
    def test_n_params_matches_param_space(self, scheme: str) -> None:
        """config.n_params must equal len(PARAM_SPACES[scheme]) for non-NN schemes."""
        config = make_training_config(scheme)
        assert config.n_params == len(PARAM_SPACES[scheme]), (
            f"scheme={scheme}: config.n_params={config.n_params}, len(PARAM_SPACES)={len(PARAM_SPACES[scheme])}"
        )

    def test_nn_n_params_uses_network_config(self) -> None:
        """For neural_network, n_params == network.n_base_coef."""
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=[16, 24, 2]),
            guidance_type="neural_network",
        )
        expected = config.network.n_base_coef
        assert config.n_params == expected, f"NN n_params={config.n_params} != n_base_coef={expected}"

    @pytest.mark.parametrize(
        "layer_sizes,activations",
        [
            ([6, 12, 2], ["tanh", "asinh"]),
            ([6, 8, 8, 2], ["tanh", "tanh", "asinh"]),
            ([6, 6, 2], ["tanh", "asinh"]),
        ],
    )
    def test_nn_n_params_varies_with_architecture(self, layer_sizes: list[int], activations: list[str]) -> None:
        """n_params reflects the actual NN architecture."""
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=layer_sizes, activations=activations),
            guidance_type="neural_network",
        )
        # Manual calculation
        expected = sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
        assert config.n_params == expected


class TestNetworkConfigProperties:
    def test_n_base_coef_matches_layer_computation(self) -> None:
        """n_base_coef: sum of (n_in * n_out + n_out) for each layer transition."""
        net = NetworkConfig(layer_sizes=[16, 24, 2])
        # Layer 0->1: 16*24 + 24 = 408; Layer 1->2: 24*2 + 2 = 50
        assert net.n_base_coef == 458

    def test_n_coef_is_double_n_base_coef(self) -> None:
        """n_coef includes sign bits -- it is 2 * n_base_coef."""
        net = NetworkConfig(layer_sizes=[16, 24, 2])
        assert net.n_coef == 2 * net.n_base_coef

    def test_input_output_properties(self) -> None:
        """n_input and n_output reflect the first and last layer sizes."""
        net = NetworkConfig(layer_sizes=[4, 8, 3])
        assert net.n_input == 4
        assert net.n_output == 3
