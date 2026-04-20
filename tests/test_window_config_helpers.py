"""Phase 2b config.py + encoding.py + initialization_v2.py Window arms.

Covers: _layer_n_params window arm (zero), _layer_output_size window arm
(n_steps * input_size), describe_architecture window rendering,
_layer_param_specs returns [], init_v2_population handles mixed Window+Dense.
"""

from __future__ import annotations

import numpy as np

from aerocapture.training.config import (
    NetworkConfig,
    _layer_n_params,
    _layer_output_size,
    describe_architecture,
)
from aerocapture.training.encoding import _layer_param_specs, nn_param_specs_from_v2
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.rl.schemas import DenseSpec, WindowSpec


def test_layer_n_params_window_returns_zero() -> None:
    entry = {"type": "window", "input_size": 16, "n_steps": 8}
    assert _layer_n_params(entry) == 0


def test_layer_output_size_window_is_input_times_n_steps() -> None:
    entry = {"type": "window", "input_size": 16, "n_steps": 8}
    assert _layer_output_size(entry) == 128

    entry = {"type": "window", "input_size": 4, "n_steps": 3}
    assert _layer_output_size(entry) == 12


def test_describe_architecture_renders_window_layer() -> None:
    architecture = [
        {"type": "window", "input_size": 16, "n_steps": 8},
        {"type": "dense", "input_size": 128, "output_size": 32, "activation": "swish"},
    ]
    # NetworkConfig's n_base_coef / n_input / n_output are properties; pass
    # architecture in a v2 context where layer_sizes/activations are ignored.
    net = NetworkConfig(
        layer_sizes=[16, 32],  # kept non-empty for dataclass __post_init__ path
        activations=["tanh"],
        input_mask=None,
        architecture=architecture,
    )
    s = describe_architecture(net, output_interpretation="atan2")
    assert "window" in s
    assert "n_steps=8" in s
    assert "128" in s  # output = n_steps * input_size


def test_layer_param_specs_window_returns_empty() -> None:
    spec = WindowSpec(type="window", input_size=16, n_steps=8)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=2.0)
    assert specs == []


def test_nn_param_specs_from_v2_handles_mixed_window_dense() -> None:
    architecture = [
        WindowSpec(type="window", input_size=4, n_steps=3),
        DenseSpec(type="dense", input_size=12, output_size=2, activation="linear"),
    ]
    specs = nn_param_specs_from_v2(architecture, bound_multiplier=2.0)
    # Only Dense contributes: 12*2 weights + 2 biases = 26 specs.
    assert len(specs) == 26


def test_init_v2_population_skips_window_layer_with_zero_params() -> None:
    architecture = [
        {"type": "window", "input_size": 4, "n_steps": 3},
        {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture, n_pop=8, bound_multiplier=2.0, rng=rng)

    # Total params = 0 (window) + 12*2 + 2 (dense) = 26.
    assert pop.shape == (8, 26)
    assert np.all(np.isfinite(pop))


def test_init_v2_population_window_first_arch_matches_training_config_shape() -> None:
    # Mirrors the shipping training config arch:
    # Window(16, 8) -> Dense(128->32,swish) -> Dense(32->8,swish) -> Dense(8->2,linear).
    architecture = [
        {"type": "window", "input_size": 16, "n_steps": 8},
        {"type": "dense", "input_size": 128, "output_size": 32, "activation": "swish"},
        {"type": "dense", "input_size": 32, "output_size": 8, "activation": "swish"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(7)
    pop = init_v2_population(architecture, n_pop=4, bound_multiplier=2.0, rng=rng)
    # 128*32 + 32 + 32*8 + 8 + 8*2 + 2 = 4410.
    assert pop.shape == (4, 4410)
    assert np.all(np.isfinite(pop))
