"""_policy_to_flat_weights_v2 dispatches per layer type and matches per-layer to_flat."""

import numpy as np
import torch
from pydantic import TypeAdapter

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import LayerSpec
from aerocapture.training.warm_start import _policy_to_flat_weights_v2


def _build(arch):
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    return V2Policy(architecture=validated, input_mask=None).double()


def test_dense_only_matches_concat_of_to_flat():
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build(arch)
    with torch.no_grad():
        for p in policy.parameters():
            p.uniform_(-0.1, 0.1)
    expected = np.concatenate([layer.to_flat() for layer in policy.layers])
    actual = _policy_to_flat_weights_v2(policy, arch)
    assert np.allclose(actual, expected, atol=0.0)  # bitwise equal


def test_mixed_arch_dense_gru_dense():
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "gru", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build(arch)
    with torch.no_grad():
        for p in policy.parameters():
            p.uniform_(-0.1, 0.1)
    expected = np.concatenate([layer.to_flat() for layer in policy.layers])
    actual = _policy_to_flat_weights_v2(policy, arch)
    assert np.allclose(actual, expected, atol=0.0)
