"""Chain consistency + describe_architecture for NetworkConfig v2 architectures."""

from __future__ import annotations

import pytest
from aerocapture.training.config import NetworkConfig, describe_architecture


def test_v2_consistent_chain_is_accepted() -> None:
    arch = [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
        {"type": "gru", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    cfg = NetworkConfig(architecture=arch, input_mask=list(range(23)))
    assert cfg.n_input == 23
    assert cfg.n_output == 2


def test_v2_dense_to_dense_input_output_mismatch_raises() -> None:
    arch = [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"},  # expects 32
    ]
    with pytest.raises(ValueError, match="chain mismatch at layer 0->1"):
        NetworkConfig(architecture=arch, input_mask=list(range(23)))


def test_v2_dense_to_gru_mismatch_raises() -> None:
    arch = [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
        {"type": "gru", "input_size": 16, "hidden_size": 32},  # expects 32
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    with pytest.raises(ValueError, match="chain mismatch"):
        NetworkConfig(architecture=arch, input_mask=list(range(23)))


def test_v2_gru_to_dense_mismatch_raises() -> None:
    """GRU's output feed to the next layer is its hidden_size, not anything else."""
    arch = [
        {"type": "dense", "input_size": 10, "output_size": 8, "activation": "tanh"},
        {"type": "gru", "input_size": 8, "hidden_size": 16},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},  # expects 16
    ]
    with pytest.raises(ValueError, match="layer 1->2.*output_size=16.*input_size=8"):
        NetworkConfig(architecture=arch, input_mask=list(range(10)))


def test_describe_architecture_v2() -> None:
    arch = [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
        {"type": "gru", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    cfg = NetworkConfig(architecture=arch, input_mask=list(range(23)))
    summary = describe_architecture(cfg)
    assert "Network architecture" in summary
    assert "layer 0: dense" in summary
    assert "layer 1: gru" in summary
    assert "layer 2: dense" in summary
    assert "23 -> 32" in summary
    assert "32 -> 32" in summary
    assert "32 -> 2" in summary
    # Dense(23->32,tanh): 23*32+32=768; Gru(32,32): 3*32*32*2 + 6*32=6336;
    # Dense(32->2,linear): 32*2+2=66. Total: 7170.
    assert "7170" in summary


def test_describe_architecture_v1_dense_only() -> None:
    cfg = NetworkConfig(
        layer_sizes=[16, 24, 2],
        activations=["tanh", "linear"],
        input_mask=list(range(16)),
    )
    summary = describe_architecture(cfg)
    assert "Network architecture" in summary
    assert "layer 0: dense" in summary
    assert "16 -> 24" in summary
    assert "24 -> 2" in summary
    assert "tanh" in summary
    assert "linear" in summary
