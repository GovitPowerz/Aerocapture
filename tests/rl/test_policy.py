"""Policy network tests."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.policy import GaussianPolicy, build_mlp  # noqa: E402


def test_gaussian_policy_deterministic_shape() -> None:
    policy = GaussianPolicy(input_dim=16, layer_sizes=[64, 64, 2], activations=["tanh", "tanh", "linear"])
    obs = torch.randn(4, 16)
    mean, log_std = policy.forward_mean_logstd(obs)
    assert mean.shape == (4, 2)
    assert log_std.shape == (2,)


def test_build_mlp_asinh_activation_no_keyerror() -> None:
    """build_mlp must accept 'asinh' without KeyError (N2 regression guard)."""
    mlp = build_mlp(input_dim=4, layer_sizes=[8, 1], activations=["tanh", "asinh"])
    x = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    out = mlp(x)
    assert out.shape == (1, 1)
    # asinh(0) == 0; verify the activation is applied correctly via a known value
    import torch as _torch

    mlp_asinh_only = build_mlp(input_dim=1, layer_sizes=[1], activations=["asinh"])
    # With identity weights the output equals asinh(input).
    # Set weight=1 bias=0 explicitly so the check is deterministic.
    with _torch.no_grad():
        mlp_asinh_only[0].weight.fill_(1.0)
        mlp_asinh_only[0].bias.fill_(0.0)
    result = mlp_asinh_only(_torch.tensor([[1.0]])).item()
    assert abs(result - math.asinh(1.0)) < 1e-6


def test_gaussian_policy_deterministic_bank_angle() -> None:
    policy = GaussianPolicy(input_dim=16, layer_sizes=[64, 64, 2], activations=["tanh", "tanh", "linear"])
    obs = torch.randn(4, 16)
    bank = policy.deterministic_bank(obs)
    assert bank.shape == (4,)
    assert torch.all(bank >= -torch.pi)
    assert torch.all(bank <= torch.pi)
