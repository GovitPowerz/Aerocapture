"""Policy network tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.policy import GaussianPolicy  # noqa: E402


def test_gaussian_policy_deterministic_shape() -> None:
    policy = GaussianPolicy(input_dim=16, layer_sizes=[64, 64, 2], activations=["tanh", "tanh", "linear"])
    obs = torch.randn(4, 16)
    mean, log_std = policy.forward_mean_logstd(obs)
    assert mean.shape == (4, 2)
    assert log_std.shape == (2,)


def test_gaussian_policy_deterministic_bank_angle() -> None:
    policy = GaussianPolicy(input_dim=16, layer_sizes=[64, 64, 2], activations=["tanh", "tanh", "linear"])
    obs = torch.randn(4, 16)
    bank = policy.deterministic_bank(obs)
    assert bank.shape == (4,)
    assert torch.all(bank >= -torch.pi)
    assert torch.all(bank <= torch.pi)
