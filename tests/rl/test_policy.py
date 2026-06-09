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
        mlp_asinh_only[0].weight.fill_(1.0)  # type: ignore[operator]
        mlp_asinh_only[0].bias.fill_(0.0)  # type: ignore[operator]
    result = mlp_asinh_only(_torch.tensor([[1.0]])).item()
    assert abs(result - math.asinh(1.0)) < 1e-6


def test_gaussian_policy_deterministic_bank_angle() -> None:
    policy = GaussianPolicy(input_dim=16, layer_sizes=[64, 64, 2], activations=["tanh", "tanh", "linear"])
    obs = torch.randn(4, 16)
    bank = policy.deterministic_bank(obs)
    assert bank.shape == (4,)
    assert torch.all(bank >= -torch.pi)
    assert torch.all(bank <= torch.pi)


def test_log_std_clamped_to_max_and_min() -> None:
    """log_std clamps to [min_log_std, max_log_std]. The ceiling prevents the
    entropy-bonus runaway seen in the last PPO run (log_std drifted to ~+2.9,
    action std ~18). V2Policy shares the identical clamp (covered by the PPO smoke).
    """
    policy = GaussianPolicy(input_dim=16, layer_sizes=[8, 2], activations=["tanh", "linear"], min_log_std=-2.0, max_log_std=0.0)
    with torch.no_grad():
        policy.log_std.copy_(torch.tensor([5.0, -9.0]))  # above ceiling, below floor
    _, log_std = policy.forward_mean_logstd(torch.randn(3, 16))
    log_std = log_std.detach()
    assert float(log_std[0]) == pytest.approx(0.0)  # 5.0 -> max 0.0
    assert float(log_std[1]) == pytest.approx(-2.0)  # -9.0 -> min -2.0
