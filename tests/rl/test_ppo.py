"""Unit tests for PPO update internals."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.policy import GaussianPolicy, ValueNetwork  # noqa: E402
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, ppo_update  # noqa: E402


def test_gae_known_values() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (3,)
    assert np.isfinite(adv).all()
    assert np.isfinite(ret).all()


def test_rollout_buffer_create() -> None:
    buf = RolloutBuffer.create(n_steps=8, n_envs=4, obs_dim=16)
    assert buf.obs.shape == (8, 4, 16)
    assert buf.actions.shape == (8, 4)
    assert buf.log_probs.shape == (8, 4)
    assert buf.rewards.shape == (8, 4)
    assert buf.values.shape == (8, 4)
    assert buf.dones.shape == (8, 4)


def test_ppo_update_runs_without_crashing() -> None:
    torch.manual_seed(0)
    policy = GaussianPolicy(16, [32, 32, 2], ["tanh", "tanh", "linear"])
    value = ValueNetwork(16, [32, 32], ["tanh", "tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)

    n = 256
    obs = torch.randn(n, 16)
    actions = torch.rand(n) * (2 * torch.pi) - torch.pi
    old_log_probs = torch.randn(n) * 0.1
    advantages = torch.randn(n)
    returns = torch.randn(n)

    metrics = ppo_update(
        policy,
        value,
        optim,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        clip_range=0.2,
        update_epochs=2,
        minibatches=4,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert "clip_frac" in metrics
