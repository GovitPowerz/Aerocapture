"""SAC structural unit test."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.sac import SACAgent  # noqa: E402


def test_sac_update_runs() -> None:
    agent = SACAgent(
        obs_dim=16,
        layer_sizes=[32, 32, 2],
        activations=["tanh", "tanh", "linear"],
    )
    obs = torch.randn(256, 16)
    actions = torch.rand(256) * (2 * torch.pi) - torch.pi
    rewards = torch.randn(256)
    next_obs = torch.randn(256, 16)
    dones = torch.zeros(256, dtype=torch.bool)
    metrics = agent.update(obs, actions, rewards, next_obs, dones)
    assert "q_loss" in metrics
    assert "policy_loss" in metrics
    assert "alpha" in metrics
