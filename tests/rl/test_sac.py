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
    raw_actions = torch.randn(256, 2)
    rewards = torch.randn(256)
    next_obs = torch.randn(256, 16)
    dones = torch.zeros(256, dtype=torch.bool)
    metrics = agent.update(obs, raw_actions, rewards, next_obs, dones)
    assert "q_loss" in metrics
    assert "policy_loss" in metrics
    assert "alpha" in metrics


def test_sac_replay_buffer_roundtrip() -> None:
    import numpy as np
    from aerocapture.training.rl.sac import ReplayBuffer

    buf = ReplayBuffer(capacity=128, obs_dim=8)
    obs = np.random.randn(4, 8).astype(np.float32)
    raw = np.random.randn(4, 2).astype(np.float32)
    rewards = np.random.randn(4).astype(np.float32)
    next_obs = np.random.randn(4, 8).astype(np.float32)
    dones = np.zeros(4, dtype=bool)
    buf.push(obs, raw, rewards, next_obs, dones)

    clone = ReplayBuffer(capacity=128, obs_dim=8)
    clone.load_state_dict(buf.state_dict())
    assert len(clone) == 4
