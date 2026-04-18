"""RolloutBuffer v2 hidden-state fields."""

from __future__ import annotations

import numpy as np
from aerocapture.training.rl.ppo import RolloutBuffer


def test_rollout_buffer_dense_only_has_empty_hidden_state() -> None:
    """Dense-only policy: hidden-state fields are populated with None sentinels,
    one per layer. Zero memory overhead."""
    buf = RolloutBuffer.create(n_steps=8, n_envs=2, obs_dim=4, hidden_shapes=[None, None])
    assert buf.h_initial == [None, None]
    assert buf.h_final == [None, None]
    assert buf.states == [None, None]


def test_rollout_buffer_default_hidden_shapes_omitted_is_backward_compatible() -> None:
    """Omitting hidden_shapes entirely: all three fields are empty lists.

    This is the backward-compat path -- every pre-Phase-1.5 caller constructs
    without the hidden_shapes kwarg and must continue to work.
    """
    buf = RolloutBuffer.create(n_steps=4, n_envs=2, obs_dim=4)
    assert buf.h_initial == []
    assert buf.h_final == []
    assert buf.states == []


def test_rollout_buffer_gru_has_time_axis_state_storage() -> None:
    """Dense->GRU->Dense: layer 1 gets (n_steps, n_envs, H) state storage; layers 0 and 2 stay None."""
    buf = RolloutBuffer.create(n_steps=8, n_envs=2, obs_dim=4, hidden_shapes=[None, (8,), None])
    assert buf.h_initial[0] is None
    h_init_1 = buf.h_initial[1]
    assert h_init_1 is not None
    assert h_init_1.shape == (2, 8)
    assert buf.h_initial[2] is None
    states_1 = buf.states[1]
    assert states_1 is not None
    assert states_1.shape == (8, 2, 8)
    h_final_1 = buf.h_final[1]
    assert h_final_1 is not None
    assert h_final_1.shape == (2, 8)


def test_rollout_buffer_write_and_read_state_roundtrip() -> None:
    buf = RolloutBuffer.create(n_steps=4, n_envs=3, obs_dim=2, hidden_shapes=[(5,)])
    h = np.random.randn(3, 5).astype(np.float32)
    states_0 = buf.states[0]
    assert states_0 is not None
    states_0[1] = h
    assert np.array_equal(states_0[1], h)
