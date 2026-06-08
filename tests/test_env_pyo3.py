"""Smoke tests for the BatchedSimulation PyO3 env class."""

from __future__ import annotations

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


TOML = "configs/test/test_neural_golden.toml"


def test_batched_simulation_construct_and_close() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4, seed_base=3_000_000)
    env.close()


def test_batched_simulation_reset_shape() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4, seed_base=3_000_000)
    obs, aux = env.reset()
    assert obs.shape == (4, 16)  # default input_mask is 16 elements
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert aux.shape == (4, 5)
    assert aux.dtype == np.float32
    assert np.isfinite(aux).all()
    env.close()


def test_reset_wrong_seed_length_raises() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4)
    env.reset()  # first reset works
    with pytest.raises(ValueError, match="seeds length"):
        env.reset(seeds=np.array([0, 1], dtype=np.int64))
    env.close()


def test_reset_default_draws_distinct_seeds() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=3_000_000)
    env.reset()
    seeds_after_first = env.current_seeds().copy()
    env.reset()
    seeds_after_second = env.current_seeds().copy()
    # episode_ids must advance by n_envs each default reset
    assert (seeds_after_second == seeds_after_first + 2).all()
    env.close()


def test_step_advances_and_returns_correct_shapes() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4, seed_base=3_000_000)
    obs, _ = env.reset()
    actions = np.zeros(4, dtype=np.float32)  # bank = 0 rad
    obs2, reward, done, info, aux = env.step(actions)
    assert obs2.shape == obs.shape
    assert reward.shape == (4,)
    assert reward.dtype == np.float32
    assert done.shape == (4,)
    assert done.dtype == np.bool_
    assert isinstance(info, list)
    assert len(info) == 4
    assert np.isfinite(obs2).all()
    assert np.isfinite(reward).all()
    assert aux.shape == (4, 5)
    assert np.isfinite(aux).all()
    env.close()


def test_step_eventually_terminates() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=3_000_000)
    env.reset()
    dones_seen = np.zeros(2, dtype=np.bool_)
    for _ in range(2000):
        _, _, done, _, _ = env.step(np.zeros(2, dtype=np.float32))
        dones_seen |= done
        if dones_seen.all():
            break
    assert dones_seen.all(), "both envs should have terminated at least once"
    env.close()


def test_step_seed_determinism() -> None:
    """Two envs constructed with the same seed_base produce identical first-obs + first-step results."""
    env_a = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=7_777_777)
    env_b = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=7_777_777)
    obs_a, aux_a = env_a.reset()
    obs_b, aux_b = env_b.reset()
    assert np.allclose(obs_a, obs_b, atol=0.0)
    assert np.allclose(aux_a, aux_b, atol=0.0)

    act = np.full(2, 0.25, dtype=np.float32)
    o_a, r_a, d_a, _, a_a = env_a.step(act)
    o_b, r_b, d_b, _, a_b = env_b.step(act)
    assert np.allclose(o_a, o_b, atol=0.0)
    assert np.allclose(r_a, r_b, atol=0.0)
    assert np.array_equal(d_a, d_b)
    assert np.allclose(a_a, a_b, atol=0.0)
    env_a.close()
    env_b.close()


def test_step_action_clipping() -> None:
    """Actions outside [-pi, pi] must be clipped and still produce finite obs."""
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=3_000_000)
    env.reset()
    obs, _, _, _, aux = env.step(np.array([10.0, -10.0], dtype=np.float32))
    assert np.isfinite(obs).all()
    assert np.isfinite(aux).all()
    env.close()


def test_terminal_observation_in_info() -> None:
    """On done, info must contain 'terminal_observation' matching the pre-reset obs_dim."""
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=1, seed_base=3_000_000)
    env.reset()
    for _ in range(2000):
        obs, _, done, info, _ = env.step(np.zeros(1, dtype=np.float32))
        if done[0]:
            assert "terminal_observation" in info[0]
            t = info[0]["terminal_observation"]
            assert len(t) == env.obs_dim
            return
    pytest.fail("env did not terminate within 2000 steps")


def test_aux_carries_dv_components() -> None:
    """Aux columns 2-4 are the raw predicted-DV correction budget (finite, live)."""
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=4, seed_base=3_000_000)
    _, aux = env.reset()
    assert aux.shape == (4, 5)
    assert np.isfinite(aux).all()
    seen_nonzero = np.zeros(3, dtype=bool)
    for _ in range(50):
        _, _, _, _, aux = env.step(np.zeros(4, dtype=np.float32))
        assert aux.shape == (4, 5)
        assert np.isfinite(aux).all()
        seen_nonzero |= np.abs(aux[:, 2:5]).max(axis=0) > 0.0
    assert seen_nonzero.any(), "predicted-DV aux columns never became nonzero"
    env.close()
