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
    obs = env.reset()
    assert obs.shape == (4, 16)  # default input_mask is 16 elements
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
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
    obs = env.reset()
    actions = np.zeros(4, dtype=np.float32)  # bank = 0 rad
    obs2, reward, done, info = env.step(actions)
    assert obs2.shape == obs.shape
    assert reward.shape == (4,)
    assert reward.dtype == np.float32
    assert done.shape == (4,)
    assert done.dtype == np.bool_
    assert isinstance(info, list)
    assert len(info) == 4
    assert np.isfinite(obs2).all()
    assert np.isfinite(reward).all()
    env.close()


def test_step_eventually_terminates() -> None:
    env = aerocapture_rs.BatchedSimulation(TOML, n_envs=2, seed_base=3_000_000)
    env.reset()
    dones_seen = np.zeros(2, dtype=np.bool_)
    for _ in range(2000):
        _, _, done, _ = env.step(np.zeros(2, dtype=np.float32))
        dones_seen |= done
        if dones_seen.all():
            break
    assert dones_seen.all(), "both envs should have terminated at least once"
    env.close()
