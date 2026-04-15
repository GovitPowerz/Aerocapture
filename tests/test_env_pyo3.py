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
