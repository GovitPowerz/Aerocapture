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
