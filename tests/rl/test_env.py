"""VectorEnv wrapper tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.env import AerocaptureVecEnv  # noqa: E402

TOML = "configs/test/test_neural_golden.toml"


def test_reset_returns_expected_shape() -> None:
    env = AerocaptureVecEnv(TOML, n_envs=4, seed_base=3_000_000)
    obs, aux = env.reset()
    assert obs.shape == (4, env.obs_dim)
    assert obs.dtype == np.float32
    assert aux.shape == (4, 2)
    assert aux.dtype == np.float32
    env.close()


def test_step_shapes() -> None:
    env = AerocaptureVecEnv(TOML, n_envs=4, seed_base=3_000_000)
    env.reset()
    obs, reward, done, info, aux = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (4, env.obs_dim)
    assert reward.shape == (4,)
    assert done.shape == (4,)
    assert len(info) == 4
    assert aux.shape == (4, 2)
    assert aux.dtype == np.float32
    # aux[0] = energy (J/kg), aux[1] = pdyn (Pa); both should be finite
    assert np.all(np.isfinite(aux))
    env.close()


def test_done_info_contains_terminal_keys() -> None:
    env = AerocaptureVecEnv(TOML, n_envs=2, seed_base=3_000_000)
    env.reset()
    for _ in range(2000):
        _, _, done, info, _ = env.step(np.zeros(2, dtype=np.float32))
        for i, d in enumerate(done):
            if d:
                assert "final_record" in info[i]
                assert "captured" in info[i]
                assert "dv_m_s" in info[i]
                assert "terminal_observation" in info[i]
                assert "truncated" in info[i], "RL training relies on truncation vs termination distinction"
                # truncated == (ifinal == Timeout == 2); must agree with the final_record.
                ifinal = int(info[i]["final_record"][31])
                assert info[i]["truncated"] == (ifinal == 2)
                env.close()
                return
    env.close()
    pytest.fail("no episode terminated within 2000 steps")
