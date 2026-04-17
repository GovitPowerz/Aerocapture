"""Thin vectorized env wrapper over BatchedSimulation.

API resembles Gymnasium's VecEnv contract (reset returns obs only, step
returns (obs, reward, done, info)) but does not depend on gymnasium --
the RL training loop consumes this object directly.
"""

from __future__ import annotations

from typing import Any

import aerocapture_rs
import numpy as np
import numpy.typing as npt


class AerocaptureVecEnv:
    def __init__(
        self,
        toml_path: str,
        n_envs: int,
        overrides: dict[str, Any] | None = None,
        seed_base: int = 3_000_000,
    ) -> None:
        self._env = aerocapture_rs.BatchedSimulation(
            toml_path,
            n_envs=n_envs,
            overrides=overrides,
            seed_base=seed_base,
        )
        self.n_envs = n_envs
        self.obs_dim = int(self._env.obs_dim)

    def reset(self, seeds: npt.NDArray[np.int64] | None = None) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        obs, aux = self._env.reset(seeds)
        return np.asarray(obs, dtype=np.float32), np.asarray(aux, dtype=np.float32)

    def step(
        self, actions: npt.NDArray[np.float32]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        list[dict[str, Any]],
        npt.NDArray[np.float32],
    ]:
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        obs, reward, done, info, aux = self._env.step(actions)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(done, dtype=np.bool_),
            info,
            np.asarray(aux, dtype=np.float32),
        )

    def close(self) -> None:
        self._env.close()
