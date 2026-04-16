"""Phase-aware per-step reward calculator and terminal cost for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import compute_cost

# Full-input indices from build_nn_input (neural.rs).
_IDX_ECC_EXCESS = 0
_IDX_HEAT_FLUX_FRAC = 6
_IDX_HEAT_LOAD_FRAC = 7
_IDX_SMA_ERROR = 13
_IDX_BOUNCE_FLAG = 15
_IDX_PDYN_ERROR = 19


@dataclass
class StepRewardCalculator:
    """Phase-aware per-step reward from obs + aux.

    Obs indices refer to the full 23-element input vector. When a subset
    input_mask is used, the constructor builds a reverse lookup so that
    obs[:, mapped_idx] corresponds to the correct full-input index.
    """

    input_mask: list[int]
    corridor_weight: float = 0.1
    energy_rate_weight: float = 0.05
    constraint_weight: float = 0.2
    apoapsis_weight: float = 0.2
    eccentricity_weight: float = 0.1
    energy_scale: float = 1.0e6

    def __post_init__(self) -> None:
        self._rev: dict[int, int] = {v: i for i, v in enumerate(self.input_mask)}
        required = [_IDX_ECC_EXCESS, _IDX_HEAT_FLUX_FRAC, _IDX_HEAT_LOAD_FRAC, _IDX_SMA_ERROR, _IDX_BOUNCE_FLAG, _IDX_PDYN_ERROR]
        missing = [r for r in required if r not in self._rev]
        if missing:
            raise ValueError(f"input_mask missing required indices: {missing}")

    def _col(self, full_idx: int) -> int:
        return self._rev[full_idx]

    def step_reward(
        self,
        obs: npt.NDArray[np.float32],
        aux_cur: npt.NDArray[np.float32],
        aux_next: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        """Compute per-step reward for all envs. Shape: (n_envs,)."""
        n = obs.shape[0]
        reward = np.zeros(n, dtype=np.float64)

        bounce = obs[:, self._col(_IDX_BOUNCE_FLAG)]
        in_capture = bounce < 0
        in_exit = ~in_capture

        # -- Shared: constraint proximity (both phases) --
        # obs[6] = heat_flux_fraction * 2.0 - 1.0, so frac = (obs[6] + 1) / 2
        hf_frac = (obs[:, self._col(_IDX_HEAT_FLUX_FRAC)].astype(np.float64) + 1.0) / 2.0
        hl_frac = (obs[:, self._col(_IDX_HEAT_LOAD_FRAC)].astype(np.float64) + 1.0) / 2.0
        reward -= self.constraint_weight * (hf_frac**2 + hl_frac**2)

        # -- Capture phase --
        if np.any(in_capture):
            pdyn_err = obs[:, self._col(_IDX_PDYN_ERROR)].astype(np.float64)
            reward -= np.where(in_capture, self.corridor_weight * pdyn_err**2, 0.0)

            delta_e = (aux_next[:, 0] - aux_cur[:, 0]).astype(np.float64) / self.energy_scale
            reward -= np.where(in_capture, self.energy_rate_weight * np.maximum(delta_e, 0.0), 0.0)

        # -- Exit phase --
        if np.any(in_exit):
            sma_err = obs[:, self._col(_IDX_SMA_ERROR)].astype(np.float64)
            reward -= np.where(in_exit, self.apoapsis_weight * sma_err**2, 0.0)

            ecc_excess = obs[:, self._col(_IDX_ECC_EXCESS)].astype(np.float64)
            reward -= np.where(in_exit, self.eccentricity_weight * np.maximum(ecc_excess, 0.0) ** 2, 0.0)

        return reward


def compute_terminal_cost(final_record: npt.NDArray[np.float64]) -> float:
    """Per-episode cost matching evaluate.compute_cost on a single record."""
    return compute_cost(final_record.reshape(1, -1))
