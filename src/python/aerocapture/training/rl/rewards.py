"""Potential-based per-step shaping and terminal cost for RL training.

Per Ng, Harada, Russell (1999), shaping of the form `F = gamma * Phi(s') - Phi(s)`
leaves the set of optimal policies unchanged. The potential `Phi` is phase-aware:
capture phase (pre-bounce) penalizes corridor + constraint proximity + energy;
exit phase (post-bounce) penalizes apoapsis and eccentricity errors.

The raw (non-shaped) signal is the terminal cost applied once at episode end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import compute_cost

_IDX_ECC_EXCESS = 0
_IDX_HEAT_FLUX_FRAC = 6
_IDX_HEAT_LOAD_FRAC = 7
_IDX_SMA_ERROR = 13
_IDX_BOUNCE_FLAG = 15
_IDX_PDYN_ERROR = 19


@dataclass
class StepRewardCalculator:
    """Potential-based shaping built from phase-aware obs components.

    `Phi(obs, aux)` is the (negative) potential; lower is worse. The per-step
    shaped reward is `gamma * Phi(next) - Phi(cur)`, so the return telescopes
    to `gamma^T * Phi(terminal) - Phi(initial)` which is a fixed offset for
    any policy -- the optimum is preserved.
    """

    input_mask: list[int]
    gamma: float = 0.99
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

    def _potential(
        self,
        obs: npt.NDArray[np.float32],
        aux: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        """Negative potential Phi(obs, aux), shape (n_envs,)."""
        bounce = obs[:, self._col(_IDX_BOUNCE_FLAG)]
        in_capture = bounce < 0
        in_exit = ~in_capture

        hf_frac = (obs[:, self._col(_IDX_HEAT_FLUX_FRAC)].astype(np.float64) + 1.0) / 2.0
        hl_frac = (obs[:, self._col(_IDX_HEAT_LOAD_FRAC)].astype(np.float64) + 1.0) / 2.0
        phi = -self.constraint_weight * (hf_frac**2 + hl_frac**2)

        pdyn_err = obs[:, self._col(_IDX_PDYN_ERROR)].astype(np.float64)
        energy = aux[:, 0].astype(np.float64) / self.energy_scale
        phi_capture = -(self.corridor_weight * pdyn_err**2 + self.energy_rate_weight * np.maximum(energy, 0.0))

        sma_err = obs[:, self._col(_IDX_SMA_ERROR)].astype(np.float64)
        ecc_excess = obs[:, self._col(_IDX_ECC_EXCESS)].astype(np.float64)
        phi_exit = -(self.apoapsis_weight * sma_err**2 + self.eccentricity_weight * np.maximum(ecc_excess, 0.0) ** 2)

        phi += np.where(in_capture, phi_capture, 0.0)
        phi += np.where(in_exit, phi_exit, 0.0)
        return phi

    def step_reward(
        self,
        obs_cur: npt.NDArray[np.float32],
        obs_next: npt.NDArray[np.float32],
        aux_cur: npt.NDArray[np.float32],
        aux_next: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        """PBRS shaped reward: gamma * Phi(next) - Phi(cur), shape (n_envs,)."""
        phi_cur = self._potential(obs_cur, aux_cur)
        phi_next = self._potential(obs_next, aux_next)
        return self.gamma * phi_next - phi_cur


def compute_terminal_cost(final_record: npt.NDArray[np.float64]) -> float:
    """Per-episode cost matching evaluate.compute_cost on a single record."""
    return compute_cost(final_record.reshape(1, -1))
