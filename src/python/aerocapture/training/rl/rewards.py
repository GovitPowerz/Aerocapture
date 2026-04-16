"""Potential-based reward shaping and terminal cost for RL training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import compute_cost


@dataclass
class PBRSShaper:
    enabled: bool
    alpha: float = 1.0
    energy_scale: float = 1.0e6
    pdyn_scale: float = 1.0e3
    ref_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None

    def phi(self, energy: npt.NDArray[np.float64], pdyn: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not self.enabled or self.ref_fn is None:
            return np.zeros_like(energy)
        pdyn_ref = self.ref_fn(energy)
        p_norm = (pdyn - pdyn_ref) / self.pdyn_scale
        return -self.alpha * np.abs(p_norm)

    def step_reward(
        self,
        aux_cur: npt.NDArray[np.float32],
        aux_next: npt.NDArray[np.float32],
        gamma: float,
    ) -> npt.NDArray[np.float64]:
        """PBRS step reward from aux arrays (n_envs, 2): [energy, pdyn]."""
        if not self.enabled:
            return np.zeros(aux_cur.shape[0], dtype=np.float64)
        e_cur = aux_cur[..., 0].astype(np.float64)
        p_cur = aux_cur[..., 1].astype(np.float64)
        e_nxt = aux_next[..., 0].astype(np.float64)
        p_nxt = aux_next[..., 1].astype(np.float64)
        return gamma * self.phi(e_nxt, p_nxt) - self.phi(e_cur, p_cur)


def compute_terminal_cost(final_record: npt.NDArray[np.float64]) -> float:
    """Per-episode cost matching evaluate.compute_cost on a single record."""
    return compute_cost(final_record.reshape(1, -1))


def load_reference_pdyn(path: Path) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Load ref_trajectory.dat; return a callable: energy[J/kg] -> pdyn[Pa]."""
    if not path.exists():
        return lambda e: np.zeros_like(e)
    # 7-column format: [energy, cos_bank, pdyn, hdot, ...].
    table = np.loadtxt(path)
    energies = table[:, 0]
    pdyns = table[:, 2]
    order = np.argsort(energies)
    energies = energies[order]
    pdyns = pdyns[order]

    def interp(e: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.interp(e, energies, pdyns)

    return interp
