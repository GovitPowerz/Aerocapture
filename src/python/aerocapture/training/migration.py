"""Subpopulation migration for island-model GA.

Replaces MATLAB Migration_Aerocap.m.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.local_search import improve_chromosome


def migrate(
    populations: list[npt.NDArray[np.int8]],
    costs: list[npt.NDArray[np.float64]],
    generation: int,
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[npt.NDArray[np.int8]], list[npt.NDArray[np.float64]]]:
    """Perform ring-topology migration between subpopulations.

    Best individual from each subpopulation replaces worst in previous.
    Triggered every migration_interval generations.

    Args:
        populations: List of population arrays, one per subpopulation.
        costs: List of cost arrays, one per subpopulation.
        generation: Current generation number.
        base_network: Base network weights.
        config: Training configuration.
        cwd: Working directory.
        rng: Random number generator.

    Returns:
        Updated (populations, costs) tuple.
    """
    if generation % config.ga.migration_interval != 0:
        return populations, costs

    n_subpop = len(populations)
    if n_subpop <= 1:
        return populations, costs

    if rng is None:
        rng = np.random.default_rng()

    # Ring migration: best of subpop i+1 -> worst of subpop i
    for i in range(n_subpop - 1):
        # Best from next subpopulation
        best_idx = np.argmin(costs[i + 1])
        # Worst in current subpopulation
        worst_idx = np.argmax(costs[i])

        populations[i][worst_idx] = populations[i + 1][best_idx].copy()
        costs[i][worst_idx] = costs[i + 1][best_idx]

    # Wrap: best of first -> worst of last
    best_idx = np.argmin(costs[0])
    worst_idx = np.argmax(costs[-1])
    populations[-1][worst_idx] = populations[0][best_idx].copy()
    costs[-1][worst_idx] = costs[0][best_idx]

    # Post-migration local improvement on best of each subpopulation
    for k in range(n_subpop):
        best_idx = np.argmin(costs[k])
        improved, improved_cost, _ = improve_chromosome(
            populations[k][best_idx], base_network, config,
            mode=0, cwd=cwd, rng=rng,
        )
        # Place improved at second-worst position
        order = np.argsort(costs[k])
        second_worst = order[-2]
        populations[k][second_worst] = improved
        costs[k][second_worst] = improved_cost

    return populations, costs
