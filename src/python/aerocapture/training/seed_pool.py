"""Adaptive seed pool for difficulty-aware GA training.

Manages a growing pool of MC seeds scored by population-relative
difficulty. Ensures coverage across the difficulty spectrum via
eviction of redundant seeds. Fitness is aggregated as a blend
of mean cost and CVaR (Conditional Value at Risk).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_cvar(costs: npt.NDArray[np.float64], percentile: int) -> float:
    """Compute CVaR (mean of the worst p% of costs).

    Args:
        costs: 1D array of costs for one individual across seeds.
        percentile: Tail fraction (e.g. 20 = worst 20%).

    Returns:
        Mean of the worst ceil(max(1, n * p/100)) costs.
    """
    n = len(costs)
    k = max(1, int(np.ceil(n * percentile / 100)))
    sorted_costs = np.sort(costs)
    return float(np.mean(sorted_costs[-k:]))


def aggregate_fitness(
    cost_matrix: npt.NDArray[np.float64],
    alpha: float,
    cvar_percentile: int,
) -> npt.NDArray[np.float64]:
    """Aggregate per-seed costs into scalar fitness per individual.

    fitness[i] = alpha * mean(costs[i]) + (1 - alpha) * CVaR_p(costs[i])

    Args:
        cost_matrix: Shape (n_individuals, n_seeds).
        alpha: Blend weight (1.0 = pure mean, 0.0 = pure CVaR).
        cvar_percentile: Tail fraction for CVaR.

    Returns:
        1D array of scalar fitness values (n_individuals,).
    """
    means = np.mean(cost_matrix, axis=1)
    cvars = np.array([compute_cvar(row, cvar_percentile) for row in cost_matrix])
    result: npt.NDArray[np.float64] = alpha * means + (1 - alpha) * cvars
    return result
