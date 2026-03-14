"""Adaptive seed pool for difficulty-aware GA training.

Manages a growing pool of MC seeds scored by population-relative
difficulty. Ensures coverage across the difficulty spectrum via
eviction of redundant seeds. Fitness is aggregated as a blend
of mean cost and CVaR (Conditional Value at Risk).
"""

from __future__ import annotations

from typing import Any

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


class SeedPool:
    """Adaptive MC seed pool with difficulty-based eviction."""

    def __init__(
        self,
        base_seed: int,
        max_size: int = 100,
        alpha: float = 0.7,
        cvar_percentile: int = 20,
    ) -> None:
        self.base_seed = base_seed
        self.max_size = max_size
        self.alpha = alpha
        self.cvar_percentile = cvar_percentile
        self.seeds: list[int] = []
        self.difficulty: dict[int, float] = {}
        self.generation_added: dict[int, int] = {}
        self.n_evictions: int = 0

    def add_seeds(self, generation: int) -> None:
        """Add seeds to the pool for a given generation.

        Generation 0 bootstraps 5 seeds; subsequent generations add 1 per call.
        Duplicate seeds are silently ignored.
        """
        new_seeds = [self.base_seed + i for i in range(5)] if generation == 0 else [self.base_seed + generation + 4]
        for seed in new_seeds:
            if seed not in self.generation_added:
                self.seeds.append(seed)
                self.generation_added[seed] = generation

    def score_difficulty(self, cost_matrix: npt.NDArray[np.float64], best_idx: int) -> None:
        """Update per-seed difficulty scores using the best individual's costs.

        Difficulty is defined as the best individual's cost on each seed —
        high cost means the seed is hard even for the best solution found.

        Args:
            cost_matrix: Shape (n_individuals, n_seeds), costs per individual per seed.
            best_idx: Row index of the best individual in cost_matrix.
        """
        best_costs = cost_matrix[best_idx]
        for i, seed in enumerate(self.seeds):
            self.difficulty[seed] = float(best_costs[i])

    def evict_redundant(self) -> None:
        """Evict seeds until pool size <= max_size.

        Eviction strategy: find the adjacent pair (when sorted by difficulty)
        with the smallest gap, and remove the older of the two. This preserves
        coverage across the difficulty spectrum while removing redundant seeds.
        """
        while len(self.seeds) > self.max_size:
            scored = sorted(self.seeds, key=lambda s: self.difficulty.get(s, 0.0))
            min_gap = float("inf")
            evict_candidate = scored[0]
            for i in range(len(scored) - 1):
                gap = abs(self.difficulty.get(scored[i + 1], 0.0) - self.difficulty.get(scored[i], 0.0))
                if gap < min_gap:
                    min_gap = gap
                    a, b = scored[i], scored[i + 1]
                    evict_candidate = a if self.generation_added.get(a, 0) <= self.generation_added.get(b, 0) else b
            self.seeds.remove(evict_candidate)
            del self.difficulty[evict_candidate]
            del self.generation_added[evict_candidate]
            self.n_evictions += 1

    @property
    def difficulty_range(self) -> tuple[float, float]:
        """Return (min, max) of current difficulty scores."""
        if not self.difficulty:
            return (0.0, 0.0)
        vals = list(self.difficulty.values())
        return (min(vals), max(vals))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pool to a JSON-compatible dict for checkpointing."""
        return {
            "base_seed": self.base_seed,
            "max_size": self.max_size,
            "alpha": self.alpha,
            "cvar_percentile": self.cvar_percentile,
            "seeds": self.seeds,
            "difficulty": {str(k): v for k, v in self.difficulty.items()},
            "generation_added": {str(k): v for k, v in self.generation_added.items()},
            "n_evictions": self.n_evictions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeedPool:
        """Restore a SeedPool from a checkpointed dict."""
        pool = cls(
            base_seed=int(data["base_seed"]),
            max_size=int(data["max_size"]),
            alpha=float(data.get("alpha", 0.7)),
            cvar_percentile=int(data.get("cvar_percentile", 20)),
        )
        pool.seeds = list(data["seeds"])
        pool.difficulty = {int(k): float(v) for k, v in dict(data["difficulty"]).items()}
        pool.generation_added = {int(k): int(v) for k, v in dict(data["generation_added"]).items()}
        pool.n_evictions = int(data.get("n_evictions", 0))
        return pool
