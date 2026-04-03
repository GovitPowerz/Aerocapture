"""Adaptive seed pool for difficulty-aware GA training.

Manages a growing pool of MC seeds scored by population-relative
difficulty. Ensures coverage across the difficulty spectrum via
eviction of redundant seeds. Fitness is aggregated as a blend
of mean cost and CVaR (Conditional Value at Risk).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def _pool_seed(base: int, index: int) -> int:
    """Generate a well-spread seed from (base, index) using SHA-256."""
    h = hashlib.sha256(f"{base}:pool:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _stress_seed(base: int, generation: int, index: int) -> int:
    """Generate a stress-test seed from a separate hash namespace."""
    h = hashlib.sha256(f"{base}:stress:{generation}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


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
        excluded_seeds: set[int] | None = None,
    ) -> None:
        self.base_seed = base_seed
        self.max_size = max_size
        self.alpha = alpha
        self.cvar_percentile = cvar_percentile
        self.excluded_seeds: set[int] = excluded_seeds or set()
        self.seeds: list[int] = []
        self.difficulty: dict[int, float] = {}
        self.generation_added: dict[int, int] = {}
        self.n_evictions: int = 0
        self._next_index: int = 0

    def add_seeds(self, generation: int) -> None:
        """Add seeds to the pool for a given generation.

        Generation 0 bootstraps 5 seeds; subsequent generations add 1 per call.
        Seeds are generated via hash-based spread. Excluded seeds are skipped.
        """
        n_to_add = 5 if generation == 0 and self._next_index == 0 else 1
        added = 0
        while added < n_to_add:
            seed = _pool_seed(self.base_seed, self._next_index)
            self._next_index += 1
            if seed in self.excluded_seeds or seed in self.generation_added:
                continue
            self.seeds.append(seed)
            self.generation_added[seed] = generation
            added += 1

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

        Keep-hardest strategy: drop the seed with the lowest difficulty
        score (easiest for the best individual). Hard seeds survive
        unconditionally.
        """
        while len(self.seeds) > self.max_size:
            easiest = min(self.seeds, key=lambda s: self.difficulty.get(s, 0.0))
            self.seeds.remove(easiest)
            del self.difficulty[easiest]
            del self.generation_added[easiest]
            self.n_evictions += 1

    def evaluate_population(
        self,
        population: npt.NDArray[np.int8],
        evaluator: Callable[[npt.NDArray[np.int8], int], float],
        batch_evaluator: Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate all individuals on all pool seeds.

        If batch_evaluator is provided, uses it for per-individual batched
        evaluation. Falls back to the scalar evaluator otherwise.

        Args:
            population: Shape (n_pop, chrom_length).
            evaluator: Callable(chromosome, mc_seed) -> cost (scalar fallback).
            batch_evaluator: Callable(chromosome, seeds) -> costs array (n_seeds,).

        Returns:
            1D fitness array (n_pop,) with aggregated fitness values.
        """
        n_pop = len(population)
        n_seeds = len(self.seeds)
        cost_matrix = np.full((n_pop, n_seeds), np.inf)

        if batch_evaluator is not None:
            for i in range(n_pop):
                cost_matrix[i] = batch_evaluator(population[i], self.seeds)
        else:
            for i in range(n_pop):
                for j, seed in enumerate(self.seeds):
                    cost_matrix[i, j] = evaluator(population[i], seed)

        fitness = aggregate_fitness(cost_matrix, self.alpha, self.cvar_percentile)
        best_idx = int(np.argmin(fitness))
        self.score_difficulty(cost_matrix, best_idx)

        return fitness

    def stress_test(
        self,
        generation: int,
        evaluator: Callable[[list[int]], npt.NDArray[np.float64]],
        n_probes: int = 200,
        n_inject: int = 20,
    ) -> dict[str, Any]:
        """Probe fresh seeds and inject the hardest into the pool.

        Generates n_probes seeds from a separate hash namespace, evaluates
        the best individual on all of them, and injects the worst n_inject
        into the pool. Returns metrics including an estimated capture rate.
        """
        probe_seeds = []
        idx = 0
        while len(probe_seeds) < n_probes:
            s = _stress_seed(self.base_seed, generation, idx)
            idx += 1
            if s not in self.excluded_seeds and s not in self.generation_added:
                probe_seeds.append(s)

        costs = evaluator(probe_seeds)

        capture_rate = float(np.mean(costs < 10000.0))

        worst_indices = np.argsort(costs)[::-1][:n_inject]
        n_injected = 0
        for i in worst_indices:
            s = probe_seeds[i]
            if s not in self.generation_added:
                self.seeds.append(s)
                self.difficulty[s] = float(costs[i])
                self.generation_added[s] = generation
                n_injected += 1

        self.evict_redundant()

        return {
            "n_probes": n_probes,
            "n_injected": n_injected,
            "worst_cost": float(np.max(costs)),
            "median_cost": float(np.median(costs)),
            "capture_rate": capture_rate,
        }

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
            "next_index": self._next_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], excluded_seeds: set[int] | None = None) -> SeedPool:
        """Restore a SeedPool from a checkpointed dict."""
        pool = cls(
            base_seed=int(data["base_seed"]),
            max_size=int(data["max_size"]),
            alpha=float(data.get("alpha", 0.7)),
            cvar_percentile=int(data.get("cvar_percentile", 20)),
            excluded_seeds=excluded_seeds,
        )
        pool.seeds = list(data["seeds"])
        pool.difficulty = {int(k): float(v) for k, v in dict(data["difficulty"]).items()}
        pool.generation_added = {int(k): int(v) for k, v in dict(data["generation_added"]).items()}
        pool.n_evictions = int(data.get("n_evictions", 0))
        pool._next_index = int(data.get("next_index", 0))
        return pool
