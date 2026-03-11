"""Pure functions for computing derived training metrics.

Used by both TrainingLogger (during training) and report.py (post-hoc analysis).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def cost_stats(costs: npt.NDArray[np.float64]) -> dict[str, float]:
    """Compute best/mean/worst/median/std cost, filtering np.inf and np.nan.

    Returns np.nan for all stats when no finite values exist.
    """
    finite = costs[np.isfinite(costs)]
    if len(finite) == 0:
        return {"best": math.nan, "worst": math.nan, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "best": float(np.min(finite)),
        "worst": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def population_diversity(chromosomes: npt.NDArray[np.int8]) -> float:
    """Mean pairwise Hamming distance, normalized 0-1.

    Assumes binary {0, 1} input. For a single individual, returns 0.0.
    """
    n = len(chromosomes)
    if n < 2:
        return 0.0
    chrom_len: int = chromosomes.shape[1]
    total_distance: int = 0
    n_pairs: int = 0
    for i in range(n):
        diffs = np.sum(chromosomes[i] != chromosomes[i + 1 :], axis=1)
        total_distance += int(np.sum(diffs))
        n_pairs += len(diffs)
    return float(total_distance) / float(n_pairs * chrom_len)


def capture_rate(costs: npt.NDArray[np.float64], capture_threshold: float = 1e6) -> float:
    """Fraction of individuals with cost below capture threshold.

    Default threshold 1e6 is the floor of the hyperbolic-branch cost
    in compute_cost (non-capturing trajectories get 1e6 + 1e3*|energy|).
    """
    return float(int(np.sum(costs < capture_threshold)) / len(costs))


def convergence_speed(cost_history: list[float], threshold: float = 0.9) -> int:
    """Generation at which threshold% of final improvement was achieved.

    Returns 0 if no improvement occurred.
    """
    if len(cost_history) < 2:
        return 0
    initial = cost_history[0]
    final = cost_history[-1]
    total_improvement = initial - final
    if total_improvement <= 0:
        return 0
    target = initial - threshold * total_improvement
    for i, cost in enumerate(cost_history):
        if cost <= target:
            return i
    return len(cost_history) - 1


def stagnation_count(cost_history: list[float]) -> int:
    """Number of consecutive generations without improvement at end of history."""
    if len(cost_history) < 2:
        return 0
    best = cost_history[0]
    last_improvement = 0
    for i in range(1, len(cost_history)):
        if cost_history[i] < best:
            best = cost_history[i]
            last_improvement = i
    return len(cost_history) - 1 - last_improvement
