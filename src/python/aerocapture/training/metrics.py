"""Pure functions for computing derived training metrics.

Used by both TrainingLogger (during training) and report.py (post-hoc analysis).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist


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


def population_diversity(population: npt.NDArray[np.floating]) -> float:
    """Mean pairwise L2 distance normalized to [0, 1].

    For real-valued populations in [0, 1]^n, max pairwise distance is sqrt(n).
    Returns 0.0 for a single individual.
    """
    n = len(population)
    if n < 2:
        return 0.0
    n_dims = population.shape[1]
    # pdist returns the n*(n-1)/2 unordered pairwise L2 distances; its mean is the
    # same statistic the prior O(n^2) double-loop computed (sum over pairs / n_pairs).
    max_distance = math.sqrt(n_dims)
    mean_distance = float(np.mean(pdist(population)))
    return mean_distance / max_distance


def capture_rate(costs: npt.NDArray[np.float64], capture_threshold: float = 3000.0) -> float:
    """Fraction of sims with cost below capture_threshold.

    Defaults to 3000.0 (the Rust CRASH_FLOOR). Captures produce the real
    orbital-correction DV, which is far below this; non-captures use virtual
    DV at or above CRASH_FLOOR, so the threshold cleanly separates the two.
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
