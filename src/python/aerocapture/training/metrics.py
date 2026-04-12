"""Pure functions for computing derived training metrics.

Used by both TrainingLogger (during training) and report.py (post-hoc analysis).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# Cost above this threshold indicates a non-capture (crash, hyperbolic escape, timeout).
# Rust virtual DV assigns >= 10000 m/s to all non-capture outcomes.
CAPTURE_COST_THRESHOLD = 10000.0


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
    total_distance = 0.0
    n_pairs = 0
    for i in range(n):
        diffs = population[i] - population[i + 1 :]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_distance += float(np.sum(distances))
        n_pairs += len(distances)
    max_distance = np.sqrt(float(n_dims))
    return total_distance / (n_pairs * max_distance) if n_pairs > 0 else 0.0


def capture_rate(costs: npt.NDArray[np.float64], capture_threshold: float = 3000.0) -> float:
    """Fraction of individuals with cost below capture threshold.

    Default threshold 3000 separates captured trajectories (max ~2600
    after log compression) from non-captures (min ~3300).

    Note: this default assumes dv_threshold=1000 in the cost function.
    If dv_threshold is changed, this threshold should be adjusted
    accordingly — the gap is log_cap(HYPERBOLIC_BASE, dv_threshold).
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
