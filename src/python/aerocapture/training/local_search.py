"""Bit-flip hill climbing local search.

Replaces MATLAB Improve_Chrom_Aerocap.m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import evaluate_chromosome


def improve_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    mode: int = 0,
    cwd: str | Path | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.int8], float, float]:
    """Improve a chromosome via greedy bit-flip local search.

    For each coefficient (in random order), tries flipping each bit
    (in random order) and keeps the flip if it improves cost.

    Args:
        xbit: Binary chromosome to improve.
        base_network: Base network weights.
        config: Training configuration.
        mode: 0 = optimize single random coefficient, 1 = optimize all.
        cwd: Working directory.
        rng: Random number generator.

    Returns:
        (improved_chromosome, new_cost, gain_percent).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_coef = config.n_params
    n_bit = config.ga.n_bit

    # Initial cost
    current = xbit.copy()
    current_cost, _ = evaluate_chromosome(current, base_network, config, cwd=cwd)
    initial_cost = current_cost

    # Determine which coefficients to optimize
    coef_indices = rng.permutation(n_coef)[:1] if mode == 0 else rng.permutation(n_coef)

    for coef_idx in coef_indices:
        # Random bit order within this coefficient
        bit_order = rng.permutation(n_bit)
        for bit_offset in bit_order:
            pos = coef_idx * n_bit + bit_offset
            # Flip bit
            current[pos] = 1 - current[pos]
            new_cost, _ = evaluate_chromosome(current, base_network, config, cwd=cwd)

            if new_cost < current_cost:
                current_cost = new_cost  # Keep improvement
            else:
                current[pos] = 1 - current[pos]  # Revert

    gain = 100.0 * (1.0 - current_cost / max(initial_cost, 1e-300))
    return current, current_cost, gain
