"""GA initial population generation.

Replaces MATLAB Initial_Population_Aerocap.m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import evaluate_chromosome
from aerocapture.training.local_search import improve_chromosome


def encode_weights_to_chromosome(
    weights: npt.NDArray[np.float64],
    config: TrainingConfig,
) -> npt.NDArray[np.int8]:
    """Encode real-valued weights into a binary chromosome (direct encoding).

    Inverse of decode_direct: maps weights from [p_min, p_max] to binary.
    """
    n_base = config.network.n_base_coef
    n_bit = config.ga.n_bit
    p_range = config.ga.p_max - config.ga.p_min

    # Clip and normalize to [0, 1]
    clipped = np.clip(weights[:n_base], config.ga.p_min, config.ga.p_max)
    normalized = (clipped - config.ga.p_min) / p_range
    int_vals = np.round(normalized * (2**n_bit - 1)).astype(np.int64)

    # Convert each integer to n_bit binary digits
    chrom = np.zeros(n_base * n_bit, dtype=np.int8)
    for i in range(n_base):
        for b in range(n_bit):
            chrom[i * n_bit + b] = (int_vals[i] >> (n_bit - 1 - b)) & 1
    return chrom


def create_initial_population(
    config: TrainingConfig,
    base_network: npt.NDArray[np.float64],
    rng: np.random.Generator | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    seed_weights: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """Generate and evaluate initial GA population.

    Creates 3x oversized population, evaluates all, keeps best.
    Optionally seeds population with encoded versions of known weights.

    Args:
        config: Training configuration.
        base_network: Base network weights (ignored in direct encoding).
        rng: Random number generator.
        cwd: Working directory for simulation.
        verbose: Print progress.
        seed_weights: Known weight vector to seed population (direct encoding only).

    Returns:
        (population, costs) where population has shape (n_pop, chromosome_length)
        and costs has shape (n_pop,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_pop = config.ga.n_pop
    n_bit = config.ga.n_bit
    if config.ga.direct_encoding:
        chrom_len = n_bit * config.network.n_base_coef
    else:
        n_coef = config.network.n_coef
        chrom_len = n_bit * n_coef + n_coef  # bits + sign bits
    n_candidates = 3 * n_pop

    if verbose:
        print(f"Generating {n_candidates} candidate chromosomes...")

    # Generate random binary chromosomes
    candidates = rng.integers(0, 2, size=(n_candidates, chrom_len), dtype=np.int8)

    # Seed with known weights: exact encoding + mutated variants
    if seed_weights is not None and config.ga.direct_encoding:
        seed_chrom = encode_weights_to_chromosome(seed_weights, config)
        candidates[0] = seed_chrom
        n_seeded = min(n_pop // 2, n_candidates - 1)
        for i in range(1, 1 + n_seeded):
            mutant = seed_chrom.copy()
            # Flip ~5% of bits randomly
            flip_mask = rng.random(chrom_len) < 0.05
            mutant[flip_mask] = 1 - mutant[flip_mask]
            candidates[i] = mutant
        if verbose:
            print(f"  Seeded {1 + n_seeded} chromosomes from known weights")

    costs = np.full(n_candidates, np.inf)

    # Evaluate all candidates
    for i in range(n_candidates):
        cost, _ = evaluate_chromosome(candidates[i], base_network, config, cwd=cwd)
        costs[i] = cost
        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{n_candidates}, best so far: {np.min(costs[: i + 1]):.4e}")

    # Sort by cost and keep best n_pop
    order = np.argsort(costs)
    population = candidates[order[:n_pop]]
    pop_costs = costs[order[:n_pop]]

    if verbose:
        print(f"Best initial cost: {pop_costs[0]:.4e}")

    # Local improvement on best chromosome (skip if all costs are identical — no gradient)
    if pop_costs[0] < pop_costs[-1]:
        improved, improved_cost, gain = improve_chromosome(
            population[0],
            base_network,
            config,
            mode=0,
            cwd=cwd,
        )
        if improved_cost < pop_costs[-1]:
            population[-1] = improved
            pop_costs[-1] = improved_cost
            order = np.argsort(pop_costs)
            population = population[order]
            pop_costs = pop_costs[order]
            if verbose:
                print(f"Local improvement gain: {gain:.2f}%")

    return population, pop_costs
