"""GA initial population generation.

Replaces MATLAB Initial_Population_Aerocap.m.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import evaluate_chromosome
from aerocapture.training.local_search import improve_chromosome


def create_initial_population(
    config: TrainingConfig,
    base_network: npt.NDArray[np.float64],
    rng: np.random.Generator | None = None,
    cwd: str | None = None,
    verbose: bool = True,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """Generate and evaluate initial GA population.

    Creates 5x oversized population, evaluates all, keeps best.

    Args:
        config: Training configuration.
        base_network: Base network weight vector.
        rng: Random number generator.
        cwd: Working directory for simulation.
        verbose: Print progress.

    Returns:
        (population, costs) where population has shape (n_pop, chromosome_length)
        and costs has shape (n_pop,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_pop = config.ga.n_pop
    n_bit = config.ga.n_bit
    n_coef = config.network.n_coef
    chrom_len = n_bit * n_coef + n_coef  # bits + sign bits
    n_candidates = 5 * n_pop

    if verbose:
        print(f"Generating {n_candidates} candidate chromosomes...")

    # Generate random binary chromosomes
    candidates = rng.integers(0, 2, size=(n_candidates, chrom_len), dtype=np.int8)
    costs = np.full(n_candidates, np.inf)

    # Evaluate all candidates
    for i in range(n_candidates):
        cost, _ = evaluate_chromosome(candidates[i], base_network, config, cwd=cwd)
        costs[i] = cost
        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{n_candidates}, best so far: {np.min(costs[:i + 1]):.4e}")

    # Sort by cost and keep best n_pop
    order = np.argsort(costs)
    population = candidates[order[:n_pop]]
    pop_costs = costs[order[:n_pop]]

    if verbose:
        print(f"Best initial cost: {pop_costs[0]:.4e}")

    # Local improvement on best chromosome
    improved, improved_cost, gain = improve_chromosome(
        population[0], base_network, config, mode=0, cwd=cwd,
    )
    if improved_cost < pop_costs[-1]:
        population[-1] = improved
        pop_costs[-1] = improved_cost
        # Re-sort
        order = np.argsort(pop_costs)
        population = population[order]
        pop_costs = pop_costs[order]
        if verbose:
            print(f"Local improvement gain: {gain:.2f}%")

    return population, pop_costs
