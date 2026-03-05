"""Main GA optimization loop for neural network guidance training.

Replaces MATLAB Train_Net_Aerocap.m.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import evaluate_chromosome
from aerocapture.training.migration import migrate
from aerocapture.training.population import create_initial_population


def roulette_selection(
    costs: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> int:
    """Select an individual via roulette wheel selection.

    Args:
        costs: Array of costs (lower is better).
        rng: Random number generator.

    Returns:
        Index of selected individual.
    """
    max_cost = np.max(costs)
    fitness = (max_cost - costs) / max(max_cost, 1e-300)
    total = np.sum(fitness)
    if total <= 0:
        return rng.integers(len(costs))
    probs = np.cumsum(fitness / total)
    r = rng.random()
    return int(np.searchsorted(probs, r))


def crossover_and_mutate(
    population: npt.NDArray[np.int8],
    costs: npt.NDArray[np.float64],
    config: TrainingConfig,
    rng: np.random.Generator,
) -> npt.NDArray[np.int8]:
    """Apply uniform crossover and mutation to create offspring.

    Args:
        population: Current population array (n_pop, chrom_length).
        costs: Current costs array (n_pop,).
        config: Training configuration.
        rng: Random number generator.

    Returns:
        New population of offspring (n_pop, chrom_length).
    """
    n_pop = len(population)
    chrom_len = population.shape[1]
    offspring = np.zeros((n_pop, chrom_len), dtype=np.int8)

    # Create offspring pairs via crossover
    for i in range(0, n_pop, 2):
        p1 = roulette_selection(costs, rng)
        p2 = roulette_selection(costs, rng)
        while p2 == p1:
            p2 = roulette_selection(costs, rng)

        # Uniform crossover
        mask = rng.integers(0, 2, size=chrom_len, dtype=np.int8)
        offspring[i] = mask * population[p1] + (1 - mask) * population[p2]
        if i + 1 < n_pop:
            offspring[i + 1] = (1 - mask) * population[p1] + mask * population[p2]

    # Mutation: flip random bits
    n_mut = int(np.ceil(config.ga.mutation_rate * offspring.size))
    mut_positions = rng.integers(0, offspring.size, size=n_mut)
    flat = offspring.ravel()
    flat[mut_positions] = 1 - flat[mut_positions]

    return offspring


def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
) -> dict:
    """Run the full GA training pipeline.

    Args:
        config: Training configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        cwd: Working directory for simulations.
        verbose: Print progress.
        checkpoint_interval: Save checkpoint every N generations.

    Returns:
        Dictionary with training results:
            - 'best_cost': Best cost found
            - 'best_chromosome': Best chromosome
            - 'best_network': Best network weights
            - 'cost_history': Cost per generation
    """
    if config is None:
        config = TrainingConfig()

    rng = np.random.default_rng(seed)

    # Initialize base network (used for perturbation encoding, ignored for direct)
    base_network = config.random_network(rng)

    # Try loading existing weights for population seeding
    seed_weights = None
    if config.ga.direct_encoding:
        nn_param_path = Path(cwd or config.sim.exec_dir) / config.sim.nn_param_file
        if nn_param_path.exists():
            try:
                loaded = config.load_base_network(str(nn_param_path))
                seed_weights = loaded[: config.network.n_base_coef]
                if verbose:
                    print(f"Loaded seed weights from {nn_param_path} ({len(seed_weights)} params)")
            except Exception as e:
                if verbose:
                    print(f"Could not load seed weights: {e}")

    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_overall_cost = np.inf
    best_overall_chrom = None
    cost_history = []

    for run in range(config.ga.n_runs):
        if verbose:
            print(f"\n=== Run {run + 1}/{config.ga.n_runs} ===")

        # Create initial population (seeded on first run only)
        population, costs = create_initial_population(
            config, base_network, rng=rng, cwd=cwd, verbose=verbose,
            seed_weights=seed_weights if run == 0 else None,
        )

        # Wrap in list for subpopulation support
        populations = [population]
        all_costs = [costs]

        gen_best_costs = []

        for gen in range(config.ga.n_gen):
            for k in range(config.ga.n_subpop):
                pop = populations[k]
                pop_costs = all_costs[k]

                # Create offspring
                offspring = crossover_and_mutate(pop, pop_costs, config, rng)

                # Evaluate offspring
                offspring_costs = np.full(len(offspring), np.inf)
                for i in range(len(offspring)):
                    cost, _ = evaluate_chromosome(
                        offspring[i], base_network, config, cwd=cwd,
                    )
                    offspring_costs[i] = cost

                # Tournament selection: combine parents + offspring, keep best
                combined = np.vstack([pop, offspring])
                combined_costs = np.concatenate([pop_costs, offspring_costs])
                order = np.argsort(combined_costs)
                n_pop = len(pop)
                populations[k] = combined[order[:n_pop]]
                all_costs[k] = combined_costs[order[:n_pop]]

                # Track best
                gen_best = all_costs[k][0]
                if gen_best < best_overall_cost:
                    best_overall_cost = gen_best
                    best_overall_chrom = populations[k][0].copy()

            # Migration
            populations, all_costs = migrate(
                populations, all_costs, gen + 1,
                base_network, config, cwd=cwd, rng=rng,
            )

            gen_best_costs.append(best_overall_cost)

            if verbose and (gen + 1) % 5 == 0:
                print(f"  Gen {gen + 1}/{config.ga.n_gen}: best={best_overall_cost:.4e}")

            # Checkpoint
            if (gen + 1) % checkpoint_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint = {
                    "run": run,
                    "generation": gen + 1,
                    "best_cost": float(best_overall_cost),
                    "cost_history": [float(c) for c in gen_best_costs],
                }
                checkpoint_path = save_dir / f"checkpoint_{timestamp}.json"
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint, f, indent=2)

        cost_history.extend(gen_best_costs)

    return {
        "best_cost": best_overall_cost,
        "best_chromosome": best_overall_chrom,
        "cost_history": cost_history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NN guidance via GA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--n-pop", type=int, default=20)
    parser.add_argument("--cwd", type=str, default="old_codebase/exec")
    args = parser.parse_args()

    cfg = TrainingConfig()
    cfg.ga.n_gen = args.n_gen
    cfg.ga.n_pop = args.n_pop
    cfg.ga.n_runs = 1

    result = train(cfg, seed=args.seed, cwd=args.cwd)
    print(f"\nFinal best cost: {result['best_cost']:.4e}")
