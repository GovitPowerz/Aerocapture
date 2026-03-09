"""Main GA optimization loop for guidance parameter training.

Supports both NN weight optimization and generic guidance parameter optimization.
Replaces MATLAB Train_Net_Aerocap.m.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import decode_direct, decode_params_from_chromosome, evaluate_chromosome, write_nn_json
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
        return int(rng.integers(len(costs)))
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


def save_checkpoint(
    save_dir: Path,
    run: int,
    generation: int,
    populations: list[npt.NDArray[np.int8]],
    all_costs: list[npt.NDArray[np.float64]],
    best_cost: float,
    best_chrom: npt.NDArray[np.int8] | None,
    cost_history: list[float],
    rng: np.random.Generator,
    config: TrainingConfig,
    cwd: str | Path | None,
) -> None:
    """Save full training state for later resumption."""
    prefix = f"checkpoint_r{run:03d}_g{generation:05d}"

    # Serialize RNG state — convert large ints to strings for JSON compatibility
    raw_state = rng.bit_generator.state
    rng_state_json = {
        "bit_generator": raw_state["bit_generator"],
        "state": {k: str(v) if isinstance(v, int) and v.bit_length() > 53 else v for k, v in raw_state["state"].items()},
        "has_uint32": raw_state["has_uint32"],
        "uinteger": raw_state["uinteger"],
    }
    meta = {
        "run": run,
        "generation": generation,
        "best_cost": best_cost,
        "cost_history": [float(c) for c in cost_history],
        "rng_state": rng_state_json,
    }
    with open(save_dir / f"{prefix}.json", "w") as f:
        json.dump(meta, f, indent=2)

    arrays: dict[str, npt.NDArray] = {}
    for k, pop in enumerate(populations):
        arrays[f"pop_{k}"] = pop
        arrays[f"costs_{k}"] = all_costs[k]
    arrays["n_subpops"] = np.array([len(populations)])
    if best_chrom is not None:
        arrays["best_chromosome"] = best_chrom
    np.savez(save_dir / f"{prefix}.npz", **arrays)  # type: ignore[arg-type]  # mypy vs numpy stubs kwargs issue

    # Save best model/params (immediately usable by Rust)
    if best_chrom is not None:
        if config.guidance_type == "neural_network":
            weights = decode_direct(best_chrom, config)
            write_nn_json(weights, config.network, save_dir / "best_model.json")
            if cwd is not None:
                nn_path = Path(cwd) / config.sim.nn_param_file
                write_nn_json(weights, config.network, nn_path)
        else:
            params = decode_params_from_chromosome(best_chrom, config)
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(params, fp, indent=2)


def load_checkpoint(
    save_dir: Path,
) -> dict | None:
    """Find and load the latest checkpoint from save_dir.

    Returns dict with: run, generation, populations, all_costs, best_cost,
    best_chromosome, cost_history, rng_state. Or None if no checkpoint found.
    """
    json_files = sorted(save_dir.glob("checkpoint_r*_g*.json"))
    if not json_files:
        return None

    latest = json_files[-1]
    npz_path = latest.with_suffix(".npz")
    if not npz_path.exists():
        return None

    with open(latest) as f:
        meta = json.load(f)

    data = np.load(npz_path)
    n_subpops = int(data["n_subpops"][0])
    populations = [data[f"pop_{k}"] for k in range(n_subpops)]
    all_costs = [data[f"costs_{k}"] for k in range(n_subpops)]
    best_chrom = data.get("best_chromosome", None)

    return {
        "run": meta["run"],
        "generation": meta["generation"],
        "populations": populations,
        "all_costs": all_costs,
        "best_cost": meta["best_cost"],
        "best_chromosome": best_chrom,
        "cost_history": meta["cost_history"],
        "rng_state": meta.get("rng_state"),
    }


def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
) -> dict:
    """Run the full GA training pipeline.

    Args:
        config: Training configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        cwd: Working directory for simulations.
        verbose: Print progress.
        checkpoint_interval: Save checkpoint every N generations.
        resume_dir: Directory to resume training from (loads latest checkpoint).

    Returns:
        Dictionary with training results:
            - 'best_cost': Best cost found
            - 'best_chromosome': Best chromosome
            - 'cost_history': Cost per generation
    """
    if config is None:
        config = TrainingConfig()

    rng = np.random.default_rng(seed)

    # Initialize base network (used for perturbation encoding, ignored for direct)
    base_network = config.random_network(rng)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Try resuming from checkpoint
    resumed = None
    if resume_dir is not None:
        resumed = load_checkpoint(Path(resume_dir))
        if resumed is not None:
            # Restore RNG state
            if resumed["rng_state"] is not None:
                try:
                    state = resumed["rng_state"]
                    # Convert stringified large ints back
                    state["state"] = {k: int(v) if isinstance(v, str) else v for k, v in state["state"].items()}
                    rng.bit_generator.state = state
                except Exception:
                    pass  # Fall back to seeded RNG if state restore fails
            if verbose:
                print(f"Resumed from run {resumed['run']}, gen {resumed['generation']}, best={resumed['best_cost']:.4e}")

    # Try loading existing NN weights for population seeding (NN only)
    seed_weights = None
    if config.guidance_type == "neural_network" and config.ga.direct_encoding and resumed is None:
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

    best_overall_cost = resumed["best_cost"] if resumed else np.inf
    best_overall_chrom = resumed["best_chromosome"] if resumed else None
    cost_history: list[float] = resumed["cost_history"] if resumed else []

    start_run = resumed["run"] if resumed else 0
    start_gen = resumed["generation"] if resumed else 0

    for run in range(start_run, config.ga.n_runs):
        if verbose:
            print(f"\n=== Run {run + 1}/{config.ga.n_runs} ===")

        if resumed is not None and run == start_run:
            # Restore population from checkpoint
            populations = resumed["populations"]
            all_costs = resumed["all_costs"]
            gen_start = start_gen
        else:
            # Create initial population
            population, costs = create_initial_population(
                config,
                base_network,
                rng=rng,
                cwd=cwd,
                verbose=verbose,
                seed_weights=seed_weights if run == 0 and resumed is None else None,
            )
            populations = [population]
            all_costs = [costs]
            gen_start = 0

        gen_best_costs: list[float] = []

        for gen in range(gen_start, config.ga.n_gen):
            for k in range(config.ga.n_subpop):
                pop = populations[k]
                pop_costs = all_costs[k]

                # Create offspring
                offspring = crossover_and_mutate(pop, pop_costs, config, rng)

                # Evaluate offspring
                offspring_costs = np.full(len(offspring), np.inf)
                for i in range(len(offspring)):
                    cost, _ = evaluate_chromosome(
                        offspring[i],
                        base_network,
                        config,
                        cwd=cwd,
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
                populations,
                all_costs,
                gen + 1,
                base_network,
                config,
                cwd=cwd,
                rng=rng,
            )

            gen_best_costs.append(best_overall_cost)

            if verbose and (gen + 1) % 5 == 0:
                print(f"  Gen {gen + 1}/{config.ga.n_gen}: best={best_overall_cost:.4e}")

            # Checkpoint
            if (gen + 1) % checkpoint_interval == 0:
                save_checkpoint(
                    save_dir,
                    run,
                    gen + 1,
                    populations,
                    all_costs,
                    best_overall_cost,
                    best_overall_chrom,
                    cost_history + gen_best_costs,
                    rng,
                    config,
                    cwd,
                )
                if verbose:
                    print(f"  Checkpoint saved: r{run:03d}_g{gen + 1:05d}")

        cost_history.extend(gen_best_costs)

    return {
        "best_cost": best_overall_cost,
        "best_chromosome": best_overall_chrom,
        "cost_history": cost_history,
    }


if __name__ == "__main__":
    import argparse

    from aerocapture.training.evaluate import compute_cost, run_simulation, write_guidance_toml

    parser = argparse.ArgumentParser(description="Train guidance parameters via GA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--n-pop", type=int, default=20)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--toml", type=str, default=None, help="TOML config path (enables TOML mode, runs from repo root)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume training from")
    parser.add_argument(
        "--guidance",
        type=str,
        default="neural_network",
        choices=["neural_network", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag", "ftc"],
        help="Guidance scheme to optimize (default: neural_network)",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    cfg.ga.n_gen = args.n_gen
    cfg.ga.n_pop = args.n_pop
    cfg.ga.n_runs = 1
    cfg.guidance_type = args.guidance

    cwd = args.cwd
    if args.toml:
        cfg.sim.toml_config = args.toml
        cfg.sim.executable = "src/rust/target/release/aerocapture"
        cfg.sim.nn_param_file = "old_codebase/donnees/nn_model.json"
        cfg.sim.final_file = "old_codebase/sorties/final.train_nn_temp"
        cfg.sim.exec_dir = "."
        if cwd is None:
            cwd = "."
    else:
        if cwd is None:
            cwd = "old_codebase/exec"

    # Non-NN schemes require TOML mode
    if cfg.guidance_type != "neural_network" and not args.toml:
        print("ERROR: Non-NN guidance schemes require --toml <config.toml>")
        raise SystemExit(1)

    # Save dir per scheme
    cfg.save_dir = f"save_net/{cfg.guidance_type}"

    if args.resume:
        cfg.save_dir = args.resume

    result = train(cfg, seed=args.seed, cwd=cwd, resume_dir=args.resume)
    print(f"\nFinal best cost: {result['best_cost']:.4e}")

    # Save best result and run final evaluation
    if result["best_chromosome"] is not None:
        if cfg.guidance_type == "neural_network":
            weights = decode_direct(result["best_chromosome"], cfg)
            nn_path = Path(cwd) / cfg.sim.nn_param_file
            write_nn_json(weights, cfg.network, nn_path)
            print(f"Best weights saved to {nn_path}")
        else:
            params = decode_params_from_chromosome(result["best_chromosome"], cfg)
            params_path = Path(cfg.save_dir) / "best_params.json"
            with open(params_path, "w") as fp:
                json.dump(params, fp, indent=2)
            print(f"Best params saved to {params_path}")
            print(f"  Params: {params}")

            # Write optimized TOML for easy re-use
            assert cfg.sim.toml_config is not None
            base_toml = Path(cwd) / cfg.sim.toml_config
            opt_toml = Path(cfg.save_dir) / f"optimized_{cfg.guidance_type}.toml"
            write_guidance_toml(base_toml, cfg.guidance_type, params, opt_toml)
            print(f"  Optimized TOML: {opt_toml}")

        final = run_simulation(cfg, cwd=cwd)
        if final is not None:
            cost = compute_cost(final)
            print(f"Final re-evaluation cost: {cost:.4e}")
            energy = final[:, 8]
            ecc = final[:, 10]
            captured = (ecc < 1.0) & (energy < 0)
            print(f"  Captured: {captured.sum()}/{len(final)}")
            if captured.any():
                print(f"  Apoapsis err (km):  mean={np.abs(final[captured, 31]).mean():.1f}")
                print(f"  Periapsis err (km): mean={np.abs(final[captured, 30]).mean():.1f}")
                print(f"  Delta-V (m/s):      mean={final[captured, 42].mean():.1f}")
