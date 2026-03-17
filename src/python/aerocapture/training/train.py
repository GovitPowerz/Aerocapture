"""Main GA optimization loop for guidance parameter training.

Supports both NN weight optimization and generic guidance parameter optimization.
Replaces MATLAB Train_Net_Aerocap.m.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import (
    _HAS_PYO3,
    _aero_rs,
    compute_cost,
    decode_direct,
    decode_params_from_chromosome,
    evaluate_chromosome,
    perturb_network,
    write_nn_json,
)
from aerocapture.training.migration import migrate
from aerocapture.training.population import create_initial_population
from aerocapture.training.seed_pool import SeedPool
from aerocapture.training.weight_stats import compute_weight_stats


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
    seed_pool: SeedPool | None = None,
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
    if seed_pool is not None:
        meta["seed_pool"] = seed_pool.to_dict()
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
        "seed_pool": meta.get("seed_pool"),
    }


def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
    no_tui: bool = False,
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

    # Fail fast if Rust binary is missing
    exe = Path(cwd or config.sim.exec_dir) / config.sim.executable
    if not exe.exists():
        msg = f"Rust simulator not found at {exe.resolve()}. Build it first: cd src/rust && cargo build --release"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)

    # Initialize base network (used for perturbation encoding, ignored for direct)
    base_network = config.random_network(rng)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load TOML config once (used for cost function params, seed rotation, adaptive seeds)
    from aerocapture.training.toml_utils import load_toml_with_bases

    _toml: dict = {}
    cost_kwargs: dict[str, float] = {}
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        _toml = load_toml_with_bases(toml_path)

        # Parse cost function config (with defaults)
        cost_cfg = _toml.get("cost_function", {})
        cost_kwargs = {
            "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
            "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
            "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
            "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
        }

    # Read base MC seed from TOML for seed rotation
    base_mc_seed: int | None = None
    if config.ga.rotate_seeds:
        if not config.sim.toml_config:
            msg = "rotate_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        base_mc_seed = _toml.get("monte_carlo", {}).get("seed")
        if base_mc_seed is None:
            msg = "rotate_seeds requires [monte_carlo].seed in the TOML config"
            raise ValueError(msg)

    # Initialize adaptive seed pool
    seed_pool: SeedPool | None = None
    if config.ga.adaptive_seeds:
        if not config.sim.toml_config:
            msg = "adaptive_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        pool_base_seed = _toml.get("monte_carlo", {}).get("seed")
        if pool_base_seed is None:
            msg = "adaptive_seeds requires [monte_carlo].seed in the TOML config"
            raise ValueError(msg)
        seed_pool = SeedPool(
            base_seed=pool_base_seed,
            max_size=config.ga.seed_pool_cap,
            alpha=config.ga.cost_alpha,
            cvar_percentile=config.ga.cvar_percentile,
        )

    # Compute config hash for experiment grouping
    config_hash = hashlib.sha256(repr(config).encode()).hexdigest()[:12]

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
            if seed_pool is not None and resumed.get("seed_pool") is not None:
                seed_pool = SeedPool.from_dict(resumed["seed_pool"])
            # Make --n-gen mean "N additional" on resume (only safe with n_runs=1,
            # which is the CLI default; with multiple runs, subsequent runs would
            # inherit the inflated n_gen and loop range(0, inflated) = too many gens)
            if config.ga.n_runs == 1:
                config.ga.n_gen += resumed["generation"]

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

    from aerocapture.training.display import create_display
    from aerocapture.training.logger import TrainingLogger

    display = create_display(
        scheme=config.guidance_type,
        n_runs=config.ga.n_runs,
        n_generations=config.ga.n_gen,
        enabled=not no_tui and verbose,
    )

    interrupted = False

    with display:
        try:
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
                        cost_kwargs=cost_kwargs,
                    )
                    populations = [population]
                    all_costs = [costs]
                    gen_start = 0

                # Set up decode function for logger (typed for mypy disallow_untyped_defs)
                decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None
                if config.guidance_type == "neural_network":
                    decode_fn = None
                else:

                    def _decode(chrom: npt.NDArray[np.int8]) -> dict[str, float]:
                        return decode_params_from_chromosome(chrom, config)

                    decode_fn = _decode

                logger = TrainingLogger(
                    scheme=config.guidance_type,
                    run=run,
                    output_dir=save_dir,
                    config_hash=config_hash,
                )

                gen_best_costs: list[float] = []

                # Build evaluator callbacks for adaptive seed pool
                def _pool_evaluator(chrom: npt.NDArray[np.int8], mc_seed: int) -> float:
                    """Scalar fallback: one (chromosome, seed) pair."""
                    cost, _ = evaluate_chromosome(chrom, base_network, config, cwd=cwd, mc_seed=mc_seed, cost_kwargs=cost_kwargs)
                    return cost

                _batch_evaluator: Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]] | None = None
                if seed_pool is not None and _HAS_PYO3 and config.sim.toml_config:

                    def _make_batch_eval(
                        base_net: npt.NDArray[np.float64],
                        cfg: TrainingConfig,
                        working_dir: str | Path | None,
                        cost_kw: dict[str, float],
                    ) -> Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]]:
                        """Factory to avoid closure over mutable loop variables."""

                        def _batch_eval(chrom: npt.NDArray[np.int8], seeds: list[int]) -> npt.NDArray[np.float64]:
                            if cfg.guidance_type == "neural_network":
                                weights = decode_direct(chrom, cfg) if cfg.ga.direct_encoding else perturb_network(chrom, base_net, cfg)
                                nn_path = Path(working_dir or cfg.sim.exec_dir) / cfg.sim.nn_param_file
                                write_nn_json(weights, cfg.network, nn_path)
                                overrides_list: list[dict[str, object]] = [{"monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]
                            else:
                                params = decode_params_from_chromosome(chrom, cfg)
                                from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

                                section = GUIDANCE_TOML_SECTIONS[cfg.guidance_type]
                                base_overrides: dict[str, object] = {f"guidance.{section}.{k}": v for k, v in params.items()}
                                base_overrides["guidance.type"] = cfg.guidance_type
                                overrides_list = [{**base_overrides, "monte_carlo.seed": s, "simulation.n_sims": 1} for s in seeds]

                            assert cfg.sim.toml_config is not None
                            toml_path = str((Path(working_dir or cfg.sim.exec_dir) / cfg.sim.toml_config).resolve())
                            results = _aero_rs.run_batch(  # type: ignore[union-attr]
                                toml_path=toml_path,
                                overrides_list=overrides_list,
                            )
                            final_records = results.final_records  # (N, 52) numpy array
                            costs: npt.NDArray[np.float64] = np.array(
                                [compute_cost(final_records[i : i + 1], **cost_kw) for i in range(final_records.shape[0])]
                            )
                            return costs

                        return _batch_eval

                    _batch_evaluator = _make_batch_eval(base_network, config, cwd, cost_kwargs)

                for gen in range(gen_start, config.ga.n_gen):
                    if seed_pool is not None:
                        # === Adaptive seed pool path ===
                        seed_pool.add_seeds(gen)

                        for k in range(config.ga.n_subpop):
                            pop = populations[k]
                            pop_costs = all_costs[k]

                            offspring = crossover_and_mutate(pop, pop_costs, config, rng)

                            combined = np.vstack([pop, offspring])
                            combined_fitness = seed_pool.evaluate_population(
                                combined,
                                _pool_evaluator,
                                batch_evaluator=_batch_evaluator,
                            )

                            seed_pool.evict_redundant()

                            n_pop = len(pop)
                            order = np.argsort(combined_fitness)
                            populations[k] = combined[order[:n_pop]]
                            all_costs[k] = combined_fitness[order[:n_pop]]

                            gen_best = all_costs[k][0]
                            if gen_best < best_overall_cost:
                                best_overall_cost = gen_best
                                best_overall_chrom = populations[k][0].copy()

                        # Migration: skip local improvement in adaptive mode
                        if (gen + 1) % config.ga.migration_interval == 0 and config.ga.n_subpop > 1:
                            for i in range(config.ga.n_subpop - 1):
                                best_idx = int(np.argmin(all_costs[i + 1]))
                                worst_idx = int(np.argmax(all_costs[i]))
                                populations[i][worst_idx] = populations[i + 1][best_idx].copy()
                                all_costs[i][worst_idx] = all_costs[i + 1][best_idx]
                            best_idx = int(np.argmin(all_costs[0]))
                            worst_idx = int(np.argmax(all_costs[-1]))
                            populations[-1][worst_idx] = populations[0][best_idx].copy()
                            all_costs[-1][worst_idx] = all_costs[0][best_idx]

                    else:
                        # === Original path (fixed seed or rotate-seeds) ===
                        mc_seed = (base_mc_seed + gen) if base_mc_seed is not None else None

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
                                    mc_seed=mc_seed,
                                    cost_kwargs=cost_kwargs,
                                )
                                offspring_costs[i] = cost

                            # Re-evaluate parents on current seed when rotating
                            if mc_seed is not None:
                                for i in range(len(pop)):
                                    cost, _ = evaluate_chromosome(
                                        pop[i],
                                        base_network,
                                        config,
                                        cwd=cwd,
                                        mc_seed=mc_seed,
                                        cost_kwargs=cost_kwargs,
                                    )
                                    pop_costs[i] = cost

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

                    # === Common path (both adaptive and original) ===
                    gen_best_costs.append(best_overall_cost)

                    # Compute per-layer weight stats for NN (instrumentation for future adaptive bounds)
                    ws = None
                    if config.guidance_type == "neural_network" and best_overall_chrom is not None:
                        best_weights = decode_direct(best_overall_chrom, config)
                        ws = compute_weight_stats(best_weights, config.network.layer_sizes)

                    # Pool metrics for logger
                    pool_metrics: dict | None = None
                    if seed_pool is not None:
                        d_min, d_max = seed_pool.difficulty_range
                        pool_metrics = {
                            "pool_size": len(seed_pool.seeds),
                            "difficulty_min": d_min,
                            "difficulty_max": d_max,
                            "n_evictions": seed_pool.n_evictions,
                        }

                    # Log metrics
                    logger.log_generation(
                        gen + 1,
                        populations,
                        all_costs,
                        best_overall_chrom if best_overall_chrom is not None else populations[0][0],
                        decode_fn,
                        weight_stats=ws,
                        mc_seed=(base_mc_seed + gen) if base_mc_seed is not None else None,
                        pool_metrics=pool_metrics,
                    )
                    display.update(logger, current_run=run)

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
                            seed_pool=seed_pool,
                        )
                        if verbose:
                            print(f"  Checkpoint saved: r{run:03d}_g{gen + 1:05d}")

                cost_history.extend(gen_best_costs)

                # Always save a final checkpoint at end of run
                last_gen = config.ga.n_gen
                if last_gen % checkpoint_interval != 0:
                    save_checkpoint(
                        save_dir,
                        run,
                        last_gen,
                        populations,
                        all_costs,
                        best_overall_cost,
                        best_overall_chrom,
                        cost_history,
                        rng,
                        config,
                        cwd,
                        seed_pool=seed_pool,
                    )
                    if verbose:
                        print(f"  Final checkpoint saved: r{run:03d}_g{last_gen:05d}")

                logger.close()

        except KeyboardInterrupt:
            interrupted = True
            display.stop()
            print(f"\nInterrupted at run {run + 1}, gen {gen + 1}. Saving checkpoint...")
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
                seed_pool=seed_pool,
            )
            logger.close()

    return {
        "best_cost": best_overall_cost,
        "best_chromosome": best_overall_chrom,
        "cost_history": cost_history,
        "interrupted": interrupted,
    }


if __name__ == "__main__":
    import argparse

    from aerocapture.training.evaluate import compute_cost, write_guidance_toml

    parser = argparse.ArgumentParser(description="Train guidance parameters via GA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gen", type=int, default=100, help="Number of generations (additional when resuming)")
    parser.add_argument("--n-pop", type=int, default=20)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--toml", type=str, default=None, help="TOML config path (enables TOML mode, runs from repo root)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from (auto-detected if omitted and checkpoint exists)")
    parser.add_argument("-fs", "--from-scratch", action="store_true", help="Wipe existing training output and start fresh (deletes checkpoints, logs, reports)")
    parser.add_argument(
        "--guidance",
        type=str,
        default="neural_network",
        choices=["neural_network", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag", "ftc"],
        help="Guidance scheme to optimize (default: neural_network)",
    )
    parser.add_argument("--no-tui", action="store_true", help="Disable Rich TUI (use plain-text output)")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--rotate-seeds", action="store_true", help="Rotate MC dispersion seed each generation (prevents overfitting to fixed scenarios)")
    seed_group.add_argument("--adaptive-seeds", action="store_true", help="Use adaptive seed pool with difficulty-based eviction")
    parser.add_argument("--seed-pool-cap", type=int, default=100, help="Maximum adaptive seed pool size (default: 100)")
    parser.add_argument("--cost-alpha", type=float, default=0.7, help="Mean/CVaR blend weight: 1.0=pure mean, 0.0=pure CVaR (default: 0.7)")
    parser.add_argument("--cvar-percentile", type=int, default=20, help="CVaR tail fraction in percent (default: 20)")
    parser.add_argument("--skip-final-report", action="store_true", help="Skip final re-evaluation report")
    parser.add_argument("--final-n-sims", type=int, default=1000, help="Number of MC sims for final re-evaluation (default: 1000)")
    args = parser.parse_args()

    cfg = TrainingConfig()
    cfg.ga.n_gen = args.n_gen
    cfg.ga.n_pop = args.n_pop
    cfg.ga.n_runs = 1
    cfg.guidance_type = args.guidance
    cfg.ga.rotate_seeds = args.rotate_seeds
    cfg.ga.adaptive_seeds = args.adaptive_seeds
    cfg.ga.seed_pool_cap = args.seed_pool_cap
    cfg.ga.cost_alpha = args.cost_alpha
    cfg.ga.cvar_percentile = args.cvar_percentile

    cwd = args.cwd
    if args.toml:
        cfg.sim.toml_config = args.toml
        cfg.sim.executable = "src/rust/target/release/aerocapture"
        # Read nn_param_file from TOML [data] neural_network field if present
        from aerocapture.training.toml_utils import load_toml_with_bases

        _toml_data = load_toml_with_bases(Path(args.toml))
        cfg.sim.nn_param_file = _toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
        # Override NN architecture from TOML [network] section if present
        _net = _toml_data.get("network", {})
        if "layer_sizes" in _net:
            cfg.network.layer_sizes = _net["layer_sizes"]
        if "activations" in _net:
            cfg.network.activations = _net["activations"]
        cfg.sim.final_file = "output/final.train_nn_temp"
        cfg.sim.exec_dir = "."
        if cwd is None:
            cwd = "."
    else:
        if cwd is None:
            cwd = "."

    # Non-NN schemes require TOML mode
    if cfg.guidance_type != "neural_network" and not args.toml:
        print("ERROR: Non-NN guidance schemes require --toml <config.toml>")
        raise SystemExit(1)

    # Save dir per scheme
    cfg.save_dir = f"training_output/{cfg.guidance_type}"

    if args.resume:
        cfg.save_dir = args.resume

    if args.from_scratch:
        if args.resume:
            print("ERROR: --from-scratch and --resume are mutually exclusive")
            raise SystemExit(1)
        save_path = Path(cfg.save_dir)
        if save_path.exists():
            import shutil

            shutil.rmtree(save_path)
            print(f"Wiped existing output: {save_path}")

    # Auto-resume: if no --resume and no -fs, check for existing checkpoint
    resume_dir = args.resume
    if resume_dir is None and not args.from_scratch:
        save_path = Path(cfg.save_dir)
        if list(save_path.glob("checkpoint_*.json")):
            resume_dir = cfg.save_dir

    result = train(cfg, seed=args.seed, cwd=cwd, resume_dir=resume_dir, no_tui=args.no_tui)
    print(f"\nFinal best cost: {result['best_cost']:.4e}")

    # Generate convergence report from JSONL training logs
    from aerocapture.training.report import generate_single_report

    scheme_dir = Path(cfg.save_dir)
    if list(scheme_dir.glob("*.jsonl")):
        generate_single_report(scheme_dir)
    else:
        print("No JSONL logs found, skipping convergence report")

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

        # Final evaluation report (large-MC re-evaluation)
        if not args.skip_final_report:
            from aerocapture.training.final_report import (
                _COL_APO_ERR,
                _COL_DV_TOTAL,
                _COL_ECC,
                _COL_ENERGY,
                _COL_INCL,
                _COL_PERI_ERR,
                _read_target_inclination,
                generate_final_report,
                run_final_evaluation,
            )

            # For non-NN schemes, use the optimized TOML (contains best guidance params)
            # For NN, the base TOML already references the NN JSON on disk
            if cfg.guidance_type != "neural_network":
                opt_toml = Path(cfg.save_dir) / f"optimized_{cfg.guidance_type}.toml"
                if opt_toml.exists():
                    cfg.sim.toml_config = str(opt_toml)

            # Read target inclination and reference trajectory from the base TOML
            base_toml_path = Path(cwd or ".") / args.toml
            target_incl = _read_target_inclination(base_toml_path)

            final_seed = args.seed + 9999
            print(f"\nRunning {args.final_n_sims}-sim final evaluation (seed={final_seed})...")
            eval_data = run_final_evaluation(cfg, n_sims=args.final_n_sims, seed=final_seed, cwd=cwd)
            if eval_data is not None:
                # Print summary statistics to stdout
                final_eval = eval_data.final_array
                n_sims = len(final_eval)
                energy = final_eval[:, _COL_ENERGY]
                ecc = final_eval[:, _COL_ECC]
                captured = (ecc < 1.0) & (energy < 0)
                n_cap = int(captured.sum())
                print(f"\n  Final evaluation ({n_sims} sims):")
                print(f"    Capture rate:       {n_cap}/{n_sims} ({100 * n_cap / n_sims:.1f}%)")
                if n_cap > 0:
                    dv = final_eval[captured, _COL_DV_TOTAL]
                    apo_err = np.abs(final_eval[captured, _COL_APO_ERR])
                    peri_err = np.abs(final_eval[captured, _COL_PERI_ERR])
                    print(f"    Delta-V (m/s):      p50={np.median(dv):.1f}  p95={np.percentile(dv, 95):.1f}  mean={dv.mean():.1f}")
                    print(f"    Apoapsis err (km):  p50={np.median(apo_err):.1f}  p95={np.percentile(apo_err, 95):.1f}  mean={apo_err.mean():.1f}")
                    print(f"    Periapsis err (km): p50={np.median(peri_err):.1f}  p95={np.percentile(peri_err, 95):.1f}  mean={peri_err.mean():.1f}")
                    incl_err = final_eval[captured, _COL_INCL] - target_incl
                    print(f"    Inclin. err (deg):  p50={np.median(incl_err):.2f}  p95={np.percentile(incl_err, 95):.2f}  mean={incl_err.mean():.2f}")
                    print(
                        "\n  Note: Final evaluation stats are only meaningful in comparison to other schemes or configurations on the same scenario and seed.\n"
                    )

                # Compute corridor boundaries (or load cached) — shared across schemes per mission
                # Derive mission name from the first base TOML (the mission config)
                import tomllib

                with open(base_toml_path, "rb") as _f:
                    _raw_toml = tomllib.load(_f)
                _bases = _raw_toml.get("base", [])
                if isinstance(_bases, str):
                    _bases = [_bases]
                # First base that contains "missions/" is the mission config
                _mission_base = next((b for b in _bases if "missions/" in b), _bases[0] if _bases else "")
                mission_name = Path(_mission_base).stem if _mission_base else Path(args.toml).stem
                corr_dir = Path(cfg.save_dir).parent / mission_name
                corr_dir.mkdir(parents=True, exist_ok=True)
                corr_npz = corr_dir / "corridor_boundaries.npz"
                if not corr_npz.exists():
                    from aerocapture.training.corridor import compute_corridor, save_corridor

                    print("Computing corridor boundaries...")
                    corr_data = compute_corridor(str(base_toml_path), seed=final_seed)
                    save_corridor(corr_data, corr_npz)

                report_path = Path(cfg.save_dir) / "final_report.html"
                generate_final_report(eval_data, cfg.guidance_type, target_incl, report_path, corridor_path=corr_npz)
                print(f"Final report saved to {report_path}")
            else:
                print("WARNING: Final evaluation simulation failed, skipping report")
