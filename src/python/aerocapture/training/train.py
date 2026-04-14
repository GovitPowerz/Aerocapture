"""Main optimization loop for guidance parameter training.

Uses pymoo for real-valued optimization. Supports both NN weight optimization
and generic guidance parameter optimization.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pymoo.core.evaluator import Evaluator  # type: ignore[import-untyped]
from pymoo.core.population import Population  # type: ignore[import-untyped]

from aerocapture.training.config import TrainingConfig
from aerocapture.training.corridor import CorridorAccumulator
from aerocapture.training.encoding import decode_normalized, nn_param_specs_from_architecture
from aerocapture.training.evaluate import (
    _HAS_PYO3,
    FINAL_EVAL_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    _aero_rs,
    make_reserved_seeds,
    write_nn_json,
)
from aerocapture.training.metrics import capture_rate
from aerocapture.training.optimizer import OptimizerConfig, create_algorithm
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.population import create_initial_population, create_nn_initial_population
from aerocapture.training.problem import AerocaptureProblem
from aerocapture.training.seed_curator import SeedCurator
from aerocapture.training.weight_stats import compute_weight_stats

# Constant bank angles for corridor boundary sentinels (degrees).
# 0 = full lift-up (hyperbolic boundary), 180 = full lift-down (crash boundary).
# Only magnitude affects energy-vs-pdyn corridor; sign only affects lateral track.
_SENTINEL_BANK_ANGLES = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]


def _draw_disjoint_seeds(
    rng: np.random.Generator,
    n: int,
    excluded: set[int],
) -> list[int]:
    """Draw `n` random seeds disjoint from `excluded`."""
    drawn: list[int] = []
    while len(drawn) < n:
        batch = rng.integers(0, 2**31, size=n - len(drawn)).tolist()
        drawn.extend(s for s in batch if s not in excluded)
    return drawn[:n]


def _compute_fixed_seeds(base_mc_seed: int, n_sims: int, excluded: set[int]) -> list[int]:
    """Deterministic seed list for the `fixed` strategy.

    Raises ValueError if any seed in the range overlaps `excluded`.
    """
    seeds = [base_mc_seed + i for i in range(n_sims)]
    overlap = set(seeds) & excluded
    if overlap:
        msg = (
            f"fixed seed range [{base_mc_seed}..{base_mc_seed + n_sims - 1}] "
            f"overlaps {len(overlap)} validation/final-eval reserved seeds"
        )
        raise ValueError(msg)
    return seeds


def save_checkpoint(
    save_dir: Path,
    generation: int,
    population: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    best_cost: float,
    best_individual: npt.NDArray[np.float64] | None,
    cost_history: list[float],
    rng: np.random.Generator,
    config: TrainingConfig,
    cwd: str | Path | None,
    param_specs: list[ParamSpec],
    seed_curator: SeedCurator | None = None,
    corridor_acc: CorridorAccumulator | None = None,
    best_val_cost: float = np.inf,
) -> None:
    """Save full training state for later resumption."""
    prefix = f"checkpoint_g{generation:05d}"

    # Serialize RNG state -- convert large ints to strings for JSON compatibility
    raw_state = rng.bit_generator.state
    rng_state_json = {
        "bit_generator": raw_state["bit_generator"],
        "state": {k: str(v) if isinstance(v, int) and v.bit_length() > 53 else v for k, v in raw_state["state"].items()},
        "has_uint32": raw_state["has_uint32"],
        "uinteger": raw_state["uinteger"],
    }
    meta = {
        "generation": generation,
        "best_cost": best_cost,
        "best_val_cost": best_val_cost,
        "cost_history": [float(c) for c in cost_history],
        "rng_state": rng_state_json,
    }
    if seed_curator is not None:
        meta["seed_curator"] = seed_curator.to_dict()
    with open(save_dir / f"{prefix}.json", "w") as f:
        json.dump(meta, f, indent=2)

    arrays: dict[str, npt.NDArray] = {}
    arrays["population"] = population
    arrays["costs"] = costs
    if best_individual is not None:
        arrays["best_individual"] = best_individual
    if corridor_acc is not None:
        for ck, cv in corridor_acc.to_checkpoint().items():
            arrays[ck] = cv
    np.savez(save_dir / f"{prefix}.npz", **arrays)  # type: ignore[arg-type]  # mypy vs numpy stubs kwargs issue

    # Save best model/params (immediately usable by Rust)
    if best_individual is not None:
        if config.guidance_type == "neural_network":
            weights = _decode_nn_weights(best_individual, param_specs)
            write_nn_json(weights, config.network, save_dir / "best_model.json", input_mask=config.network.input_mask)
            if cwd is not None:
                nn_path = Path(cwd) / config.sim.nn_param_file
                write_nn_json(weights, config.network, nn_path, input_mask=config.network.input_mask)
        else:
            params = decode_normalized(best_individual, param_specs)
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(params, fp, indent=2)


def load_checkpoint(
    save_dir: Path,
) -> dict | None:
    """Find and load the latest checkpoint from save_dir.

    Returns dict with: generation, population, costs, best_cost,
    best_individual, cost_history, rng_state. Or None if no checkpoint found.
    """
    # Support both new (checkpoint_g*.json) and old (checkpoint_r*_g*.json) naming
    json_files = sorted(save_dir.glob("checkpoint_g*.json"))
    if not json_files:
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

    if "population" not in data:
        return None  # Incompatible legacy checkpoint; start fresh

    population = data["population"]
    costs = data["costs"]
    best_individual = data.get("best_individual", None)

    # Restore corridor accumulator if present in checkpoint
    corridor_acc_restored: CorridorAccumulator | None = None
    if "corridor_energy_bins" in data:
        corridor_state = {k: data[k] for k in data if k.startswith("corridor_")}
        corridor_acc_restored = CorridorAccumulator.from_checkpoint(corridor_state)

    return {
        "generation": meta["generation"],
        "population": population,
        "costs": costs,
        "best_cost": meta["best_cost"],
        "best_individual": best_individual,
        "cost_history": meta["cost_history"],
        "rng_state": meta.get("rng_state"),
        "best_val_cost": meta.get("best_val_cost", float("inf")),
        "seed_curator": meta.get("seed_curator"),
        "corridor_acc": corridor_acc_restored,
    }


def _decode_nn_weights(x: npt.NDArray[np.float64], specs: list[ParamSpec]) -> npt.NDArray[np.float64]:
    """Decode normalized [0,1] vector to NN weight values."""
    weights = np.empty(len(specs), dtype=np.float64)
    for i, s in enumerate(specs):
        weights[i] = s.p_min + float(x[i]) * (s.p_max - s.p_min)
    return weights


def train(
    config: TrainingConfig | None = None,
    seed: int | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    checkpoint_interval: int = 10,
    resume_dir: str | Path | None = None,
    no_tui: bool = False,
    corridor_acc: CorridorAccumulator | None = None,
    from_scratch: bool = False,
) -> dict:
    """Run the full optimization training pipeline.

    Args:
        config: Training configuration. Uses defaults if None.
        seed: Random seed for reproducibility.
        cwd: Working directory for simulations.
        verbose: Print progress.
        checkpoint_interval: Save checkpoint every N generations.
        resume_dir: Directory to resume training from (loads latest checkpoint).
        no_tui: Disable Rich TUI (use plain-text output).
        corridor_acc: Optional CorridorAccumulator for piecewise_constant training.
        from_scratch: Ignore existing checkpoints and start fresh.

    Returns:
        Dictionary with training results:
            - 'best_cost': Best cost found
            - 'best_individual': Best individual (normalized [0,1] vector)
            - 'cost_history': Cost per generation
            - 'corridor_acc': CorridorAccumulator (if piecewise_constant)
    """
    if config is None:
        config = TrainingConfig()

    # Fail fast if Rust binary is missing
    exe = Path(cwd or config.sim.exec_dir) / config.sim.executable
    if not exe.exists():
        msg = f"Rust simulator not found at {exe.resolve()}. Build it first: cd src/rust && cargo build --release"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load TOML config once (used for cost function params, curator config)
    from aerocapture.training.toml_utils import load_toml_with_bases

    _toml: dict = {}
    cost_kwargs: dict[str, float] = {}
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        _toml = load_toml_with_bases(toml_path)

        # Parse cost function config
        cost_cfg = _toml.get("cost_function", {})
        constraints = _toml.get("flight", {}).get("constraints", {})
        cost_kwargs = {
            "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
            "g_load_limit": float(constraints.get("max_load_factor", 15.0)),
            "heat_flux_limit": float(constraints.get("max_heat_flux", 200.0)),
            "heat_load_limit": float(constraints.get("max_heat_load", 25000.0)),
            "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
            "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
            "heat_load_weight": float(cost_cfg.get("heat_load_weight", 1000.0)),
        }

    # Seed strategy: three mutually exclusive training seed paths.
    #   fixed    -- deterministic [mc_seed + i]; seeds never change.
    #   rotating -- fresh random seeds drawn each generation (handled in loop body).
    #   adaptive -- bootstrap random + curated-CDF refreshes (SeedCurator).
    seed_curator: SeedCurator | None = None
    strategy = config.optimizer.seed_strategy
    if strategy == "adaptive" and config.optimizer.training_n_sims > 1:
        seed_curator = SeedCurator(
            sample_size=config.optimizer.curation_sample_size,
            n_bins=config.optimizer.training_n_sims,
            excluded_seeds=set(),  # populated once val/final-eval sets are computed
            rng=rng,
        )

    # Build parameter specifications
    from aerocapture.training.param_spaces import PARAM_SPACES

    if config.guidance_type == "neural_network":
        param_specs = nn_param_specs_from_architecture(
            config.network.layer_sizes,
            config.network.activations,
        )
    else:
        param_specs = PARAM_SPACES[config.guidance_type]

    n_params = len(param_specs)

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
                print(f"Resumed from gen {resumed['generation']}, best={resumed['best_cost']:.4e}")
            if seed_curator is not None and resumed.get("seed_curator") is not None:
                seed_curator = SeedCurator.from_dict(
                    resumed["seed_curator"],
                    excluded_seeds=seed_curator.excluded_seeds,
                    rng=rng,
                )
            if corridor_acc is not None and resumed.get("corridor_acc") is not None:
                corridor_acc = resumed["corridor_acc"]
            # Make --n-gen mean "N additional" on resume
            config.optimizer.n_gen += resumed["generation"]

    # Try loading existing NN weights for population seeding (NN only)
    seed_weights = None
    if config.guidance_type == "neural_network" and resumed is None and not from_scratch:
        nn_param_path = Path(cwd or config.sim.exec_dir) / config.sim.nn_param_file
        if nn_param_path.exists():
            try:
                seed_weights = config.load_base_network(str(nn_param_path))
                if verbose:
                    print(f"Loaded seed weights from {nn_param_path} ({len(seed_weights)} params)")
            except Exception as e:
                if verbose:
                    print(f"Could not load seed weights: {e}")

    best_overall_cost = resumed["best_cost"] if resumed else np.inf
    best_overall_individual: npt.NDArray[np.float64] | None = resumed["best_individual"] if resumed else None
    best_val_cost: float = resumed["best_val_cost"] if resumed else np.inf
    cost_history: list[float] = resumed["cost_history"] if resumed else []
    # Identity of the last individual we ran validation on. Used to detect
    # "new best individual" by parameter comparison -- cost comparison is
    # unreliable under rotating or curated seeds.
    last_validated_individual: npt.NDArray[np.float64] | None = (
        resumed["best_individual"].copy() if resumed and resumed["best_individual"] is not None else None
    )

    start_gen = resumed["generation"] if resumed else 0

    from aerocapture.training.display import create_display
    from aerocapture.training.logger import TrainingLogger

    display = create_display(
        scheme=config.guidance_type,
        n_runs=1,
        n_generations=config.optimizer.n_gen,
        enabled=not no_tui and verbose,
    )

    interrupted = False

    # Build MC seed list for problem evaluation
    mc_seed_val = _toml.get("monte_carlo", {}).get("seed")
    problem_seeds = [mc_seed_val] if mc_seed_val is not None else [42]

    # Set up problem
    toml_abs_path = str((Path(cwd or config.sim.exec_dir) / config.sim.toml_config).resolve()) if config.sim.toml_config else ""

    problem = AerocaptureProblem(
        param_specs=param_specs,
        toml_path=toml_abs_path,
        seeds=problem_seeds,
        cost_kwargs=cost_kwargs,
        scheme=config.guidance_type,
        sim_timeout=config.sim.sim_timeout_secs,
        nn_config=config.network if config.guidance_type == "neural_network" else None,
    )

    # Reserved seed sets for validation and final evaluation.
    # Uses well-separated RNG streams so training, validation, and final eval
    # never share seeds.
    base_mc_seed = mc_seed_val if mc_seed_val is not None else 42
    val_seeds: list[int] | None = None
    excluded_seeds: set[int] = set()
    if config.optimizer.validation_n_sims > 0 and toml_abs_path:
        val_seeds = make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, config.optimizer.validation_n_sims)
        final_eval_n = max(config.optimizer.validation_n_sims, 10000)
        final_eval_seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, final_eval_n)
        excluded_seeds = set(val_seeds) | set(final_eval_seeds)
        overlap = set(val_seeds) & set(final_eval_seeds)
        if overlap:
            msg = f"BUG: {len(overlap)} seeds overlap between validation and final eval sets"
            raise RuntimeError(msg)
        if seed_curator is not None:
            seed_curator.excluded_seeds = excluded_seeds
            if seed_curator.seed_list is not None:
                problem.update_seeds(seed_curator.seed_list)

    if strategy == "fixed":
        fixed_seeds = _compute_fixed_seeds(
            base_mc_seed=base_mc_seed,
            n_sims=config.optimizer.training_n_sims,
            excluded=excluded_seeds,
        )
        problem.update_seeds(fixed_seeds)

    # Create initial population
    if resumed is not None:
        pop_array = resumed["population"]
        pop_costs = resumed["costs"]
        # Ensure pop_array is float64 (legacy checkpoints may have int8)
        if pop_array.dtype != np.float64:
            pop_array = pop_array.astype(np.float64)
    else:
        if config.guidance_type == "neural_network":
            pop_array = create_nn_initial_population(
                config.network.layer_sizes,
                config.network.activations,
                config.optimizer.n_pop,
                rng,
                seed_weights=seed_weights,
            )
        else:
            pop_array = create_initial_population(
                param_specs,
                config.optimizer.n_pop,
                rng,
            )
        pop_costs = None  # Will be evaluated by pymoo

    # Set up algorithm
    algorithm = create_algorithm(config.optimizer, n_params=n_params)
    if verbose:
        print(f"  Algorithm: {type(algorithm).__name__} ({config.optimizer.algorithm}), n_params={n_params}, n_pop={config.optimizer.n_pop}")

    # Inject initial population into pymoo
    initial_pop = Population.new("X", pop_array)
    if pop_costs is not None:
        initial_pop.set("F", pop_costs.reshape(-1, 1))
    else:
        Evaluator().eval(problem, initial_pop)
        pop_costs = initial_pop.get("F").flatten()

    algorithm.setup(problem, pop=initial_pop)

    # Update best from initial evaluation
    init_best_idx = int(np.argmin(pop_costs))
    init_best_cost = float(pop_costs[init_best_idx])
    if init_best_cost < best_overall_cost:
        best_overall_cost = init_best_cost
        best_overall_individual = pop_array[init_best_idx].copy()

    # Set up decode function for logger
    decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None
    if config.guidance_type == "neural_network":
        decode_fn = None
    else:

        def _decode(x: npt.NDArray[np.float64]) -> dict[str, float]:
            return decode_normalized(x, param_specs)

        decode_fn = _decode

    logger = TrainingLogger(
        scheme=config.guidance_type,
        run=0,
        output_dir=save_dir,
        config_hash=config_hash,
    )

    gen_best_costs: list[float] = []
    pending_seed_change = False
    # Pre-bind for KeyboardInterrupt handler safety (in case interrupt fires during algorithm.next())
    X = pop_array
    costs = np.full(config.optimizer.n_pop, np.inf)
    gen = start_gen

    with display:
        try:
            # Validate gen-0 best (first candidate) on fresh starts
            if val_seeds is not None and best_overall_individual is not None and start_gen == 0:
                gen0_val_costs = problem.evaluate_individual_per_seed(best_overall_individual, val_seeds)
                best_val_cost = float(np.sqrt(np.mean(gen0_val_costs**2)))
                last_validated_individual = best_overall_individual.copy()
                gen0_val_metrics = {
                    "rms_cost": best_val_cost,
                    "mean_cost": float(np.mean(gen0_val_costs)),
                    "median_cost": float(np.median(gen0_val_costs)),
                    "std_cost": float(np.std(gen0_val_costs)),
                    "p95_cost": float(np.percentile(gen0_val_costs, 95)),
                    "worst_cost": float(np.max(gen0_val_costs)),
                    "capture_rate": capture_rate(gen0_val_costs),
                    "n_sims": len(val_seeds),
                }
                logger.log_generation(
                    0,
                    pop_array,
                    pop_costs if pop_costs is not None else np.full(config.optimizer.n_pop, np.inf),
                    best_overall_individual,
                    decode_fn,
                    validation=gen0_val_metrics,
                    improved=True,
                )
                display.update(logger, current_run=0)
                if verbose:
                    print(f"  Gen 0 validation: mean={best_val_cost:.4e} cap={gen0_val_metrics['capture_rate']:.0%}")

            for gen in range(start_gen, config.optimizer.n_gen):
                gen_wall_start = time.perf_counter()

                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False

                # Bootstrap: first iteration draws a random seed list if curator has none.
                if seed_curator is not None and seed_curator.seed_list is None:
                    bootstrap: list[int] = []
                    while len(bootstrap) < config.optimizer.training_n_sims:
                        batch = rng.integers(0, 2**31, size=config.optimizer.training_n_sims - len(bootstrap)).tolist()
                        bootstrap.extend(s for s in batch if s not in excluded_seeds)
                    problem.update_seeds(bootstrap[: config.optimizer.training_n_sims])
                    # Note: intentionally do NOT set seed_curator.seed_list -- the first
                    # real curation will set it. Bootstrap means "no curation has run yet".
                    seeds_changed_this_gen = True

                # Pre-next re-eval: only fire when seeds changed. Skip for CMA-ES.
                if seeds_changed_this_gen:
                    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES  # noqa: PLC0415

                    if not isinstance(algorithm, (CMAES, SimpleCMAES)) and algorithm.pop is not None:
                        parent_X = algorithm.pop.get("X")
                        fresh_F = problem._run_batch(parent_X)
                        algorithm.pop.set("F", fresh_F.reshape(-1, 1))

                # Advance one generation via pymoo
                algorithm.next()
                pop = algorithm.pop
                X = pop.get("X")
                F = pop.get("F")
                costs = F[:, 0]

                # Gen best by parameter identity -- cost comparison across gens is
                # unreliable under rotating or curated seeds.
                gen_best_idx = int(np.argmin(costs))
                gen_best_individual = X[gen_best_idx].copy()
                gen_best_cost = float(costs[gen_best_idx])
                new_gen_best = last_validated_individual is None or not np.array_equal(gen_best_individual, last_validated_individual)

                # Corridor accumulation for piecewise_constant
                if config.guidance_type == "piecewise_constant" and corridor_acc is not None and _HAS_PYO3 and config.sim.toml_config:
                    _accumulate_corridor(
                        X,
                        param_specs,
                        config,
                        corridor_acc,
                        toml_abs_path,
                        problem=problem,
                    )

                # Validation gate: fires whenever the gen-best individual differs
                # (by parameter identity) from the last validated individual.
                # Promotion to best_overall_individual gated on validation improvement.
                validation_metrics: dict | None = None
                validated_improvement = False
                if val_seeds is not None and new_gen_best:
                    val_costs = problem.evaluate_individual_per_seed(gen_best_individual, val_seeds)
                    val_rms = float(np.sqrt(np.mean(val_costs**2)))
                    validation_metrics = {
                        "rms_cost": val_rms,
                        "mean_cost": float(np.mean(val_costs)),
                        "median_cost": float(np.median(val_costs)),
                        "std_cost": float(np.std(val_costs)),
                        "p95_cost": float(np.percentile(val_costs, 95)),
                        "worst_cost": float(np.max(val_costs)),
                        "capture_rate": capture_rate(val_costs),
                        "n_sims": len(val_seeds),
                    }
                    last_validated_individual = gen_best_individual
                    if val_rms < best_val_cost:
                        best_val_cost = val_rms
                        best_overall_individual = gen_best_individual
                        best_overall_cost = gen_best_cost
                        validated_improvement = True

                # Curation trigger: on validated promotion OR periodic fallback.
                if seed_curator is not None:
                    elapsed = gen - seed_curator.last_curation_gen
                    periodic = elapsed >= config.optimizer.seed_pool_interval
                    if validated_improvement or periodic:
                        k = min(config.optimizer.curation_top_k, len(costs))
                        top_k_idx = np.argsort(costs)[:k]
                        top_k_X = X[top_k_idx]
                        new_seeds = seed_curator.curate(problem, top_k_X)
                        seed_curator.last_curation_gen = gen
                        problem.update_seeds(new_seeds)
                        pending_seed_change = True  # next gen's pre-next re-eval picks up

                # Common logging
                gen_best_costs.append(best_overall_cost)

                # Compute per-layer weight stats for NN
                ws = None
                if config.guidance_type == "neural_network" and best_overall_individual is not None:
                    best_weights = _decode_nn_weights(best_overall_individual, param_specs)
                    ws = compute_weight_stats(best_weights, config.network.layer_sizes)

                # Pool metrics for logger
                pool_metrics: dict | None = None
                if seed_curator is not None and seed_curator.seed_list is not None:
                    pool_metrics = {
                        "pool_size": len(seed_curator.seed_list),
                        "last_curation_gen": seed_curator.last_curation_gen,
                    }

                # Log metrics
                gen_elapsed_s = time.perf_counter() - gen_wall_start
                logger.log_generation(
                    gen + 1,
                    X,
                    costs,
                    best_overall_individual if best_overall_individual is not None else X[0],
                    decode_fn,
                    weight_stats=ws,
                    pool_metrics=pool_metrics,
                    gen_elapsed_s=gen_elapsed_s,
                    gen_best_individual=gen_best_individual,
                    validation=validation_metrics,
                    improved=validated_improvement if val_seeds is not None else None,
                )
                display.update(logger, current_run=0)

                if verbose and (gen + 1) % 5 == 0:
                    print(f"  Gen {gen + 1}/{config.optimizer.n_gen}: best={best_overall_cost:.4e} ({gen_elapsed_s:.1f}s)")

                # Checkpoint
                if (gen + 1) % checkpoint_interval == 0:
                    save_checkpoint(
                        save_dir,
                        gen + 1,
                        X,
                        costs,
                        best_overall_cost,
                        best_overall_individual,
                        cost_history + gen_best_costs,
                        rng,
                        config,
                        cwd,
                        param_specs,
                        seed_curator=seed_curator,
                        corridor_acc=corridor_acc,
                        best_val_cost=best_val_cost,
                    )
                    if verbose:
                        print(f"  Checkpoint saved: g{gen + 1:05d}")

            cost_history.extend(gen_best_costs)

            # Always save a final checkpoint
            last_gen = config.optimizer.n_gen
            if last_gen % checkpoint_interval != 0:
                save_checkpoint(
                    save_dir,
                    last_gen,
                    X,
                    costs,
                    best_overall_cost,
                    best_overall_individual,
                    cost_history,
                    rng,
                    config,
                    cwd,
                    param_specs,
                    seed_curator=seed_curator,
                    corridor_acc=corridor_acc,
                    best_val_cost=best_val_cost,
                )
                if verbose:
                    print(f"  Final checkpoint saved: g{last_gen:05d}")

            logger.close()

        except KeyboardInterrupt:
            interrupted = True
            display.stop()
            print(f"\nInterrupted at gen {gen + 1}. Saving checkpoint...")
            save_checkpoint(
                save_dir,
                gen + 1,
                X,
                costs,
                best_overall_cost,
                best_overall_individual,
                cost_history + gen_best_costs,
                rng,
                config,
                cwd,
                param_specs,
                seed_curator=seed_curator,
                corridor_acc=corridor_acc,
                best_val_cost=best_val_cost,
            )
            logger.close()

    return {
        "best_cost": best_overall_cost,
        "best_individual": best_overall_individual,
        "cost_history": cost_history,
        "interrupted": interrupted,
        "corridor_acc": corridor_acc,
        "param_specs": param_specs,
    }


def _accumulate_corridor(
    X: npt.NDArray[np.float64],
    param_specs: list[ParamSpec],
    config: TrainingConfig,
    corridor_acc: CorridorAccumulator,
    toml_path: str,
    problem: object | None = None,
) -> None:
    """Run corridor accumulation for piecewise_constant training."""
    from aerocapture.training.corridor import classify_trajectories as classify_traj
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    section = GUIDANCE_TOML_SECTIONS[config.guidance_type]
    pop_overrides: list[dict[str, object]] = []
    for i in range(X.shape[0]):
        params = decode_normalized(X[i], param_specs)
        if problem is not None and hasattr(problem, "_build_overrides"):
            ovr = problem._build_overrides(params)
        else:
            ovr = {f"guidance.{section}.{k_}": v for k_, v in params.items()}
            ovr["simulation.n_sims"] = 1
        ovr["guidance.type"] = config.guidance_type
        pop_overrides.append(ovr)

    batch_results = _aero_rs.run_batch(  # type: ignore[union-attr]
        toml_path=toml_path,
        overrides_list=pop_overrides,
        include_trajectories=True,
        sim_timeout_secs=config.sim.sim_timeout_secs,
    )
    labels = classify_traj(batch_results.final_records, delta_za_low=corridor_acc.delta_za_low, delta_za_high=corridor_acc.delta_za_high)
    corridor_acc.update(batch_results.trajectories, labels)

    # Sentinel chromosomes: constant bank angles for corridor boundary resolution
    sentinel_overrides: list[dict[str, object]] = []
    for bank in _SENTINEL_BANK_ANGLES:
        ovr_s: dict[str, object] = {f"guidance.{section}.bank_angle_{i}": float(bank) for i in range(10)}
        ovr_s["guidance.type"] = config.guidance_type
        ovr_s["simulation.n_sims"] = 1
        sentinel_overrides.append(ovr_s)

    sentinel_results = _aero_rs.run_batch(  # type: ignore[union-attr]
        toml_path=toml_path,
        overrides_list=sentinel_overrides,
        include_trajectories=True,
        sim_timeout_secs=config.sim.sim_timeout_secs,
    )
    sentinel_labels = classify_traj(
        sentinel_results.final_records,
        delta_za_low=corridor_acc.delta_za_low,
        delta_za_high=corridor_acc.delta_za_high,
    )
    corridor_acc.update(sentinel_results.trajectories, sentinel_labels)


if __name__ == "__main__":
    import argparse

    from aerocapture.training.evaluate import write_guidance_toml

    parser = argparse.ArgumentParser(description="Train guidance parameters via pymoo optimization")
    parser.add_argument("toml", type=str, help="TOML training config path (must contain [guidance] type)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=None, help="Number of generations (additional when resuming; default: from TOML [optimizer])")
    parser.add_argument("--n-pop", type=int, default=None, help="Population size (default: from TOML [optimizer])")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint directory to resume from (auto-detected if omitted and checkpoint exists)")
    parser.add_argument("-fs", "--from-scratch", action="store_true", help="Wipe existing training output and start fresh (deletes checkpoints, logs, reports)")
    parser.add_argument("--no-tui", action="store_true", help="Disable Rich TUI (use plain-text output)")
    parser.add_argument("--skip-report", "--skip-final-report", action="store_true", dest="skip_report", help="Skip PDF report generation at end of training")
    parser.add_argument("--final-n-sims", type=int, default=1000, help="Number of MC sims for final re-evaluation (default: 1000)")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Wall-clock timeout per simulation in seconds (default: no limit)")
    parser.add_argument("--algorithm", type=str, default=None, help="Optimization algorithm: ga, cma_es, de, pso (default: from TOML [optimizer])")
    args = parser.parse_args()

    cfg = TrainingConfig()

    # Load TOML first -- optimizer config comes from TOML, CLI overrides on top
    from aerocapture.training.toml_utils import load_toml_with_bases

    _toml_data = load_toml_with_bases(Path(args.toml))

    # Parse optimizer config from TOML (uses OptimizerConfig defaults for missing keys)
    cfg.optimizer = OptimizerConfig.from_dict(_toml_data.get("optimizer", {}))

    # CLI overrides -- only when explicitly provided (not None / default False)
    if args.n_gen is not None:
        cfg.optimizer.n_gen = args.n_gen
    if args.n_pop is not None:
        cfg.optimizer.n_pop = args.n_pop
    if args.algorithm is not None:
        cfg.optimizer.algorithm = args.algorithm
    guidance_type = _toml_data.get("guidance", {}).get("type")
    if guidance_type is None:
        print("ERROR: TOML config must contain [guidance] type = '<scheme>'")
        print("  Valid schemes: neural_network, equilibrium_glide, energy_controller, pred_guid, fnpag, ftc, piecewise_constant")
        raise SystemExit(1)

    from aerocapture.training.param_spaces import PARAM_SPACES

    _valid_types = set(PARAM_SPACES.keys()) | {"neural_network"}
    if guidance_type not in _valid_types:
        print(f"ERROR: Unknown guidance type '{guidance_type}' in TOML")
        print(f"  Valid schemes: {', '.join(sorted(_valid_types))}")
        raise SystemExit(1)

    cfg.guidance_type = guidance_type
    cfg.sim.toml_config = args.toml
    cfg.sim.sim_timeout_secs = args.sim_timeout
    cfg.sim.executable = "src/rust/target/release/aerocapture"
    cfg.sim.nn_param_file = _toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
    # Override NN architecture from TOML [network] section if present
    _net = _toml_data.get("network", {})
    if "layer_sizes" in _net:
        cfg.network.layer_sizes = _net["layer_sizes"]
    if "activations" in _net:
        cfg.network.activations = _net["activations"]
    if "input_mask" in _net:
        cfg.network.input_mask = _net["input_mask"]
    cfg.sim.final_file = "output/final.train_nn_temp"
    cfg.sim.exec_dir = "."
    cwd = "."

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

        # For piecewise_constant, also wipe corridor/ref trajectory in the mission directory
        if cfg.guidance_type == "piecewise_constant":
            mission_dir = save_path.parent
            for stale in ("corridor_boundaries.npz", "ref_trajectory.dat"):
                stale_path = mission_dir / stale
                if stale_path.exists():
                    stale_path.unlink()
                    print(f"  Removed stale {stale_path}")

    # Auto-resume: if no --resume and no -fs, check for existing checkpoint
    resume_dir = args.resume
    if resume_dir is None and not args.from_scratch:
        save_path = Path(cfg.save_dir)
        if list(save_path.glob("checkpoint_*.json")):
            resume_dir = cfg.save_dir

    # Derive mission name from the first base TOML (the mission config).
    import tomllib

    base_toml_path = Path(cwd) / args.toml
    with open(base_toml_path, "rb") as _f:
        _raw_toml = tomllib.load(_f)
    _bases = _raw_toml.get("base", [])
    if isinstance(_bases, str):
        _bases = [_bases]
    _mission_base = next((b for b in _bases if "missions/" in b), _bases[0] if _bases else "")
    mission_name = Path(_mission_base).stem if _mission_base else Path(args.toml).stem
    corr_dir = Path(cfg.save_dir).parent / mission_name
    corr_dir.mkdir(parents=True, exist_ok=True)

    # Check for reference trajectory requirement
    from aerocapture.training.param_spaces import REQUIRES_REF_TRAJECTORY

    if cfg.guidance_type in REQUIRES_REF_TRAJECTORY:
        ref_traj_path = corr_dir / "ref_trajectory.dat"
        if not ref_traj_path.exists():
            print(f"\nERROR: No reference trajectory found for mission '{mission_name}'.")
            print("Run piecewise_constant training first:")
            print("  uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml")
            sys.exit(1)
        print(f"  Using reference trajectory: {ref_traj_path}")

    # Initialize corridor accumulator for piecewise_constant training
    corridor_acc_init: CorridorAccumulator | None = None
    if cfg.guidance_type == "piecewise_constant":
        _pc_toml = _toml_data
        pc_section = _pc_toml.get("guidance", {}).get("piecewise_constant", {})
        energy_min = float(pc_section.get("energy_min", -6.0))
        energy_max = float(pc_section.get("energy_max", 5.0))
        corr_section = _pc_toml.get("corridor", {})
        delta_za_r = float(corr_section.get("delta_za_restricted", 200.0))
        delta_za_low = float(corr_section.get("delta_za_restricted_low", -delta_za_r))
        delta_za_high = float(corr_section.get("delta_za_restricted_high", delta_za_r))
        corridor_acc_init = CorridorAccumulator(energy_min, energy_max, delta_za_restricted=delta_za_r, delta_za_low=delta_za_low, delta_za_high=delta_za_high)

    result = train(cfg, seed=args.seed, cwd=cwd, resume_dir=resume_dir, no_tui=args.no_tui, corridor_acc=corridor_acc_init, from_scratch=args.from_scratch)
    print(f"\nFinal best training cost (RMS over {cfg.optimizer.training_n_sims} seeds): {result['best_cost']:.4e}")

    param_specs = result["param_specs"]

    # Update corridor_acc from train() result (may have been restored from checkpoint)
    corridor_acc_final = result.get("corridor_acc")

    # Save corridor data and reference trajectory for piecewise_constant
    if cfg.guidance_type == "piecewise_constant" and corridor_acc_final is not None and result["best_individual"] is not None:
        import aerocapture_rs as _aero_pc  # type: ignore[import-not-found, import-untyped]

        from aerocapture.training.corridor import save_corridor as _save_corr
        from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS as _GTS

        best_params = decode_normalized(result["best_individual"], param_specs)
        _pc_section = _GTS[cfg.guidance_type]
        best_ovr: dict[str, object] = {}
        for k_, v in best_params.items():
            if k_ == "lateral.max_reversals":
                v = int(round(v))
            if k_.startswith("lateral."):
                best_ovr[f"guidance.lateral.{k_.removeprefix('lateral.')}"] = v
            elif k_.startswith("exit."):
                best_ovr[f"guidance.ftc.{k_.removeprefix('exit.')}"] = v
            elif k_.startswith("nav."):
                best_ovr[f"navigation.{k_.removeprefix('nav.')}"] = v
            elif k_.startswith("thermal."):
                best_ovr[f"guidance.thermal_limiter.{k_.removeprefix('thermal.')}"] = v
            elif k_.startswith("shaping."):
                best_ovr[f"guidance.command_shaping.{k_.removeprefix('shaping.')}"] = v
                best_ovr["guidance.command_shaping.enabled"] = True
            else:
                best_ovr[f"guidance.{_pc_section}.{k_}"] = v
        best_ovr["guidance.type"] = cfg.guidance_type
        best_ovr["simulation.n_sims"] = 1
        # Disable dispersions so the nominal is the true undispersed trajectory
        best_ovr["monte_carlo.initial_state.level"] = "off"
        best_ovr["monte_carlo.atmosphere.level"] = "off"
        best_ovr["monte_carlo.aerodynamics.level"] = "off"
        best_ovr["monte_carlo.navigation.level"] = "off"
        best_ovr["monte_carlo.mass.level"] = "off"

        assert cfg.sim.toml_config is not None
        _pc_toml_path = str((Path(cwd) / cfg.sim.toml_config).resolve())
        best_batch = _aero_pc.run_batch(
            toml_path=_pc_toml_path,
            overrides_list=[best_ovr],
            include_trajectories=True,
            sim_timeout_secs=cfg.sim.sim_timeout_secs,
        )
        nom_traj = np.asarray(best_batch.trajectories[0]) if best_batch.trajectories else np.empty((0, 12))
        nom_dv_total = float(best_batch.final_records[0, 41]) if best_batch.final_records.shape[0] > 0 else 0.0

        # Save corridor_boundaries.npz from accumulated envelopes
        corr_data = corridor_acc_final.to_corridor_data(nominal=nom_traj)
        corr_data["nominal_dv"] = np.array([nom_dv_total])
        corr_npz = corr_dir / "corridor_boundaries.npz"
        _save_corr(corr_data, corr_npz)

        # Generate ref_trajectory.dat (7-column format)
        if nom_traj.ndim == 2 and nom_traj.shape[0] > 0:
            vel = nom_traj[:, 3]
            fpa_rad = np.radians(nom_traj[:, 4])
            radial_vel = vel * np.sin(fpa_rad)
            energy_j = nom_traj[:, 8] * 1e6
            pdyn_pa = nom_traj[:, 9] * 1e3
            incl_rad = np.radians(nom_traj[:, 11])
            time_s = nom_traj[:, 7]
            bank_rad = np.radians(nom_traj[:, 10])
            cos_bank = np.cos(bank_rad)

            ref_data = np.column_stack([energy_j, pdyn_pa, radial_vel, radial_vel, incl_rad, time_s, cos_bank])
            ref_path = corr_dir / "ref_trajectory.dat"
            np.savetxt(str(ref_path), ref_data, fmt="  %.16E")
            print(f"  Reference trajectory saved to {ref_path} ({ref_data.shape[0]} points)")

    # Save best result and run final evaluation
    if result["best_individual"] is not None:
        if cfg.guidance_type == "neural_network":
            weights = _decode_nn_weights(result["best_individual"], param_specs)
            nn_path = Path(cwd) / cfg.sim.nn_param_file
            write_nn_json(weights, cfg.network, nn_path, input_mask=cfg.network.input_mask)
            print(f"Best weights saved to {nn_path}")
        else:
            params = decode_normalized(result["best_individual"], param_specs)
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

        # Report Generation
        if not args.skip_report:
            from aerocapture.training.report import generate_report

            toml_path_report = Path(args.toml)
            generate_report(Path(cfg.save_dir), toml_path_report, n_sims_override=args.final_n_sims, sim_timeout_secs=cfg.sim.sim_timeout_secs)
