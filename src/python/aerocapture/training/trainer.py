"""The training-loop seam: one loop contract, two adapters, one loop.

`Trainer` is the contract `run_loop` drives per generation (`prologue /
should_continue / re_evaluate / advance / observe / top_k / emit /
maybe_checkpoint / finalize / on_interrupt / interrupted_result`, plus the
`finalize_in_display_scope` / `start_gen` / `excluded_seeds` / `seed_curator` /
`problem` / `rng` attributes the loop reads). `SingleAlgoTrainer` wraps a bare pymoo algorithm
plus the gate / checkpoint / selection logic; `IslandsTrainer` is a thin adapter
over `IslandModel`. Each adapter owns its setup in `from_config(config, problem,
save_dir, ...)`: resume detection, reserved seed pools, the initial population
and pymoo seeding; the explicit keyword `__init__` is the state-in, no-IO entry
the conformance test uses. `train.train` is resolve -> build -> `run_loop`.

Per-adapter conventions deliberately preserved from the legacy loops:
- JSONL generation labels: single-algo logs `gen + 1`, islands logs `gen`.
- Checkpoint cadence: single-algo saves at `(gen+1) % interval == 0` with
  label `gen + 1`; islands at `(gen+1) % interval == 0 or last gen` with
  label `gen`.
- Post-loop placement: single-algo finalizes INSIDE the display/interrupt
  scope, islands outside it (`finalize_in_display_scope`).
- RNG draw order (bit-reproducibility, `experiments/trainer_seam_gate/`):
  resume restore -> `_build_initial_population` -> one pymoo seed draw per
  algorithm / island, in `__init__`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES  # type: ignore[import-untyped]
from pymoo.core.evaluator import Evaluator  # type: ignore[import-untyped]
from pymoo.core.population import Population  # type: ignore[import-untyped]

from aerocapture.training.artifacts import write_best_artifacts
from aerocapture.training.checkpoint import _prune_old_checkpoints, _restore_seed_curator, load_checkpoint, save_checkpoint
from aerocapture.training.corridor import _accumulate_corridor
from aerocapture.training.encoding import _decode_nn_weights, decode_normalized
from aerocapture.training.evaluate import _HAS_PYO3, GateStatus, constraint_violation_rates, format_violation_rates, is_feasible, run_validation_gate
from aerocapture.training.final_select import KnownCandidate, SelectionResult, format_selection_summary, select_final_individual, write_final_selection_json
from aerocapture.training.initial_population import _build_initial_population
from aerocapture.training.island_model import IslandModel, compute_migration_origin_stats, summarize_latest_migration, val_generalization_gap
from aerocapture.training.metrics import capture_rate
from aerocapture.training.optimizer import create_algorithm, warm_start_algorithm
from aerocapture.training.problem import AerocaptureProblem, PerSeedEvaluator, training_rms
from aerocapture.training.seed_curator import SeedCurator
from aerocapture.training.seeds import (
    FINAL_EVAL_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    _compute_fixed_seeds,
    _draw_disjoint_seeds,
    base_mc_seed_from_toml,
    make_reserved_seeds,
)
from aerocapture.training.weight_stats import compute_weight_stats

if TYPE_CHECKING:
    from aerocapture.training.config import TrainingConfig
    from aerocapture.training.corridor import CorridorAccumulator
    from aerocapture.training.display import DisplayProtocol
    from aerocapture.training.logger import TrainingLogger
    from aerocapture.training.param_spaces import ParamSpec


def _build_validation_payload(
    costs: npt.NDArray[np.float64],
    final_records: npt.NDArray[np.float64] | None,
    n_sims: int,
    cost_kwargs: dict[str, Any] | None,
) -> tuple[dict, dict | None]:
    """Compose the flat metrics dict (back-compat for charts/report) and the
    rich `compute_eval_summary` dashboard (consumed by the TUI).

    The summary is None when `final_records` is unavailable (e.g. legacy
    fallback paths or future callers that haven't switched yet).
    """
    metrics = {
        "rms_cost": float(np.sqrt(np.mean(costs**2))),
        "mean_cost": float(np.mean(costs)),
        "median_cost": float(np.median(costs)),
        "std_cost": float(np.std(costs)),
        "p95_cost": float(np.percentile(costs, 95)),
        "worst_cost": float(np.max(costs)),
        "capture_rate": capture_rate(costs, cost_transform=(cost_kwargs or {}).get("cost_transform", "linear")),
        "n_sims": n_sims,
    }
    summary: dict | None = None
    if final_records is not None:
        from aerocapture.training.report import compute_eval_summary  # noqa: PLC0415

        summary = compute_eval_summary(final_records, n_sims, cost_kwargs)
    return metrics, summary


def _apply_seed_strategy(
    *,
    strategy: str,
    rng: np.random.Generator,
    n_sims: int,
    excluded_seeds: set[int],
    problem: PerSeedEvaluator,
    seed_curator: SeedCurator | None,
    pending_seed_change: bool,
) -> bool:
    """Per-gen training-seed draw shared by the single-algorithm and islands loops.

    `rotating` redraws a disjoint seed list every gen; `adaptive` does the same
    (a per-gen bootstrap draw) until the first curation has populated
    `seed_curator.seed_list`. Returns `seeds_changed_this_gen` (OR'd with the
    incoming `pending_seed_change`); `fixed` changes nothing and just echoes it.
    """
    seeds_changed = pending_seed_change
    rotating = strategy == "rotating"
    adaptive_bootstrap = strategy == "adaptive" and seed_curator is not None and seed_curator.seed_list is None
    if rotating or adaptive_bootstrap:
        problem.update_seeds(_draw_disjoint_seeds(rng, n=n_sims, excluded=excluded_seeds))
        seeds_changed = True
    return seeds_changed


def _maybe_curate(
    *,
    seed_curator: SeedCurator | None,
    problem: PerSeedEvaluator,
    gen: int,
    seed_pool_interval: int,
    curation_top_k: int,
    promoted: bool,
    top_k_provider: Callable[[int], npt.NDArray[np.float64]],
) -> bool:
    """Adaptive curation trigger shared by both loops.

    Fires on a validated promotion OR the periodic fallback interval. When it
    fires, `top_k_provider(curation_top_k)` yields the search-space slice the
    curator probes (single-algo: this gen's argmin slice; islands: the union
    across all 3 populations). Returns True when seeds changed (the caller is
    responsible for setting `pending_seed_change` so next gen re-evaluates).
    """
    if seed_curator is None:
        return False
    elapsed = gen - seed_curator.last_curation_gen
    if promoted or elapsed >= seed_pool_interval:
        new_seeds = seed_curator.curate(problem, top_k_provider(curation_top_k))
        seed_curator.last_curation_gen = gen
        problem.update_seeds(new_seeds)
        return True
    return False


def _persist_islands_promotion(
    island_model: Any,  # IslandModel; tests pass a fake
    selection: SelectionResult,
    save_dir: Path,
    gen: int,
    seed_curator: SeedCurator | None,
    keep_last: int | None,
) -> None:
    """Persist a selection-promoted winner into the islands checkpoint.

    `from_checkpoint` restores best_overall_* verbatim, so without this write-back
    a later resume restores the stale pre-selection champion and its next
    checkpoint save / end-of-run write silently overwrites the deployed
    best_model.json. Called BEFORE write_best_artifacts (durable state first) and
    AFTER final_eval (the per-island report numbers still quote the pre-selection
    champions; the promoted winner keeps its clean single-candidate quote).
    """
    win_name = selection.provenance.split(":", 1)[0]
    for isl in island_model.islands:
        if isl.name == win_name:
            isl.best_overall_individual = selection.individual.copy()
            isl.best_val_cost = float(selection.val_rms)
            break
    island_model.checkpoint(
        save_dir / f"checkpoint_g{gen:05d}.npz",
        generation=gen,
        seed_curator_state=seed_curator.to_dict() if seed_curator is not None else None,
    )
    _prune_old_checkpoints(save_dir, keep_last)


@runtime_checkable
class Trainer(Protocol):
    """What `run_loop` drives, once per generation, on both adapters."""

    finalize_in_display_scope: bool
    start_gen: int
    excluded_seeds: set[int]
    seed_curator: SeedCurator | None
    problem: PerSeedEvaluator
    rng: np.random.Generator

    def prologue(self, logger: TrainingLogger, display: DisplayProtocol) -> None: ...
    def should_continue(self, gen: int) -> bool: ...
    def re_evaluate(self) -> None: ...
    def advance(self, gen: int) -> None: ...
    def observe(self, gen: int) -> bool: ...
    def top_k(self, k: int) -> npt.NDArray[np.float64]: ...
    def emit(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None: ...
    def maybe_checkpoint(self, gen: int) -> None: ...
    def finalize(self, logger: TrainingLogger) -> dict[str, Any]: ...
    def on_interrupt(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None: ...
    def interrupted_result(self) -> dict[str, Any]: ...


def run_loop(
    trainer: Trainer,
    *,
    config: TrainingConfig,
    logger: TrainingLogger,
    display: DisplayProtocol,
) -> dict[str, Any]:
    """The per-generation loop, identical for both adapters."""
    problem, rng = trainer.problem, trainer.rng
    strategy = config.optimizer.seed_strategy
    interrupted = False
    pending_seed_change = False
    gen = trainer.start_gen
    result: dict[str, Any] | None = None

    with display:
        try:
            trainer.prologue(logger, display)
            for gen in range(trainer.start_gen, config.optimizer.n_gen):
                if not trainer.should_continue(gen):
                    break

                seeds_changed_this_gen = _apply_seed_strategy(
                    strategy=strategy,
                    rng=rng,
                    n_sims=config.optimizer.training_n_sims,
                    excluded_seeds=trainer.excluded_seeds,
                    problem=problem,
                    seed_curator=trainer.seed_curator,
                    pending_seed_change=pending_seed_change,
                )
                pending_seed_change = False

                if seeds_changed_this_gen:
                    trainer.re_evaluate()

                trainer.advance(gen)
                promoted = trainer.observe(gen)

                if _maybe_curate(
                    seed_curator=trainer.seed_curator,
                    problem=problem,
                    gen=gen,
                    seed_pool_interval=config.optimizer.seed_pool_interval,
                    curation_top_k=config.optimizer.curation_top_k,
                    promoted=promoted,
                    top_k_provider=trainer.top_k,
                ):
                    pending_seed_change = True

                trainer.emit(gen, logger, display)
                trainer.maybe_checkpoint(gen)

            if trainer.finalize_in_display_scope:
                result = trainer.finalize(logger)
        except KeyboardInterrupt:
            interrupted = True
            trainer.on_interrupt(gen, logger, display)

    if interrupted:
        return trainer.interrupted_result()
    if result is None:
        result = trainer.finalize(logger)
    return result


# -- setup shared by both adapters' `from_config` --


def _make_seed_curator(config: TrainingConfig, rng: np.random.Generator) -> SeedCurator | None:
    """`adaptive` seeds: bootstrap random + curated-CDF refreshes. `fixed` and
    `rotating` have no curator (deterministic list / per-gen redraw in the loop)."""
    if config.optimizer.seed_strategy != "adaptive":
        return None
    return SeedCurator(
        sample_size=config.optimizer.curation_sample_size,
        n_bins=config.optimizer.training_n_sims,
        excluded_seeds=set(),  # populated once val/final-eval sets are computed
        rng=rng,
        trim_fraction=config.optimizer.curation_trim_fraction,
        bucket_selection=config.optimizer.curation_bucket_selection,
    )


def _restore_rng_state(rng: np.random.Generator, state: dict | None) -> None:
    if state is None:
        return
    try:
        # Convert stringified large ints back
        state["state"] = {k: int(v) if isinstance(v, str) else v for k, v in state["state"].items()}
        rng.bit_generator.state = state
    except Exception as e:
        print(f"Warning: RNG state restore failed ({type(e).__name__}: {e}); using seeded RNG", file=sys.stderr)


def _load_seed_weights(config: TrainingConfig, cwd: str | Path | None, verbose: bool) -> npt.NDArray[np.float64] | None:
    """Existing v1 dense-only NN weights seed the population (`create_nn_initial_population`).

    v2 architectures use `init_v2_population` and discard seed weights, AND
    `load_base_network` only knows the v1 JSON layout (on a v2 best_model.json
    it raises "list indices must be integers or slices, not str").
    """
    if config.guidance_type != "neural_network" or config.network.architecture is not None:
        return None
    nn_param_path = Path(cwd or config.sim.exec_dir) / config.sim.nn_param_file
    if not nn_param_path.exists():
        return None
    try:
        seed_weights = config.load_base_network(str(nn_param_path))
    except Exception as e:
        if verbose:
            print(f"Could not load seed weights: {e}")
        return None
    if verbose:
        print(f"Loaded seed weights from {nn_param_path} ({len(seed_weights)} params)")
    return seed_weights


def _reserved_seed_pools(
    config: TrainingConfig,
    problem: PerSeedEvaluator,
    base_mc_seed: int,
    seed_curator: SeedCurator | None,
) -> tuple[list[int] | None, set[int]]:
    """Validation and final-eval pools from well-separated RNG streams (never
    shared with training), pushed into the curator's exclusion set; a restored
    curated list or the `fixed` list goes into the problem here."""
    val_seeds: list[int] | None = None
    excluded_seeds: set[int] = set()
    if config.optimizer.validation_n_sims > 0 and problem.toml_path:
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
    if config.optimizer.seed_strategy == "fixed":
        problem.update_seeds(_compute_fixed_seeds(base_mc_seed=base_mc_seed, n_sims=config.optimizer.training_n_sims, excluded=excluded_seeds))
    return val_seeds, excluded_seeds


def _override_cmaes_sigma0(config: TrainingConfig) -> None:
    """CMA-ES + warm-start: shrink the initial step size. Applied unconditionally
    (resume or fresh) so the checkpointed sigma reflects the warm-start tunable;
    gated on NN training so a non-NN config inheriting `[warm_start]` keeps its sigma0."""
    warm_start_active = bool(config.network.warm_start_from) or config.warm_start.enabled
    if config.guidance_type == "neural_network" and warm_start_active and config.optimizer.algorithm == "cma_es":
        config.optimizer.cma_es.sigma0 = config.warm_start.cmaes_sigma0


def _make_decode_fn(config: TrainingConfig, param_specs: list[ParamSpec]) -> Callable[[npt.NDArray[np.float64]], dict[str, float]] | None:
    """Logger decode (analytic schemes decode; NN bypasses)."""
    if config.guidance_type == "neural_network":
        return None

    def _decode(x: npt.NDArray[np.float64]) -> dict[str, float]:
        return decode_normalized(x, param_specs)

    return _decode


class SingleAlgoTrainer:
    """Adapter over one pymoo algorithm: the legacy single-algorithm path."""

    finalize_in_display_scope = True

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        problem: PerSeedEvaluator,
        save_dir: Path,
        *,
        toml: dict,
        cwd: str | Path | None,
        rng: np.random.Generator,
        resume_dir: str | Path | None,
        from_scratch: bool,
        corridor_acc: CorridorAccumulator | None,
        verbose: bool,
        checkpoint_interval: int,
    ) -> SingleAlgoTrainer:
        """Resume detection (RNG / curator / corridor restore, `n_gen` bump),
        reserved seed pools, the initial population, then `__init__` (pymoo
        seeding). The checkpointed best is restored verbatim in `__init__`."""
        seed_curator = _make_seed_curator(config, rng)
        resumed = load_checkpoint(Path(resume_dir)) if resume_dir is not None else None
        if resumed is not None:
            _restore_rng_state(rng, resumed["rng_state"])
            if verbose:
                print(f"Resumed from gen {resumed['generation']}, best={resumed['best_cost']:.4e}")
            if seed_curator is not None and resumed.get("seed_curator") is not None:
                seed_curator = _restore_seed_curator(resumed["seed_curator"], seed_curator, verbose)
            if corridor_acc is not None and resumed.get("corridor_acc") is not None:
                corridor_acc = resumed["corridor_acc"]
            # Make --n-gen mean "N additional" on resume
            config.optimizer.n_gen += resumed["generation"]
            if config.optimizer.algorithm == "qpso" and verbose:
                # n_iter is not checkpointed: alpha re-anneals from alpha_start
                # over the stretched schedule, re-expanding a converged swarm.
                print("  NOTE: QPSO resume restarts the alpha anneal at alpha_start (paper runs are single-shot --from-scratch)")
        seed_weights = _load_seed_weights(config, cwd, verbose) if resumed is None and not from_scratch else None
        if resumed is not None and verbose:
            # The checkpointed best is re-validated unconditionally in `prologue`
            # (gated on val_seeds), so best_val_cost is already recomputed under
            # the current transform; this is just an informative notice.
            saved_transform = resumed.get("cost_transform")
            current_transform = problem.cost_kwargs.get("cost_transform", "linear")
            if saved_transform is None or saved_transform != current_transform:
                will_revalidate = config.optimizer.validation_n_sims > 0 and bool(problem.toml_path)
                suffix = "; re-validating best under new metric" if will_revalidate else ""
                print(f"  cost_transform changed {saved_transform!r} -> {current_transform!r}{suffix}")
        base_mc_seed = base_mc_seed_from_toml(toml)
        val_seeds, excluded_seeds = _reserved_seed_pools(config, problem, base_mc_seed, seed_curator)
        pop_array, pop_costs = _build_initial_population(resumed, config, problem.param_specs, seed_weights, problem, val_seeds, base_mc_seed, rng, verbose)
        _override_cmaes_sigma0(config)
        return cls(
            config=config,
            problem=problem,
            param_specs=problem.param_specs,
            save_dir=save_dir,
            cwd=cwd,
            corridor_acc=corridor_acc,
            resumed=resumed,
            pop_array=pop_array,
            pop_costs=pop_costs,
            val_seeds=val_seeds,
            excluded_seeds=excluded_seeds,
            rng=rng,
            seed_curator=seed_curator,
            verbose=verbose,
            start_gen=resumed["generation"] if resumed else 0,
            checkpoint_interval=checkpoint_interval,
            toml_abs_path=problem.toml_path,
            decode_fn=_make_decode_fn(config, problem.param_specs),
        )

    def __init__(
        self,
        *,
        config: TrainingConfig,
        problem: PerSeedEvaluator,
        param_specs: list[ParamSpec],
        save_dir: Path,
        cwd: str | Path | None,
        corridor_acc: CorridorAccumulator | None,
        resumed: dict | None,
        pop_array: npt.NDArray[np.float64],
        pop_costs: npt.NDArray[np.float64] | None,
        val_seeds: list[int] | None,
        excluded_seeds: set[int],
        rng: np.random.Generator,
        seed_curator: SeedCurator | None,
        verbose: bool,
        start_gen: int,
        checkpoint_interval: int,
        toml_abs_path: str,
        decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None,
    ) -> None:

        self.config = config
        self.problem = problem
        self.param_specs = param_specs
        self.save_dir = save_dir
        self.cwd = cwd
        self.corridor_acc = corridor_acc
        self.val_seeds = val_seeds
        self.excluded_seeds = excluded_seeds
        self.rng = rng
        self.seed_curator = seed_curator
        self.verbose = verbose
        self.start_gen = start_gen
        self.checkpoint_interval = checkpoint_interval
        self.toml_abs_path = toml_abs_path
        self.decode_fn = decode_fn

        self.best_overall_cost: float = resumed["best_cost"] if resumed else np.inf
        self.best_overall_individual: npt.NDArray[np.float64] | None = resumed["best_individual"] if resumed else None
        self.best_val_cost: float = resumed["best_val_cost"] if resumed else np.inf
        self.cost_history: list[float] = resumed["cost_history"] if resumed else []
        # Identity of the last individual we ran validation on. Used to detect
        # "new best individual" by parameter comparison -- cost comparison is
        # unreliable under rotating or curated seeds.
        self.last_validated_individual: npt.NDArray[np.float64] | None = (
            resumed["best_individual"].copy() if resumed and resumed["best_individual"] is not None else None
        )

        self.algorithm = create_algorithm(config.optimizer, n_params=len(param_specs))
        if verbose:
            opt = config.optimizer
            print(f"  Algorithm: {type(self.algorithm).__name__} ({opt.algorithm}), n_params={len(param_specs)}, n_pop={opt.n_pop}, n_gen={opt.n_gen}")
            print(f"  Seeds:     strategy={opt.seed_strategy}, training_n_sims={opt.training_n_sims}, validation_n_sims={opt.validation_n_sims}")
            if opt.seed_strategy == "adaptive":
                print(
                    f"  Curation:  seed_pool_interval={opt.seed_pool_interval}, "
                    f"curation_top_k={opt.curation_top_k}, curation_sample_size={opt.curation_sample_size}"
                )
            if opt.algorithm == "ga":
                print(f"  GA:        crossover_eta={opt.ga.crossover_eta}, mutation_eta={opt.ga.mutation_eta}, mutation_prob={opt.ga.mutation_prob}")
            elif opt.algorithm == "cma_es":
                print(f"  CMA-ES:    sigma0={opt.cma_es.sigma0}, restart_strategy={opt.cma_es.restart_strategy}")
            elif opt.algorithm == "de":
                print(f"  DE:        variant={opt.de.variant}, crossover_prob={opt.de.crossover_prob}, scaling_factor={opt.de.scaling_factor}")
            elif opt.algorithm == "pso":
                print(f"  PSO:       w={opt.pso.w}, c1={opt.pso.c1}, c2={opt.pso.c2}")
            elif opt.algorithm == "qpso":
                print(f"  QPSO:      alpha_start={opt.qpso.alpha_start}, alpha_end={opt.qpso.alpha_end}")

        # Inject initial population into pymoo. NOTE: `setup(pop=…)` alone is
        # insufficient — pymoo's first `next()` would call `_initialize()` and
        # `_initialize_infill()`, wiping the seeded pop with an LHS sample.
        initial_pop = Population.new("X", pop_array)
        if pop_costs is not None:
            initial_pop.set("F", pop_costs.reshape(-1, 1))
        else:
            Evaluator().eval(problem, initial_pop)
            pop_costs = initial_pop.get("F").flatten()
        # Seed pymoo's per-algorithm RNG from the training RNG: without it the
        # operators draw from an unseeded stream and full-run reproducibility
        # never holds, even under seed_strategy = "fixed".
        warm_start_algorithm(self.algorithm, problem, initial_pop, seed=int(rng.integers(2**31)))

        # Initialize best from the first population eval -- but ONLY on a fresh
        # start. On resume the checkpointed best was validated under a different
        # seed list, so a `<` comparison against the current pop is meaningless
        # (cross-gen incomparability rule).
        if self.best_overall_individual is None:
            finite = np.isfinite(pop_costs)
            if finite.any():
                init_best_idx = int(np.flatnonzero(finite)[np.argmin(pop_costs[finite])])
                self.best_overall_cost = float(pop_costs[init_best_idx])
            else:
                init_best_idx = 0
                self.best_overall_cost = float("inf")
            self.best_overall_individual = pop_array[init_best_idx].copy()

        # CMA-ES self-terminates (wraps pycma); cache the instance check once.
        self.is_cmaes = isinstance(self.algorithm, (CMAES, SimpleCMAES))

        # Interrupt-safety pre-binds + per-gen scratch state.
        self.X: npt.NDArray[np.float64] = pop_array
        self.costs: npt.NDArray[np.float64] = pop_costs if pop_costs is not None else np.full(config.optimizer.n_pop, np.inf)
        self._initial_pop_costs = self.costs
        self.gen_best_costs: list[float] = []
        self.completed_gen = start_gen
        self._gen_best_individual: npt.NDArray[np.float64] | None = None
        self._gen_best_cost: float = np.inf
        self._validation_metrics: dict | None = None
        self._validation_summary: dict | None = None
        self._validated_improvement = False
        self._gen_wall_start: float = 0.0

    # ── loop contract ──────────────────────────────────────────────

    def prologue(self, logger: TrainingLogger, display: DisplayProtocol) -> None:
        """Validate the starting best: gen-0 individual on fresh starts, the
        checkpointed best on resume (keeps "Best val" + stagnation honest)."""

        if self.val_seeds is None or self.best_overall_individual is None:
            return
        init_val_costs, init_val_records = self.problem.evaluate_individual_records_per_seed(self.best_overall_individual, self.val_seeds)
        init_rms = float(np.sqrt(np.mean(init_val_costs**2)))
        # An infeasible initial / resumed champion must not anchor the promotion bar (ADR-0005).
        ceiling = self.config.optimizer.max_violation_rate
        init_rates = constraint_violation_rates(init_val_records, self.problem.cost_kwargs)
        init_feasible = is_feasible(init_rates, ceiling)
        if init_feasible:
            self.best_val_cost = init_rms
        else:
            self.best_val_cost = float("inf")
            if self.verbose:
                print(
                    f"  Initial champion validation: INFEASIBLE ({format_violation_rates(init_rates)} > ceiling {ceiling:.3%}) "
                    f"- not anchoring best_val_cost (rms was {init_rms:.4g})"
                )
        self.last_validated_individual = self.best_overall_individual.copy()
        init_val_metrics, init_val_summary = _build_validation_payload(
            init_val_costs,
            init_val_records,
            len(self.val_seeds),
            self.problem.cost_kwargs,
        )
        init_val_metrics["feasible"] = init_feasible
        init_val_metrics["violation_rates"] = init_rates
        logger.log_generation(
            self.start_gen,
            self.X,
            self._initial_pop_costs,
            self.best_overall_individual,
            self.decode_fn,
            validation=init_val_metrics,
            validation_summary=init_val_summary,
            improved=init_feasible,
        )
        display.update(logger, current_run=0)
        if self.verbose:
            label = f"Gen {self.start_gen}" if self.start_gen > 0 else "Gen 0"
            print(f"  {label} validation: mean={init_rms:.4e} cap={init_val_metrics['capture_rate']:.0%}")

    def re_evaluate(self) -> None:
        """Pre-next re-eval after a seed-list change (skipped for CMA-ES)."""
        if not self.is_cmaes and self.algorithm.pop is not None:
            parent_X = self.algorithm.pop.get("X")
            fresh_F = training_rms(self.problem, parent_X)
            self.algorithm.pop.set("F", fresh_F.reshape(-1, 1))

    def should_continue(self, gen: int) -> bool:
        """False = the algorithm terminated internally (CMA-ES convergence /
        restarts exhausted); GA/DE/PSO/QPSO use NoTermination and never do.
        Checked at loop top, BEFORE the seed strategy runs (legacy order)."""
        if self.is_cmaes and not self.algorithm.has_next():
            if self.verbose:
                print(f"  CMA-ES terminated internally at gen {gen} (converged / restarts exhausted); ending training loop.")
            return False
        return True

    def advance(self, gen: int) -> None:

        self._gen_wall_start = time.perf_counter()
        self.algorithm.next()
        pop = self.algorithm.pop
        self.X = pop.get("X")
        self.costs = pop.get("F")[:, 0]
        gen_best_idx = int(np.argmin(self.costs))
        self._gen_best_individual = self.X[gen_best_idx].copy()
        self._gen_best_cost = float(self.costs[gen_best_idx])

    def observe(self, gen: int) -> bool:
        """Corridor accumulation + validation gate / no-validation promotion.
        Returns True when a validated promotion happened this gen."""

        if self.config.guidance_type == "piecewise_constant" and self.corridor_acc is not None and _HAS_PYO3 and self.config.sim.toml_config:
            assert isinstance(self.problem, AerocaptureProblem)  # corridor sims route through the problem's override builder
            _accumulate_corridor(
                self.X,
                self.param_specs,
                self.config,
                self.corridor_acc,
                self.toml_abs_path,
                problem=self.problem,
            )

        self._validation_metrics = None
        self._validation_summary = None
        self._validated_improvement = False
        if self.val_seeds is not None:
            ceiling = self.config.optimizer.max_violation_rate
            gate = run_validation_gate(
                self.X,
                self.costs,
                self.last_validated_individual,
                self.best_val_cost,
                self.problem,
                self.val_seeds,
                max_violation_rate=ceiling,
                cost_kwargs=self.problem.cost_kwargs,
            )
            if gate.status is GateStatus.VALIDATED:
                assert gate.individual is not None and gate.val_costs is not None and gate.val_rms is not None
                self._validation_metrics, self._validation_summary = _build_validation_payload(
                    gate.val_costs,
                    gate.val_records,
                    len(self.val_seeds),
                    self.problem.cost_kwargs,
                )
                self._validation_metrics["feasible"] = gate.feasible
                self._validation_metrics["violation_rates"] = gate.violation_rates
                self.last_validated_individual = gate.individual
                if gate.promoted:
                    self.best_val_cost = gate.val_rms
                    self.best_overall_individual = gate.individual
                    self.best_overall_cost = gate.argmin_cost
                    self._validated_improvement = True
                elif not gate.feasible and gate.val_rms < self.best_val_cost and self.verbose:
                    print(
                        f"  Gen {gen} validation: REJECTED (infeasible: {format_violation_rates(gate.violation_rates)} > ceiling {ceiling:.3%}) "
                        f"despite rms {gate.val_rms:.4g}"
                    )
            # SKIP_UNCHANGED / SKIP_ALL_INF: no validation, no promotion.
        elif np.isfinite(self._gen_best_cost):
            # No validation gate: promote each generation's finite training
            # argmin directly. The final MC eval re-ranks on a disjoint pool,
            # so cross-gen seed incomparability is bounded.
            assert self._gen_best_individual is not None
            self.best_overall_individual = self._gen_best_individual
            self.best_overall_cost = self._gen_best_cost
        return self._validated_improvement

    def top_k(self, k: int) -> npt.NDArray[np.float64]:
        return self.X[np.argsort(self.costs)[: min(k, len(self.costs))]]

    def emit(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None:

        self.gen_best_costs.append(self.best_overall_cost)

        # Per-layer weight stats for NN (dense-only; v2 heterogeneous skip).
        ws = None
        if self.config.guidance_type == "neural_network" and self.best_overall_individual is not None and self.config.network.architecture is None:
            best_weights = _decode_nn_weights(self.best_overall_individual, self.param_specs)
            ws = compute_weight_stats(best_weights, self.config.network.layer_sizes)

        pool_metrics: dict | None = None
        if self.seed_curator is not None and self.seed_curator.seed_list is not None:
            pool_metrics = {
                "pool_size": len(self.seed_curator.seed_list),
                "last_curation_gen": self.seed_curator.last_curation_gen,
            }

        gen_elapsed_s = time.perf_counter() - self._gen_wall_start
        logger.log_generation(
            gen + 1,
            self.X,
            self.costs,
            self.best_overall_individual if self.best_overall_individual is not None else self.X[0],
            self.decode_fn,
            weight_stats=ws,
            pool_metrics=pool_metrics,
            gen_elapsed_s=gen_elapsed_s,
            gen_best_individual=self._gen_best_individual,
            validation=self._validation_metrics,
            validation_summary=self._validation_summary,
            improved=self._validated_improvement if self.val_seeds is not None else None,
        )
        display.update(logger, current_run=0)
        if self.verbose and not display.is_live and (gen + 1) % 5 == 0:
            print(f"  Gen {gen + 1}/{self.config.optimizer.n_gen}: best={self.best_overall_cost:.4e} ({gen_elapsed_s:.1f}s)")

    def maybe_checkpoint(self, gen: int) -> None:

        if (gen + 1) % self.checkpoint_interval == 0:
            save_checkpoint(
                self.save_dir,
                gen + 1,
                self.X,
                self.costs,
                self.best_overall_cost,
                self.best_overall_individual,
                self.cost_history + self.gen_best_costs,
                self.rng,
                self.config,
                self.cwd,
                self.param_specs,
                seed_curator=self.seed_curator,
                corridor_acc=self.corridor_acc,
                best_val_cost=self.best_val_cost,
                cost_transform=self.problem.cost_kwargs.get("cost_transform", "linear"),
            )
            if self.verbose:
                print(f"  Checkpoint saved: g{gen + 1:05d}")
        self.completed_gen = gen + 1

    def finalize(self, logger: TrainingLogger) -> dict[str, Any]:
        self.cost_history.extend(self.gen_best_costs)
        # Cleared so the KeyboardInterrupt save can't double-count the tail
        # when Ctrl+C lands during the final-selection MC sweeps below.
        self.gen_best_costs.clear()

        final_sel = None
        selection_promoted = False
        if self.val_seeds is not None:
            known: list[KnownCandidate] = []
            cand_X, cand_prov = self.X, [f"last_gen[{i}]" for i in range(self.X.shape[0])]
            if self.best_overall_individual is not None:
                if np.isfinite(self.best_val_cost):
                    known.append(KnownCandidate(x=self.best_overall_individual, provenance="champion", val_rms=float(self.best_val_cost)))
                else:
                    # Infeasible initial / resumed champion (prologue): competes as a candidate, not as a trusted champion.
                    cand_X = np.vstack([self.X, self.best_overall_individual[None, :]])
                    cand_prov.append("champion[infeasible]")
            try:
                sel = select_final_individual(
                    self.problem,
                    cand_X,
                    cand_prov,
                    known,
                    self.val_seeds,
                    max_violation_rate=self.config.optimizer.max_violation_rate,
                    cost_kwargs=self.problem.cost_kwargs,
                )
            except ValueError:
                # Pathological all-inf run with no champion: nothing to select.
                sel = None
            if sel is not None:
                final_sel = sel
                if sel.promoted:
                    self.best_overall_individual = sel.individual.copy()
                    self.best_val_cost = sel.val_rms
                    assert sel.winner_index is not None
                    # Training-cost-at-promotion semantics (resume-incomparability rule);
                    # the appended champion row keeps its own recorded cost.
                    if sel.winner_index < self.X.shape[0]:
                        self.best_overall_cost = float(self.costs[sel.winner_index])
                    selection_promoted = True

        # Always save a final checkpoint labeled with the last gen that ran.
        last_gen = self.completed_gen
        if last_gen % self.checkpoint_interval != 0 or selection_promoted:
            save_checkpoint(
                self.save_dir,
                last_gen,
                self.X,
                self.costs,
                self.best_overall_cost,
                self.best_overall_individual,
                self.cost_history,
                self.rng,
                self.config,
                self.cwd,
                self.param_specs,
                seed_curator=self.seed_curator,
                corridor_acc=self.corridor_acc,
                best_val_cost=self.best_val_cost,
                cost_transform=self.problem.cost_kwargs.get("cost_transform", "linear"),
            )
            if self.verbose:
                print(f"  Final checkpoint saved: g{last_gen:05d}")

        # Sidecar AFTER the durable save (crash-window ordering).
        if final_sel is not None and self.val_seeds is not None:
            write_final_selection_json(self.save_dir, final_sel, len(self.val_seeds))
            if self.verbose:
                print(format_selection_summary(final_sel))

        logger.close()
        return self._result(interrupted=False)

    def on_interrupt(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None:

        display.stop()
        print(f"\nInterrupted at gen {gen + 1}. Saving checkpoint...")
        save_checkpoint(
            self.save_dir,
            gen + 1,
            self.X,
            self.costs,
            self.best_overall_cost,
            self.best_overall_individual,
            self.cost_history + self.gen_best_costs,
            self.rng,
            self.config,
            self.cwd,
            self.param_specs,
            seed_curator=self.seed_curator,
            corridor_acc=self.corridor_acc,
            best_val_cost=self.best_val_cost,
            cost_transform=self.problem.cost_kwargs.get("cost_transform", "linear"),
        )
        logger.close()

    def interrupted_result(self) -> dict[str, Any]:
        return self._result(interrupted=True)

    def _result(self, *, interrupted: bool) -> dict[str, Any]:
        return {
            "best_cost": self.best_overall_cost,
            "best_individual": self.best_overall_individual,
            "cost_history": self.cost_history,
            "interrupted": interrupted,
            "corridor_acc": self.corridor_acc,
            "param_specs": self.param_specs,
        }


class IslandsTrainer:
    """Adapter over `IslandModel`."""

    finalize_in_display_scope = False

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        problem: PerSeedEvaluator,
        save_dir: Path,
        *,
        toml: dict,
        cwd: str | Path | None,
        rng: np.random.Generator,
        resume_dir: str | Path | None,
        from_scratch: bool,
        corridor_acc: CorridorAccumulator | None,
        verbose: bool,
        checkpoint_interval: int,
    ) -> IslandsTrainer:
        """Reserved seed pools and the initial population, then `__init__`
        (islands resume from the latest v2 npz in `save_dir` there: `resume_dir`
        and `corridor_acc` are single-algo concerns, accepted for a uniform
        signature and ignored)."""
        seed_curator = _make_seed_curator(config, rng)
        seed_weights = None if from_scratch else _load_seed_weights(config, cwd, verbose)
        base_mc_seed = base_mc_seed_from_toml(toml)
        val_seeds, excluded_seeds = _reserved_seed_pools(config, problem, base_mc_seed, seed_curator)
        pop_array, pop_costs = _build_initial_population(None, config, problem.param_specs, seed_weights, problem, val_seeds, base_mc_seed, rng, verbose)
        return cls(
            config=config,
            cwd=cwd,
            save_dir=save_dir,
            problem=problem,
            param_specs=problem.param_specs,
            pop_array=pop_array,
            pop_costs=pop_costs,
            val_seeds=val_seeds,
            base_mc_seed=base_mc_seed,
            excluded_seeds=excluded_seeds,
            rng=rng,
            seed_curator=seed_curator,
            verbose=verbose,
            start_gen=0,
            checkpoint_interval=checkpoint_interval,
            decode_fn=_make_decode_fn(config, problem.param_specs),
        )

    def __init__(
        self,
        *,
        config: TrainingConfig,
        cwd: str | Path | None,
        save_dir: Path,
        problem: PerSeedEvaluator,
        param_specs: list[ParamSpec],
        pop_array: npt.NDArray[np.float64],
        pop_costs: npt.NDArray[np.float64] | None,
        val_seeds: list[int] | None,
        base_mc_seed: int,
        excluded_seeds: set[int],
        rng: np.random.Generator,
        seed_curator: SeedCurator | None,
        verbose: bool,
        start_gen: int,
        checkpoint_interval: int,
        decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None,
    ) -> None:

        self.config = config
        self.cwd = cwd
        self.save_dir = save_dir
        self.problem = problem
        self.param_specs = param_specs
        self.val_seeds = val_seeds
        self.rng = rng
        self.seed_curator = seed_curator
        self.verbose = verbose
        self.checkpoint_interval = checkpoint_interval
        self.decode_fn = decode_fn
        self.gen = start_gen

        # Reserved final-eval seeds (disjoint from training + validation pools).
        final_eval_n = max(config.optimizer.validation_n_sims, 10000)
        final_eval_seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, final_eval_n)

        # Keep training-seed draws disjoint from the reserved pools. `_reserved_seed_pools`
        # only unions these when validation_n_sims > 0; do it unconditionally.
        self.excluded_seeds = excluded_seeds | set(final_eval_seeds)
        if val_seeds:
            self.excluded_seeds = self.excluded_seeds | set(val_seeds)
        # The curator's candidate draws honour the same widened set (a resume restore inherits it).
        if self.seed_curator is not None:
            self.seed_curator.excluded_seeds = self.excluded_seeds

        self.island_model = IslandModel(
            config=config.optimizer,
            problem=problem,
            n_params=len(param_specs),
            validation_seeds=val_seeds or [],
            final_eval_seeds=final_eval_seeds,
            base_mc_seed=base_mc_seed,
            rng=rng,
        )

        # Probe for a resumable islands checkpoint FIRST (skip the cold-start
        # eval on resume). Pick the LATEST npz carrying the islands v2 marker.
        ckpt_files = sorted(save_dir.glob("checkpoint_g*.npz"))
        resume_ckpt: Path | None = None
        for cand in reversed(ckpt_files):
            try:
                with np.load(cand, allow_pickle=True) as probe:
                    if "version" in probe and int(probe["version"]) == 2:
                        resume_ckpt = cand
                        break
            except Exception:
                continue
        if resume_ckpt is None and ckpt_files and verbose:
            print(
                f"  Found {len(ckpt_files)} checkpoint_g*.npz in {save_dir} but none are islands v2 checkpoints; starting fresh.",
            )

        # Fan out the (possibly warm-started) initial population to all 3 islands.
        if pop_costs is None:
            if resume_ckpt is not None:
                # from_checkpoint overwrites pop.F immediately.
                pop_costs = np.zeros(pop_array.shape[0], dtype=np.float64)
            else:
                # Evaluate once and share F across all islands.
                shared_eval_pop = Population.new("X", pop_array.copy())
                Evaluator().eval(problem, shared_eval_pop)
                pop_costs = shared_eval_pop.get("F").flatten()
        for island in self.island_model.islands:
            init_pop = Population.new("X", pop_array.copy())
            init_pop.set("F", pop_costs.reshape(-1, 1).copy())
            # Per-island pymoo RNG seed (unseeded operators break reproducibility).
            warm_start_algorithm(island.algorithm, problem, init_pop, seed=int(rng.integers(2**31)))

        self.start_gen = start_gen
        if resume_ckpt is not None:
            resumed_gen, resumed_curator_state, resumed_cost_transform = self.island_model.from_checkpoint(resume_ckpt)
            self.start_gen = resumed_gen + 1
            # `--n-gen N` after resume means "N additional gens". Done here
            # because the single-algo bump (`SingleAlgoTrainer.from_config`) reads json checkpoints only.
            config.optimizer.n_gen += resumed_gen + 1
            if resumed_curator_state is not None and self.seed_curator is not None:
                self.seed_curator = _restore_seed_curator(resumed_curator_state, self.seed_curator, verbose)
                # Push the restored curated seed list into the problem.
                if self.seed_curator.seed_list is not None:
                    problem.update_seeds(self.seed_curator.seed_list)
            # Reconcile population size AFTER the curated seeds are pushed.
            old_n = self.island_model.islands[0].algorithm.pop.get("X").shape[0]
            if config.optimizer.n_pop != old_n:
                if verbose:
                    print(f"  Resizing islands populations {old_n} -> {config.optimizer.n_pop}")
                self.island_model.resize_populations(
                    target_n=config.optimizer.n_pop,
                    rng=rng,
                    fresh_fraction=config.optimizer.grow_fresh_fraction,
                    velocity_scale=config.optimizer.islands.pso_inject_velocity_scale,
                )
            # Re-validate each island's best under the current config.
            if val_seeds:
                self.island_model.revalidate_each()
            current_transform = problem.cost_kwargs.get("cost_transform", "linear")
            if resumed_cost_transform is None or resumed_cost_transform != current_transform:
                if verbose:
                    print(f"  cost_transform changed {resumed_cost_transform!r} -> {current_transform!r}; re-validated best under new metric")
                for island in self.island_model.islands:
                    island.stagnation_counter = 0
            if verbose:
                print(f"  Resumed islands from gen {resumed_gen}, continuing from {self.start_gen}")

        self._events: list[Any] = []
        self._val_records: list[dict[str, Any]] = []

    # ── loop contract ──────────────────────────────────────────────

    def prologue(self, logger: TrainingLogger, display: DisplayProtocol) -> None:
        pass

    def should_continue(self, gen: int) -> bool:
        return True

    def re_evaluate(self) -> None:
        self.island_model.re_evaluate_all_populations()

    def advance(self, gen: int) -> None:
        self.gen = gen
        self._events = self.island_model.step(current_gen=gen)

    def observe(self, gen: int) -> bool:
        if self.val_seeds:
            self._val_records = self.island_model.validate_each(current_gen=gen)
        else:
            # No validation gate — promote each island's finite training argmin
            # directly so final_eval() still selects a deployable winner.
            self._val_records = []
            for i in self.island_model.islands:
                F = i.algorithm.pop.get("F").flatten()
                finite_mask = np.isfinite(F)
                if finite_mask.any():
                    X = i.algorithm.pop.get("X")
                    amin = int(np.argmin(np.where(finite_mask, F, np.inf)))
                    i.best_overall_individual = X[amin].copy()
                    i.best_overall_cost = float(F[amin])
                    argmin_cost = float(F[amin])
                else:
                    argmin_cost = float("inf")
                self._val_records.append(
                    {
                        "island": i.name,
                        "validated": False,
                        "promoted": False,
                        "argmin_train_cost": argmin_cost,
                        "stagnation": i.stagnation_counter,
                    }
                )
        return any(r.get("promoted") for r in self._val_records)

    def top_k(self, k: int) -> npt.NDArray[np.float64]:
        return self.island_model.pool_top_k_X(k)

    def emit(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None:

        for island, val_rec in zip(self.island_model.islands, self._val_records, strict=True):
            X = island.algorithm.pop.get("X")
            F = island.algorithm.pop.get("F").flatten()
            validation_dict: dict | None = None
            if val_rec["validated"]:
                validation_dict = {
                    "rms_cost": val_rec["val_rms"],
                    "mean_cost": val_rec["val_mean"],
                    "p95_cost": val_rec["val_p95"],
                    "capture_rate": val_rec["val_capture_rate"],
                    "n_sims": len(self.val_seeds) if self.val_seeds else 0,
                    "feasible": val_rec.get("feasible", True),
                    "violation_rates": val_rec.get("violation_rates"),
                }
            logger.log_generation(
                generation=gen,
                population=X,
                costs=F,
                best_individual=island.best_overall_individual,
                decode_fn=self.decode_fn,
                validation=validation_dict,
                validation_summary=val_rec.get("val_summary") if val_rec["validated"] else None,
                improved=val_rec["promoted"],
                island_name=island.name,
            )

        island_records: dict[str, Any] = {
            island.name: {
                "best_val": island.best_val_cost,
                "val_rms": val_rec.get("val_rms", float("inf")),
                "stagnation": island.stagnation_counter,
                "argmin_train_cost": val_rec.get("argmin_train_cost", float("inf")),
                # Sticky: last validated dashboard shown even on non-validating gens.
                "val_summary": island.latest_val_summary,
            }
            for island, val_rec in zip(
                self.island_model.islands,
                self._val_records,
                strict=True,
            )
        }
        island_records["_gen"] = gen
        island_records["_n_gen"] = self.config.optimizer.n_gen
        island_records["_total_migrations"] = len(self.island_model.migration_log)

        # Migration summary: recomputed only on migration gens; cached otherwise.
        if self._events:
            self.island_model.latest_migration_summary = summarize_latest_migration(self._events)
            self.island_model.latest_migration_gen = gen
            self.island_model.origin_stats_cache = compute_migration_origin_stats(self.island_model.migration_log)

        island_records["_latest_migration_summary"] = self.island_model.latest_migration_summary
        island_records["_latest_migration_gen"] = self.island_model.latest_migration_gen
        island_records["_origin_stats"] = self.island_model.origin_stats_cache

        display.update(logger, current_run=0, island_records=island_records)

        if self.verbose and not display.is_live and (gen + 1) % 5 == 0:
            parts = ", ".join(f"{r['island']}={r.get('argmin_train_cost', float('inf')):.3e}" for r in self._val_records)
            print(f"  Gen {gen + 1}/{self.config.optimizer.n_gen}: argmin {parts}")

    def maybe_checkpoint(self, gen: int) -> None:

        if (gen + 1) % self.checkpoint_interval == 0 or gen == self.config.optimizer.n_gen - 1:
            self.island_model.checkpoint(
                self.save_dir / f"checkpoint_g{gen:05d}.npz",
                generation=gen,
                seed_curator_state=self.seed_curator.to_dict() if self.seed_curator is not None else None,
            )
            _prune_old_checkpoints(self.save_dir, self.config.checkpoints.keep_last)

    def on_interrupt(self, gen: int, logger: TrainingLogger, display: DisplayProtocol) -> None:

        self.island_model.checkpoint(
            self.save_dir / f"checkpoint_g{gen:05d}.npz",
            generation=gen,
            seed_curator_state=self.seed_curator.to_dict() if self.seed_curator is not None else None,
        )
        _prune_old_checkpoints(self.save_dir, self.config.checkpoints.keep_last)
        if self.verbose:
            print(f"\n  Interrupted at gen {gen}; checkpoint saved.")
        logger.close()

    def interrupted_result(self) -> dict[str, Any]:
        # Ctrl+C means stop NOW: no validation-pool selection, no final_eval,
        # deployed artifacts stay whatever the last full run wrote.
        return {
            "best_cost": float("inf"),
            "best_individual": None,
            "cost_history": [],
            "interrupted": True,
            "corridor_acc": None,
            "param_specs": self.param_specs,
            "winner": None,
            "results": [],
            "migration_log": self.island_model.migration_log,
        }

    def finalize(self, logger: TrainingLogger) -> dict[str, Any]:

        config, problem, save_dir = self.config, self.problem, self.save_dir
        island_model, param_specs = self.island_model, self.param_specs

        # Validation-pool final selection across islands: union of last-gen
        # pops + champions decides the ARTIFACTS; final_eval is report-only.
        selection = None
        if island_model.validation_seeds:
            known = [
                KnownCandidate(
                    x=np.asarray(isl.best_overall_individual, dtype=np.float64),
                    provenance=f"{isl.name}:champion",
                    val_rms=float(isl.best_val_cost),
                )
                for isl in island_model.islands
                if isl.best_overall_individual is not None and np.isfinite(isl.best_val_cost)
            ]
            cand_rows: list[npt.NDArray[np.float64]] = []
            cand_prov: list[str] = []
            for isl in island_model.islands:
                if isl.best_overall_individual is not None and not np.isfinite(isl.best_val_cost):
                    # Infeasible resumed champion (revalidate_each): competes as a candidate.
                    cand_rows.append(np.asarray(isl.best_overall_individual, dtype=np.float64))
                    cand_prov.append(f"{isl.name}:champion[infeasible]")
                pop = isl.algorithm.pop
                if pop is None:
                    continue
                pop_x = pop.get("X")
                for j in range(pop_x.shape[0]):
                    cand_rows.append(np.asarray(pop_x[j], dtype=np.float64))
                    cand_prov.append(f"{isl.name}:last_gen[{j}]")
            if known or cand_rows:
                try:
                    selection = select_final_individual(
                        problem,
                        np.vstack(cand_rows) if cand_rows else np.empty((0, len(param_specs))),
                        cand_prov,
                        known,
                        island_model.validation_seeds,
                        max_violation_rate=config.optimizer.max_violation_rate,
                        cost_kwargs=problem.cost_kwargs,
                    )
                except ValueError:
                    # Pathological all-inf run with no champions.
                    selection = None

        # Final eval (report-only when selection ran).
        results = island_model.final_eval()
        if selection is None and not results:
            # validation off AND no island promoted -- stale-artifact removal.
            if self.verbose:
                print("  No island had a validated best — skipping final-eval / artifact write.")
            for stale in (
                save_dir / "best_model.json",
                save_dir / "best_params.json",
                Path(self.cwd or ".") / config.sim.nn_param_file if config.guidance_type == "neural_network" else None,
            ):
                if stale is not None and stale.exists():
                    stale.unlink()
                    if self.verbose:
                        print(f"  Removed stale {stale}")
            logger.close()
            return {
                "best_cost": float("inf"),
                "best_individual": None,
                "cost_history": [],
                "interrupted": False,
                "corridor_acc": None,
                "param_specs": param_specs,
                "winner": None,
                "results": [],
                "migration_log": island_model.migration_log,
            }

        if selection is not None:
            # Winner = validation-pool selection; quote its UNBIASED final-eval rms.
            match = next((r for r in results if r["island"] + ":champion" == selection.provenance), None)
            if match is not None:
                final_rms = float(match["rms"])
                win_island = str(match["island"])
                capture = float(match["capture_rate"])
            else:
                fe_costs = problem.evaluate_individual_per_seed(selection.individual, island_model.final_eval_seeds)
                final_rms = float(np.sqrt(np.mean(np.asarray(fe_costs, dtype=np.float64) ** 2)))
                win_island = selection.provenance.split(":", 1)[0]
                capture = float(capture_rate(np.asarray(fe_costs), cost_transform=str(problem.cost_kwargs.get("cost_transform", "linear"))))
            winner: dict[str, Any] = {
                "island": win_island,
                "X": selection.individual.copy(),
                "rms": final_rms,
                "val_rms": float(selection.val_rms),
                "capture_rate": capture,
                "n_sims": len(island_model.final_eval_seeds),
                "selection_provenance": selection.provenance,
            }
        else:
            winner = results[0]

        if selection is not None and selection.promoted:
            _persist_islands_promotion(island_model, selection, save_dir, self.gen, self.seed_curator, config.checkpoints.keep_last)

        if self.verbose:
            gap, overfit = val_generalization_gap(winner["val_rms"], winner["rms"])
            gap_detail = ""
            if winner["val_rms"] < float("inf"):
                gap_detail = f" (val_rms={winner['val_rms']:.4e}, gap={gap:+.1%}{'  [WARN: overfit to validation?]' if overfit else ''})"
            print(
                f"  Winner: {winner['island']} rms={winner['rms']:.4e} cap={winner['capture_rate']:.0%}{gap_detail}",
            )

        write_best_artifacts(winner["X"], config, param_specs, save_dir, cwd=self.cwd)

        if selection is not None:
            # Sidecar AFTER checkpoint + artifacts (crash-window ordering).
            write_final_selection_json(save_dir, selection, len(island_model.validation_seeds))
            if self.verbose:
                print(format_selection_summary(selection))

        logger.close()
        return {
            "best_cost": float(winner["rms"]),
            "best_individual": winner["X"],
            "cost_history": [],
            "interrupted": False,
            "corridor_acc": None,
            "param_specs": param_specs,
            "winner": winner,
            "results": results,
            "migration_log": island_model.migration_log,
        }
