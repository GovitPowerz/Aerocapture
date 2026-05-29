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
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.core.evaluator import Evaluator  # type: ignore[import-untyped]
from pymoo.core.population import Population  # type: ignore[import-untyped]

from aerocapture.training.config import CheckpointConfig, TrainingConfig, WarmStartConfig  # noqa: F401  (CheckpointConfig re-exported for downstream tests)
from aerocapture.training.corridor import CorridorAccumulator
from aerocapture.training.encoding import decode_normalized, nn_param_specs_from_architecture, nn_param_specs_from_v2
from aerocapture.training.evaluate import (
    _HAS_PYO3,
    FINAL_EVAL_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    _aero_rs,
    make_reserved_seeds,
    write_nn_json,
)
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.metrics import capture_rate
from aerocapture.training.optimizer import OptimizerConfig, create_algorithm
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.population import create_initial_population, create_nn_initial_population
from aerocapture.training.problem import AerocaptureProblem
from aerocapture.training.seed_curator import SeedCurator
from aerocapture.training.weight_stats import compute_weight_stats

_DEFAULT_PIECEWISE_N_SEGMENTS = 10

# Constant bank angles for corridor boundary sentinels (degrees).
# 0 = full lift-up (hyperbolic boundary), 180 = full lift-down (crash boundary).
# Only magnitude affects energy-vs-pdyn corridor; sign only affects lateral track.
_SENTINEL_BANK_ANGLES = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]


def _resolve_piecewise_n_segments(toml: dict) -> int:
    """Mirror of Rust TomlPiecewiseConstantParams::resolve_bank_angles_deg.

    Order of precedence: explicit n_segments > bank_angles array length >
    highest bank_angle_N key index + 1 > default (10).
    """
    pc = toml.get("guidance", {}).get("piecewise_constant", {})
    if "n_segments" in pc:
        n = int(pc["n_segments"])
        if n < 1:
            raise ValueError(f"[guidance.piecewise_constant] n_segments must be >= 1, got {n}")
        return n
    if "bank_angles" in pc:
        return len(pc["bank_angles"])
    max_idx = max(
        (int(k.removeprefix("bank_angle_")) for k in pc if k.startswith("bank_angle_") and k.removeprefix("bank_angle_").isdigit()),
        default=-1,
    )
    if max_idx >= 0:
        return max_idx + 1
    return _DEFAULT_PIECEWISE_N_SEGMENTS


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
        "capture_rate": capture_rate(costs),
        "n_sims": n_sims,
    }
    summary: dict | None = None
    if final_records is not None:
        from aerocapture.training.report import compute_eval_summary  # noqa: PLC0415

        summary = compute_eval_summary(final_records, n_sims, cost_kwargs)
    return metrics, summary


def _compute_fixed_seeds(base_mc_seed: int, n_sims: int, excluded: set[int]) -> list[int]:
    """Deterministic seed list for the `fixed` strategy.

    Raises ValueError if any seed in the range overlaps `excluded`.
    """
    seeds = [base_mc_seed + i for i in range(n_sims)]
    overlap = set(seeds) & excluded
    if overlap:
        msg = f"fixed seed range [{base_mc_seed}..{base_mc_seed + n_sims - 1}] overlaps {len(overlap)} validation/final-eval reserved seeds"
        raise ValueError(msg)
    return seeds


def _apply_seed_strategy(
    *,
    strategy: str,
    rng: np.random.Generator,
    n_sims: int,
    excluded_seeds: set[int],
    problem: Any,
    seed_curator: SeedCurator | None,
    pending_seed_change: bool,
) -> bool:
    """Per-gen training-seed draw shared by the single-algorithm and islands loops.

    `rotating` redraws a disjoint seed list every gen; `adaptive` draws a
    one-time bootstrap list before the first curation has populated
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
    problem: Any,
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


def _prune_old_checkpoints(save_dir: Path, keep_last: int | None) -> None:
    """Retain only the `keep_last` most recent checkpoints; no-op when unset.

    Shared by the single-algorithm `save_checkpoint` and the islands path.
    `prune_checkpoints` matches both `checkpoint_g*.json` and `checkpoint_g*.npz`
    (islands writes npz-only), and leaves JSONL logs / best_* / warm_start_* /
    report.pdf untouched, so post-training analysis still works.
    """
    if keep_last is None or keep_last < 1:
        return
    from aerocapture.training.cleanup_checkpoints import prune_checkpoints  # noqa: PLC0415

    prune_checkpoints(save_dir, keep_last=keep_last)


def _check_resume_chromosome_shape(
    saved_population: npt.NDArray[np.float64],
    expected_n_params: int,
) -> None:
    """Fail loudly if a resumed checkpoint's chromosome width disagrees with current ParamSpec count.

    Catches the user flipping `scaffolding` (or `output_parameterization`,
    which changes last-layer width) between training runs.
    """
    saved_n_params = saved_population.shape[1]
    if saved_n_params != expected_n_params:
        msg = (
            f"checkpoint chromosome shape mismatch: saved {saved_n_params} params, "
            f"current ParamSpec list has {expected_n_params}. This usually means "
            f"`[guidance.neural_network] scaffolding` or "
            f"`output_parameterization` was changed since the checkpoint was saved. "
            f"To resume, revert the TOML knob; to start fresh, pass --from-scratch."
        )
        raise ValueError(msg)


def _seed_initial_population(
    algorithm_name: str,
    chromosome: np.ndarray,
    n_pop: int,
    jitter: float,
    rng: np.random.Generator,
    n_weights: int | None = None,
) -> np.ndarray:
    """Build the initial population from a warm-started chromosome.

    Row 0 of the returned population is ALWAYS the exact warm-start
    chromosome (no jitter), so the supervised-pretrained vector is
    guaranteed to be evaluated by the optimizer at generation 0 regardless
    of what jitter does on the other rows. This protects the warm-start
    "anchor": even if every jittered draw turns out worse, the pristine
    warm-start chromosome stays in the population and any reasonable
    selection / elitism propagates it forward.

    GA / DE / PSO: tile chromosome to `n_pop` rows; add per-row N(0, jitter)
    noise to the first `n_weights` columns of rows 1..n_pop-1 (or all
    columns if `n_weights` is None); clip to [0, 1]. The scaffolding tail
    (if scaffolding != "off", `chromosome[n_weights:]`) is NOT
    jittered here -- caller is responsible for overwriting that slab with
    `scaffolding_slab` when applicable, AND restoring row 0's tail to the
    un-jittered warm-start values afterward.

    CMA-ES: tile chromosome to `n_pop` rows without jitter; pymoo's CMA-ES
    uses the population mean as its initial mean. sigma0 is configured via
    `OptimizerConfig.cma_es.sigma0` (separate path in create_algorithm).
    Row 0 is trivially the warm-start chromosome since the entire tile is.
    """
    pop = np.tile(chromosome, (n_pop, 1))
    if algorithm_name == "cma_es":
        return pop
    if algorithm_name not in ("ga", "de", "pso", "islands"):
        raise ValueError(f"unknown algorithm {algorithm_name!r} for warm-start seeding")
    if n_pop < 1:
        raise ValueError(f"n_pop must be >= 1, got {n_pop}")
    nw = n_weights if n_weights is not None else chromosome.size
    # Jitter rows 1..n_pop-1 only; row 0 stays as the exact warm-start chromosome.
    if n_pop > 1:
        pop[1:, :nw] += rng.normal(0.0, jitter, size=(n_pop - 1, nw))
        pop[1:, :nw] = np.clip(pop[1:, :nw], 0.0, 1.0)
    return pop


def build_initial_population_for_v2(
    architecture: list[dict],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
    param_specs: list[ParamSpec],
    scaffolding_slab: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Activation-aware initial population for v2 architectures.

    When `scaffolding_slab` is provided (shape `(n_pop, n_scaffolding)`),
    appends it as the trailing slab of every individual. Used when
    `scaffolding = "full"` to seed scaffolding from FTC's optimum.
    """
    physical = init_v2_population(architecture, n_pop, bound_multiplier, rng)
    n_pop_actual, n_params = physical.shape
    n_scaff = 0 if scaffolding_slab is None else scaffolding_slab.shape[1]
    n_weight_specs = len(param_specs) - n_scaff
    assert n_params == n_weight_specs, (
        f"init_v2_population produced {n_params} params, ParamSpec has {n_weight_specs} weight specs (total {len(param_specs)}, scaff {n_scaff})"
    )
    normalized = np.empty((n_pop_actual, len(param_specs)), dtype=np.float64)
    for j in range(n_weight_specs):
        s = param_specs[j]
        normalized[:, j] = np.clip((physical[:, j] - s.p_min) / (s.p_max - s.p_min), 0.0, 1.0)
    if scaffolding_slab is not None:
        normalized[:, n_weight_specs:] = scaffolding_slab
    return normalized


def build_scaffolding_initial_slab(
    ftc_params_path: str | Path,
    scaffolding_specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    jitter: float = 0.02,
) -> npt.NDArray[np.float64]:
    """Seed the scaffolding slab of the PSO chromosome from FTC's GA optimum.

    Reads `<ftc_params_path>` (a JSON file with the same keys FTC writes,
    e.g. "lateral.tau", "exit.exit_pdyn_margin", ...), encodes each value
    to its [0, 1] slot via `encode_to_normalized`, replicates `n_pop`
    times, then adds `N(0, jitter)` per-individual noise clipped to [0, 1].

    Raises FileNotFoundError if `ftc_params_path` does not exist.
    Raises KeyError if any scaffolding spec name is missing from the JSON.
    """
    from aerocapture.training.encoding import encode_to_normalized

    ftc_params_path = Path(ftc_params_path)
    if not ftc_params_path.exists():
        msg = (
            f"scaffolding='full' requires a source params file; '{ftc_params_path}' "
            f"does not exist. Run FTC training first (./train_all.sh ftc) or correct the path."
        )
        raise FileNotFoundError(msg)

    with open(ftc_params_path) as f:
        ftc_params: dict[str, float] = json.load(f)

    spec_names = {s.name for s in scaffolding_specs}
    missing = spec_names - set(ftc_params.keys())
    if missing:
        msg = f"FTC params file '{ftc_params_path}' missing scaffolding keys: {sorted(missing)}. Re-run FTC training so its best_params.json includes them."
        raise KeyError(msg)

    center = encode_to_normalized(ftc_params, list(scaffolding_specs))
    slab = np.tile(center, (n_pop, 1))
    if jitter > 0.0:
        slab = slab + rng.normal(0.0, jitter, size=slab.shape)
        slab = np.clip(slab, 0.0, 1.0)
    return slab


def build_default_scaffolding_slab(
    scaffolding_specs: list[ParamSpec],
    n_pop: int,
    rng: np.random.Generator,
    jitter: float = 0.02,
) -> npt.NDArray[np.float64]:
    """Seed a scaffolding slab from each spec's default (no FTC file read).

    Mirrors `build_scaffolding_initial_slab`'s shape/jitter contract but sources
    the center from `ParamSpec.default` instead of an FTC JSON. Used for
    `scaffolding = "live"`, where the params have standalone defaults and no
    FTC dependency.
    """
    from aerocapture.training.encoding import encode_to_normalized

    center = encode_to_normalized({s.name: s.default for s in scaffolding_specs}, list(scaffolding_specs))
    slab = np.tile(center, (n_pop, 1))
    if jitter > 0.0:
        slab = slab + rng.normal(0.0, jitter, size=slab.shape)
        slab = np.clip(slab, 0.0, 1.0)
    return slab


def _make_warm_start_eval_callback(
    problem: Any,
    config: TrainingConfig,
    warm_seeds: list[int],
    val_seeds: list[int],
) -> Callable[[int, Any], None]:
    """Build the closure invoked by `_chunked_bptt_train` every
    `eval_interval` epochs.

    The closure:
      1. Extracts the policy's current flat weights via `_policy_to_flat_weights_v2`.
      2. Writes them to a temp NN JSON via `aerocapture_rs.flat_weights_to_json`.
      3. Runs MC on both `warm_seeds` (training corpus) and `val_seeds`
         (reserved validation pool) via `problem.evaluate_individual_records_per_seed`
         -- adapted to a "weights from temp JSON" path.
      4. Computes `compute_eval_summary` for each pool and prints
         `format_eval_summary` lines with a clear pool header.

    Two pools are evaluated separately because they answer different questions:
      - warm seeds: "how well does the NN approximate the supervised target on
        the EXACT seeds we trained on?" -- in-sample fit.
      - val seeds: "how well does it generalize to unseen scenarios?" -- the
        same metric the validation gate later uses for promotion decisions.
    """
    import tempfile

    from aerocapture.training.report import compute_eval_summary, format_eval_summary
    from aerocapture.training.warm_start import _policy_to_flat_weights_v2

    save_dir = Path(config.save_dir)

    def _evaluate_pool(label: str, seeds: list[int], temp_nn_json_path: Path) -> dict[str, Any]:
        """Run MC on `seeds` with the current temp NN JSON; compute the eval summary."""
        # Mirror evaluate_individual_records_per_seed's logic, but skip the
        # chromosome -> weights step (we already have the weights on disk).
        decoded_params: dict[str, float] = {}
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _eval_pack = active_scaffolding_specs(config.network.scaffolding)
        if config.network.scaffolding == "full":
            # Pull the scaffolding values from FTC's best_params.json so the
            # eval runs with the same scaffolding the chromosome will carry.
            ftc_path = Path("training_output/ftc/best_params.json")
            if ftc_path.exists():
                ftc_params = json.loads(ftc_path.read_text())
                for spec in _eval_pack:
                    if spec.name in ftc_params:
                        decoded_params[spec.name] = float(ftc_params[spec.name])
        elif config.network.scaffolding == "live":
            # live tail is seeded from defaults; eval with the same.
            for spec in _eval_pack:
                decoded_params[spec.name] = float(spec.default)

        from aerocapture.training.evaluate import _aero_rs as _aero  # noqa: PLC0415

        overrides_list = []
        for seed in seeds:
            ovr = problem._build_overrides(decoded_params, mc_seed=int(seed))
            ovr["data.neural_network"] = str(temp_nn_json_path)
            overrides_list.append(ovr)
        result = _aero.run_batch(
            problem.toml_path,
            overrides_list,
            n_threads=None,
            include_trajectories=False,
            sim_timeout_secs=problem.sim_timeout,
        )
        final_records = np.asarray(result.final_records, dtype=np.float64)
        return compute_eval_summary(final_records, len(seeds), problem.cost_kwargs)

    def _callback(epoch: int, policy: Any) -> None:
        from aerocapture.training.evaluate import _aero_rs as _aero  # noqa: PLC0415

        if config.network.architecture is None:
            return  # v1 dense-only warm-start cannot use the v2 callback path

        flat_weights = _policy_to_flat_weights_v2(policy, config.network.architecture)
        fd, tmp_str = tempfile.mkstemp(suffix=".json", prefix=f"warm_eval_epoch_{epoch:04d}_")
        import os

        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            _aero.flat_weights_to_json(
                flat_weights.tolist(),
                json.dumps(config.network.architecture),
                str(tmp_path),
                config.network.input_mask,
                config.network.output_parameterization,
            )
            print()
            print(f"  [warm_start] === In-training evaluation at epoch {epoch} ===")
            for label, seeds in (("warm-start corpus (training seeds)", warm_seeds), ("validation pool (reserved val_seeds)", val_seeds)):
                summary = _evaluate_pool(label, seeds, tmp_path)
                print(f"  [warm_start] {label}:")
                for line in format_eval_summary(summary, indent="      "):
                    print(f"  {line}" if not line.startswith(" ") else line)
                # Snapshot the val-pool summary to warm_start_eval_summary.json
                # so the report has a fresh-state copy if training is interrupted
                # between epochs. The post-warm-start gen-0 baseline path
                # overwrites this with the FINAL chromosome's stats.
                if label.startswith("validation"):
                    (save_dir / "warm_start_eval_summary.json").write_text(json.dumps(summary, indent=2))
            print()
        finally:
            tmp_path.unlink(missing_ok=True)

    return _callback


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
            from aerocapture.training.param_spaces import active_scaffolding_specs

            _pack = active_scaffolding_specs(config.network.scaffolding)
            n_scaff = len(_pack)
            n_weights = len(param_specs) - n_scaff
            weights = _decode_nn_weights(best_individual[:n_weights], param_specs[:n_weights])
            write_nn_json(
                weights, config.network, save_dir / "best_model.json", input_mask=config.network.input_mask, output_param=config.network.output_parameterization
            )
            if cwd is not None:
                nn_path = Path(cwd) / config.sim.nn_param_file
                write_nn_json(weights, config.network, nn_path, input_mask=config.network.input_mask, output_param=config.network.output_parameterization)
            if n_scaff > 0:
                scaff_params = decode_normalized(best_individual[n_weights:], list(_pack))
                for s in _pack:
                    if s.is_integer and s.name in scaff_params:
                        scaff_params[s.name] = int(round(scaff_params[s.name]))
                with open(save_dir / "best_params.json", "w") as fp:
                    json.dump(scaff_params, fp, indent=2)
        else:
            params = decode_normalized(best_individual, param_specs)
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(params, fp, indent=2)

    # Auto-prune older checkpoints when retention is configured.
    _prune_old_checkpoints(save_dir, config.checkpoints.keep_last)


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


def warm_start_algorithm(
    algorithm: Any,
    problem: Any,
    pop: Population,
    *,
    seed: int | None = None,
    n_iter: int = 1,
) -> None:
    """Seed a pymoo Algorithm with a pre-evaluated population.

    pymoo's `algorithm.setup(problem, pop=init_pop)` writes the pop onto the
    algorithm but does NOT prevent the first `algorithm.next()` from calling
    `_initialize()` (which wipes `self.pop`) and `_initialize_infill()` (which
    resamples via LHS). The seeded population is then silently discarded.

    This helper does the work `Algorithm.advance(infills=…)` does on the
    first call, but with our pre-evaluated pop instead of LHS infills:
    sets `self.pop`, runs `_initialize_advance` (which for PSO initializes
    velocity + sets `self.particles = self.pop`), flips `is_initialized`,
    and computes `self.opt`.

    Use this in place of `algorithm.setup(problem, pop=init_pop)` whenever
    the seeded chromosomes must survive into gen 0.
    """
    import time as _time  # noqa: PLC0415

    algorithm.setup(problem, seed=seed)
    algorithm.pop = pop
    algorithm.n_iter = n_iter
    # pymoo's `_initialize()` is what normally stamps `start_time`. We
    # bypass it here, so set it explicitly — otherwise `algorithm.result()`
    # (called by `advance()` when internal termination fires) crashes with
    # `unsupported operand type(s) for -: 'float' and 'NoneType'` on
    # `res.end_time - res.start_time`.
    algorithm.start_time = _time.time()
    algorithm._initialize_advance(infills=pop)
    algorithm.is_initialized = True
    algorithm._set_optimum()


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

    from aerocapture.training.optimizer import _VALID_SEED_STRATEGIES

    if config.optimizer.seed_strategy not in _VALID_SEED_STRATEGIES:
        msg = (
            f"config.optimizer.seed_strategy must be one of {_VALID_SEED_STRATEGIES}, "
            f"got {config.optimizer.seed_strategy!r}. Did the TOML [optimizer] section "
            f"set it, or did you pass a TrainingConfig without overriding the default?"
        )
        raise ValueError(msg)

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
    cost_kwargs: dict[str, Any] = {}
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
            "cost_transform": str(cost_cfg.get("cost_transform", "linear")),
        }

    # Seed strategy: three mutually exclusive training seed paths.
    #   fixed    -- deterministic [mc_seed + i]; seeds never change.
    #   rotating -- fresh random seeds drawn each generation (handled in loop body).
    #   adaptive -- bootstrap random + curated-CDF refreshes (SeedCurator).
    seed_curator: SeedCurator | None = None
    strategy = config.optimizer.seed_strategy
    if strategy == "adaptive":
        seed_curator = SeedCurator(
            sample_size=config.optimizer.curation_sample_size,
            n_bins=config.optimizer.training_n_sims,
            excluded_seeds=set(),  # populated once val/final-eval sets are computed
            rng=rng,
        )

    # Build parameter specifications
    from aerocapture.training.param_spaces import PARAM_SPACES

    if config.guidance_type == "neural_network":
        if config.network.architecture is not None:
            from pydantic import TypeAdapter

            from aerocapture.training.rl.schemas import LayerSpec

            specs_adapter = TypeAdapter(list[LayerSpec])
            validated = specs_adapter.validate_python(config.network.architecture)
            # bound_multiplier=2.0 matches create_nn_initial_population's Phase 1
            # convention AND build_initial_population_for_v2 below. Keeping them
            # aligned avoids ~49% boundary-saturation on the initial PSO population.
            # When warm-start is on, use the wider [warm_start] bound_multiplier
            # (default 4.0) so the search space envelops the warm-started chromosome's
            # post-supervised-training drift past Xavier bounds.
            warm_start_active = bool(config.network.warm_start_from) or config.warm_start.enabled
            bound_mult = config.warm_start.bound_multiplier if warm_start_active else 2.0
            param_specs = nn_param_specs_from_v2(validated, bound_multiplier=bound_mult)
        else:
            param_specs = nn_param_specs_from_architecture(
                config.network.layer_sizes,
                config.network.activations,
            )

        if config.network.scaffolding != "off":
            if config.network.architecture is None:
                msg = "scaffolding != 'off' requires v2 [[network.architecture]]; v1 layer_sizes/activations is not supported. Convert your config."
                raise ValueError(msg)
            from aerocapture.training.param_spaces import active_scaffolding_specs

            param_specs = [*param_specs, *active_scaffolding_specs(config.network.scaffolding)]
            if verbose:
                if config.network.scaffolding == "live":
                    print("scaffolding optimization: LIVE — 3 params (nav density filter ×2, command shaping); no FTC dependency")
                else:  # full
                    print("scaffolding optimization: FULL — 17 params, seeded from training_output/ftc/best_params.json")
        else:
            if verbose and config.network.architecture is not None:
                print("scaffolding optimization: OFF — NN weights only")
    elif config.guidance_type == "piecewise_constant":
        from aerocapture.training.param_spaces import make_piecewise_constant_specs

        n_segments = _resolve_piecewise_n_segments(_toml)
        param_specs = make_piecewise_constant_specs(n_segments)
        if verbose:
            pc = _toml.get("guidance", {}).get("piecewise_constant", {})
            e_min = float(pc.get("energy_min", -6.0))
            e_max = float(pc.get("energy_max", 5.0))
            seg_width = (e_max - e_min) / n_segments if n_segments > 0 else float("nan")
            initial = pc.get("bank_angles")
            init_label = f"seeded from TOML bank_angles ({len(initial)} values)" if initial is not None else "GA-initialized (no TOML bank_angles)"
            n_shaping = len(param_specs) - n_segments
            print(
                f"piecewise_constant: {n_segments} segments over E in [{e_min:.2f}, {e_max:.2f}] MJ/kg "
                f"(width {seg_width:.3f} MJ/kg), {init_label}; "
                f"chromosome = {n_segments} bank + {n_shaping} shaping = {len(param_specs)} params"
            )
    else:
        param_specs = PARAM_SPACES[config.guidance_type]

    n_params = len(param_specs)

    # Compute config hash for experiment grouping
    config_hash = hashlib.sha256(repr(config).encode()).hexdigest()[:12]

    # Try resuming from checkpoint. The islands path manages its own .npz-only
    # resume inside `_train_islands`; skip the single-algorithm `load_checkpoint`
    # here so a stale single-algo `checkpoint.json` left in a shared save_dir
    # can't bump `n_gen` a second time (it would also be bumped in _train_islands).
    resumed = None
    if resume_dir is not None and config.optimizer.algorithm != "islands":
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

    # Try loading existing NN weights for population seeding. Only meaningful
    # under the v1 dense-only init path (`create_nn_initial_population`,
    # which is the only consumer of `seed_weights`). v2 architectures use
    # `init_v2_population` and discard seed_weights, AND `load_base_network`
    # only knows the v1 JSON layout — calling it on a v2 best_model.json
    # raises "list indices must be integers or slices, not str" because the
    # v2 `architecture` key is a list of layer dicts, not a dict-with-"layers".
    seed_weights = None
    is_v1_nn = config.guidance_type == "neural_network" and config.network.architecture is None
    if is_v1_nn and resumed is None and not from_scratch:
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
        _check_resume_chromosome_shape(pop_array, expected_n_params=len(param_specs))
    else:
        if config.guidance_type == "neural_network" and config.network.architecture is None:
            # v1 dense-only NN: existing activation-aware Xavier/He/LeCun init.
            pop_array = create_nn_initial_population(
                config.network.layer_sizes,
                config.network.activations,
                config.optimizer.n_pop,
                rng,
                seed_weights=seed_weights,
            )
        elif config.guidance_type == "neural_network" and config.network.architecture is not None:
            # v2 heterogeneous NN: per-layer activation-aware init with LSTM forget-bias-1.
            scaffolding_slab = None
            if config.network.scaffolding != "off":
                from aerocapture.training.param_spaces import active_scaffolding_specs

                _slab_pack = active_scaffolding_specs(config.network.scaffolding)
                if config.network.scaffolding == "full":
                    # full pack is seeded from FTC's best_params.json (FTC's bounds,
                    # FTC's optimum). warm_start_from is independent -- it points at
                    # a behavioural-cloning source, not a scaffolding source.
                    scaffolding_slab = build_scaffolding_initial_slab(
                        "training_output/ftc/best_params.json",
                        list(_slab_pack),
                        config.optimizer.n_pop,
                        rng,
                        jitter=config.warm_start.jitter,
                    )
                else:
                    # live pack: 3 params seeded from their defaults, no FTC dep.
                    scaffolding_slab = build_default_scaffolding_slab(
                        list(_slab_pack),
                        config.optimizer.n_pop,
                        rng,
                        jitter=config.warm_start.jitter,
                    )

            if warm_start_active:
                from aerocapture.training.warm_start import WARM_START_SEED_OFFSET, build_warm_start_chromosome

                # Build the periodic in-training eval callback. When
                # `[warm_start] eval_interval > 0`, this fires every N epochs
                # AND on the final epoch (see _chunked_bptt_train) -- writes
                # the current policy to a temp NN JSON, runs MC on BOTH the
                # warm-start seed pool and the reserved validation pool, and
                # prints two detailed stats blocks to stdout. Only built when
                # the user actually opted in to avoid pointless MC work.
                warm_eval_callback = None
                if config.warm_start.eval_interval > 0 and val_seeds is not None:
                    warm_seeds_for_eval = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, config.warm_start.n_warm_seeds)
                    warm_eval_callback = _make_warm_start_eval_callback(
                        problem=problem,
                        config=config,
                        warm_seeds=warm_seeds_for_eval,
                        val_seeds=val_seeds,
                    )

                warm_chromo, warm_weight_specs = build_warm_start_chromosome(
                    cfg=config,
                    base_mc_seed=base_mc_seed,
                    eval_callback=warm_eval_callback,
                )
                from aerocapture.training.param_spaces import active_scaffolding_specs

                n_scaff = len(active_scaffolding_specs(config.network.scaffolding))
                n_weights = len(warm_chromo) - n_scaff
                # Propagate the warm-start bounds back into param_specs so PSO/GA/DE
                # decode chromosomes under the same bounds they were encoded with.
                # ParamSpec is frozen; replace entries in-place so Problem (which
                # holds a reference to the same list) sees the new bounds at decode
                # time. Length is preserved -- only the NN-weight slab [0..n_weights)
                # is rewritten; scaffolding tail stays untouched.
                assert len(warm_weight_specs) == n_weights, f"warm_weight_specs length ({len(warm_weight_specs)}) != n_weights ({n_weights})"
                for j in range(n_weights):
                    param_specs[j] = warm_weight_specs[j]
                pop_array = _seed_initial_population(
                    algorithm_name=config.optimizer.algorithm,
                    chromosome=warm_chromo,
                    n_pop=config.optimizer.n_pop,
                    jitter=config.warm_start.jitter,
                    rng=rng,
                    n_weights=n_weights,
                )
                if scaffolding_slab is not None:
                    pop_array[:, n_weights:] = scaffolding_slab
                    # Restore row 0's scaffolding tail so the warm-start
                    # chromosome is present un-jittered in the initial
                    # population. The slab overwrite above replaced row 0's tail
                    # with jittered values ("full": FTC-seeded via
                    # build_scaffolding_initial_slab; "live": default-seeded via
                    # build_default_scaffolding_slab). The warm-start chromosome
                    # encodes the un-jittered center directly, so we copy it back.
                    pop_array[0, n_weights:] = warm_chromo[n_weights:]

                # Gen-0 validation baseline: evaluate the bare warm-started
                # chromosome on the RESERVED VALIDATION seed pool (same seeds
                # the validation gate uses) so the persisted rms/mean/p95
                # metrics are directly comparable to the `Gen N validation:`
                # line later printed by the validation gate. Best-effort:
                # failure here must not block training.
                from aerocapture.training._warm_start_baseline import write_gen0_baseline
                from aerocapture.training.report import compute_eval_summary, format_eval_summary

                try:
                    if val_seeds is None:
                        raise RuntimeError("val_seeds not initialized; gen-0 baseline requires the validation pool")
                    # Single MC pass on val_seeds returning both per-seed costs
                    # AND the (n, 52) final_records so we can derive DV / apo /
                    # peri / heat-flux statistics without re-running.
                    baseline_costs, baseline_records = problem.evaluate_individual_records_per_seed(warm_chromo, val_seeds)
                    eval_summary = compute_eval_summary(baseline_records, len(val_seeds), problem.cost_kwargs)
                    baseline_path = write_gen0_baseline(
                        save_dir=Path(config.save_dir),
                        costs=baseline_costs,
                        capture_rate=eval_summary["capture_rate"],
                        n_sims=len(val_seeds),
                    )
                    # Persist the structured eval summary so the warm-start
                    # report PDF can embed it (and CLI re-render works).
                    (Path(config.save_dir) / "warm_start_eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
                    # User-facing block: mirrors the end-of-training final-eval
                    # summary so users can compare like-for-like.
                    if verbose:
                        print()
                        for line in format_eval_summary(eval_summary, indent="    "):
                            print(f"  {line}" if not line.startswith(" ") else line)
                        baseline = json.loads(baseline_path.read_text())
                        print(f"  [warm_start] gen-0 baseline cost (val seeds): rms={baseline['rms_cost']:.4e} mean={baseline['mean_cost']:.4e}")
                except Exception as e:
                    # Best-effort: failure here must not block training, but the
                    # error is always logged so it does not silently mask real
                    # bugs in problem.evaluate_individual_records_per_seed /
                    # write_gen0_baseline / compute_eval_summary.
                    print(f"  [warm_start] WARNING: gen-0 baseline write failed: {type(e).__name__}: {e}")

                # Trajectory comparison: supervisor vs warm-started NN on both
                # training and validation pools. Runs ~2*(n_warm_seeds +
                # validation_n_sims) MC sims (e.g. ~12k for n_warm_seeds=5000,
                # validation_n_sims=1000) and renders 20 SVGs into the report
                # dir. Best-effort: failure here must not block training, but
                # the warm-start PDF will just omit the comparison section.
                try:
                    from aerocapture.training.warm_start_compare import render_trajectory_comparison

                    render_trajectory_comparison(
                        cfg=config,
                        base_mc_seed=base_mc_seed,
                        warm_chromo=warm_chromo,
                        nn_weight_specs=warm_weight_specs,
                    )
                except Exception as e:
                    print(f"  [warm_start] WARNING: trajectory comparison failed: {type(e).__name__}: {e}")

                # Intermediate warm-start report: charts + Typst PDF summarizing
                # supervised MSE convergence, supervisor selection, search-space
                # bounds, the gen-0 validation baseline + eval summary, and the
                # supervisor-vs-NN trajectory comparison panels (if rendered).
                # Best-effort.
                try:
                    from aerocapture.training.warm_start_report import render_report

                    pdf = render_report(Path(config.save_dir))
                    if verbose and pdf is not None:
                        print(f"  [warm_start] report: {pdf}")
                except Exception as e:
                    print(f"  [warm_start] WARNING: report rendering failed: {type(e).__name__}: {e}")
            else:
                pop_array = build_initial_population_for_v2(
                    config.network.architecture,
                    config.optimizer.n_pop,
                    bound_multiplier=bound_mult,
                    rng=rng,
                    param_specs=param_specs,
                    scaffolding_slab=scaffolding_slab,
                )
        else:
            # Non-NN scheme: uniform [0, 1] with ParamSpec-defaults seeding.
            pop_array = create_initial_population(
                param_specs,
                config.optimizer.n_pop,
                rng,
            )
        pop_costs = None  # Will be evaluated by pymoo

    # CMA-ES + warm-start: shrink the initial step size. Applied unconditionally
    # (independent of resume vs fresh start) so the checkpointed CMA-ES sigma
    # in `algorithm.next()` consistently reflects the warm-start tunable.
    # Gated on guidance_type == "neural_network" because [warm_start] is only
    # meaningful for NN training; a non-NN config that picked up [warm_start]
    # via base inheritance must NOT have its sigma0 silently overridden.
    warm_start_active = bool(config.network.warm_start_from) or config.warm_start.enabled
    if config.guidance_type == "neural_network" and warm_start_active and config.optimizer.algorithm == "cma_es":
        config.optimizer.cma_es.sigma0 = config.warm_start.cmaes_sigma0

    # Islands dispatch: return early so the single-algorithm path below is untouched.
    if config.optimizer.algorithm == "islands":
        return _train_islands(
            config=config,
            cwd=cwd,
            save_dir=save_dir,
            problem=problem,
            param_specs=param_specs,
            n_params=n_params,
            pop_array=pop_array,
            pop_costs=pop_costs,
            val_seeds=val_seeds,
            base_mc_seed=base_mc_seed,
            excluded_seeds=excluded_seeds,
            rng=rng,
            seed_curator=seed_curator,
            strategy=strategy,
            display=display,
            verbose=verbose,
            start_gen=start_gen,
            config_hash=config_hash,
            checkpoint_interval=checkpoint_interval,
            toml_abs_path=toml_abs_path,
        )

    # Set up algorithm
    algorithm = create_algorithm(config.optimizer, n_params=n_params)
    if verbose:
        opt = config.optimizer
        print(f"  Algorithm: {type(algorithm).__name__} ({opt.algorithm}), n_params={n_params}, n_pop={opt.n_pop}, n_gen={opt.n_gen}")
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

    # Inject initial population into pymoo. NOTE: `setup(pop=…)` alone is
    # insufficient — pymoo's first `next()` would call `_initialize()` and
    # `_initialize_infill()`, wiping the seeded pop with an LHS sample.
    # `warm_start_algorithm` flips `is_initialized` and runs
    # `_initialize_advance` so the seeded chromosomes survive into gen 0
    # (and PSO's particles/V get initialized against them).
    initial_pop = Population.new("X", pop_array)
    if pop_costs is not None:
        initial_pop.set("F", pop_costs.reshape(-1, 1))
    else:
        Evaluator().eval(problem, initial_pop)
        pop_costs = initial_pop.get("F").flatten()

    warm_start_algorithm(algorithm, problem, initial_pop)

    # Initialize best from the first population eval -- but ONLY on a fresh
    # start. On resume, `best_overall_{cost,individual}` are the checkpointed
    # validated best; overwriting them with the current population's argmin
    # would be wrong because the two training costs were computed under
    # different seed lists (adaptive/rotating seeds evolve across gens), so
    # the `<` comparison is meaningless. Swapping here would silently promote
    # an un-validated individual and make the re-validation at line 539 run
    # on the wrong chromosome -- drifting the "Best val" RMS and corrupting
    # the best_model.json that the final eval reads.
    if best_overall_individual is None:
        init_best_idx = int(np.argmin(pop_costs))
        best_overall_cost = float(pop_costs[init_best_idx])
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
            # Validate the starting best: gen-0 individual on fresh starts,
            # the checkpointed best on resume. Re-validating on resume keeps
            # the TUI's "Best val" and stagnation counter honest (val_seeds
            # are deterministic, so RMS is reproducible).
            if val_seeds is not None and best_overall_individual is not None:
                init_val_costs, init_val_records = problem.evaluate_individual_records_per_seed(best_overall_individual, val_seeds)
                best_val_cost = float(np.sqrt(np.mean(init_val_costs**2)))
                last_validated_individual = best_overall_individual.copy()
                init_val_metrics, init_val_summary = _build_validation_payload(
                    init_val_costs,
                    init_val_records,
                    len(val_seeds),
                    problem.cost_kwargs,
                )
                logger.log_generation(
                    start_gen,
                    pop_array,
                    pop_costs if pop_costs is not None else np.full(config.optimizer.n_pop, np.inf),
                    best_overall_individual,
                    decode_fn,
                    validation=init_val_metrics,
                    validation_summary=init_val_summary,
                    improved=True,
                )
                display.update(logger, current_run=0)
                if verbose:
                    label = f"Gen {start_gen}" if start_gen > 0 else "Gen 0"
                    print(f"  {label} validation: mean={best_val_cost:.4e} cap={init_val_metrics['capture_rate']:.0%}")

            for gen in range(start_gen, config.optimizer.n_gen):
                gen_wall_start = time.perf_counter()

                seeds_changed_this_gen = _apply_seed_strategy(
                    strategy=strategy,
                    rng=rng,
                    n_sims=config.optimizer.training_n_sims,
                    excluded_seeds=excluded_seeds,
                    problem=problem,
                    seed_curator=seed_curator,
                    pending_seed_change=pending_seed_change,
                )
                pending_seed_change = False

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
                validation_summary: dict | None = None
                validated_improvement = False
                if val_seeds is not None and new_gen_best:
                    val_costs, val_records = problem.evaluate_individual_records_per_seed(gen_best_individual, val_seeds)
                    validation_metrics, validation_summary = _build_validation_payload(
                        val_costs,
                        val_records,
                        len(val_seeds),
                        problem.cost_kwargs,
                    )
                    val_rms = validation_metrics["rms_cost"]
                    last_validated_individual = gen_best_individual
                    if val_rms < best_val_cost:
                        best_val_cost = val_rms
                        best_overall_individual = gen_best_individual
                        best_overall_cost = gen_best_cost
                        validated_improvement = True

                # Curation trigger: on validated promotion OR periodic fallback.
                # next gen's pre-next re-eval picks up the new seeds. Default-bind
                # X/costs so the provider closes over THIS gen's pop (it's called
                # synchronously, but binding also silences the loop-var lint).
                def _single_top_k(
                    k: int,
                    X: npt.NDArray[np.float64] = X,
                    costs: npt.NDArray[np.float64] = costs,
                ) -> npt.NDArray[np.float64]:
                    return X[np.argsort(costs)[: min(k, len(costs))]]

                if _maybe_curate(
                    seed_curator=seed_curator,
                    problem=problem,
                    gen=gen,
                    seed_pool_interval=config.optimizer.seed_pool_interval,
                    curation_top_k=config.optimizer.curation_top_k,
                    promoted=validated_improvement,
                    top_k_provider=_single_top_k,
                ):
                    pending_seed_change = True

                # Common logging
                gen_best_costs.append(best_overall_cost)

                # Compute per-layer weight stats for NN (dense-only; v2 heterogeneous
                # architectures skip -- stats are TUI decoration, not load-bearing).
                ws = None
                if config.guidance_type == "neural_network" and best_overall_individual is not None and config.network.architecture is None:
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
                    validation_summary=validation_summary,
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


def _train_islands(
    *,
    config: TrainingConfig,
    cwd: str | Path | None,
    save_dir: Path,
    problem: AerocaptureProblem,
    param_specs: list[ParamSpec],
    n_params: int,
    pop_array: npt.NDArray[np.float64],
    pop_costs: npt.NDArray[np.float64] | None,
    val_seeds: list[int] | None,
    base_mc_seed: int,
    excluded_seeds: set[int],
    rng: np.random.Generator,
    seed_curator: SeedCurator | None,
    strategy: str,
    display: Any,
    verbose: bool,
    start_gen: int,
    config_hash: str,
    checkpoint_interval: int,
    toml_abs_path: str,
) -> dict[str, Any]:
    """Outer loop for the 3-island PSO/GA/DE trainer.

    Mirrors the single-algorithm path in train() but drives an IslandModel.
    Per-island JSONL records (3 per gen) are written via TrainingLogger with
    the `island_name` field set.
    """
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds  # noqa: PLC0415
    from aerocapture.training.island_model import (  # noqa: PLC0415
        IslandModel,
        compute_migration_origin_stats,
        summarize_latest_migration,
    )
    from aerocapture.training.logger import TrainingLogger  # noqa: PLC0415

    # Reserved final-eval seeds (disjoint from training + validation pools).
    # Match single-algorithm path: max(validation_n_sims, 10000).
    final_eval_n = max(config.optimizer.validation_n_sims, 10000)
    final_eval_seeds = make_reserved_seeds(
        base_mc_seed,
        FINAL_EVAL_SEED_OFFSET,
        final_eval_n,
    )

    # Keep training-seed draws (rotating / adaptive) disjoint from the reserved
    # final-eval and validation pools. train() only unions these into
    # excluded_seeds when validation_n_sims > 0, so do it here unconditionally.
    excluded_seeds = excluded_seeds | set(final_eval_seeds)
    if val_seeds:
        excluded_seeds = excluded_seeds | set(val_seeds)

    island_model = IslandModel(
        config=config.optimizer,
        problem=problem,
        n_params=n_params,
        validation_seeds=val_seeds or [],
        final_eval_seeds=final_eval_seeds,
        base_mc_seed=base_mc_seed,
        rng=rng,
    )

    # Probe for a resumable islands checkpoint FIRST, so the cold-start
    # population evaluation below can be skipped on resume (from_checkpoint
    # overwrites every island's pop, so evaluating the fresh pop would be
    # wasted MC work). Pick the LATEST checkpoint that actually carries the
    # islands v2 marker — not just the lexicographically-last .npz — so a
    # foreign single-algorithm .npz sharing the directory (which has no
    # "version" key) can't shadow a valid islands checkpoint.
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
    # Each island gets the same starting chromosome but its algorithm's own
    # internal state (e.g. PSO velocity init) is fresh. `warm_start_algorithm`
    # is used instead of `setup(pop=…)` because pymoo would otherwise wipe the
    # seeded pop via `_initialize()` on first `next()`; it also binds the
    # problem via setup(), which is required even on resume.
    if pop_costs is None:
        if resume_ckpt is not None:
            # Resuming: from_checkpoint overwrites pop.F immediately, so don't
            # spend a full cold-start MC batch we're about to discard.
            pop_costs = np.zeros(pop_array.shape[0], dtype=np.float64)
        else:
            # Evaluate once and share F across all islands (chromosomes are
            # identical, so the costs are too — three Evaluator passes would
            # triple the cold-start budget).
            shared_eval_pop = Population.new("X", pop_array.copy())
            Evaluator().eval(problem, shared_eval_pop)
            pop_costs = shared_eval_pop.get("F").flatten()
    for island in island_model.islands:
        init_pop = Population.new("X", pop_array.copy())
        init_pop.set("F", pop_costs.reshape(-1, 1).copy())
        warm_start_algorithm(island.algorithm, problem, init_pop)

    if resume_ckpt is not None:
        resumed_gen, resumed_curator_state = island_model.from_checkpoint(resume_ckpt)
        start_gen = resumed_gen + 1
        # Mirror the single-algorithm convention: `--n-gen N` after resume
        # means "N additional gens", so bump n_gen by the resumed
        # generation count. Done inside `_train_islands` because the outer
        # `train()` bump only runs when `load_checkpoint()` returns a
        # non-None dict — which it can't for an npz-only islands checkpoint.
        config.optimizer.n_gen += resumed_gen + 1
        if resumed_curator_state is not None and seed_curator is not None:
            from aerocapture.training.seed_curator import SeedCurator as _SeedCurator  # noqa: PLC0415

            seed_curator = _SeedCurator.from_dict(
                resumed_curator_state,
                excluded_seeds=seed_curator.excluded_seeds,
                rng=seed_curator.rng,
            )
            # Push the restored curated seed list into the problem so the
            # first post-resume gen evaluates against the right seeds. The
            # in-loop "adaptive bootstrap" branch is gated on `seed_list is
            # None`, so without this push we would silently evaluate against
            # the pre-islands-dispatch seed state.
            if seed_curator.seed_list is not None:
                problem.update_seeds(seed_curator.seed_list)
        if verbose:
            print(f"  Resumed islands from gen {resumed_gen}, continuing from {start_gen}")

    # Decode function for logger (NN bypasses, analytic schemes use decode_normalized).
    decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None = None
    if config.guidance_type != "neural_network":

        def _decode(x: npt.NDArray[np.float64]) -> dict[str, float]:
            return decode_normalized(x, param_specs)

        decode_fn = _decode

    logger = TrainingLogger(
        scheme=config.guidance_type,
        run=0,
        output_dir=save_dir,
        config_hash=config_hash,
    )

    display.set_start_gen(start_gen)
    pending_seed_change = False
    interrupted = False
    gen = start_gen

    with display:
        try:
            for gen in range(start_gen, config.optimizer.n_gen):
                seeds_changed_this_gen = _apply_seed_strategy(
                    strategy=strategy,
                    rng=rng,
                    n_sims=config.optimizer.training_n_sims,
                    excluded_seeds=excluded_seeds,
                    problem=problem,
                    seed_curator=seed_curator,
                    pending_seed_change=pending_seed_change,
                )
                pending_seed_change = False

                if seeds_changed_this_gen:
                    island_model.re_evaluate_all_populations()

                # Advance + (maybe) migrate.
                events = island_model.step(current_gen=gen)

                # Validate (identity-trigger per island).
                if val_seeds:
                    val_records = island_model.validate_each(current_gen=gen)
                else:
                    # No validation seeds — there is no validation gate to
                    # promote best_overall_*, so promote each island's finite
                    # training argmin directly. Without this no island ever sets
                    # best_overall_individual, final_eval() returns [], and the
                    # run produces zero artifacts while deleting any prior
                    # best_model.json. final_eval re-ranks on the disjoint
                    # final-eval pool, so the cross-gen incomparability is bounded
                    # to which gen's argmin seeds each island's candidate.
                    val_records = []
                    for i in island_model.islands:
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
                        val_records.append(
                            {
                                "island": i.name,
                                "validated": False,
                                "promoted": False,
                                "argmin_train_cost": argmin_cost,
                                "stagnation": i.stagnation_counter,
                            }
                        )

                # Adaptive seed curation: probe a top-K slice pooled across all
                # 3 islands (vs the single-algo per-gen argmin slice).
                if _maybe_curate(
                    seed_curator=seed_curator,
                    problem=problem,
                    gen=gen,
                    seed_pool_interval=config.optimizer.seed_pool_interval,
                    curation_top_k=config.optimizer.curation_top_k,
                    promoted=any(r.get("promoted") for r in val_records),
                    top_k_provider=island_model.pool_top_k_X,
                ):
                    pending_seed_change = True

                # Per-island JSONL records.
                for island, val_rec in zip(island_model.islands, val_records, strict=True):
                    X = island.algorithm.pop.get("X")
                    F = island.algorithm.pop.get("F").flatten()
                    validation_dict: dict | None = None
                    if val_rec["validated"]:
                        validation_dict = {
                            "rms_cost": val_rec["val_rms"],
                            "mean_cost": val_rec["val_mean"],
                            "p95_cost": val_rec["val_p95"],
                            "capture_rate": val_rec["val_capture_rate"],
                            "n_sims": len(val_seeds) if val_seeds else 0,
                        }
                    logger.log_generation(
                        generation=gen,
                        population=X,
                        costs=F,
                        best_individual=island.best_overall_individual,
                        decode_fn=decode_fn,
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
                        # Sticky: shows the last validated dashboard even on
                        # gens where this island didn't re-validate.
                        "val_summary": island.latest_val_summary,
                    }
                    for island, val_rec in zip(
                        island_model.islands,
                        val_records,
                        strict=True,
                    )
                }
                island_records["_gen"] = gen
                island_records["_n_gen"] = config.optimizer.n_gen
                island_records["_total_migrations"] = len(island_model.migration_log)

                # Migration summary: best/worst migrant per destination from THIS
                # gen's events (if any). Only (re)computed on migration gens — the
                # cached snapshot is reused on the other ~(k_period-1)/k_period
                # gens so the per-gen display refresh doesn't re-scan the full
                # migration_log every generation.
                if events:
                    island_model.latest_migration_summary = summarize_latest_migration(events)
                    island_model.latest_migration_gen = gen
                    island_model.origin_stats_cache = compute_migration_origin_stats(island_model.migration_log)

                island_records["_latest_migration_summary"] = island_model.latest_migration_summary
                island_records["_latest_migration_gen"] = island_model.latest_migration_gen
                island_records["_origin_stats"] = island_model.origin_stats_cache

                display.update(logger, current_run=0, island_records=island_records)

                if (gen + 1) % checkpoint_interval == 0 or gen == config.optimizer.n_gen - 1:
                    island_model.checkpoint(
                        save_dir / f"checkpoint_g{gen:05d}.npz",
                        generation=gen,
                        seed_curator_state=seed_curator.to_dict() if seed_curator is not None else None,
                    )
                    _prune_old_checkpoints(save_dir, config.checkpoints.keep_last)

        except KeyboardInterrupt:
            interrupted = True
            island_model.checkpoint(
                save_dir / f"checkpoint_g{gen:05d}.npz",
                generation=gen,
                seed_curator_state=seed_curator.to_dict() if seed_curator is not None else None,
            )
            _prune_old_checkpoints(save_dir, config.checkpoints.keep_last)
            if verbose:
                print(f"\n  Interrupted at gen {gen}; checkpoint saved.")

    # Final eval + winner selection.
    results = island_model.final_eval()
    if not results:
        if verbose:
            print("  No island had a validated best — skipping final-eval / artifact write.")
        # Remove any best_model.json / best_params.json left over from a
        # previous experiment so downstream tooling (compare_guidance,
        # report.py, deploy paths) doesn't silently consume a stale model.
        for stale in (
            save_dir / "best_model.json",
            save_dir / "best_params.json",
            Path(cwd or ".") / config.sim.nn_param_file if config.guidance_type == "neural_network" else None,
        ):
            if stale is not None and stale.exists():
                stale.unlink()
                if verbose:
                    print(f"  Removed stale {stale}")
        logger.close()
        return {
            "best_cost": float("inf"),
            "best_individual": None,
            "cost_history": [],
            "interrupted": interrupted,
            "corridor_acc": None,
            "param_specs": param_specs,
            "winner": None,
            "results": [],
            "migration_log": island_model.migration_log,
        }

    winner = results[0]
    if verbose:
        print(
            f"  Winner: {winner['island']} rms={winner['rms']:.4e} cap={winner['capture_rate']:.0%}",
        )

    _write_winner_artifacts(
        winner=winner,
        config=config,
        save_dir=save_dir,
        param_specs=param_specs,
    )

    logger.close()
    return {
        "best_cost": float(winner["rms"]),
        "best_individual": winner["X"],
        "cost_history": [],
        "interrupted": interrupted,
        "corridor_acc": None,
        "param_specs": param_specs,
        "winner": winner,
        "results": results,
        "migration_log": island_model.migration_log,
    }


def _write_winner_artifacts(
    *,
    winner: dict[str, Any],
    config: TrainingConfig,
    save_dir: Path,
    param_specs: list[ParamSpec],
) -> None:
    """Write best_model.json / best_params.json from the winning island's chromosome.

    Writes only to save_dir. main() handles the deploy-path write to cwd.
    """
    best_individual = winner["X"]

    if config.guidance_type == "neural_network":
        from aerocapture.training.param_spaces import active_scaffolding_specs

        _pack = active_scaffolding_specs(config.network.scaffolding)
        n_scaff = len(_pack)
        n_weights = len(param_specs) - n_scaff
        weights = _decode_nn_weights(
            best_individual[:n_weights],
            param_specs[:n_weights],
        )
        write_nn_json(
            weights,
            config.network,
            save_dir / "best_model.json",
            input_mask=config.network.input_mask,
            output_param=config.network.output_parameterization,
        )
        if n_scaff > 0:
            scaff_params = decode_normalized(
                best_individual[n_weights:],
                list(_pack),
            )
            for s in _pack:
                if s.is_integer and s.name in scaff_params:
                    scaff_params[s.name] = int(round(scaff_params[s.name]))
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(scaff_params, fp, indent=2)
    else:
        params = decode_normalized(best_individual, param_specs)
        with open(save_dir / "best_params.json", "w") as fp:
            json.dump(params, fp, indent=2)


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
    n_segments = sum(1 for s in param_specs if s.name.startswith("bank_angle_"))
    sentinel_overrides: list[dict[str, object]] = []
    for bank in _SENTINEL_BANK_ANGLES:
        ovr_s: dict[str, object] = {f"guidance.{section}.bank_angle_{i}": float(bank) for i in range(n_segments)}
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
    if "architecture" in _net:
        # v2 heterogeneous arch (list of per-layer dicts, e.g. dense + gru + dense).
        cfg.network.architecture = list(_net["architecture"])
    if "layer_sizes" in _net:
        cfg.network.layer_sizes = _net["layer_sizes"]
    if "activations" in _net:
        cfg.network.activations = _net["activations"]
    if "input_mask" in _net:
        cfg.network.input_mask = _net["input_mask"]
    _gnn = _toml_data.get("guidance", {}).get("neural_network", {})
    if "scaffolding" in _gnn:
        cfg.network.scaffolding = str(_gnn["scaffolding"])
    if "output_parameterization" in _gnn:
        cfg.network.output_parameterization = str(_gnn["output_parameterization"])
    if "scaled_pi_n" in _gnn:
        cfg.network.scaled_pi_n = float(_gnn["scaled_pi_n"])
    if "delta_max" in _gnn:
        cfg.network.delta_max = float(_gnn["delta_max"])
    if "warm_start_from" in _gnn:
        cfg.network.warm_start_from = str(_gnn["warm_start_from"])
    if cfg.network.warm_start_from is not None:
        warm_path = Path(cfg.network.warm_start_from)
        if not warm_path.exists():
            print(f"ERROR: warm_start_from='{warm_path}' does not exist")
            raise SystemExit(1)
    if "warm_start" in _toml_data:
        cfg.warm_start = WarmStartConfig.from_dict(_toml_data["warm_start"])

    # `[checkpoints]` block: optional disk-retention policy. `keep_last = N`
    # auto-prunes older `checkpoint_g*.{json,npz}` pairs after each save,
    # keeping only the N most recent. The JSONL log + best_* artifacts are
    # untouched.
    if "checkpoints" in _toml_data:
        _ckpt = _toml_data["checkpoints"]
        known_keys = {"keep_last"}
        unknown = set(_ckpt.keys()) - known_keys
        if unknown:
            print(f"ERROR: unknown [checkpoints] keys: {sorted(unknown)}")
            raise SystemExit(1)
        if "keep_last" in _ckpt:
            kl_raw = _ckpt["keep_last"]
            if kl_raw is not None and not isinstance(kl_raw, int):
                print(f"ERROR: [checkpoints] keep_last must be an int or null, got {type(kl_raw).__name__}")
                raise SystemExit(1)
            if isinstance(kl_raw, int) and kl_raw < 1:
                print(f"ERROR: [checkpoints] keep_last must be >= 1, got {kl_raw}")
                raise SystemExit(1)
            cfg.checkpoints.keep_last = kl_raw

    # Warm-start contract: supervised targets are the post-lateral, pre-shaper
    # SIGNED bank command (tick.rs captures `guidance_out.pre_shaper_signed`).
    # warm_start.py collapses the sign to magnitude only when mode = "magnitude_only".
    # Two matched setups:
    #   - mode = "magnitude_only" + output_parameterization = "acos_tanh":
    #     single-output tanh head -> acos in [0, pi], runtime decoder .abs()'s
    #     the NN output and lateral guidance re-selects the sign.
    #   - mode = "full_neural" + output_parameterization = "atan2_signed":
    #     two-output atan2 head -> signed bank in [-pi, pi], no runtime
    #     lateral/thermal/shaping interception.
    # acos_tanh + full_neural is REJECTED here because the Rust runtime
    # (src/rust/src/data/mod.rs::validate_output_parameterization) hard-errors
    # at config load: "output_parameterization=acos_tanh is only legal with
    # mode=magnitude_only". Catching it before warm-start compute saves the
    # ~10 minutes of supervised collection + BPTT pretrain that would
    # otherwise be wasted on a config Rust will reject at gen-0.
    warm_start_active = bool(cfg.network.warm_start_from) or cfg.warm_start.enabled
    if warm_start_active:
        nn_mode = str(_gnn.get("mode", "full_neural"))
        out_param = cfg.network.output_parameterization or "atan2_signed"
        if nn_mode == "full_neural" and out_param == "acos_tanh":
            print(
                "ERROR: output_parameterization='acos_tanh' requires mode='magnitude_only' "
                "(Rust runtime enforces this at config load). Either set mode='magnitude_only' "
                "or switch to output_parameterization='atan2_signed' for full_neural."
            )
            raise SystemExit(1)
        matched = (nn_mode == "magnitude_only" and out_param == "acos_tanh") or (nn_mode == "full_neural" and out_param == "atan2_signed")
        if not matched:
            print(
                f"  [warm_start] WARNING: (mode='{nn_mode}', output_parameterization='{out_param}') "
                f"is not a matched pair. The matched setups are "
                f"(magnitude_only, acos_tanh) and (full_neural, atan2_signed). "
                f"Training will still run, but the supervised target and runtime decoder may be suboptimal."
            )
    if cfg.network.architecture is not None:
        cfg.network.__post_init__()  # re-validate once all fields are set
    cfg.sim.final_file = "output/final.train_nn_temp"
    cfg.sim.exec_dir = "."
    cwd = "."

    # Save dir per (variant × algorithm). For NN schemes the scheme name is
    # encoded in `[data] neural_network` (e.g. "training_output/neural_network_gru_pso/best_model.json"),
    # so we derive save_dir from its parent -- that's the single source of truth
    # and it lines up exactly with what compare_guidance / deploy paths expect.
    # For non-NN schemes (ftc, eqglide, piecewise_constant, etc.), guidance_type
    # already IS the scheme name, so training_output/{guidance_type} is correct.
    if cfg.guidance_type == "neural_network":
        nn_parent = Path(cfg.sim.nn_param_file).parent
        if not str(nn_parent).startswith("training_output/"):
            print(
                f"ERROR: [data] neural_network = '{cfg.sim.nn_param_file}' must live "
                f"under 'training_output/' so checkpoints and report artifacts land alongside the "
                f"deploy JSON. Fix the TOML to point at e.g. 'training_output/neural_network_<variant>/best_model.json'."
            )
            raise SystemExit(1)
        cfg.save_dir = str(nn_parent)
    else:
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

    # Auto-resume: if no --resume and no -fs, check for existing checkpoint.
    # Single-algorithm runs write paired checkpoint_g*.{json,npz}; the
    # islands path writes .npz-only checkpoints. Glob both so islands runs
    # auto-resume just like single-algo runs do.
    resume_dir = args.resume
    if resume_dir is None and not args.from_scratch:
        save_path = Path(cfg.save_dir)
        if list(save_path.glob("checkpoint_*.json")) or list(save_path.glob("checkpoint_g*.npz")):
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

    # Architecture summary (NN schemes only).
    if cfg.guidance_type == "neural_network":
        from aerocapture.training.config import describe_architecture

        print(describe_architecture(cfg.network))

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
            from aerocapture.training.param_spaces import active_scaffolding_specs

            _pack = active_scaffolding_specs(cfg.network.scaffolding)
            n_scaff = len(_pack)
            n_weights = len(param_specs) - n_scaff
            weights = _decode_nn_weights(result["best_individual"][:n_weights], param_specs[:n_weights])
            nn_path = Path(cwd) / cfg.sim.nn_param_file
            write_nn_json(weights, cfg.network, nn_path, input_mask=cfg.network.input_mask, output_param=cfg.network.output_parameterization)
            print(f"Best weights saved to {nn_path}")
            if n_scaff > 0:
                scaff_params = decode_normalized(result["best_individual"][n_weights:], list(_pack))
                for s in _pack:
                    if s.is_integer and s.name in scaff_params:
                        scaff_params[s.name] = int(round(scaff_params[s.name]))
                params_path = Path(cfg.save_dir) / "best_params.json"
                with open(params_path, "w") as fp:
                    json.dump(scaff_params, fp, indent=2)
                print(f"Best scaffolding params saved to {params_path}")
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
