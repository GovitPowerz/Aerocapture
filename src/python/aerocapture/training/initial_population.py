"""The initial population: resume, v1 / v2 NN init, scaffolding slabs and the
warm-start chromosome path (`_build_initial_population`).

Called once per training run by the trainer adapters' `from_config`; consumes
the training RNG BEFORE pymoo's per-algorithm seed is drawn (order matters for
bit-reproducibility, see `experiments/trainer_seam_gate/`). A leaf: no trainer
or train import.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.artifacts import _emit_warm_start_artifacts
from aerocapture.training.checkpoint import _check_resume_chromosome_shape
from aerocapture.training.config import TrainingConfig
from aerocapture.training.evaluate import write_nn_json
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.population import create_initial_population, create_nn_initial_population
from aerocapture.training.problem import AerocaptureProblem, PerSeedEvaluator
from aerocapture.training.seeds import make_reserved_seeds
from aerocapture.training.training_config import _resolve_config_normalization

# scaffolding = "full" seeds the chromosome slab from FTC's GA optimum; the
# warm-start eval callback must read the SAME file so eval == chromosome.
_FTC_SCAFFOLDING_PARAMS_PATH = Path("training_output/ftc/best_params.json")


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
    if algorithm_name not in ("ga", "de", "pso", "qpso", "islands"):
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
            # build_scaffolding_initial_slab already hard-errored on a missing
            # file / missing keys before this callback could fire, so read
            # unconditionally -- a silent fallback here would score the eval
            # on TOML defaults while the chromosome carries FTC values.
            ftc_params = json.loads(_FTC_SCAFFOLDING_PARAMS_PATH.read_text())
            for spec in _eval_pack:
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
        if config.network.architecture is None:
            return  # v1 dense-only warm-start cannot use the v2 callback path

        flat_weights = _policy_to_flat_weights_v2(policy, config.network.architecture)
        fd, tmp_str = tempfile.mkstemp(suffix=".json", prefix=f"warm_eval_epoch_{epoch:04d}_")
        import os

        os.close(fd)
        tmp_path = Path(tmp_str)
        try:
            write_nn_json(
                flat_weights,
                config.network,
                tmp_path,
                input_mask=config.network.input_mask,
                output_param=config.network.output_parameterization,
                normalization=_resolve_config_normalization(config, None),
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


def _build_initial_population(
    resumed: dict | None,
    config: TrainingConfig,
    param_specs: list[ParamSpec],
    seed_weights: npt.NDArray[np.float64] | None,
    problem: PerSeedEvaluator,
    val_seeds: list[int] | None,
    base_mc_seed: int,
    rng: np.random.Generator,
    verbose: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Build the initial population (resume / v1-NN / v2-NN / non-NN), incl. the warm-start chromosome path. Returns (pop_array, pop_costs).

    CRITICAL: in the warm-start branch this MUTATES `param_specs` IN PLACE
    (param_specs[j] = warm_weight_specs[j]) so the caller's AerocaptureProblem --
    which holds the same list reference -- decodes under the warm-start bounds.
    The list object must NOT be reassigned; only its elements are overwritten.
    """
    # Recomputed here (formerly set in the param-spec block) for the v2-NN
    # population-build branch below; both are pure functions of config.
    warm_start_active = bool(config.network.warm_start_from) or config.warm_start.enabled
    bound_mult = config.warm_start.bound_multiplier if warm_start_active else 2.0

    # Create initial population
    if resumed is not None:
        pop_array = resumed["population"]
        pop_costs = resumed["costs"]
        # Ensure pop_array is float64 (legacy checkpoints may have int8)
        if pop_array.dtype != np.float64:
            pop_array = pop_array.astype(np.float64)
        _check_resume_chromosome_shape(pop_array, expected_n_params=len(param_specs))
        if config.optimizer.n_pop != pop_array.shape[0]:
            from aerocapture.training.population import resize_population  # noqa: PLC0415

            if verbose:
                print(f"  Resizing resumed population {pop_array.shape[0]} -> {config.optimizer.n_pop}")
            pop_array = resize_population(
                pop_array,
                pop_costs,
                config.optimizer.n_pop,
                rng,
                fresh_fraction=config.optimizer.grow_fresh_fraction,
            )
            pop_costs = None  # force a single re-eval of the resized pop
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
                        _FTC_SCAFFOLDING_PARAMS_PATH,
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

                assert isinstance(problem, AerocaptureProblem)  # the eval callback flies the simulator through the problem's override builder

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

                _emit_warm_start_artifacts(config, base_mc_seed, problem, val_seeds, warm_chromo, warm_weight_specs, verbose)
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

    return pop_array, pop_costs
