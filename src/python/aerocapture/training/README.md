# Training package (`src/python/aerocapture/training/`)

The population-search training pipeline (pymoo GA / CMA-ES / DE / PSO / QPSO / islands) for every
guidance scheme, its seed pools, cost, reports, and the paper's evaluation tooling. It drives the
Rust simulator through the PyO3 seam only ([src/rust/aerocapture-py/README.md](../../../rust/aerocapture-py/README.md)).

Read first: [docs/ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md) ("A training run in fifteen
lines", "`training/` by role"), [CONTEXT.md](../../../../CONTEXT.md) (champion, final selection vs
final eval, sizing tail), and the ADRs this package implements: ADR-0001 (adaptive seed
strategy), ADR-0002 (final selection on the validation pool), ADR-0005 (feasibility before
performance in selection), ADR-0004 (`run_grid` bit identity). The TOML knobs are documented in
[configs/README.md](../../../../configs/README.md); the NN runtime and the torch mirror in
[src/rust/src/data/neural/README.md](../../../rust/src/data/neural/README.md); the RL trainer in
[rl/README.md](rl/README.md). The lessons a change here must respect ("Silent Config Wiring",
"Reference Trajectory Design", "Resume: cross-gen training-cost incomparability", "pymoo CMA-ES
self-terminates", "GA Parameter Routing") live in the root `CLAUDE.md` and are not repeated.

## Commands

```bash
# Train all schemes with optimized settings (see train_all.sh)
./train_all.sh                     # all schemes in dependency order
./train_all.sh eqglide fnpag       # specific schemes only

# Optimize a guidance scheme (with Rich TUI)
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 2500 --n-pop 60

# Disable TUI (CI, piped output)
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 2500 --n-pop 60 --no-tui

# Resume (auto-detects the checkpoint; --n-gen means "N additional")
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 50

# 3-island PSO/GA/DE (algorithm = "islands"): per-island n_pop from common.toml (60 -> 180
# individuals per gen; --n-pop sets the per-island size), migration every k_period=20 gens
# (top-2 from each source -> worst-4 in each destination; common.toml [optimizer.islands])
uv run python -m aerocapture.training.train configs/training/msr_aller_islands_train.toml --n-gen 2500

# Compare schemes on identical MC scenarios (each scheme uses its own training TOML)
uv run python -m aerocapture.training.compare_guidance --n-sims 500 \
    --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network piecewise_constant

# PDF report (auto-generated at end of training; standalone:)
uv run python -m aerocapture.training.report training_output/equilibrium_glide/ --toml configs/training/msr_aller_eqglide_train.toml
uv run python -m aerocapture.training.report --compare training_output/

# Training-evolution GIF from checkpoints
uv run python -m aerocapture.training.animate training_output/piecewise_constant/ \
    --toml configs/training/msr_aller_piecewise_constant_train.toml --n-sims 100 --fps 4 --every 5

# Sensitivity analysis (Morris screening + Sobol decomposition; --morris-only for the quick ranking)
uv run python -m aerocapture.training.sensitivity configs/training/msr_aller_eqglide_train.toml --morris-n 1000 --sobol-n 1024 --top-k 10
```

`train.py` CLI: `<config.toml> [--n-gen N] [--n-pop N] [--training-n-sims N] [--resume DIR]
[--from-scratch] [--seed N] [--sim-timeout S] [--no-tui] [--skip-report] [--final-n-sims N]
[--algorithm ALG] [--seed-strategy fixed|rotating|adaptive] [--output-dir DIR]`.
Mission artifacts (corridor, reference trajectory) always live at the canonical
`training_output/<mission>/`; `--output-dir` / `--resume` relocate only `save_dir`.

## The training loop

`train.train` is resolve -> build -> run: it resolves the config, builds the `AerocaptureProblem`,
picks the adapter and lets it build itself (`SingleAlgoTrainer.from_config` /
`IslandsTrainer.from_config(config, problem, save_dir, ...)`: resume detection, reserved seed
pools, initial population, pymoo seeding), then hands it to `trainer.run_loop`. `trainer.Trainer`
(a `runtime_checkable` Protocol) is the loop contract: `prologue / should_continue / re_evaluate /
advance / observe / top_k / emit / maybe_checkpoint / finalize / on_interrupt / interrupted_result`
plus the `finalize_in_display_scope`, `start_gen`, `excluded_seeds`, `seed_curator`, `problem`, `rng`
attributes the loop reads (the loop takes no separate problem or RNG). `SingleAlgoTrainer` is a pymoo algorithm + the gate / checkpoint / final-selection
logic, `IslandsTrainer` a thin adapter over `IslandModel`. Per generation: `should_continue` (the
CMA-ES self-termination guard) -> `_apply_seed_strategy` -> `re_evaluate` on seed change ->
`advance` (one `algorithm.next()`) -> `observe` (validation gate / no-validation promotion) ->
`_maybe_curate(top_k_provider=trainer.top_k)` -> `emit` (JSONL + display + heartbeat) ->
`maybe_checkpoint`; then `finalize` (final selection + final checkpoint + sidecar + artifacts).
Per-path conventions are preserved bit-exactly (the 3-run bit-equivalence gate in
`experiments/trainer_seam_gate/`, a script, not a pytest): single-algo logs gen+1 / checkpoints at
(gen+1)%interval labeled gen+1 / finalizes INSIDE the display+interrupt scope; islands logs gen /
checkpoints labeled gen with last-gen force / finalizes outside (`finalize_in_display_scope`).
The adapters talk to the problem through `problem.PerSeedEvaluator` only (`training_rms` is the
RMS-over-seeds cost pymoo minimizes; no `_run_batch` pokes), so `tests/test_trainer_contract.py`
drives both adapters through `run_loop` against `tests/fixtures/fake_problem.py` with no
simulator and no patched privates; `tests/test_training_imports.py` pins that `trainer.py` never
loads `train.py` (the leaf modules: `training_config`, `checkpoint`, `artifacts`,
`initial_population`, `corridor`, `optimizer`, `seeds`).

**Seeding pymoo.** `warm_start_algorithm(algorithm, problem, init_pop, *, seed, n_iter)` hands the
seeded population to the algorithm without letting pymoo re-initialize it (pymoo's first
`.next()` would otherwise call `_initialize()`, which sets `self.pop = Population.empty()`, and
`_initialize_infill()`, an LHS resample). It sets `pop` / `n_iter` / `start_time` (bypassing
`_initialize()` leaves `start_time = None`, which crashes `algorithm.result()` past
`n_max_gen=1000`), runs `_initialize_advance` (PSO sets V + `self.particles = self.pop`; QPSO sets
`self.particles = self.pop`, no velocity; DE/GA's `FitnessSurvival` stamps `rank`), flips
`is_initialized = True`, calls `_set_optimum()`, and seeds pymoo's per-algorithm RNG from the
training RNG (one draw per algorithm/island) so a full run is bit-reproducible under
`seed_strategy = "fixed"`. Both paths use it.

**Seed strategies** (`[optimizer] seed_strategy`, required; ADR-0001; spec
`docs/design/2026-04-14-explicit-seed-strategy-design.md`). `"fixed"`: the deterministic range
`[mc_seed + 0, ..., mc_seed + (training_n_sims - 1)]`, never changes. `"rotating"`: draws
`training_n_sims` fresh random seeds every generation, disjoint from the reserved pools, so the
optimizer cannot memorize scenarios. `"adaptive"`: the curated-CDF path. Bootstrap draws a random
`training_n_sims` seed list once; the list is refreshed on (a) validated best promoted, or (b)
every `seed_pool_interval` generations (from `last_curation_gen`). Each curation draws
`curation_sample_size` probe seeds, runs the top `curation_top_k` individuals on them in ONE
batched `problem.evaluate_population_per_seed` call, averages per-seed costs, DROPS seeds with
non-finite cost (one un-simulatable seed in the training list makes every individual's RMS
non-finite), sorts, splits into `training_n_sims` equal-count quantile bins (optionally trimming
the CDF tails by `curation_trim_fraction`), and picks one seed per bin per
`curation_bucket_selection` (`random` / `min` / `max` / `middle`). On resume the curator restores
STATE only (seed_list, last_curation_gen) via `_restore_seed_curator`; the knobs come from the
current TOML, with a printed notice when the checkpointed values differ; a checkpointed seed_list
whose width differs from `training_n_sims` is dropped and `last_curation_gen` reset to -1 (notice
printed), so the resume behaves like an adaptive fresh start: a random `n_sims`-wide draw plus a
re-eval every generation while `seed_list` is None, and the periodic trigger re-curates at the
first resumed generation once `gen >= seed_pool_interval - 1`. `algorithm.pop` is
re-evaluated pre-`algorithm.next()` only when the seeds actually changed; CMA-ES skips the re-eval
entirely. `fixed` and `rotating` have no class (inline in `trainer.py`); `adaptive` is
`seed_curator.SeedCurator` (`curate(problem, top_k_X)`, `to_dict()` / `from_dict()`). The single
difference between the two adapters is the `top_k_provider` (single-algo: this gen's argmin
slice; islands: `IslandModel.pool_top_k_X`, the union across the 3 populations).

**Validation gate** (identity-triggered). If the current gen's argmin differs from
`last_validated_individual` (`np.array_equal`), the candidate is scored on the reserved
validation pool (`validation_n_sims`, default 1000) by `evaluate.run_validation_gate`
(`GateStatus` / `GateResult`). Cost-based new-best detection is unreliable under
non-stationary seeds, hence the identity trigger. Records carry `rms_cost` (the promotion
metric), mean, p95, worst, capture rate, `feasible` and `violation_rates`.
`best_overall_individual` is promoted only when the candidate is FEASIBLE (every configured
constraint's validation-pool violation rate <= `[optimizer] max_violation_rate`, default 0.0
strict; `evaluate.constraint_violation_rates`, ADR-0005; the single-algo loop prints REJECTED for
an RMS-better infeasible candidate) AND `val_rms < best_val_cost`. The same check gates the
pre-loop initial/resumed champion and the islands `revalidate_each` on resume (an infeasible
resumed champion stays `last_validated` but its RMS does not anchor `best_val_cost`). The
`improvement` flag in logger records reflects validation promotions, keeping the TUI's "Stagnant
for N gens" counter honest. The gate also feeds the full `(n_seeds, 52)` final-records matrix
through `compute_eval_summary` and stores the dashboard payload (DV / apoapsis / periapsis /
inclination percentiles + cost/DV min·max + per-burn `dv1`/`dv2`/`dv3` spread blocks + heat-flux /
g-load / heat-load p50/p95/max + violation %) under `record["validation_summary"]` in the JSONL,
mirroring the end-of-training "Final evaluation (...)" stats block 1-for-1.

**Final selection** (ADR-0002; spec `docs/design/2026-06-10-final-selection-design.md`; skipped
on Ctrl+C and when `validation_n_sims = 0`). `final_select.select_final_individual(problem,
candidates, provenances, known, val_seeds)` re-ranks the last generation (plus, for islands, the
union of all 3 islands' last-gen pops + champions) against `known: list[KnownCandidate]`
(pre-scored champions, NEVER re-simulated; incumbent = lowest val RMS) on the VALIDATION pool.
Fresh rows byte-identical to a known row or an earlier fresh row are deduplicated (for PSO/QPSO
pbest pops the champion row dedups automatically); fresh candidates are scored in ONE batched
`problem.evaluate_population_records_per_seed` call so the feasibility check reads the same
records; a fresh candidate wins only among FEASIBLE candidates and with STRICTLY lower val RMS
(ties keep the incumbent; no champion + all infeasible: the best-RMS infeasible deploys with a
loud warning and `winner_feasible = false` in the sidecar). The deployed individual can never get
worse than the champion, and the final-eval pool stays report-only (no min-of-N selection bias
on quoted numbers). Provenance strings are structured (`"champion"` / `"last_gen[i]"` /
`"<island>:champion"` / `"<island>:last_gen[i]"`; island names must not contain `:`).
`SelectionResult` carries `winner_index`, `incumbent_val_rms` and per-candidate `candidate_rms`.
Promotion forces a final checkpoint save, and the `final_selection.json` sidecar (winner
provenance / val_rms / promoted, `champion_val_rms`, `n_candidates` vs `n_deduped`,
`validation_n_sims`, the full `candidate_rms` array) is written AFTER the durable save so it never
describes a winner the artifacts don't have. Cost: <= n_candidates x validation_n_sims sims,
once (~5% of a 2000-gen budget at paper scale). Retro CLI: `python -m
aerocapture.training.final_select <training_dir> --toml <config> [--no-checkpoint-patch]
[--sim-timeout S]` rebuilds the config via `build_training_config_from_toml`, overlays
`warm_start_bounds.json` weight-spec bounds when present (decoding under rebuilt Xavier bounds
would corrupt NN weights), reconciles `base_mc_seed` (islands npz value cross-checked against
the TOML), validates the chromosome width, errors when `validation_n_sims = 0`, then re-runs the
rule and rewrites artifacts + sidecar + checkpoint (`load_selection_state` reads BOTH checkpoint
formats into a `SelectionState`; `patch_checkpoint` atomically rewrites ONLY the best fields,
temp files `.tmp_`-prefixed so a crashed patch never shadows the real checkpoint in the
`checkpoint_g*` resume globs); non-NN winners also get `optimized_<scheme>.toml` rewritten and,
for joint-ref runs, the reference table regenerated via `deploy_optimized_artifacts`. Writes are
UNCONDITIONAL even on champion-kept, so legacy dirs are rewritten to the validation-pool champion.

**Artifacts and reports.** The winner's `best_model.json` / `best_params.json` are written to
`training_output/<scheme>/` via the shared `write_best_artifacts` (when an NN scheme is trained
with `scaffolding = "full"` / `"live"`, both files are written: the weights and the co-tuned
scaffolding values, 17 or 3; `compare_guidance` and `report.py` pick both up). For islands, a
selection-PROMOTED winner is persisted into the winning island's `best_overall_*` and the npz
checkpoint re-saved (`_persist_islands_promotion`, after `final_eval` so the report keeps the
pre-selection champions, before the artifact write); when no island promotes a validated best
the deploy-path `best_model.json` is removed so `compare_guidance` cannot consume a stale one.
`final_eval` (islands) re-evaluates each island's `best_overall_individual` on the disjoint
final-eval pool for REPORTING only; a promoted fresh winner gets one single-candidate final-eval
run. Then `report.py` renders the PDF unless `--skip-report`.

**Checkpoints and resume.** Auto-resume when the output dir holds a checkpoint (`--resume` not
needed; the probe globs single-algo `checkpoint_g*.json` AND islands `checkpoint_g*.npz`, taking
the LATEST npz that carries the v2 marker so a foreign single-algo checkpoint neither crashes nor
shadows it; only `SingleAlgoTrainer.from_config` calls `load_checkpoint`, so a stale
`checkpoint.json` cannot double-bump an islands run's `n_gen`). On resume `--n-gen` means "N additional
generations" (`config.optimizer.n_gen += resumed_gen + 1`, mirrored inside `IslandsTrainer`). A
checkpoint is always saved at end of training, labeled with the last generation that ACTUALLY
ran (`completed_gen`, so a CMA-ES internal early stop does not inflate the resume math);
single-algo writes are atomic (`.tmp_` + rename, npz before json: the json is the resume-glob
key) and `load_checkpoint` falls back past corrupt pairs newest-first. `[checkpoints] keep_last`
is honored via the shared `_prune_old_checkpoints` (`cleanup_checkpoints.prune_checkpoints`
matches both `.json` and `.npz`). *Changed population size:* `population.resize_population(pop_X,
pop_F, target_n, rng, fresh_fraction, jitter_sigma)` reshapes the resumed pop (grow keeps the
resumed individuals verbatim and fills with `grow_fresh_fraction` fresh-random plus round-robin
clone+jitter; shrink keeps the best-N by checkpoint cost); single-algo sets `pop_costs = None` to
force one re-eval before `warm_start_algorithm`, islands call
`IslandModel.resize_populations(target_n, rng, fresh_fraction, velocity_scale)`. The chromosome
WIDTH is guarded (`_check_resume_chromosome_shape` -> ValueError pointing at `--from-scratch`
when `scaffolding` / `output_parameterization` / `input_mask` flipped between runs). *Changed
`cost_transform`:* persisted in both checkpoint formats (legacy checkpoints read back `None` =
"assume changed"); single-algo re-validates its checkpointed best unconditionally, islands call
`IslandModel.revalidate_each()` and reset each island's `stagnation_counter`. `IslandsTrainer`
resume order: `from_checkpoint` -> curator seed push -> `resize_populations` ->
`revalidate_each` -> transform notice. The checkpointed best is trusted VERBATIM (never swapped
via a `<` comparison against the resumed population's cost; see the root `CLAUDE.md` lesson).
Spec: `docs/design/2026-06-02-resume-enhancements-design.md`.

**Interrupts and headless runs.** Ctrl+C saves a checkpoint and returns cleanly with
`interrupted: True` on both paths; the islands trainer returns before selection and
`final_eval`, so an interrupt never launches the post-loop MC sweeps or rewrites artifacts.
`--no-tui` / non-tty -> `NoopDisplay` (`is_live = False`) with a plain per-5-gen heartbeat print
plus checkpoint-saved lines (a silent multi-hour run is indistinguishable from a NaN-hung batch;
use `--sim-timeout` against NaN hangs).

## Modules

### Config and chromosome

- `config.py` — `TrainingConfig` / `NetworkConfig` (`architecture: list[dict] | None`,
  `_layer_n_params`, `_layer_output_size`, `resolve_mamba_dt_rank`; `__post_init__` validates the
  decoder vs the head and shallow-copies the architecture list), `WarmStartConfig.from_dict`,
  `candidate_input_names()` / `candidate_input_index()` / `candidate_input_normalization()`
  (derived from `aerocapture_rs.candidate_inputs()`, with a fallback tuple for machines without
  the extension), `build_training_config_from_toml` (in `training_config.py`).
- `toml_utils.py` — `load_toml_with_bases()` (mirrors Rust `resolve_toml_bases`),
  `set_dot_path()`, `write_toml()` (the minimal machine-consumed writer), `find_mission_name()`
  (recursive walk of the base chain to the first `missions/` entry), `reject_unknown_keys(section,
  table, known)` (the ONE unknown-key check for the Python-owned sections: `[cost_function]` in
  `cost.build_cost_kwargs`, `[corridor]` / `[reference]` / `[checkpoints]` in
  `build_training_config_from_toml`, `[warm_start]` / `[warm_start.adam]` in
  `WarmStartConfig.from_dict`).
- `param_spaces.py` — Per-scheme parameter bounds (`ParamSpec`, optional log-scale), the
  prefix groups `_NAV_PARAMS` / `_LATERAL_PARAMS` / `_EXIT_PARAMS` / `_THERMAL_LIMITER_PARAMS` /
  `_SHAPING_PARAMS`, `_NN_LIVE_PARAMS` (3) / `_NN_SCAFFOLDING_PARAMS` (17),
  `active_scaffolding_specs(scaffolding)`, `make_piecewise_constant_specs(N)`,
  `JOINT_REF_BANK_SCHEMES`, `route_param_path` (the prefix -> TOML dot-path rule: `nav.` ->
  `navigation.*`, `lateral.` -> `guidance.lateral.*`, `exit.` -> `guidance.ftc.*`, `thermal.` ->
  `guidance.thermal_limiter.*`, `shaping.` -> `guidance.command_shaping.*`, unprefixed ->
  `guidance.<scheme>.*`).
- `deploy_overrides.py` — `overrides_from_params` (the full deploy rule: routing + `shaping.* =>
  guidance.command_shaping.enabled = true` + `ParamSpec.is_integer` coercion + `ref_bank` skip),
  `load_scaffolding_overrides` / `resolve_eval_toml` (consumed by `cell_eval`), `LEGACY_NOISE_REGIME`.
- `cell_eval.py` — the ONE deploy-side evaluation path. `evaluate_cell(cell_dir, base_toml,
  seeds | pool=(offset, n), *, model, extra_overrides, per_seed_overrides, include_trajectories,
  sim_timeout_secs, n_threads)` flies a cell one sim per seed through `run_batch`: TOML via
  `resolve_eval_toml` (optimized TOML wins, else base + `best_params.json` scaffolding), NN pinned
  to `<cell_dir>/best_model.json` unless `model=` (a bundle's frozen weights, a temp ablated or
  checkpoint model; a `model=` that does not exist raises), `extra_overrides` (noise regime,
  stress levels, guidance type) win over the cell, the seed is applied last; `cell_dir=None` flies
  the bare TOML. A cell holding `best_params.json` but neither `optimized_<scheme>.toml` nor
  `best_model.json` (an undeployed classical cell) is refused unless `model=` is given: only its
  prefixed scaffolding keys would route and the gains would fly at TOML defaults. `fly_mc` is the same
  resolution through `run_mc` (the config's own Monte Carlo: `compare_guidance`, `ablation`,
  `animate`), `fly_nominal` the undispersed nominal (every `_MC_DISPERSION_DOMAINS` level off).
  `CellResult` carries `final_records (N,52)`, `dispersions (N,26)`, `trajectories` (only when
  requested), `seeds`, the resolved `toml_path` and the shared `overrides` (provenance), plus
  `captured` (`is_captured`, the ONE captured predicate; `charts.is_captured` re-exports it) and
  `dv` (captured-only). `reserved_pool(toml, offset, n)` keys a registered pool by the TOML's
  `[monte_carlo] seed`. Callers: `report.py` (final MC + nominal overlay), `demo.py`, `rl/train.py`
  (`_evaluate_model`), `rl/report_rl.py`, `compare_guidance`, `param_sweep --eval`, `quantize`,
  `probe_common.score_model`, `mamba3_962_compare`, `warm_start_compare`, `ablation`, `animate`,
  every paper script, `experiments/ou_marginal`, `experiments/fnpag_ab`. The seam is still called
  directly where the override builder is genuinely different: `problem.py` (`run_grid`, the
  training chokepoint), `initial_population.py` (the warm-start eval callback on
  `problem._build_overrides`), `corridor.py` (the piecewise corridor accumulation over a
  population), `train.py` (the piecewise best nominal via `nominal_flight_overrides`), `reference.py` / `make_reference.py` (reference generation),
  `sensitivity.py` (`run_with_draws`), `aerocapture.physics_crosscheck` (undispersed AMAT cells),
  and `articles/paper/scripts/compute_benchmark.py`'s timed repeats (the warmup resolves the cell
  through `evaluate_cell`; the repeats replay that batch on the bare seam so cell resolution stays
  out of the timing). Migration was checked bit-for-bit per caller (#72).
- `encoding.py` — All algorithms work on normalized `np.ndarray[float64]` in [0, 1].
  `decode_normalized(x, specs)`, `encode_to_normalized(params, specs)`,
  `decode_normalized_array(X, specs)`, `nn_param_specs_from_architecture(layer_sizes, activations,
  bound_multiplier)` (v1 dense) and `nn_param_specs_from_v2(architecture, bound_multiplier)`
  (per-layer-type `_layer_param_specs` walking `layer_schema`; identical bounds to v1 for
  all-dense architectures via the shared `compute_layer_bound`).
- `initialization.py` / `initialization_v2.py` — Activation-aware weight init (Xavier / He /
  LeCun uniform) for v1 populations (`create_nn_initial_population`; the uniform
  `create_initial_population` for classical schemes); `init_v2_population(architecture, n_pop, bound_multiplier,
  rng)` per layer type for v2, normalized to [0, 1] by `initial_population.py::build_initial_population_for_v2`
  (details in the NN runtime README). `population.py`:
  `resize_population`.
- `layer_schema.py` — per-layer tensor names / shapes / flat order, read from
  `aerocapture_rs.layer_schema` with a `_fallback_layer_schema` mirror
  (`tests/test_layer_schema_drift.py`).

### Loop and optimizers

- `train.py` - `train()` (resolve -> build -> `run_loop`) + the CLI. Nothing imports it as a
  library.
- `trainer.py` - the `Trainer` Protocol, `run_loop`, `SingleAlgoTrainer`, `IslandsTrainer` (each
  with `from_config`), the loop's `_apply_seed_strategy` / `_maybe_curate`,
  `_build_validation_payload`, `_persist_islands_promotion`.
- `training_config.py` - `build_training_config_from_toml`, `_setup_param_specs`,
  `check_ref_trajectory_wiring`, `_resolve_piecewise_n_segments`, `_resolve_config_normalization`.
- `checkpoint.py` - `save_checkpoint` / `load_checkpoint` (paired json+npz), `_prune_old_checkpoints`,
  `_restore_seed_curator`, `_check_resume_chromosome_shape`.
- `artifacts.py` - `write_best_artifacts`, `deploy_optimized_artifacts`, `_emit_warm_start_artifacts`.
- `initial_population.py` - `_build_initial_population` (resume / v1 / v2 / warm-start),
  `_seed_initial_population`, `build_initial_population_for_v2`, the scaffolding slabs, the
  warm-start eval callback.
- `seeds.py` also carries the training-side draws (`_draw_disjoint_seeds`, `_compute_fixed_seeds`,
  `base_mc_seed_from_toml`); `optimizer.py` carries `warm_start_algorithm`; `corridor.py` carries
  `_accumulate_corridor`; `encoding.py` carries `_decode_nn_weights`.
- `optimizer.py` — `OptimizerConfig` (with `GASettings`, `CMAESSettings`, `DESettings`,
  `PSOSettings`, `QPSOSettings` (`alpha_start` / `alpha_end` validated in (0, 2]),
  `IslandSettings`; `from_dict()` for TOML-like dicts) and `create_algorithm(config, n_params)`:
  GA (SBX crossover + PM mutation), CMA-ES, DE, PSO, or QPSO (`max_iter = n_gen` for the alpha
  anneal). CMA-ES falls back to GA with a warning when n_params > 20000 (`_CMAES_MAX_PARAMS`;
  dense_p3998 runs real CMA-ES). `algorithm = "islands"` is rejected here (ValueError pointing to
  `IslandModel`); the islands path constructs each sub-algorithm via `IslandModel.__init__`.
- `qpso.py` — `QPSO(Algorithm)`: canonical mbest quantum-behaved PSO (Sun/Feng/Xu 2004) as a
  custom pymoo Algorithm. Velocity-free: each generation every position is resampled around a
  per-particle local attractor `phi*pbest + (1-phi)*gbest` (`phi ~ U(0,1)` per element) with
  characteristic length `alpha * |mbest - x|` (`mbest` = mean of all pbest; `u = 1 - U(0,1)` in
  (0,1] keeps `ln(1/u)` finite; sign ±1 at p=0.5), `alpha` annealed linearly `alpha_start ->
  alpha_end` over `max_iter = n_gen`. State conventions mirror pymoo PSO (`pop` = pbest,
  `particles` = current positions) so `warm_start_algorithm`, checkpoint/resume, resize and the
  seed strategies work untouched; out-of-bounds positions repaired with `repair_random_init`. On
  resume `n_iter` restarts at 1 against the bumped `max_iter`, so alpha restarts at `alpha_start`
  on the stretched schedule. Tests: `tests/test_qpso.py`. Paper configs
  `configs/training/paper/dense_p515_qpso.toml` + `dense_p3998_qpso.toml`; runners
  `experiments/paper/02_optimizer_budget.sh` + `03_optimizer_dimensionality.sh`. Spec:
  `docs/design/2026-06-10-qpso-optimizer-design.md`.
- `island_model.py` — `IslandModel` owns 3 pymoo `Algorithm` instances (PSO / GA / DE) sharing
  one problem and one seed list (migrant F values are comparable across islands at migration
  time). `step(current_gen)` advances each island then applies migration every `k_period` gens
  (skip at gen 0 and when `enabled=false`); `validate_each(current_gen)` runs the identity-trigger
  per-island validation (skipping islands whose entire `pop.F` is non-finite: `np.argmin` on an
  all-inf array returns 0); `pool_top_k_X(k)` feeds the curator from the UNION of the 3
  populations; `re_evaluate_all_populations()`; `final_eval()` (`max(validation_n_sims, 10000)`
  sims on the final-eval pool, report-only). `migrate(islands, k_top, current_gen, rng,
  velocity_scale)` is a pure function: snapshots top-k emigrants from every island BEFORE in-place
  replacement (each emigrant `.copy()`'d so two destinations never alias one ndarray), sorts
  incoming by descending F so the BEST migrant lands in the absolute-worst destination slot,
  overwrites the `k_top * (n_islands - 1)` worst-F slots; only FINITE-F individuals are eligible
  emigrants (a collapsed island contributes none; a destination whose every other island collapsed
  is skipped); PSO destinations get fresh velocity via `inject_into_pso` (`U(-velocity_scale,
  velocity_scale)` per dim plus pbest reset) and each destination's `_set_optimum()` is called so
  PSO's social attractor reflects the post-migration pop. `k_top * (n_islands - 1) > n_pop` raises
  (pre-validated in `__init__`). Checkpoint v2 atomic `.npz` (`np.savez_compressed` + tempfile +
  rename) holds all 3 islands' state + migration log + RNG state + optional `seed_curator_state`;
  per-island `pop_X` / `pop_F`, `n_iter`, `is_initialized`, and for PSO `particles_X` /
  `particles_F` / `particles_V`. Deliberately NOT saved: `pop.get("V")` / `.get("pbest")` (pymoo's
  `Population.get` returns an object ndarray of `None` entries when the attribute was never set,
  so an `is not None` guard would persist junk). `from_checkpoint` raises `ValueError` on version /
  `base_mc_seed` / island-name / chromosome-width mismatch, restores `best_overall_*` VERBATIM,
  rebuilds `algorithm.particles` for PSO, sets `is_initialized = True` and `n_iter = saved + 1`,
  stamps `start_time`, calls `_set_optimum()`, re-runs `FitnessSurvival().do(...)` on non-PSO
  islands to re-stamp `rank` (DE's `_infill` picks its target via `pop.get("rank") == 0`), and
  returns `(generation, seed_curator_state, saved_cost_transform)`. When `validation_n_sims = 0`
  each island promotes its finite training argmin into `best_overall_*`, and `IslandsTrainer`
  unions the reserved pools into `excluded_seeds` unconditionally (the curator shares that set). `resize_populations` /
  `revalidate_each` back the resume features above. Spec:
  `docs/design/2026-05-28-island-model-pso-ga-de-design.md`.
- `seed_curator.py` — `SeedCurator` (above).
- `final_select.py` — `select_final_individual`, `KnownCandidate`, `SelectionResult`,
  `load_selection_state` / `SelectionState` / `patch_checkpoint`, the CLI (above).

### Evaluation

- `problem.py` — `AerocaptureProblem(Problem)`: normalized [0,1] decision variables decoded via
  `decode_normalized_array()` at eval time; `_evaluate(X, out)` sets `out["F"]` as (n_pop, 1)
  through `_run_batch()` -> `training_rms(problem, X)` (RMS over the seed axis; the module-level
  function is what the trainer seam and `IslandModel` re-evaluate through). `PerSeedEvaluator` is
  the Protocol the seam drives (the four per-seed methods + `update_seeds`, `seeds`, `cost_kwargs`,
  `param_specs`, `toml_path`).
  `evaluate_population_records_per_seed(X, seeds) -> ((n_pop, n_seeds) costs, (n_pop, n_seeds, 52)
  records)` is the ONE batched kernel over `_run_grid_records` (one `run_grid` call, SimData built
  once); `evaluate_population_per_seed` is its costs-only view and `evaluate_individual_per_seed`
  the 1-row case; final selection and curation route through it. For NN schemes the decode step
  only materializes the scaffolding-tail params (the [0, n_w) weight slice is re-decoded
  vectorized and carried in-memory by run_grid). `_build_grid_overrides(params)` routes with
  `route_param_path` + its own `is_integer` set and deliberately does not add the shaping enable
  flag (a section created by a `max_bank_acceleration` override is enabled by default in Rust);
  `_build_overrides(params, mc_seed)` adds seed + n_sims=1. `update_seeds(seeds)` injects curated
  seed lists between generations.
- `evaluate.py` — `write_nn_json` (chromosome -> model JSON via
  `aerocapture_rs.flat_weights_to_json`, QAT rounding applied), `build_v2_architecture`,
  `write_guidance_toml` (base TOML + `deploy_overrides.overrides_from_params` -> `toml_utils.write_toml`),
  `run_validation_gate` / `GateStatus` / `GateResult`, `constraint_violation_rates`,
  `_parse_final_to_legacy_array` (test-only: the Rust-CLI subprocess oracle
  `tests/fixtures/subprocess_oracle.py` backing `test_pyo3_matches_subprocess`).
- `cost.py` — `build_cost_kwargs(toml_data)`, the ONE `[cost_function]` + `[flight.constraints]`
  reader (train, gate, final selection, report, compare; `report.read_cost_kwargs(path)`
  delegates; unknown keys raise) + the per-sim objective `compute_cost`, built on `dv_cost(dv)`, a
  C-infinity softplus-quadratic (linear below `dv_threshold`, softplus-quadratic above, smooth
  knee) plus normalized soft constraint penalties for g-load, heat flux and heat load
  exceedances, optionally wrapped in `cost_transform`. The DV every termination outcome carries is
  described in the Rust README ("Termination outcomes and virtual DV").
- `seeds.py` — the reserved seed-pool registry: every `*_SEED_OFFSET` (`VALIDATION_SEED_OFFSET` 1M,
  `FINAL_EVAL_SEED_OFFSET` 2M, `RL_TRAINING_SEED_OFFSET` 3M, `WARM_START_SEED_OFFSET` 4M,
  `NN_INPUT_REPORT_SEED_OFFSET` 5M, `CALIBRATION_SEED_OFFSET` 6M, `SWEEP_EVAL_SEED_OFFSET` 7M, headline
  requote 8M, stress 9M, `PROBE_EVAL_SEED_OFFSET` 10M, `CONFIRM_EVAL_SEED_OFFSET` 20M; `tests/test_seed_offsets.py` asserts the
  list AND that no other module defines one), `make_reserved_seeds(base_mc_seed, offset, n)`, and
  `make_confirmatory_pools(base, n_replicates=10, n=100_000)` (`CONFIRM_EVAL_SEED_OFFSET` names
  the stream; duplicate-free seeds from `[2^31, 2^32)`, structurally disjoint from every
  historical pool, which all draw from `integers(0, 2^31)`; consumed by
  `articles/paper/scripts/confirmatory_eval.py`). Pool disjointness is otherwise probabilistic
  (independent streams, ~n²/2³¹ collision odds per pair); rotating/adaptive draws exclude the
  reserved pools explicitly.
- `compare_guidance.py` — head-to-head comparison on identical MC scenarios: each scheme's cell
  (`<params_dir>/<scheme>/`) flown through `cell_eval.fly_mc` on its own training TOML with
  `n_sims` dispersed sims from the shared `[monte_carlo] seed` (no temp TOML, no subprocess, no
  CSV parse; the old CLI transport agreed to CSV precision, ~5e-8; `--sim-timeout` defaults to 30 s
  per sim so a non-terminating sim cannot hang the table); `compare_guidance.SCHEMES` /
  `_NN_DEPLOY_SCHEMES` register every deployable cell (each NN scheme deploys through the Rust
  `neural_network` runtime, RL included as `neural_network_rl`); cost kwargs from
  `report.read_cost_kwargs` so heat-load weight/limit match the training objective.
- `reference.py` (leaf module) — `ref_trajectory_array` (7-column table from a trajectory
  matrix; column 0 in MJ/kg, the Rust loader multiplies by 1e6), `piecewise_commanded_cos_bank`
  (the COMMANDED segment profile for the cos_bank column: the realized bank carries shaper sweeps
  through 0 deg that whipsaw tracker feedforward), `nominal_flight_overrides` (a nominal with ALL
  10 MC dispersion domains off plus any config-declared domain), `generate_constant_bank_tables`
  (per-bank undispersed 1-segment nominals in ONE batched run_batch; slot files).
- `make_reference.py` — the target-energy-matched constant-bank reference CLI: bisects the bank
  angle until the undispersed nominal's exit energy hits the target orbit energy minus
  `--overshoot-mj` (default 0.7, the legacy margin), writes the mission ref with constant
  commanded-cos feedforward. `python -m aerocapture.training.make_reference --toml
  configs/training/msr_aller_pc_ref_train.toml [--overshoot-mj F]`.
- `sensitivity.py` — SALib support: `DISPERSION_COLUMNS` (26 names matching
  `DispersionDraw::to_array()`), `build_problem(mc_config)` (a `[monte_carlo]` dict -> SALib
  problem with per-dimension distribution types and SI-unit bounds mirroring
  `build_dim_transforms()`; Gaussian dims are `truncnorm` `[lo, hi, mean, sd]` at ±4σ
  (`_TRUNCNORM_SIGMA_SPAN`), since an unbounded `'norm'` maps the Morris/Sobol grid endpoints to
  ±inf draws; wind-absent pins scale at the neutral 1.0), `_active_indices_and_fixed(problem)`
  (SALib raises on norm sd=0 / unif lo==hi, so off domains are excluded from the sampled problem
  and injected as fixed constants), `run_morris(toml_path, n, ...)` (via `run_with_draws()`,
  full-26 outputs with zeros for inactive dims), `run_sobol(toml_path, n, param_indices, ...)`
  (S1/ST/S2), `run_full_analysis(toml_path, ...)` (Morris ranking -> Sobol on top-k ->
  `output_dir/sensitivity_results.json`). CLI: `python -m aerocapture.training.sensitivity <toml>
  [--morris-n N] [--sobol-n N] [--top-k K] [--morris-only] [--sobol-only] [--output-dir DIR]
  [--sim-timeout S]`.
- `parquet_output.py` — `write_parquet(path, final_records, dispersions, config,
  toml_path=None)` writes 65-column Parquet (39 final-record + 26 `disp_`-prefixed dispersion
  columns) with schema-level metadata (resolved TOML as JSON, toml_path, timestamp,
  guidance_scheme, n_sims); `read_parquet(path)` -> `(DataFrame, metadata_dict)`;
  `FINAL_RECORD_INDICES` (39 of 52, matching Rust `extract_final_csv_values()`) and
  `FINAL_COLUMNS` (`FINAL_CSV_COLUMNS` minus sim_number). CSV output is unchanged (39 columns).
- `paper_stats.py`, `cleanup_checkpoints.py`.

### NN specifics (the runtime contract is in the NN runtime README)

- `torch_mirror/` — the differentiable PyTorch mirror of the Rust runtime (`schemas.py`, one
  torch module per cell under `layers/`, `policy.py::V2Policy`, `export.py`); `model_io.py` is the
  v2 JSON reader. Load-bearing for the population path (warm-start, chromosome sizing); `rl/`
  depends on it, never the reverse.
- `warm_start.py` / `warm_start_compare.py` / `warm_start_report.py` — the multi-supervisor
  BPTT warm-start (below).
- `calibrate_inputs.py` — NN input-scale calibration: runs `collect_nn_inputs` over the
  calibration pool (`CALIBRATION_SEED_OFFSET`), inverts the normalized trace back to raw using the
  normalization the SIM ACTUALLY applied (`_resolve_normalization`: `[network.normalization]`
  override > embedded model `normalization` > the Rust `candidate_inputs()` table; inverting with
  constants that differ from the forward pass distorts the recovered raw by `s_forward/s_invert`
  and the proposed scale never converges across retrain+recalibrate cycles), then derives new
  `{transform, scale, center}` entries so each input's `[p_lo, p_hi]` (default `[5, 95]`,
  `--target-percentiles LO HI`) maps EXACTLY to `[-1, 1]` via a two-parameter endpoint fit
  (`derive_asinh_endpoints` / `derive_affine`: `center = (p_lo+p_hi)/2`, `scale` from the span).
  `choose_transform` picks from the data: `tail_ratio = (p99.9-p0.1)/(p_hi-p_lo)`
  (`--tail-threshold`, default 1.6) selects `asinh` (heavy-tailed) vs affine `none`; `tanh` is
  never auto-selected. `_FORCE_ASINH` (empty `set()` by default; `--no-force-asinh`) forces
  listed indices; `_SKIP` (15 / 20-30: binary / tanh / sin-cos inputs already in [-1,1]) keeps the
  current transform. The DV inputs (32-34) calibrate over their full distribution. Emit via
  `--write-model PATH` (the 35-entry block into a model JSON's `normalization` field, no Rust
  rebuild) or `--emit-toml PATH` (a paste-ready `[network.normalization]` snippet regenerated from
  `NN_INPUT_NAMES`; the committed blocks' `# N name` comment columns are guarded by
  `tests/test_config_normalization_blocks.py`). Pure helpers (`invert_transform`,
  `derive_asinh_endpoints`, `derive_affine`, `tail_ratio`, `choose_transform`,
  `_resolve_normalization`) are unit-tested. CLI: `python -m aerocapture.training.calibrate_inputs
  --toml <config.toml> [--n-sims N] [--target-percentiles LO HI] [--tail-threshold T]
  [--no-force-asinh] [--write-model PATH] [--emit-toml PATH] [--output PATH]`.
- `ablation.py` — NN input importance: `NN_INPUT_NAMES` (the 35 candidate names),
  `run_ablation(toml_path, n_sims, ..., model_path=None, extra_overrides=None)` zeros out each
  input via a temp JSON model with `ablated_input` set, measures DV cost degradation vs baseline,
  ranks by delta (`model_path` pins the evaluated model; `extra_overrides` carries the co-trained
  `best_params.json` scaffolding; inputs outside the model's `input_mask` are skipped with
  `masked_out=True`). `--flip` runs `run_flip_ablation`: FREEZES a binary ±1 flag (default
  `bounce_flag`, index 15) to -1 and +1 separately via the model's `ablated_value` field,
  separating a flag's phase-gating effect from the out-of-distribution artifact of zeroing it;
  writes `flip_ablation_results.json`. CLI: `python -m aerocapture.training.ablation
  <training_dir> --toml <config.toml> [--n-sims N] [--flip] [--model PATH]` (defaults the model to
  `<training_dir>/best_model.json` and auto-applies the dir's scaffolding overrides). Outputs JSON
  + SVG (`charts_ablation.chart_ablation_bar`).
- `nn_input_report.py` — runs the deployed NN over the report pool
  (`NN_INPUT_REPORT_SEED_OFFSET`) via `collect_nn_inputs`, classifies trajectories blue (low
  final DV) / red (high) by `classify_by_dv` (default threshold `cost_function.dv_threshold`,
  `--dv-threshold`), renders 70 panels (35 inputs x {time, energy};
  `charts_nn_inputs.chart_nn_input_panel` with per-class p5-p95 envelopes via `binned_band` and
  ±1 guide lines, greyed "(unused)" for inputs outside the mask; `_resolve_mask` prefers the
  deployed `best_model.json`'s embedded `input_mask`) plus `summary.json` (`input_summary`:
  p1/p50/p99, `frac_out_of_range` = fraction of samples with |value|>1, blue-vs-red `separation` =
  |Δmean|/pooled-σ) and `nn_input_report.pdf` (`report_render.render_pdf` over
  `src/typst/nn_input_report.typ`; degrades to SVGs+JSON without Typst). CLI: `python -m
  aerocapture.training.nn_input_report <training_dir> --toml <config.toml> [--n-sims N]
  [--dv-threshold F] [--output-dir DIR]`.
- `quantize.py` — weight-only symmetric fake-quantization (paper Appendix C):
  `quantize_model_weights(model_json, n_bits, granularity, tensor_policy, only_tensor=None)`
  (values stored back as f64, runtime untouched; `proj_only` keeps `a_log` / `d_skip` / biases fp;
  `only_tensor` = the leave-one-out probe), `quantize_flat_weights_batch` (the QAT-in-the-loop
  path shared with `evaluate.write_nn_json`), `memory_footprint`, and the sweep CLI (PTQ grid bits
  x granularity x policy + LOO + verdict rule max-capture-then-min-CVaR95 + finalists at n=10k)
  writing `quantization_results.json` / `finalists_results.json`. Runner
  `experiments/paper/17_quantization.sh {ptq|bench|qat_finetune|qat_scratch|finalists|collect}`;
  QAT configs `configs/training/quant/`; criterion micro-bench `src/rust/benches/quant_forward.rs`
  (`cargo bench --bench quant_forward`). The materialized PTQ-verdict model is
  `training_output/quant/ptq4_verdict/`; sanity-gate any re-materialization against the committed
  grid cell (capture 1.000 / CVaR95 147.9 on the fresh pool).
- `param_sweep.py` — architecture parameter-budget sweep -> Pareto curve (cost vs trainable
  weights): `--generate` writes `configs/training/sweep/*.toml` + `manifest.json` (each
  base-inherits `msr_aller_nn_atan2_train.toml`; one capacity knob per family, exact counts via
  `NetworkConfig`; dense family floor hidden=2 so sub-500 budgets resolve), `--train`
  subprocess-trains each point (skip-if-`best_model.json`-exists, `--force` to retrain,
  `--from-scratch` passes through, `--training-n-sims N`), `--eval` re-scores every deployed model
  on ONE reserved pool (`SWEEP_EVAL_SEED_OFFSET`) WITH the co-trained `best_params.json`
  scaffolding (`_score_entry` -> `cell_eval.evaluate_cell`; the sweep configs inherit
  `scaffolding = "live"`, so scoring without them mis-ranks architectures), `--plot` renders the
  Pareto frontier SVG (`--metric
  capture_rate` flips to a higher-better front). `--out-tag <tag>` isolates a sub-sweep's
  `manifest_<tag>.json` / `pareto_results_<tag>.json` (the sub-500 floor sweep uses
  `manifest_floor.json`, whose 515/3998 anchors point at
  `paper/optimizer_dimensionality/dense_p515_ga` / `paper/optimizer_budget/ga_300`). CLI: `python
  -m aerocapture.training.param_sweep --generate|--train|--eval|--plot|--all [--archs ...]
  [--budgets ...] [--training-n-sims N] [--out-tag T] [--from-scratch]`.
- `weight_stats.py` — per-layer weight statistics (v1 dense only; skipped for v2).
- `experiments/` — the architecture probes (`mamba3_probe`, `cfc_probe`, `xlstm_probe`) on the
  shared `probe_common.py` machinery (generate / train / eval / report, seed repeats x sigma_run
  significance, tail-led reporting). Probe leaf configs base-inherit
  `msr_aller_nn_atan2_train.toml` and the sweep regime (GA n_pop 300, adaptive + bucket=max,
  training_n_sims 2); `eval_arms` applies each arm's deployed `best_params.json` scaffolding at
  scoring; all probes score on the shared `PROBE_EVAL_SEED_OFFSET` pool and print the deployed
  GRU/LSTM/Mamba champions as reference rows. `train_jobs` is resumable per arm via
  `probe_common._completion` (done / partial / stale / absent: "done" = deployed arch matches the
  config AND the latest `checkpoint_g*` reached n_gen; "partial" auto-resumes; "stale" relaunches
  `--from-scratch`; keying on bare `best_model.json` existence would deploy an under-trained arm,
  since that file is rewritten on every mid-run promotion). Leave all ARMS uncommented and let
  the guard skip finished ones. Specs: `docs/design/2026-07-07-mamba3-ablation-design.md`,
  `docs/design/2026-07-07-cfc-xlstm-probes-design.md`. The full-budget `mamba3_962` campaign (GA 512 x
  10k, `configs/training/mamba3_962/`, `mamba3_962_compare.py`) is separate from the probe.

### Reports and visualization

- `metrics.py` — pure metric functions (cost stats, diversity, capture rate, convergence speed,
  stagnation) + `apply_cost_transform` (the single source of the monotonic rescale, reused by
  `compute_cost`); `capture_rate(costs, capture_threshold, cost_transform)` maps the linear-scale
  threshold (3000 = CRASH_FLOOR) into the transformed space.
- `logger.py` — `TrainingLogger` (takes `cost_transform` at construction): one JSONL line per
  generation (`all_costs`, `constraint_violation_rate`, `best_params`, `gen_best_params`, and the
  optional `validation` / `validation_summary` dicts when the gate fires); in-memory buffer for
  the display.
- `display.py` — `LiveDisplay` (Rich, `Live` at 2 Hz; `NoopDisplay` when `--no-tui` or
  non-interactive, `is_live` tells the loops to print heartbeats instead).
  `update(logger, current_run, island_records=None)` switches to the 3-column islands layout
  when `island_records` is provided; single-algo runs render the dashboard built by
  `_build_dashboard` from four builders: `_build_header` (scheme · algorithm · Gen g/N · pop n ·
  elapsed · rate · ETA + progress line; resume-aware via `set_start_gen` and `_rate_and_eta`,
  shared with `_update_islands`), `_build_optimization_panel` (best/mean cost + worst+σ, capture,
  diversity sparklines, a 16-bin log-spaced `_cost_histogram` of `all_costs` with an `∞×k` caption
  for non-finite sims, `gen wall` from `gen_elapsed_s`, a `Seeds` row from the
  `seed_strategy`/`training_n_sims` constructor args), `_build_validation_panels` ("Last
  validation" framed GREEN when promoted / RED when rejected, and "Best validation"; each an RMS
  headline + the stats block from `_summary_renderables`), `_build_footer` (stagnation, last
  improvement, best-params preview; NN schemes show `{n} NN params`). Both modes shape the
  validation content with `_validation_summary_rows(summary) -> (label, cells, style)` (islands
  render via `_rows_to_text`, sticky across gens via `Island.latest_val_summary`).
  `create_display(..., algorithm=)` threads the optimizer name into the header.
- `report.py` — the PDF report orchestrator: loads the JSONL logs (`load_run_data` dedups by
  `(generation, island_name)` so islands runs keep all 3 per-gen records), runs the final MC
  re-evaluation on the final-eval pool via `cell_eval.evaluate_cell` (`run_final_evaluation`; the
  evaluated NN is pinned to `<scheme_dir>/best_model.json` when present, because the TOML's shared
  `[data] neural_network` deploy path is rewritten by every `--output-dir` sibling run; the noise
  regime it resolved is passed into the run and printed) and the undispersed nominal overlay via
  `cell_eval.fly_nominal` (same TOML, scaffolding and model pin as the final MC; before #72 the
  overlay flew the TOML's shared model path), generates the SVG charts (`charts.py`), writes
  metadata/summary JSON and hands off to `report_render.py` (`staged_assets` + `render_pdf`).
  Three parts: Training Convergence (cost curves, diversity, cost distribution, parameter
  evolution; in islands mode `chart_island_convergence_overlay` + `chart_migration_timeline` fed
  by the `migration_log` from the latest `checkpoint_g*.npz` via `_load_migration_log`, while the
  regular panels and the cover stats read the winning island's slice), Mission Performance
  (corridor plots with zone fills + nominal overlays, altitude / heat flux / g-load / bank /
  density ratio vs time with limit lines, DV distributions, entry/exit conditions, summary table
  with violation rates, dispersion correlations), optional Sensitivity Analysis (`--sensitivity`
  when `<scheme_dir>/sensitivity/sensitivity_results.json` exists). `_generate_training_charts`
  returns the `has_*` flags (`has_cost_distribution`, `has_islands`) that `metadata.json` carries
  and `report.typ` branches on. Also cross-scheme comparison PDFs (`--compare`), and
  `final_eval.parquet` next to the PDF. `read_cost_kwargs` delegates to `cost.build_cost_kwargs`.
- `charts.py` — one matplotlib/seaborn function per panel (25), SVG output, seaborn `whitegrid`
  + `muted`. Three-way trajectory classification: blue (captured + constraints OK), orange
  (captured + violation), red (crash / hyperbolic / timeout); captured = `(ifinal==3) &
  (ecc<1.0)`. Limits (including `heat_load_limit`) from `[flight.constraints]`.
  `chart_dispersion_grid()` takes an optional `traj_class`; `chart_heat_load_time()`;
  sensitivity charts `chart_morris_scatter`, `chart_sobol_bars`, `chart_sobol_heatmap`.
- `report_render.py` — the ONE render spine: `render_pdf` compiles every Typst template with
  `--root / --input dir=<assets>` (templates read their SVGs + JSON via `sys.inputs.at("dir")`;
  `TEMPLATES_DIR` is the one template-path constant), `staged_assets` owns the temp staging dir
  (removed unless `keep_artifacts`) for the report.py / report_rl.py renders; the warm-start and
  nn-input reports stage into their persistent output dirs. Gate: `tests/test_report_render.py`
  (the real drivers run with `render_pdf` faked and every staged SVG is checked against the
  template's static `dir + "/..."` set both ways, plus a real compile per template; CI installs
  Typst 0.15.1 for it). Templates under `src/typst/`: `report.typ` (cover + Part 1 incl. the
  islands panels behind `has_islands` + Part 2 + optional Part 3), `report_rl.typ` (RL Part 1 + a
  verbatim copy of Parts 2/3), `comparison.typ`, `warm_start_report.typ`, `nn_input_report.typ`,
  `lib.typ` (page style, colors, headings). External dependency: the `typst` CLI (`brew install
  typst` / `cargo install typst-cli`); without it charts are still generated, no PDF.
- `animate.py` — GIF of the training evolution: replays checkpoints, re-runs MC per frame, 2x2
  panels (corridor with envelope fills, inclination, bank angle, cost CDF with ECDF overlay).
- `corridor.py` — `CorridorAccumulator`: during `piecewise_constant` training each generation's
  trajectories (plus 11 constant-bank sentinel chromosomes from 0° to 180° in 18° steps, tracing
  the full lift-up / lift-down range) are classified (`classify_trajectories`, asymmetric
  `delta_za_low` / `delta_za_high`, `ifinal=4` pending crash recognized) and their pdyn envelopes
  updated incrementally (running max/min per energy bin). Produces the schema-v4
  `training_output/<mission>/corridor_boundaries.npz` (4 envelopes: crash, restricted upper/lower,
  capture; nominal trajectory; DV; Gaussian-smoothed at save) and `ref_trajectory.dat`.

## Reference trajectory and training order

The mission reference `training_output/<mission>/ref_trajectory.dat` is the product of
`make_reference.py`: a target-energy-matched CONSTANT-BANK nominal (seconds to run, no GA). This
is the reference family that trains FTC to legacy parity; GA-optimal open-loop profiles
systematically under-reach the target energy and make poor references (root `CLAUDE.md`,
"Reference Trajectory Design"). Piecewise runs can still write the reference (with the
COMMANDED-segment cos_bank) via `reference_only = true` under `[guidance.piecewise_constant]`
(`configs/training/msr_aller_pc_ref_train.toml`) without clobbering `corridor_boundaries.npz`,
which the full piecewise baseline run (train it first) produces. The three table-reading training
TOMLs (ftc, energy_controller, pred_guid; fnpag is in `REQUIRES_REF_TRAJECTORY` but never reads
the table) wire `data.reference_trajectory` to the mission file EXPLICITLY; base
`missions/mars.toml` keeps the legacy `data/reference_trajectory/msr_aller.dat` for test/nominal
configs and the goldens. `training_config.py::check_ref_trajectory_wiring` hard-errors when a ref-tracking
scheme's resolved config does not point at the mission's optimized ref. SimData re-reads the
file every generation, so never regenerate the reference while a ref-tracking scheme is
training. The file is exempted from .gitignore and committed so CI/e2e resolve it. NN training
configs stay on the legacy reference (the deployed models were trained with legacy-ref inputs
17-20).

**Joint reference optimization.** `[reference] joint_bank = true` (leaf:
`configs/training/msr_aller_ftc_joint_ref_train.toml`; eligible schemes `JOINT_REF_BANK_SCHEMES`,
others hard-error) appends a `ref_bank` gene (`bank_low` / `bank_high`, default [55, 80] deg).
Every evaluation path generates per-individual constant-bank tables (one batched undispersed
run_batch per evaluation, ~+10% cost) and injects per-individual `data.reference_trajectory`
overrides via the `_run_grid_records` chokepoint; the gene itself is NEVER routed to a guidance
TOML key. `run_grid` honors the override (`SimData::from_toml_with_tables` reloads the reference
when the patched path differs from `SharedTables.ref_trajectory_path`; gates
`tests/test_joint_ref_bank.py::test_run_grid_honors_reference_trajectory_override` and the Rust
`shared_tables_honor_per_individual_reference_override`). The winner deploys via
`deploy_optimized_artifacts` (refuses an EMPTY regenerated table) as `best_params.json` (carries
`ref_bank`) + a regenerated `<save_dir>/ref_trajectory.dat` with `optimized_<scheme>.toml` wired at
it; `compare_guidance` and `report.py` pick up the scheme-local table.
`experiments/paper/07_joint_reference.sh` trains the three table-reading schemes jointly (gene
bounds [40, 120]; budgets matching `experiments/paper/01_classical_baselines.sh`, GA 2000 gens x
n_pop 300, final eval n=1000; `--output-dir training_output/paper/joint_reference/<scheme>` keeps
the baselines intact).

## Warm-start (multi-supervisor BPTT)

Activated by a `[warm_start]` block (flips `WarmStartConfig.enabled = True` via `from_dict`) or
the legacy `[guidance.neural_network] warm_start_from`. Spec:
`docs/design/2026-05-22-warm-start-all-archs-design.md` (and the parity bundle
`docs/design/2026-05-07-nn-ftc-parity-bundle-design.md`).

`warm_start.build_warm_start_chromosome` orchestrates: each supervisor runs over the same
`n_warm_seeds` reserved pool (`WARM_START_SEED_OFFSET`) via `aerocapture_rs.collect_supervised`
(per-trajectory X, y_signed, prev_realized, dv, captured; `prev_realized` is the previous
pilot-realized bank, consumed by the `delta` target). The supervised target is
`guidance_out.pre_shaper_signed` (post-lateral, PRE-shaper SIGNED bank in [-π, π]: it preserves
the supervisor's lateral-chosen sign for `full_neural` deploy and avoids double-shaping, since the
shaper runs exactly once on the NN's output at runtime; collapsed to magnitude under
`magnitude_only`). `_select_best_teacher_per_seed` picks the captured trajectory with the lowest
DV per seed across schemes (seeds with no captures dropped; the corpus must clear
`max(20, n_warm_seeds // 4)` captures). `_chunked_bptt_train` splits each trajectory into
`bptt_length`-sized windows (trailing partial chunks dropped with a warning), forwards via
`V2Policy.forward_seq_means` (the autograd-friendly mirror of `evaluate` minus the log-prob math)
with zero-initialized recurrent state per chunk, and runs Adam MSE for `n_epochs` epochs
(`minibatch_size` chunks per step; `torch.manual_seed(seed)` for reproducible rebuilds). Loss
targets per decoder: `cos(y)` for `acos_tanh` (the tanh head read directly, no double-tanh),
`(sin(y), cos(y))` for `atan2_signed`, `clamp(y / (scaled_pi_n*pi), -1, 1)` for `scaled_pi`,
`clamp(wrap_to_pi(y - prev_realized) / delta_max, -1, 1)` for `delta`. `_seed_policy_init` applies
`init_v2_population`'s Mamba HiPPO + LSTM forget-bias-1 centers to the V2Policy BEFORE Adam
(otherwise `MambaLayer.__init__`'s zero-init starts at a degenerate fixed point). Per-layer
`to_flat()` on every layer module mirrors Rust `LayerWeights::to_flat`, so the trained policy
round-trips into the chromosome via `_policy_to_flat_weights_v2`. The chromosome is replicated
to `n_pop` with per-individual NN-weight jitter, and the scaffolding slab overridden when
`scaffolding != "off"` (`build_scaffolding_initial_slab` seeds `"full"` at FTC's optimum from
`training_output/ftc/best_params.json` + 0.02 normalized-space jitter;
`build_default_scaffolding_slab` seeds `"live"` from the ParamSpec defaults with no FTC
dependency).

With `adaptive_bounds = true` (default) the NN-weight ParamSpec bounds are derived post-Adam from
each layer slab's max-abs value with a 2x margin (floored at the Xavier × `bound_multiplier`
half-width), so encoding never clips; they persist to `<save_dir>/warm_start_bounds.json`
(`warm_start.load_warm_start_bounds`, also read by the `final_select` CLI) and
`build_warm_start_chromosome` returns `(chromosome, weight_specs)` whose specs replace
`param_specs[0..n_weights)` so every optimizer decodes under the bounds the encoding used.
`adaptive_bounds = false` falls back to static Xavier × `bound_multiplier` bounds plus a >5%
clip-rate guard. The cache key covers architecture + input_mask + output_param +
supervisor_schemes + per-scheme mtime + scaffolding source path/mtime + bound_multiplier +
adaptive_bounds + n_epochs + bptt_length + n_warm_seeds + mode + `base_mc_seed`; chromosome and
key persist to `<save_dir>/warm_start_chromosome.npy`. Sidecars: a gen-0 validation baseline of
the bare chromosome on the RESERVED VALIDATION pool (`warm_start_baseline.json`, directly
comparable to the later `Gen N validation:` lines), per-supervisor selection counts + capture
stats (`warm_start_selection.json`), then `warm_start_compare.render_trajectory_comparison`
(supervisor vs warm-started NN on the training and validation pools, 20 SVGs under
`<save_dir>/warm_start_report/compare_{train,val}_{supervisor,nn}_{corridor_pdyn,corridor_inclination,corridor_bank,altitude_time,heat_flux_time}.svg`
+ `compare_manifest.json`; the NN written via `write_nn_json`, the supervisor batch using
`deploy_overrides.overrides_from_params`; one (pool, side) at a time, peak ~600 MB at
n_warm_seeds=5000; best-effort per panel) and `warm_start_report.render_report` (supervised MSE
convergence, supervisor capture-vs-selection bars, per-layer-slab bounds; compiled into
`<save_dir>/warm_start_report.pdf` via `src/typst/warm_start_report.typ` when Typst is present;
standalone: `python -m aerocapture.training.warm_start_report <save_dir>`). `eval_interval > 0`
additionally runs MC on both pools every N epochs and on the last one (~`n_warm_seeds +
validation_n_sims` sims each).

## Paper tooling

- `aerocapture.demo` (`src/python/aerocapture/demo.py`) — the clone-to-figure demo: flies the
  deployed cell `models/demo/ft_mamba_962/` (a copy of `training_output/ou_marginal/ft_mamba_p962/`,
  the per-scenario fine-tune) over 500 per-seed MC sims (`cell_eval.evaluate_cell`, the paper's
  evaluation path) under `per_draw` noise and writes `demo_output/demo.svg`. `--legacy` flies the
  shared-path champion `models/demo/mamba_962_legacy/` (a copy of `training_output/mamba_p962_long/`)
  under `noise_seeding = "legacy"`; model and regime are printed and stamped on the figure. Demo
  seeds come from an arbitrary RNG stream (424242), disjoint from every reserved pool.
- Paper driver `articles/paper/Makefile` (GNU make 3.81-compatible; always `make -C
  articles/paper <target>`): `figures` (default) rebuilds the 18 `fig_*.svg` from `data/` + the
  bundle with real dependency semantics (a missing input is a make error, never a thinner
  figure); `results` = `aggregate_results.py` (results.json is deliberately NOT a make rule so a
  fresh clone's mtimes never trigger it); `fetch-logs` (the 195 MB run logs
  `articles/paper/data/runs/**/*.jsonl.gz` are untracked: `articles/paper/scripts/fetch_run_logs.sh`,
  a Release asset on the `arxiv-v2` tag); `provenance` writes `data/provenance.json` (a content
  SHA-256 over every tracked paper input plus Release tag, crate/typst/matplotlib versions,
  campaign TOML hashes; `--check` fails when stale); `pdf` (typst, `SOURCE_DATE_EPOCH` = HEAD
  commit time, `--input git_head=<short sha>[-dirty]` for the colophon; `PDF=/tmp/x.pdf` leaves
  the committed PDF alone); `confirmatory-marginal` writes `data/confirmatory_marginal.json`, the
  cells and fields the paper quotes from the per_draw far-tail confirmatory
  `experiments/ou_marginal/confirmatory_marginal.json` (`extract_confirmatory_marginal.py`, issue
  #137). `articles/paper/results.typ` is the compile-time seam: accessors over `results.json` /
  `confirmatory_eval.json` / `quant/finalists_results.json` (legacy regime, `legacy_regime()`) and
  `confirmatory_marginal.json` (per_draw, asserted at load) that fill every cell of the performance,
  paired-comparison, quantization-finalists and per-scenario far-tail tables (the Viol. column
  excepted: the bundle carries no violation field) and the per-scenario headline quotes of the
  abstract, Section 9 and the conclusion, and the colophon reads `provenance.json`; other prose
  numbers are still transcribed. `paper` chains them; `check` =
  `data/SHA256SUMS` recomputed over every tracked bundle file and diffed verbatim (`sums`
  regenerates it + `SHA256SUMS.runlogs`, the latter only when no fewer run logs are present than it
  lists) + `check_results_schema.py` +
  `extract_confirmatory_marginal.py --check` + `write_provenance.py
  --check` + the `FROZEN` files present (the 7 data files with no producer in the tree) + `git
  diff HEAD --exit-code` on figures, results.json and provenance.json; the opt-in `mc-*` targets
  re-fly cells (never default, never CI). Figures are byte-reproducible across macOS and Linux:
  `figlib` forces the Agg backend, sets `text.hinting = "none"`, `save` sets `svg.hashsalt` +
  `metadata={"Date": None}`, and `style()` registers the vendored STIX Two Text
  (`articles/paper/fonts/`, OFL) as the ONLY entry of that family. Bytes are stable only under the
  pinned matplotlib; a version bump regenerates all 18 (commit that deliberately). CI's `paper`
  job runs `-B figures` + `check` + a /tmp `pdf` (Typst 0.15.1); `paper-results`
  (workflow_dispatch) fetches the logs and runs `make paper` end to end. Gate:
  `tests/test_paper_figures.py`.
- `articles/paper/scripts/*.py` evaluate deployed cells on the reserved pools
  (`confirmatory_eval.py`: 10 replicate pools sharing seeds across schemes, paired replicate
  deltas, t-based SEs, per-cell `failed_seeds`, `--extra-override` / `--scaffolding-from` for
  ablation cells). `collect_runs.OU_MARGINAL` / `OFF_CAMPAIGN` and `aggregate_results.PAIRED` /
  `PER_DRAW_PREFIXES` decide what the bundle carries and stamp each run's `noise_seeding`.
- `experiments/ou_marginal/` — the frozen-vs-marginal noise quantification
  (`experiments/ou_marginal/quote_results.json`, `RESULTS.md`) and the per_draw retrain campaign for the five NN headline
  cells: `retrain_campaign.sh` (resumable: each cell continues from its latest checkpoint toward
  the 20000-gen target; configs `configs/training/ou_marginal/`, outputs
  `training_output/ou_marginal/<cell>/`), `quote_marginal.py` (the quote table, auto-discovering
  every `training_output/ou_marginal/*/best_model.json`), `phase2_campaign.sh` (`ft_<cell>`
  fine-tunes = frozen champion checkpoint + 2000 per_draw gens, and `<cell>_s2` / `_s3` scratch
  seed repeats with `--seed 2/3`; every repeat job strips `rng_state` from its copied checkpoint,
  because a checkpoint resume restores the saved trainer RNG state and would silently override
  `--seed`). Campaign runs use the sweep config's allocation (GA n_pop 60, training_n_sims 10),
  not the headline cells' CLI allocation (n_pop 512, training_n_sims 2).
- `experiments/paper/*.sh` — the numbered campaign runners (classical baselines, optimizer
  budget / dimensionality, joint reference, architecture sweep, quantization, the RL baseline
  `18_rl_baseline.sh`); `experiments/paper/README.md` documents them.
