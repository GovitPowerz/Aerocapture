# Architecture

One page for a reader who has run the demo and wants to know how a training run and a simulation
actually flow through this code. Vocabulary is `CONTEXT.md`'s; decisions this page rests on are in
`docs/adr/`; per-module detail is in `CLAUDE.md`.

## Two languages, one seam

- **Rust** (`src/rust/`) is the simulator: physics, navigation, the seven guidance schemes, the NN
  runtime, Monte Carlo dispersions. It is a library crate plus a CLI (`aerocapture <config.toml>`).
- **Python** (`src/python/aerocapture/`) is everything around it: TOML config resolution, the
  population optimizers (pymoo), seed pools, cost, reports, the paper's evaluation scripts.
- They meet at **one seam**, the PyO3 crate `aerocapture_rs` (`src/rust/aerocapture-py/`). Training
  goes through `run_grid` (one GIL-releasing call per population x seed grid; the bit-identity
  chokepoint, ADR-0004). Deploy-side evaluation (reports, demo, paper scripts) goes through
  `run_batch` with one override dict per seed.

## A training run in fifteen lines

`python -m aerocapture.training.train <config.toml>`:

1. `build_training_config_from_toml` (train.py) resolves the TOML `base` chain
   (`toml_utils.load_toml_with_bases`) into a `TrainingConfig`; `check_ref_trajectory_wiring`
   refuses a ref-tracking scheme that does not point at the mission's optimized reference.
2. `_setup_param_specs` builds the chromosome: a `ParamSpec` per gene (NN weights from the
   `[[network.architecture]]` via `encoding.nn_param_specs_from_v2`, plus the scaffolding slab when
   `scaffolding != "off"`, plus `ref_bank` for joint-reference runs).
3. `_build_initial_population` seeds the population (activation-aware init, or the warm-start
   chromosome from `warm_start.build_warm_start_chromosome`).
4. `AerocaptureProblem` (problem.py) is the pymoo problem: decode normalized genes, route them to
   TOML dot paths, and evaluate through `evaluate_population_per_seed` -> `_run_grid_records` ->
   `aerocapture_rs.run_grid`. Every training-side evaluation (population, curation, validation
   gate, final selection) is this one call.
5. `warm_start_algorithm` hands the seeded population to the pymoo algorithm (GA / CMA-ES / DE /
   PSO / QPSO) without letting pymoo re-initialize it, and seeds pymoo's RNG from the training RNG.
6. `train.train` runs the per-generation loop against one of two adapters from `trainer.py`,
   `SingleAlgoTrainer` or `IslandsTrainer`; per generation:
   `_apply_seed_strategy` (fixed / rotating / adaptive seeds, ADR-0001) -> `re_evaluate` if the
   seeds changed -> `advance` (one pymoo step) -> `observe` (the **validation gate**:
   `evaluate.run_validation_gate` re-scores a new argmin on the reserved validation pool and
   promotes the **champion** only on strictly lower val RMS) -> `_maybe_curate` (adaptive seeds:
   `seed_curator.SeedCurator`) -> `emit` (JSONL + TUI) -> `maybe_checkpoint`.
7. `finalize`: **final selection** (`final_select.select_final_individual`, ADR-0002) re-ranks the
   last population plus the champion on the validation pool; `write_best_artifacts` writes
   `best_model.json` / `best_params.json` (+ `deploy_optimized_artifacts` for classical schemes).
8. `report.run_final_evaluation` scores the deployed winner once on the disjoint final-eval pool
   (report-only, never selects) and `report.py` renders the PDF.

Checkpoints (`checkpoint_g*.{json,npz}`) make every step resumable; `--n-gen` on resume means
"N more generations".

## A simulation tick in eight lines

`aerocapture_rs.run_batch` / `run_grid` -> `runner::run_for_api_cell` -> `run_init::build_sim_state`
(one `SimState` per dispersion draw), then `tick::step_one_tick` until termination:

1. Navigation (`runner::navigate_from_state` -> `gnc::navigation::estimator`): bias-mode or EKF
   state estimate, density factor, phase (capture / exit).
2. Guidance dispatch (`gnc::guidance::dispatch::guidance_step`): the scheme's capture law (FTC,
   NN, EqGlide, Energy Controller, PredGuid, FNPAG, Piecewise Constant); after the bounce the
   shared exit law (`exit::exit_guidance`) for unsigned-magnitude schemes.
3. Lateral guidance picks the sign (roll reversals), `thermal_limiter::apply_thermal_limit` ramps
   toward lift-up near heat limits, `CommandShaper` rate/acceleration-limits the command.
4. Pilot dynamics (`gnc::control::pilot::apply_pilot`) realize the bank angle.
5. Integration: fixed Gill RK4 (`integration::rk4::rk4_step`) or adaptive DOPRI45 with sub-tick
   events (`runner::integrate_adaptive_with_events` + `integration::events::check_events_and_locate`:
   bounce, atmosphere exit, crash, phase transition).
6. Termination bookkeeping (`SimState.term`: captured / hyperbolic / crash / pending crash /
   timeout) and NN telemetry update.
7. `finalize::build_final_record` assembles the 52-column final record; captured runs get the real
   correction DV (`orbit::maneuver::compute_deltav`), the others a virtual DV so every outcome
   stays comparable in cost.

The RL env (`aerocapture-py/src/env.rs`, `BatchedSimulation`) drives the same `step_one_tick`.

## Seed pools

All in `training/seeds.py` (`make_reserved_seeds(base_mc_seed, offset, n)`), one RNG stream each:

| pool | offset | used by |
|---|---|---|
| training | (strategy-dependent) | the optimizer, excluding every pool below |
| validation | 1 000 000 | validation gate, final selection |
| final eval | 2 000 000 | `report.run_final_evaluation` (report-only) |
| RL training | 3 000 000 | `rl/train.py` |
| warm-start | 4 000 000 | supervisor trajectory collection |
| NN input report | 5 000 000 | `nn_input_report.py` |
| calibration | 6 000 000 | `calibrate_inputs.py` |
| sweep eval | 7 000 000 | `param_sweep.py --eval` |
| headline requote | 8 000 000 | the paper's abstract number |
| stress | 9 000 000 | off-nominal robustness scripts |
| probe | 10 000 000 | architecture probes |
| confirmatory | 20 000 000 (seeds in [2^31, 2^32)) | `make_confirmatory_pools`, the 10 x 100k sizing pools |

## Where the paper's numbers come from

`articles/paper/scripts/*.py` evaluate deployed cells (`training_output/<cell>/best_model.json` +
`best_params.json`, resolved by `deploy_overrides.resolve_eval_toml`) on the pools above and write
`articles/paper/data/*.json`; the Typst source reads those files. `models/demo/mamba_962/` is the
committed copy of the headline cell that `aerocapture.demo` flies.

## `training/` by role

- **Config and chromosome**: `config.py`, `toml_utils.py`, `param_spaces.py`, `encoding.py`,
  `initialization.py`, `initialization_v2.py`, `population.py`
- **Loop and optimizers**: `train.py` (orchestration + CLI), `trainer.py` (loop contract, two
  adapters), `optimizer.py`, `qpso.py`, `island_model.py`, `seed_curator.py`, `final_select.py`
- **Evaluation**: `problem.py` (`run_grid` chokepoint), `evaluate.py` (validation gate, NN JSON
  writer), `cost.py`, `seeds.py`, `deploy_overrides.py`, `reference.py`, `make_reference.py`
- **NN specifics**: `layer_schema.py` (per-layer tensor names/shapes/flat order, read from the
  Rust `tensor_table!` via `aerocapture_rs.layer_schema`), `warm_start.py`, `model_io.py`,
  `calibrate_inputs.py`, `ablation.py`, `nn_input_report.py`, `quantize.py`, `weight_stats.py`,
  `rl/` (PPO/SAC, shelved)
- **Reports**: `report.py`, `charts*.py`, `display.py`, `logger.py`, `metrics.py`, `animate.py`,
  `corridor.py`, `parquet_output.py`, `typst_utils.py`, `warm_start_report.py`,
  `warm_start_compare.py`, `compare_guidance.py`, `sensitivity.py`, `param_sweep.py`
- **Experiments**: `experiments/` (architecture probes), `cleanup_checkpoints.py`, `paper_stats.py`
