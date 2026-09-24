# PyO3 bindings (`src/rust/aerocapture-py/`)

The one seam between the Rust simulator and the Python package: a separate workspace member crate
that imports as `aerocapture_rs`. ADR-0007 (`docs/adr/0007-the-pyo3-seam-has-five-tiers.md`) fixes
its shape (five tiers, each with a named gate); ADR-0004 makes `run_grid` the bit-identity
chokepoint every training-side evaluation goes through. Vocabulary: `CONTEXT.md`.

Build from the repo root (a build from inside the sub-crate goes stale against the root
lockfile):

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```

The training pipeline requires the extension; the batch evaluation path raises if it is absent.
Every Python module CI's pure-Python job imports must soft-import it (`tests/test_soft_import.py`).

## Module map

```
src/rust/aerocapture-py/src/
  lib.rs         — Module entry, TIERED (the module doc names each tier's gate; ADR-0007):
                     evaluate  `run_grid` / `run_batch` / `run_mc` / `run_with_draws`
                     config    `validate_config` / `load_config`
                     contract  `candidate_inputs` / `final_record_indices` / `layer_schema` + the width consts
                     nn        `flat_weights_to_json` / `collect_supervised` / `collect_nn_inputs` / `nn_forward` (gate-only stateless forward) / `nn_forward_sequence`
                     env       `BatchedSimulation` (the RL step API, see src/python/aerocapture/training/rl/README.md)
  config.rs      — TOML loading with base inheritance resolution + dot-path override merging (resolve_and_patch, shared by load_and_override and the no-IO validate_only)
  results.rs     — `BatchResults` pyclass with numpy getters (final_records (N,52), captured (N,), trajectories, dispersions (N,26)); no single-run type
  batch.rs       — Rayon parallel batch execution (`run_batch` / `run_mc` / `run_with_draws`)
  grid.rs        — `run_grid`: the (individuals x seeds) grid, one SimData per individual over Arc-shared tables
  env.rs         — `BatchedSimulation`: N `SimState`s over one `Arc<SimData>`, Rayon-parallel ticks through `tick::act_tick` then the next `tick::sense_tick` (obs = the NN input deploy reads next tick; action = the NN output; `full_neural` only), `predicted_dv_for_state` for the aux channel
```

## Evaluate tier

- One run = `aerocapture_rs.run_batch(toml_path, [{}])` row 0 (the deploy path the CLI bit-identity gate compares against; `final_record_indices()` names the columns). There is no
  single-run entry point or `SimResult` type.
- `aerocapture_rs.run_mc(toml_path, overrides=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults` with all n_sims results. When `include_trajectories=True`, populates
  per-timestep trajectory data (N, 17) for corridor/time-domain plots. Trajectory columns: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, heat_flux_kw_m2, time_s, energy_mj_kg, pdyn_kpa,
  bank_angle_deg, inclination_deg, g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2, density_perturbation]. `.dispersions` (N, 26) always populated.
- `aerocapture_rs.run_batch(toml_path, overrides_list, n_threads=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults` with `.final_records` (N, 52), `.dispersions` (N, 26). One
  sim per override (typed `BatchError`): raises `ValueError` (contract) if a resolved config has `n_sims > 1` (use `run_mc` for multi-sim per config), and `RuntimeError` for config-load / TOML-parse /
  sim runtime failures. Empty trajectory entries carry the correct `(0, 17)` column count. Preloads the base config's atmosphere/wind/ref tables once and Arc-shares them across overrides whose
  `data.atmosphere`/`data.wind_table` paths are unchanged (an override retargeting either falls back to the full per-override load, so path overrides still work — unlike `run_grid`, which
  hard-errors on them); bit-identity gated by `test_pyo3_matches_subprocess`.
- `aerocapture_rs.run_with_draws(toml_path, draws, overrides=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults`; accepts a numpy array of shape (N, 26) as pre-computed
  dispersion draws, bypassing internal draw generation. Each row is one draw; `dispersions` output echoes the input draws exactly. Use this for SALib sensitivity matrices or any externally-structured
  sampling.
- `aerocapture_rs.run_grid(toml_path, overrides_list, seeds, weights=None, architecture_json=None, input_mask=None, output_param=None, scaled_pi_n=None, delta_max=None, normalization_json=None,
  n_threads=None, sim_timeout_secs=None)` → numpy `(n_pop, n_seeds, FINAL_RECORD_LEN)` f64 array. Evaluates the full individuals×seeds grid in ONE GIL-releasing call, building each `SimData` once
  (Arc-shared atmosphere/wind/ref tables) and reusing it across the seed axis; NN individuals pass flat `weights` + `architecture_json` in-memory (no temp JSON per individual). The reference
  trajectory is the one shared table that RELOADS per individual when a `data.reference_trajectory` override retargets it (`SharedTables.ref_trajectory_path` comparison inside
  `from_toml_with_tables` -- the joint `ref_bank` delivery path; gate: `tests/test_joint_ref_bank.py::test_run_grid_honors_reference_trajectory_override`). `overrides_list` MUST NOT carry
  `monte_carlo.seed` / `simulation.n_sims` (run_grid owns the seed axis — raises `ValueError` otherwise) nor `data.atmosphere` / `data.wind_table` (genuinely shared — hard error, use `run_batch`).
  This is the GA population-eval hot path (`problem._run_batch_pyo3`), bit-identical to a per-seed `run_batch` loop.

Override dicts use dot-separated TOML key paths with type coercion (int→float when the existing
field is float).

## Config tier

- `aerocapture_rs.validate_config(toml_path, overrides)`: post-override no-IO config validation via `config::validate` — reads no table; `ValueError` on the first violation.
- `aerocapture_rs.load_config(toml_path)` → Python dict after Rust-side `base` resolution; its one job is the parity oracle for `toml_utils.load_toml_with_bases`
  (`tests/test_pyo3.py::TestLoadConfig::test_base_resolution_parity` over every `configs/**/*.toml`).

## Contract tier

- `candidate_inputs()` is the ONE candidate-input schema: 35 `{index, name, transform, scale, center}` dicts exporting the Rust-owned contract
  (see `src/rust/src/data/neural/README.md`). `training/config.py::candidate_input_names()` / `candidate_input_index()` / `candidate_input_normalization()` derive the Python name list, width and
  index lookups from it (a fallback tuple covers machines without the extension, asserted equal element-wise by `tests/test_record_index_drift.py`); its `{transform, scale, center}` projection is
  the Rust `DEFAULT_NORMALIZATION` table -- the FALLBACK for `calibrate_inputs.py`'s normalized->raw inversion (which resolves override > embedded > default via `_resolve_normalization`, matching
  the forward pass so the recovery is exact). The former `default_normalization()` / `NN_INPUT_NAMES` exports were projections of it and are gone.
- `final_record_indices()` exposes the Rust final-record column map for the Python drift tests.
- `layer_schema(spec_json)` returns one layer spec's `(tensor name, shape)` list in flat order (the `tensor_table!` declaration), asserted against the Python fallback by `tests/test_layer_schema_drift.py`.

## NN tier

- `flat_weights_to_json(...)` (PSO chromosome -> deployed `best_model.json`) embeds the `normalization` block and the decoder knobs, so deployed models are self-describing. Rust is the single
  source of truth for PSO NN weight serialization.
- `collect_supervised(...)`: per-tick candidate-input trace from a teacher scheme, for NN warm-start (returns per-trajectory X, y_signed, prev_realized, dv, captured).
- `collect_nn_inputs(...)`: per-tick candidate-input trace from the deployed NN itself — no teacher override, rejects non-NN configs; returns per-seed `{seed, X (T,35), time, energy, dv, captured}`;
  powers the NN input behavior report and the input calibration.
  Both collect helpers load the config + tables ONCE and run seeds in PARALLEL via `run_for_api_cell` (the run_grid bit-identity chokepoint). The per-tick candidate trace is recorded in `tick.rs`
  with a full mask of width `NN_FULL_INPUT_SIZE`, so all 35 inputs (incl. the live correction-DV) reach the trace.
- `nn_forward(...)` (stateless single step) and `nn_forward_sequence(...)` (stateful multi-step forward over an input sequence) exist only for the per-layer cross-language equivalence gates.
