# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerocapture is a trajectory simulation tool for aerocapture maneuvers (primarily Mars Sample Return). The **Rust simulator** with **Python analysis tools** was validated against a legacy reference implementation to bit-level precision — FTC guided trajectories matched across all 725 timesteps (22/24 photo columns exact; the remaining 2 were uninitialized variable artifacts in the reference).

The simulation models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit. Includes altitude-dependent wind model (zonal/meridional profiles with MC dispersions) and two navigation modes: legacy bias-only or 13-state EKF (IMU + star tracker with atmospheric blackout + drag-derived density estimation). The GNC chain is: Navigation (bias mode or EKF, with phase management: capture/exit/emergency) -> Guidance (one of 7 algorithms: FTC, NN, Equilibrium Glide, Energy Controller, PredGuid, FNPAG, Piecewise Constant; FTC + 4 unsigned-magnitude schemes switch to a shared exit-phase controller after trajectory nadir) -> Thermal Limiter (GA-tunable smooth ramp to lift-up near heat flux/load limits, unsigned-magnitude schemes only) -> Control (pilot dynamics + roll reversal). Schemes providing signed bank angles (NN, Piecewise Constant) bypass lateral, exit, and thermal limiter guidance -- NN computes 23 candidate inputs (16 orbital/aero/thermal state + 4 reference trajectory interpolations + 3 bounce-gated exit controller signals) and selects a configurable subset via an input mask; operates as a single phase-blind controller across capture and exit phases. All guidance schemes have TOML-configurable parameters and can be GA-optimized.

## Build & Development Commands

```bash
# ── Rust Simulator ──
cd src/rust
cargo build --release              # Build optimized binary
# Run from repo root:
./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml

# ── PyO3 Bindings ──
cd src/rust/aerocapture-py
maturin develop --release          # Build + install aerocapture_rs module
# Or via uv:
uv run maturin develop --release

# ── Python Analysis ──
uv sync                            # Install dependencies (Python >=3.14)
uv sync --group dev                # Include dev tools (pytest, ruff, mypy, maturin)
pytest tests                       # Run all tests
pytest tests/test_foo.py::test_bar -v

# ── Utility Scripts (from repo root) ──
./build.sh                         # Build Rust binary + PyO3 bindings (-c to clean artifacts)
./setup_env.sh                     # Create fresh .venv + install deps
./lint_code.sh                     # Run ruff (imports, format, lint) + mypy
./check_all.sh                     # Rust: test + fmt --check + clippy + release build
./upgrade_dependencies.sh          # uv sync --upgrade
./train_all.sh                     # Train all 7 guidance schemes with optimized GA settings
./train_all.sh eqglide             # Train a single scheme (aliases: pc, eq, ec, pg, nn, etc.)
```

## Architecture

### Rust Simulator (`src/rust/`)

The crate has both `lib.rs` (public API: `RunOutput` struct + `run_for_api()`) and `main.rs` (CLI entry). A Cargo workspace contains two members: the core `aerocapture` crate and the `aerocapture-py` PyO3 binding crate. TOML config as a CLI argument (`./aerocapture config.toml`) is the only supported input format. TOML supports all 7 guidance schemes and inline vehicle/mission data.

```
src/rust/src/
  main.rs                          — CLI entry, TOML config loading
  config.rs                        — TOML parser (PlanetConfig, MissionType, SimInput, IntegrationMode, AdaptiveConfig) + base inheritance (deep_merge, resolve_toml_bases, from_toml_file)
  data/
    mod.rs, SimData                — Top-level data container
    atmosphere.rs                  — Atmosphere density table + OnboardAtmosphereModel (piecewise exponential, auto-fitted or explicit)
    aerodynamics.rs                — Cx/Cz vs AoA tables
    capsule.rs                     — Vehicle: mass, reference area, max bank rate
    guidance_params.rs             — Guidance law config: FTC gains, LateralParams, ThermalLimiterParams, EqGlide, EnergyCtrl, PredGuid, FNPAG params
    dispersions.rs                 — Monte Carlo dispersion profiles (all domains use level presets: Off/Low/Medium/High/Custom) + Gauss-Markov density perturbation (OU process config + step function) + SamplingMethod enum (Random/LHS/Sobol) + norm_ppf (Acklam inverse normal CDF) + DimTransform enum (Gaussian/Uniform/UniformRange/Fixed) + build_dim_transforms() mapping 26 dispersion dims to their transforms + generate_lhs_unit_samples() (stratified Fisher-Yates) + generate_sobol_unit_samples() (sobol_burley) + DispersionDraw::from_array() (inverse of to_array()) + draws_from_unit_samples() (applies DimTransforms to unit samples) + generate_draws() dispatch (Random/LHS/Sobol) + generate_draws_random() (legacy PRNG path, backward-compatible)
    navigation.rs                  — Navigation error profiles
    incidence.rs                   — AoA profile tables
    pilot.rs                       — Pilot dynamics parameters
  physics/
    gravity.rs                     — J2/J3/J4 zonal harmonic gravity
    atmosphere.rs                  — Density lookup
    aerodynamics.rs                — Force computation
    winds.rs                       — Altitude-dependent wind model (WindTable loader, latitude-scaled zonal winds, MC dispersions)
  gnc/
    navigation/
      estimator.rs                 — Navigation orchestrator: bias mode (legacy) or EKF mode via NavigationFilter enum; phase management (capture/exit/emergency) gated by SimPhase config; density estimation via lift-corrected inverse dynamics (body-frame: Cx*cos(alpha) + Cz*sin(alpha) denominator); legacy filter with rate-of-change limiting (density_gain_max_delta) + gain saturation [0.1, 10.0]; NavigationOutput includes thermal fractions (heat_flux_fraction, heat_load_fraction) for guidance limiter and NN inputs
      ekf.rs                       — 13-state Extended Kalman Filter (error-state: pos/vel errors, accel/gyro biases, density correction)
      imu.rs                       — IMU sensor model (accelerometer + gyroscope with bias, scale factor, noise)
      star_tracker.rs              — Star tracker model (position updates with dynamic pressure blackout)
      coordinates.rs               — Spherical<>Cartesian, geodetic, total energy
    guidance/
      dispatch.rs                  — Central guidance dispatch (phase-aware: routes to exit guidance when guidance_phase=2), GuidanceState, GuidanceOutput; CommandShaper (acceleration-limited S-curve rate shaping with realized-angle feedback; falls back to legacy hard-clamp when config absent)
      ftc.rs                       — FTC capture-phase guidance: altitude-gain predictor-corrector (FtcCaptureState, ftc_bank_angle)
      exit.rs                      — Exit phase guidance: shared pdyn-feedback controller for ascending leg (FTC + 4 unsigned-magnitude schemes)
      lateral.rs                   — Lateral guidance (roll reversal): LateralParams, LateralState, predictive first-order inclination projection (shared by unsigned-magnitude schemes)
      reference.rs                 — Constant bank angle mode
      neural.rs                    — NN guidance (modular JSON architecture, GA-trained, signed bank via atan2, 23 candidate inputs with configurable input mask: 16 orbital/aero/thermal + 4 ref trajectory + 3 bounce-gated exit signals; ablation analysis support via ablated_input)
      equilibrium_glide.rs         — Equilibrium glide with hdot damping + velocity bias
      energy_controller.rs         — Energy dissipation tracking via pdyn/hdot feedback
      predguid.rs                  — Apollo/Shuttle-heritage drag tracking guidance
      fnpag.rs                     — Lu's numerical predictor-corrector (FNPAG): 3D 6-DOF forward predictor (J2/J3/J4 gravity, Coriolis/centrifugal, onboard atmosphere, RK4 integration, inertial exit energy via total_energy(); zero lateral lift since roll sign unknown to predictor)
      piecewise_constant.rs        — 10-segment bank angle profile (GA-optimized, produces ref trajectory + corridor)
      thermal_limiter.rs           — Thermal safety limiter: smooth bank-to-lift-up ramp near heat flux/load limits (GA-tunable, unsigned-magnitude schemes only)
    control/
      angle_utils.rs               — `shortest_angle_diff()`: wrap-aware angular difference in [-π, π]
      pilot.rs                     — Pilot dynamics (wrap-aware via angle_utils)
      attitude.rs                  — Attitude command realization
  integration/
    dopri45.rs                     — Dormand-Prince 4(5) adaptive integrator (FSAL, PI step-size control, mixed atol/rtol error norm); `dopri45_step` delegates to `dopri45_step_with_stages` (single implementation); dense output (Hermite continuous extension via `dopri45_dense`)
    events.rs                      — Event detection for adaptive integration: EventDef/EventAction/EventType framework, Brent's root-finding, `check_events_and_locate` (sign-change detection + direction filtering + earliest-event arbitration on dense output), `build_aerocapture_events` (4 events: bounce/atmosphere exit/crash/phase transition); event functions use latitude-dependent ellipsoid radius for oblateness-consistent altitude; `TriggeredEvent` carries absolute time
    rk4.rs                         — Gill-variant RK4 (fixed-step, legacy default)
    sequencer.rs                   — Module cadence scheduling
  orbit/
    elements.rs                    — Orbital elements from state vector
    maneuver.rs                    — Delta-V cost computation (only called for confirmed captures)
  simulation/
    runner.rs                      — Main sim loop: run() for CLI, run_for_api() for PyO3, run_for_api_with_draws() for external-draw API; dispatches between fixed Gill RK4 and adaptive DOPRI45 based on IntegrationMode; DOPRI45 mode uses `integrate_adaptive_with_events` (returns Vec<TriggeredEvent> for all events in a tick, processed chronologically) for sub-tick event detection (bounce, atmosphere exit, crash, phase transition) via dense output + Brent's root-finding (~1 ms precision); fixed RK4 uses legacy post-tick threshold checks (unchanged); tracks peak heat flux, g-load, dynamic pressure; NaN/Inf state termination (prevents infinite loops from extreme GA params); optional wall-clock timeout per sim (prevents Rayon batch blocking); pending crash detection (ifinal=4); atmospheric apoapsis crash (bounce_alt > 20km + descending + still in atmosphere); virtual DV for all termination outcomes; event records interleaved into trajectory output (sorted by time)
    init.rs                        — Per-run initialization
    output.rs                      — File writers (photo, final, CSV)
```

Key Rust dependency: `nalgebra` for vector/matrix ops.

### PyO3 Bindings (`src/rust/aerocapture-py/`)

Separate workspace member crate providing Python bindings via PyO3. Built with `maturin develop --release`. Imports as `aerocapture_rs` in Python.

```
src/rust/aerocapture-py/src/
  lib.rs         — Module entry: run(), run_mc(), run_batch(), run_with_draws(), load_config()
  config.rs      — TOML loading with base inheritance resolution + dot-path override merging
  results.rs     — SimResult/BatchResults pyclasses with numpy getters
  batch.rs       — Rayon parallel batch execution
```

Key API:
- `aerocapture_rs.run(toml_path, overrides=None, sim_timeout_secs=None)` → `SimResult` with `.final_record` (52,), `.captured`, `.energy`, `.ecc`, `.dispersions` (26,), etc. Returns first result only (use `run_mc` for multi-sim).
- `aerocapture_rs.run_mc(toml_path, overrides=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults` with all n_sims results. When `include_trajectories=True`, populates per-timestep trajectory data (N, 17) for corridor/time-domain plots. Trajectory columns: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, heat_flux_kw_m2, time_s, energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg, g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2, density_perturbation]. `.dispersions` (N, 26) always populated.
- `aerocapture_rs.run_batch(toml_path, overrides_list, n_threads=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults` with `.final_records` (N, 52), `.dispersions` (N, 26)
- `aerocapture_rs.run_with_draws(toml_path, draws, overrides=None, include_trajectories=False, sim_timeout_secs=None)` → `BatchResults`; accepts a numpy array of shape (N, 26) as pre-computed dispersion draws, bypassing internal draw generation. Each row is one draw; `dispersions` output echoes the input draws exactly. Use this for SALib sensitivity matrices or any externally-structured sampling.
- `aerocapture_rs.load_config(toml_path)` → Python dict

The training pipeline (`evaluate.py`) auto-detects PyO3 availability and falls back to subprocess if not installed. Override dict uses dot-separated TOML key paths with type coercion (int→float when existing field is float).

### Data Files (`data/`)

- `data/atmosphere/mars.dat` — Mars density vs altitude table (tabulated MarsGram 3.8)
- `data/atmosphere/earth.dat` — Earth atmosphere table
- `data/atmosphere/mars_winds.dat` — Mars parametric wind profile (altitude vs zonal/meridional, based on Forget et al. 1999)
- `data/atmosphere/earth_winds.dat` — Earth parametric wind profile
- `data/reference_trajectory/msr_aller.dat` — MSR reference trajectory (energy vs pdyn/hdot/cos_bank)
- `data/reference_trajectory/esr_aller.dat` — ESR reference trajectory

### Input Configuration

TOML config files in `configs/` are the only supported input format, organized into subdirectories: `configs/planets/` (planet physical constants: mu, radii, omega, J2/J3/J4), `configs/missions/` (shared per-planet base configs, inherit from planets/), `configs/nominal/` (simulation configs), `configs/training/` (GA training configs), `configs/test/` (golden test configs).

**Base inheritance:** Configs support a `base` key (string or array of strings) that references parent TOML files, resolved relative to the declaring file. The loader deep-merges bases left-to-right, then overlays the child's own keys. This eliminates duplication — mission-level content (entry, vehicle, aero, flight, orbit, success, incidence, atmosphere paths) lives in `configs/missions/mars.toml` or `earth.toml`, common training settings (MC dispersions, cost function, optimizer defaults) live in `configs/training/common.toml`, and each leaf config only specifies its overrides (guidance type, n_sims, results_suffix). Both Rust (`resolve_toml_bases()` in `config.rs`) and Python (`load_toml_with_bases()` in `toml_utils.py`) implement the same resolution logic.

Each config specifies planet, mission, guidance scheme, vehicle, entry conditions, aerodynamics, Monte Carlo settings, and data file paths. Planet constants are defined in a `[planet]` section (name, mu, equatorial_radius, polar_radius, omega, j2, j3, j4) — typically inherited from `configs/planets/*.toml` via base inheritance. J3 and J4 default to 0.0 if omitted (J2-only behavior). Adding a new planet requires only a new TOML preset file, no Rust changes. An optional `[onboard_atmosphere]` section configures the onboard atmosphere model: `mode = "identical"` (uses truth table, backward compatible), `n_segments = N` (auto-fit N piecewise exponential segments from truth), or explicit `segments = [...]`. Default (no section): auto-fit with 5 segments. Navigation and guidance query the onboard model; physics propagation uses the truth table with MC dispersions. Mission TOMLs include a `[corridor]` section with asymmetric restricted corridor bounds (`delta_za_restricted_low`, `delta_za_restricted_high` in km). The NN weight file path (`[data] neural_network`) and optional architecture override (`[network] layer_sizes`, `activations`) are read from TOML at training time. The `[network]` section also supports `input_mask` (list of indices into the 23-element candidate input vector, selecting which inputs reach the network; absent defaults to [0..16] for backward compat) and `ablated_input` (index to zero out for ablation analysis). Both are parsed by Rust config and override JSON model file values. The `[simulation]` section supports `max_time` (default: 3000.0 s) as a hard wall to prevent runaway simulations. Mission TOMLs include a `[flight.constraints]` section with the authoritative constraint limits (`max_heat_flux` in kW/m², `max_load_factor` in g, `max_dynamic_pressure` in kPa, `max_heat_load` in kJ/m²) — used by the GA cost function, report violation rates, chart limit lines, and trajectory classification. Training TOMLs include a `[cost_function]` section with penalty weights (`g_load_weight`, `heat_flux_weight`, `heat_load_weight`) and the log-cap DV compression threshold (`dv_threshold`, default 1000.0 m/s). Training TOMLs also support an `[optimizer]` section to configure the pymoo optimization algorithm without CLI flags: `algorithm` ("ga", "cma_es", "de", "pso"), `n_pop`, `n_gen`, `seed_strategy` (required: `"fixed"` | `"rotating"` | `"adaptive"`), `training_n_sims` (seed list length / sims per individual, default 20), `seed_pool_interval` (periodic curation fallback interval in generations for `"adaptive"` strategy, default 50), `validation_n_sims` (sims for validation gate, default 1000), `curation_sample_size` (candidate seeds drawn per curation for `"adaptive"` strategy, default 1000), `curation_top_k` (top individuals used for curation scoring for `"adaptive"` strategy, default 5); and nested subsections `[optimizer.ga]` (`crossover_eta`, `mutation_eta`), `[optimizer.cma_es]`, `[optimizer.de]`, `[optimizer.pso]`. Defaults live in `configs/training/common.toml`; CLI args (`--n-gen`, `--n-pop`, `--algorithm`) override TOML values when explicitly provided. An optional `[integration]` section selects the integration method: `mode = "fixed"` (default, Gill-variant RK4) or `mode = "adaptive"` (Dormand-Prince 4(5) with error control). Adaptive mode supports `rtol` (default 1e-6), `initial_dt` (default 0.1 s), `min_dt` (default 1e-6 s), and `max_dt` (default = `periods.integration`). The adaptive integrator sub-steps within each outer GNC tick — GNC cadences are unchanged. An optional `[guidance.lateral]` section configures lateral guidance (predictive roll reversal) for unsigned-magnitude schemes (EqGlide, EnergyController, PredGuid, FNPAG): `tau` (lookahead horizon in seconds), `threshold` (projected inclination error threshold in degrees), `min_reversal_interval` (anti-chatter minimum seconds between reversals), `lateral_activation` (MJ/kg, upper energy threshold), `lateral_inhibition` (MJ/kg, lower energy threshold), `max_reversals`. The algorithm projects inclination error forward by tau seconds using finite-difference rate estimation and reverses only when the projected error exceeds the threshold. If absent, lateral guidance is inactive (backward compatible). These parameters are GA-optimizable for the five unsigned-magnitude schemes. NN and PiecewiseConstant bypass lateral guidance entirely (they produce signed bank angles). An optional `[guidance.thermal_limiter]` section configures the thermal safety limiter for unsigned-magnitude schemes: `heat_flux_activation` (fraction of max, 0.6-1.0), `heat_load_activation` (fraction of max, 0.6-1.0), `heat_flux_ramp_exponent` (1.0=linear, 2.0=quadratic), `heat_load_ramp_exponent`. Default activation=1.0 (disabled). When active, smoothly blends bank angle toward full lift-up as thermal quantities approach constraint limits. These 4 parameters are GA-optimizable for the five unsigned-magnitude schemes. An optional `[guidance.command_shaping]` section enables acceleration-limited S-curve rate shaping in the dispatch layer: `enabled` (bool, default true when section present), `max_bank_acceleration` (deg/s^2, must be > 0). When absent or `enabled = false`, dispatch falls back to legacy hard-clamp rate saturation (backward compatible). Shaping uses `bank_angle_realized` (pilot feedback) as the baseline each tick, not the previous command. The `[mission] phase` key selects the simulation phase mode: `"full"` (default, capture + exit phases with automatic transition), `"capture_only"` (force phase 1 throughout, backward-compatible), `"exit_only"` (force phase 2 throughout, for testing), or `"preprogrammed"` (same as full). The phase transition fires after bounce when velocity drops below `exit_velocity_threshold`. An optional `[monte_carlo.density_perturbation]` section configures time-varying (Gauss-Markov / Ornstein-Uhlenbeck) density perturbations: `level` selects a preset ("off", "low", "medium", "high", "custom") with preset tau/sigma pairs; custom mode accepts `tau` (correlation time in seconds) and `sigma` (steady-state RMS fractional amplitude). Default (absent): disabled. The perturbation evolves during each sim run, producing correlated density noise on top of the static MC density bias. An optional `[monte_carlo.wind]` section configures wind dispersions using the same level pattern as other MC domains: `level` selects a preset ("off", "low", "medium", "high", "custom") controlling wind speed scale range and direction bias. Custom mode accepts `scale_min`, `scale_max` (uniform multiplicative bounds on wind speed) and `direction_bias_deg` (max rotation in degrees). For backward compatibility, configs without a `level` field default to "medium" and explicit values override the preset. An optional `[monte_carlo] sampling` key selects the draw generation strategy: `"random"` (default, standard pseudo-random), `"lhs"` (Latin Hypercube Sampling -- stratified coverage, better space-filling for N>1), or `"sobol"` (Owen-scrambled Sobol quasi-random sequence via `sobol_burley` crate, max 65536 samples). Absent key defaults to `"random"` -- all existing configs work unchanged. LHS/Sobol only improve batch draws (n_sims>1); single-sim runs and adaptive-strategy curation probing (1 sim per probe seed) are unaffected.

### Python Tools (`src/python/`, `pyproject.toml`)

Python analysis package (numpy, pandas, matplotlib, seaborn, pymoo, scipy, SALib, pyarrow) for:

- Output file parsers (photo, final, CSV files)
- Visualization (corridor plots, MC ensembles, CDF of correction cost)
- GA training pipeline: optimizes any guidance scheme's parameters (not just NN weights)
  - `train.py` — Hybrid pymoo training loop with checkpoint save/resume (`<config.toml> [--no-tui] [--skip-report] [--final-n-sims N] [--algorithm ALG]`). Uses pymoo `Algorithm` objects stepped manually via `algorithm.next()` in a custom outer loop. Supports GA (SBX + polynomial mutation), CMA-ES, DE, and PSO via `--algorithm` or `[optimizer] algorithm` in TOML. Auto-resumes from existing checkpoint when output dir exists (no `--resume` needed); `--resume` only needed to specify a non-default directory. On resume, `--n-gen` means "N additional generations" (not total). A checkpoint is always saved at end of training (not just at interval multiples). Graceful KeyboardInterrupt handling: Ctrl+C saves checkpoint and returns cleanly with `interrupted: True`. Seed strategies: the `[optimizer] seed_strategy` key is required and picks one of three training seed paths. `"fixed"` uses a deterministic range `[mc_seed + 0, ..., mc_seed + (training_n_sims - 1)]` and never changes (bit-reproducible across runs). `"rotating"` draws `training_n_sims` fresh random seeds every generation, disjoint from the validation/final-eval reserved sets -- the landscape shifts each gen so the optimizer can't memorize scenarios. `"adaptive"` is the curated-CDF path: bootstrap draws a random `training_n_sims` seed list once, then the list is refreshed on (a) validated best promoted, or (b) every `seed_pool_interval` generations (measured from `last_curation_gen`). Each curation draws `curation_sample_size` probe seeds (default 1000), runs the top `curation_top_k` individuals (default 5) on them, averages per-seed costs, sorts, splits into `training_n_sims` equal-count quantile bins, and picks one random seed per bin. Between seed-list changes, `algorithm.pop` is only re-evaluated pre-`algorithm.next()` when the seeds actually changed; CMA-ES skips the re-eval entirely. See `src/python/aerocapture/training/seed_curator.py` and `docs/superpowers/specs/2026-04-14-explicit-seed-strategy-design.md`. Validation gate: triggered by **parameter identity** -- if the current gen's argmin individual differs from `last_validated_individual` (`np.array_equal`), run the validation MC on the reserved seed set (`validation_n_sims`, default 1000). Cost-based new-best detection is unreliable under rotating seeds, hence the identity trigger. Validation records include `rms_cost` (the promotion metric), mean, p95, worst, and capture rate; `best_overall_individual` is promoted only when `val_rms < best_val_cost`. The `improvement` flag in logger records reflects **validation promotions**, keeping the TUI's "Stagnant for N gens" counter honest under rotating seeds. TUI shows "Last val" (most recent attempt with PROMOTED/REJECTED outcome) and "Best val" (lowest-RMS validated candidate, permanent). Three reserved seed pools (training, validation, final eval) use well-separated RNG streams via `make_reserved_seeds(base, offset, n)` with offsets `VALIDATION_SEED_OFFSET` (1M) and `FINAL_EVAL_SEED_OFFSET` (2M) to guarantee zero overlap. At end of training, generates a single PDF report (convergence + final MC evaluation) via `report.py` unless `--skip-report` is passed.
  - `param_spaces.py` — Per-scheme parameter bounds (with optional log-scale encoding)
  - `evaluate.py` — Decode chromosome -> write params (NN JSON or patched TOML) -> run sim -> cost. Uses PyO3 direct call when `aerocapture_rs` is available, subprocess fallback otherwise. Cost function uses `log_cap(dv)` -- a C1-continuous log-capped function (linear below `dv_threshold`, logarithmic above) -- as primary objective, with TOML-configurable normalized soft constraint penalties for g-load, heat flux, and heat load (integrated heat flux) exceedances. All termination outcomes (captured, hyperbolic, crash, pending crash, timeout) produce meaningful DV values from Rust, so no branching on capture status is needed. Also provides `make_reserved_seeds(base_mc_seed, offset, n)` and seed offset constants (`VALIDATION_SEED_OFFSET`, `FINAL_EVAL_SEED_OFFSET`) used by both `train.py` and `report.py` to guarantee disjoint seed sets.
  - `compare_guidance.py` — Fair head-to-head comparison on identical MC scenarios
  - `initialization.py` — Activation-aware weight init (Xavier/He/LeCun uniform) for NN population seeding
  - `seed_curator.py` — `SeedCurator` class used by the `adaptive` seed strategy: maintains a fixed-size training seed list refreshed on trigger by quantile-stratified sampling from the cost CDF of the top-K individuals. Methods: `curate(problem, top_k_X)` runs each of the top-K individuals on `sample_size` fresh probe seeds (disjoint from reserved), averages per-seed costs, sorts, splits into `n_bins` equal-count quantile bins, and picks one random seed per bin; `to_dict()` / `from_dict()` for checkpoint roundtrip. The `fixed` and `rotating` strategies have no class -- they are dispatched inline in `train.py`.
  - `toml_utils.py` — `load_toml_with_bases()`: TOML loading with `base` inheritance resolution (mirrors Rust `resolve_toml_bases`)
  - `weight_stats.py` — Per-layer weight statistics (min/max/mean/std) for training instrumentation
  - `sensitivity.py` — SALib sensitivity analysis support: `DISPERSION_COLUMNS` (26-name list matching `DispersionDraw::to_array()` field order) + `build_problem(mc_config)` (converts a `[monte_carlo]` config dict to a SALib problem dict with per-dimension distribution types and SI-unit bounds mirroring `build_dim_transforms()` in `dispersions.rs`) + `run_morris(toml_path, n, ...)` (Morris elementary effects: generates samples via SALib, evaluates via `run_with_draws()`, returns mu_star/sigma/mu_star_conf/names as lists) + `run_sobol(toml_path, n, param_indices, ...)` (Sobol variance decomposition: sub-problem for selected dims, expands to full 26-dim draw matrix with neutral defaults for unselected dims, returns S1/ST/S2 indices as lists) + `run_full_analysis(toml_path, ...)` (orchestrator: Morris first to rank by mu_star, Sobol on top-k, saves results to `output_dir/sensitivity_results.json`) + CLI entry point: `python -m aerocapture.training.sensitivity <toml> [--morris-n N] [--sobol-n N] [--top-k K] [--morris-only] [--sobol-only] [--output-dir DIR] [--sim-timeout S]`
  - `parquet_output.py` — Parquet output for MC campaign results: `write_parquet(path, final_records, dispersions, config, toml_path=None)` writes 65-column Parquet (39 final-record + 26 dispersion columns prefixed `disp_`) with schema-level metadata (full resolved TOML config as JSON, toml_path, timestamp, guidance_scheme, n_sims). `read_parquet(path)` returns `(DataFrame, metadata_dict)`. `FINAL_RECORD_INDICES` (39 indices into the 52-element array, matching Rust `extract_final_csv_values()`) and `FINAL_COLUMNS` (39 column names matching Rust `FINAL_CSV_COLUMNS` minus sim_number) are module-level constants. CSV output unchanged (39 columns); Parquet is the analysis-oriented format with dispersions embedded.
  - `encoding.py` — Real-valued encoding/decoding for pymoo optimizer: all algorithms work on normalized `np.ndarray[float64]` in [0, 1]. `decode_normalized(x, specs)` maps unit-hypercube vector to physical parameter values (linear or log-scale per `ParamSpec`). `encode_to_normalized(params, specs)` inverts to normalized vector. `decode_normalized_array(X, specs)` batch-decodes population matrices. `nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier)` generates per-weight `ParamSpec` list with Xavier/He-aware bounds from `initialization.py`.
  - `optimizer.py` — pymoo algorithm factory: `OptimizerConfig` dataclass (algorithm, n_pop, n_gen, training_n_sims, validation_n_sims, seed_pool_interval, curation_sample_size, curation_top_k) with per-algorithm sub-configs (`GASettings`, `CMAESSettings`, `DESettings`, `PSOSettings`) and `from_dict()` classmethod for TOML-like dicts. `create_algorithm(config, n_params) -> pymoo.Algorithm` instantiates GA (SBX crossover + PM mutation), CMA-ES, DE, or PSO; CMA-ES falls back to GA with a warning when n_params > 200.
  - `ablation.py` — NN input importance analysis: `NN_INPUT_NAMES` (23-name list), `run_ablation(toml_path, n_sims)` zeros out each input via temp JSON model with `ablated_input` set, measures DV cost degradation vs baseline, ranks by delta. Inputs not present in the model's `input_mask` are skipped (delta=0, `masked_out=True`) since they already don't reach the network. CLI: `python -m aerocapture.training.ablation <training_dir> --toml <config.toml> [--n-sims N]`. Outputs JSON + SVG bar chart.
  - `charts_ablation.py` — `chart_ablation_bar(ranked, output_path)`: horizontal bar chart of cost delta per input, red=positive/blue=negative, SVG output.
  - `problem.py` — `AerocaptureProblem(Problem)` pymoo subclass: bridges population-level evaluation with the Rust simulator. Operates on normalized [0,1] decision variables; decodes via `decode_normalized_array()` at eval time. `_evaluate(X, out)` calls `_run_batch()` and sets `out["F"]` as (n_pop, 1). `_run_batch_pyo3()` issues one `run_batch()` call per seed, aggregates costs by RMS across seeds. `_build_overrides(params, mc_seed)` routes param prefixes to TOML dot-paths (`lateral.*` -> `guidance.lateral.*`, `exit.*` -> `guidance.ftc.*`, `nav.*` -> `navigation.*`, `thermal.*` -> `guidance.thermal_limiter.*`, `shaping.*` -> `guidance.command_shaping.*`, unprefixed -> `guidance.<scheme>.*`). `update_seeds(seeds)` allows the training loop to inject curated seed list updates between generations.
- Training visualization:
  - `metrics.py` — Pure metric functions: cost stats, diversity, capture rate, convergence speed, stagnation
  - `logger.py` — `TrainingLogger`: writes one JSONL line per generation (includes `all_costs` array, `constraint_violation_rate`, `best_params` for global best, and `gen_best_params` for generation best); in-memory buffer for live display
  - `display.py` — `LiveDisplay`: Rich TUI with sparklines, ETA, progress bar (degrades to `NoopDisplay` when `--no-tui` or non-interactive)
  - `report.py` — PDF report orchestrator: loads JSONL training logs + runs final MC re-evaluation (using reserved seeds via `run_batch` to guarantee no overlap with training/validation), generates SVG charts via `charts.py`, writes metadata/summary JSON, invokes `typst compile` to produce a single PDF. Three-part structure: Part 1 (Training Convergence: cost curves, diversity, cost distribution, parameter evolution, seed pool), Part 2 (Mission Performance: corridor plots with zone fills + undispersed/best-DV nominal overlays, altitude/heat flux/g-load/bank angle/density ratio vs time with constraint limit lines, DV distributions, entry/exit conditions, performance summary table with constraint violation rates, dispersion correlations with three-way classification), and optional Part 3 (Sensitivity Analysis: Morris scatter, Sobol bar chart, Sobol S2 heatmap — enabled via `--sensitivity` flag when `<scheme_dir>/sensitivity/sensitivity_results.json` exists). Also produces cross-scheme comparison PDFs. Auto-writes `final_eval.parquet` (65-column Parquet with embedded config metadata) alongside the PDF when pyarrow is available. Auto-generated at end of training, also standalone CLI: `python -m aerocapture.training.report`
  - `charts.py` — All matplotlib/seaborn chart functions (one per panel, 24 total). Each function takes data + output path and writes an SVG. Consistent seaborn theme (`whitegrid`, `muted` palette, light grey background). Three-way trajectory classification: blue (captured + constraints OK), orange (captured + constraint violation), red (crash/hyperbolic/timeout). Classification uses `(ifinal==3) & (ecc<1.0)` as the canonical captured definition. Constraint limits (including `heat_load_limit`) read from `[flight.constraints]` in the mission TOML. `chart_dispersion_grid()` accepts optional `traj_class` for three-way colored scatter (red `x` markers for failed, regression on captured only). Includes `chart_heat_load_time()` for cumulative heat load vs time spaghetti. Sensitivity charts: `chart_morris_scatter` (mu*/sigma scatter with nonlinearity diagonal), `chart_sobol_bars` (S1/ST grouped bars with error bars), `chart_sobol_heatmap` (S2 interaction matrix), `chart_sobol_convergence` (S1/ST vs sample size). Includes helpers for MC spaghetti plots, envelope computation, corridor zone fills, nominal trajectory overlays, and DV log-scale handling.
  - `animate.py` — Standalone CLI for GIF animation of training evolution: replays checkpoints, re-runs MC via PyO3 per frame, renders 2x2 panels (corridor with envelope fills, inclination, bank angle, cost CDF with ECDF overlay). `python -m aerocapture.training.animate <training_dir> --toml <config.toml> [--n-sims 100] [--fps 4] [--every N] [--output animation.gif]`
  - `corridor.py` — Corridor boundary computation via `CorridorAccumulator`. During `piecewise_constant` GA training, each generation's trajectories (plus 11 constant-bank-angle sentinel chromosomes from 0° to 180° in 18° steps) are classified (`classify_trajectories` with asymmetric bounds `delta_za_low`/`delta_za_high`; recognizes `ifinal=4` pending crash) and their pdyn envelopes updated incrementally (running max/min per energy bin). Sentinel trajectories improve corridor boundary resolution by tracing the full lift-up (hyperbolic boundary) to full lift-down (crash boundary) range. Produces schema-v4 `.npz` cache with 4 envelopes (crash, restricted upper/lower, capture), nominal trajectory, and DV. Gaussian smoothing applied at save time. Cached per mission in `training_output/<mission>/corridor_boundaries.npz`. Also produces `ref_trajectory.dat` (7-column format) for schemes that track a reference trajectory.

### Typst Templates (`src/typst/`)

PDF report layout templates compiled by `typst compile`. Receives SVG charts and JSON metadata from a temp directory.

```
src/typst/
  report.typ         — Main report template (cover page + Part 1: Training + Part 2: Mission Performance)
  comparison.typ     — Cross-scheme comparison template
  lib.typ            — Shared helpers (page style, colors, heading format)
```

External dependency: `typst` CLI (install via `brew install typst` or `cargo install typst-cli`). Report generation degrades gracefully if Typst is not installed — charts are still generated, just no PDF compilation.

## GA Training & Comparison

```bash
# ── Train all schemes with optimized settings (see train_all.sh) ──
./train_all.sh                     # all schemes in dependency order
./train_all.sh eqglide fnpag       # specific schemes only

# ── Optimize a guidance scheme (with Rich TUI) ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 2500 --n-pop 60

# ── Disable TUI (e.g. in CI or when piping output) ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 2500 --n-pop 60 --no-tui

# ── Resume training (auto-detects checkpoint; --n-gen means "N additional") ──
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50

# ── Compare all schemes on identical MC scenarios ──
# Each scheme uses its own training TOML (network arch, nav params, etc.)
uv run python -m aerocapture.training.compare_guidance \
    --n-sims 500 \
    --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network piecewise_constant

# ── Generate PDF report (training convergence + final MC evaluation) ──
# Automatically generated at end of training; also available standalone:
uv run python -m aerocapture.training.report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml

# ── Generate cross-scheme comparison PDF ──
uv run python -m aerocapture.training.report --compare training_output/

# ── Animate training evolution (corridor + trajectory GIF from checkpoints) ──
uv run python -m aerocapture.training.animate \
    training_output/piecewise_constant/ \
    --toml configs/training/msr_aller_piecewise_constant_train.toml \
    --n-sims 100 --fps 4 --every 5

# ── Sensitivity analysis (Morris screening + Sobol decomposition) ──
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-n 1000 --sobol-n 1024 --top-k 10

# ── Morris screening only (quick ranking of all 26 dispersion parameters) ──
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-only --morris-n 500
```

Guidance schemes and their TOML training configs:
- `piecewise_constant` -> `configs/training/msr_aller_piecewise_constant_train.toml` **(train first — produces ref trajectory + corridor)**
- `neural_network` -> `configs/training/msr_aller_nn_train_consolidated.toml`
- `equilibrium_glide` -> `configs/training/msr_aller_eqglide_train.toml`
- `energy_controller` -> `configs/training/msr_aller_energy_controller_train.toml` *(requires ref trajectory)*
- `pred_guid` -> `configs/training/msr_aller_pred_guid_train.toml` *(requires ref trajectory)*
- `fnpag` -> `configs/training/msr_aller_fnpag_train.toml` *(requires ref trajectory)*
- `ftc` -> `configs/training/msr_aller_ftc_train.toml` *(requires ref trajectory)*

**Training order:** Run `piecewise_constant` first — it produces `training_output/<mission>/ref_trajectory.dat` (optimized reference for other schemes) and `corridor_boundaries.npz` (4-layer corridor envelopes from GA population history). Schemes marked *(requires ref trajectory)* will error at startup if the ref trajectory is missing. Schemes without the marker (`neural_network`, `equilibrium_glide`) can be trained independently.

Optimized params saved to `training_output/<scheme>/best_params.json` (or `best_model.json` for NN).

## RL Training (PPO)

Reinforcement-learning training for the `neural_network` guidance scheme, running as a parallel track to the pymoo GA. RL-trained weights deploy via the same `best_model.json` format the GA produces; `compare_guidance` treats RL as just another scheme (`neural_network_rl`) that feeds the Rust `neural_network` runtime.

```bash
# Train a PPO policy
uv run python -m aerocapture.training.rl.train \
    configs/training/msr_aller_rl_train.toml \
    --algorithm ppo --total-steps 5000000

# Head-to-head RL vs GA on identical MC scenarios
uv run python -m aerocapture.training.compare_guidance \
    --n-sims 500 \
    --schemes neural_network neural_network_rl

# All schemes (train_all.sh): nn_rl runs after piecewise_constant
./train_all.sh nn_rl
```

Training CLI flags: `--algorithm {ppo|sac}`, `--total-steps N`, `--n-envs N`, `--rollout-steps N`, `--validation-n-sims N`, `--validation-interval-updates N`, `--data-neural-network PATH`, `--no-tui`, `--skip-report`, `--resume DIR`, `--output-dir DIR`.

**Architecture:** step-able `BatchedSimulation` pyclass (N `SimState`s sharing one `Arc<SimData>`, Rayon parallel ticks, auto-reset on episode end, GIL released via `py.detach()`). CleanRL-style PPO outer loop (`src/python/aerocapture/training/rl/`): `env.py` wraps the pyclass, `policy.py` is a PyTorch MLP mirroring `NeuralNetModel` JSON (`GaussianPolicy` with `atan2(out[0], out[1])` bank mapping), `export.py` writes to `best_model.json` in the Rust loader's format (format_version=1, architecture + per-layer `w`/`b`), `ppo.py` provides `RolloutBuffer` + `compute_gae` + `ppo_update` (clipped surrogate + value + entropy), `train.py` is the CLI and outer loop with reserved-seed validation gate + checkpoint save/resume + graceful Ctrl+C, `report_rl.py` produces a three-part PDF (Part 1 RL convergence panels, Parts 2/3 reused from the GA report), `logger.py`/`display.py` provide JSONL + Rich TUI mirroring the GA contract.

**Artifacts** under `training_output/neural_network_rl/`: `best_model.json` (drop-in for Rust runtime), `rl_training_*.jsonl` (per-update metrics), `config_resolved.toml`, `checkpoint.pt`, `final_eval.parquet`, `report.pdf`.

**Seed pools:** `RL_TRAINING_SEED_OFFSET = 3_000_000` (default `seed_base`), plus the existing `VALIDATION_SEED_OFFSET = 1_000_000` and `FINAL_EVAL_SEED_OFFSET = 2_000_000` — all disjoint by construction.

**v1 limitations:** PBRS shaping is wired but disabled by default (needs an `(energy, pdyn)` side channel from `BatchedSimulation`; pure terminal reward still trains, just slower). SAC is planned in Phase 7 — currently only PPO ships.

Full spec: `docs/superpowers/specs/2026-04-15-rl-nn-guidance-design.md`.

## Key Lessons & Pitfalls

### Historical: Density Filter Gain Clamping

*Context: explains why the density filter code has careful gain-clamping logic in `estimator.rs`.*

The original codebase had a memory corruption bug that turned the density filter gain from 0.8 to 56.0, causing 55x error amplification per step. The Rust code clamps lambda to [0.01, 0.99] as a safety net. Additionally, the legacy bias-mode filter now has rate-of-change limiting (configurable `density_gain_max_delta`, default 0.1) and gain saturation bounds [0.1, 10.0] matching the EKF density correction factor range. Both navigation modes use lift-corrected drag extraction: the density inversion denominator uses `Cx*cos(alpha) + Cz*sin(alpha)` instead of just `Cx`, correcting a ~4% error at typical AoA=10 deg.

### Historical: Regression Test Tolerances

*Context: explains why regression tests tolerate 2/24 column mismatches.*

Two output columns in the reference implementation used uninitialized variables, producing non-deterministic values. The Rust validation excludes these columns.

### Energy Computation

Energy must use **absolute (inertial) velocity**, not relative velocity. The Rust `total_energy()` converts relative->absolute via `to_absolute_cartesian` before computing E = V_abs^2/2 - mu/r.

### GA Parameter Routing

`param_spaces.py` uses prefixed names to route params to TOML sections: `nav.` -> `[navigation]`, `lateral.` -> `[guidance.lateral]`, `exit.` -> `[guidance.ftc]`, `thermal.` -> `[guidance.thermal_limiter]`, unprefixed -> `[guidance.<scheme>]`. This routing must be consistent across `evaluate.py` (write), `compare_guidance.py` (load best_params.json), and `train.py` (PyO3 override dict for best-individual re-evaluation). NN training bypasses `write_guidance_toml()` entirely -- navigation-level TOML overrides must be set in the NN training config directly.

### Navigation-Level Config

Density filter params (`density_filter_gain`, `density_gain_max_delta`) live in `[navigation]` TOML section (`TomlNavigation` in config.rs), not in `[guidance.ftc]`. They affect all guidance schemes via `estimator.rs`. When adding new navigation-level tunable params, put them in `[navigation]` from the start, add to `_NAV_PARAMS` in `param_spaces.py` with `nav.` prefix.

### Golden File Regeneration

Physics changes (density estimation, gravity, aerodynamics) invalidate guidance regression golden files in `tests/reference_data/rust_golden/`. Regenerate by running the updated binary on each test config and replacing the CSV files. The 6 golden files cover: eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural.

## Conventions

- **Rust**: Edition 2024, nalgebra for linear algebra, release profile with LTO
- **Python**: Python >=3.14, Ruff (line-length 160, target py314), uv package manager, pytest, mypy strict mode. Dev tools in `[dependency-groups]` (not `[project.optional-dependencies]`). Training deps (pymoo, scipy) are core dependencies. deap has been replaced by pymoo for all optimizer algorithm support.
- **Testing (Python)**: pytest, hypothesis (property-based). Golden reference files under `tests/reference_data/`. Shared fixtures in `tests/conftest.py` (session-scoped Rust build) and `tests/fixtures/factories.py` (config/chromosome factories). ~416 tests covering parsers, regression, MC, GA pipeline (chromosome, cost, TOML patching, config, operators), training visualization (metrics, logger, display, integration, report PDF generation, chart SVG generation), training animation (checkpoint discovery, override building, axis ranges, frame rendering, GIF generation), NN weight initialization, seed rotation, curated-CDF seed framework (quantile binning, curation trigger, bootstrap, checkpoint, integration), graceful interrupt handling, TOML base inheritance resolution, PyO3 integration (bit-identical regression against subprocess path, run_with_draws shape validation and dispersions roundtrip), report resume detection and conditional panel rendering, corridor accumulator (incremental envelope building, checkpoint roundtrip, asymmetric bounds, ifinal=4 pending crash classification), unified cost function (log_cap C0/C1 continuity, monotonicity, cost ordering, heat load penalty), sensitivity analysis (DISPERSION_COLUMNS length, build_problem SALib dict structure, distribution types per domain, off-level zero bounds, medium-level SI values, wind absent/present, missing domain defaults, Morris pipeline mu_star shape/finiteness/non-negativity, Sobol pipeline S1/ST shape, subset param_indices, name mapping), sensitivity charts (morris_scatter/sobol_bars/sobol_heatmap/sobol_convergence SVG output), Parquet output (write/read roundtrip, schema validation, metadata keys, data integrity, index bounds/duplicates, dispersion grid classification), real-valued encoding (decode_normalized linear/log-scale, encode roundtrip, NN weight encoding bounds), optimizer config (all algorithms accepted/rejected, from_dict parsing, subsection defaults, algorithm factory isinstance checks, SBX/PM operator inspection, CMA-ES high-dim fallback), AerocaptureProblem (problem shape n_var/bounds/n_obj, evaluate shape/finiteness, seed update, build_overrides all 6 prefix routes + n_sims injection), ablation analysis (NN_INPUT_NAMES length/uniqueness/non-empty, DV column index canary, chart SVG generation with positive/negative deltas).
- **Testing (Rust)**: Three-tier pyramid — unit tests (inline `#[cfg(test)]` modules with proptest property tests), integration tests (`src/rust/tests/`), E2E subprocess tests. Shared test infrastructure in `tests/common/` (fixtures.rs, assertions.rs). Dev-dependencies: `approx` (float comparison), `rstest` (parameterized tests), `proptest` (property-based testing), `tempfile` (temp dirs for base inheritance tests). ~419 tests covering physics (J2/J3/J4 gravity: bit-identity when J3=J4=0, symmetry breaking, small correction bounds, proptest finiteness), GNC, guidance (all 7 schemes including piecewise_constant), exit phase guidance (pdyn feedback: finite/bounded output, density sensitivity, clamping, proptest robustness), phase dispatch (phase 2 routes to exit, signed-bank schemes ignore phase), lateral guidance (roll reversal: energy window gating, corridor boundary, max_reversals, proptest invariants), navigation (bias mode + EKF: predict/update symmetry, covariance growth/reduction, density correction clamping, density filter rate limiting + gain saturation + proptest bounds invariant, lift-corrected density inversion at zero/nonzero AoA + denominator guard, IMU noise statistics, star tracker blackout/cadence, SimPhase gating: Full/CaptureOnly/ExitOnly phase transitions), wind model (table loading, interpolation, latitude scaling, integration effect), control (angle_utils proptest: range/antisymmetry/magnitude properties + wrap-around edge cases, pilot wrap-through-±π), DOPRI45 adaptive integrator (Butcher tableau consistency, FSAL continuity, rejection/recovery, PI controller bounds, error norm scaling, harmonic oscillator, proptest robustness, E2E capture validity and Gill agreement, dense output boundary conditions + midpoint accuracy + proptest finiteness), event detection (Brent's root-finding: sin/linear/endpoint/tight-bracket convergence + same-sign panic, event function sign correctness, check_events_and_locate: zero-crossing location + direction filtering + earliest-event arbitration, E2E: bounce sub-tick precision + atmosphere exit timing + fixed RK4 non-regression + trajectory event interleaving with monotonic time + proptest bounce values finiteness), error paths, `run_for_api()`, peak value tracking, TOML base inheritance (deep_merge, resolve_toml_bases, cycle detection), virtual DV ranges (proptest: crash DV in [10k,20k], hyperbolic DV >= 10k, cost ordering invariant), trajectory heat load (monotonically non-decreasing, consistent with final_record), thermal limiter (ramp bounds, monotonicity, default-disabled invariant, proptest robustness), density perturbation (OU config presets, step function decay/determinism/statistics, TOML parsing with level/custom/absent), sampling (norm_ppf known values + symmetry, DimTransform Gaussian/Uniform/UniformRange/Fixed apply, build_dim_transforms medium config, LHS stratification + determinism, Sobol bounds + determinism + seed sensitivity, from_array roundtrip, generate_draws LHS/Sobol valid + finite, Sobol 65536 limit panic, proptest: all methods produce finite draws for seed in [0,10000) x n_sims in [1,200)), command shaper (acceleration limiting, rate cap, wraparound shortest-path, deceleration before reversal, small-correction passthrough, legacy hard-clamp path, proptest: rate bounded/rate-change bounded/always finite), NN input expansion (23-input vector finiteness, mask selection, ablation zeroing, backward-compat 16-input path, bounce-gated exit inputs zero pre-bounce, input_mask validation: length/range/duplicates, ablated_input range validation). Run with `cargo test` or `./check_all.sh`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — Rust (fmt, clippy, test), Python (ruff lint, ruff format, mypy, pytest), and PyO3 (maturin build + pytest test_pyo3.py) run on PRs to `main` and manual dispatch (`workflow_dispatch`).
- **Validation**: Validated against reference implementation — 22/24 photo columns bit-identical across 725 timesteps.

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **Challenge inefficiencies**: if I'm doing something the hard way, call it out.
