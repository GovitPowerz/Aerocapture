# Rust simulator (`src/rust/`)

The simulator crate: physics, navigation, the seven guidance schemes, the NN runtime, Monte Carlo
dispersions. A Cargo workspace with two members, the core `aerocapture` crate (this directory) and
the `aerocapture-py` PyO3 binding crate ([aerocapture-py/README.md](aerocapture-py/README.md)).
The crate has both `lib.rs` (public API: `RunOutput` struct + `run_for_api()`) and `main.rs` (CLI
entry). A TOML config as the CLI argument (`./aerocapture config.toml`) is the only supported input
format; it covers all 7 guidance schemes and inline vehicle/mission data
([configs/README.md](../../configs/README.md)).

Companion docs: [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) (a simulation tick in eight
lines), [CONTEXT.md](../../CONTEXT.md) (vocabulary), the NN runtime in
[src/data/neural/README.md](src/data/neural/README.md), the decisions in `docs/adr/`
(ADR-0004 `run_grid` bit identity, ADR-0006 per-draw noise, ADR-0008 config keys declared once).
The lessons every change must respect (energy in inertial velocity, golden regeneration, the
density filter clamps) are in the root `CLAUDE.md` "Key Lessons & Pitfalls"; nothing here repeats
them.

Key dependency: `nalgebra` for vector/matrix ops.

## Build and run

```bash
# From the repo root:
cargo build --release --manifest-path src/rust/Cargo.toml
./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml
cargo test --release --workspace --manifest-path src/rust/Cargo.toml   # or ./check_all.sh
```

## Module map

```
src/rust/src/
  main.rs                          — CLI entry, TOML config loading
  config.rs                        — TOML parser (PlanetConfig, SimInput, IntegrationMode, MissionType/SimPhase/GuidanceType::parse) + base inheritance (deep_merge, resolve_toml_bases)
                                     + validate(&TomlConfig): every no-IO rule in one pass (sections, enum strings, incidence, piecewise, command_shaping, normalization width,
                                     output_parameterization vs mode+arch, MC levels/keys/bounds via build_dispersion_config, nav mode + EKF sigmas, integration mode); run by
                                     SimData::from_toml BEFORE its table IO (CLI, run_mc, run_with_draws, collect_*, env) and by from_toml_with_tables on every post-override
                                     config (run_batch / run_grid preload SharedTables from the unvalidated base first); exposed as PyO3 validate_config.
                                     Config keys (ADR-0008): a Toml* field is its TOML key (`#[serde(rename)]` is the declared exception, as `type` is) and a runtime relay
                                     keeps it (`data.guidance.neural_network.reset_state_every_tick` greps end to end); every serde type that reads a section is
                                     `deny_unknown_fields` (root included, its seven Python-only sections declared as pass-through tables) and section-struct fields are
                                     `pub(crate)` so an unrelayed field fails clippy. Gates: `tests/config_loading.rs::every_committed_config_parses_and_validates`
                                     (configs/** recursive, parse + validate) and `src/config_tests.rs::toml_keys_reachable_in_sim_data`.
  data/
    mod.rs, SimData                — Top-level data container; from_toml_with_tables = ~70 lines of orchestration: config::validate first, then named builders (build_capsule/
                                     pilot/entry/aero/flight/success/incidence/guidance_params/onboard_atmosphere/neural_net, resolve_reference_trajectory = the per-individual ref
                                     reload rule, build_dispersion_config, NavMode::from_toml). legacy_ftc_defaults() is the no-[guidance.ftc] default set (pinned
                                     bit-identical by legacy_ftc_defaults_match_historical_literals). Model-dependent rules (mask width, NN-without-model, decoder knob vs model
                                     output_param, `[[network.architecture]]` == the loaded model's `LayerSpec` list) stay in build_neural_net; SharedTables::from_toml is the
                                     only other table IO. Runtime structs carry no `#[allow(dead_code)]`: a `[guidance.*]` key that drives nothing stays declared+inert on its
                                     `Toml*` struct (6 FTC keys, EC `gain`, star-tracker `attitude_sigma`)
    atmosphere.rs                  — Atmosphere density table (binary-search lookup; altitude column validated non-decreasing at load; NaN altitude returns NaN via the exponential tail exactly like
                                       the legacy linear scan — it falls through both range guards and would otherwise underflow the bracket index and panic a Rayon worker) + OnboardAtmosphereModel
                                       (piecewise exponential, auto-fitted or explicit)
    aerodynamics.rs                — Cx/Cz vs AoA tables
    capsule.rs                     — Vehicle: mass, reference area, max bank rate
    guidance_params.rs             — Guidance law config: FTC gains, LateralParams, ThermalLimiterParams, EqGlide, EnergyCtrl, PredGuid, FNPAG params (`energy_tol` is an apoapsis-radius tolerance in
                                       meters since the apoapsis retarget; TOML key name kept for compat); ReferenceTrajectory loader (file contract: energy column in MJ/kg, converted to J/kg at load;
                                       hard-errors when the post-conversion magnitude exceeds 1e9 J/kg — i.e. the file is in J/kg — and on a MISSING/unreadable file, which would otherwise be a
                                       silent empty table that ref-tracking schemes interpolate as 0.0; strictly-descending tables get an O(log n) bracket lookup, bit-identical to the legacy walk)
    dispersions.rs                 — Monte Carlo dispersion profiles (all domains use level presets: Off/Low/Medium/High/Custom) + Gauss-Markov density perturbation (OU process config + step
                                       function) + SamplingMethod enum (Random/LHS/Sobol) + norm_ppf (Acklam inverse normal CDF) + DimTransform enum (Gaussian/Uniform/UniformRange/Fixed) +
                                       build_dim_transforms() mapping 26 dispersion dims to their transforms + generate_lhs_unit_samples() (stratified Fisher-Yates) + generate_sobol_unit_samples()
                                       (sobol_burley) + DispersionDraw::from_array() (inverse of to_array()) + draws_from_unit_samples() (applies DimTransforms to unit samples) + generate_draws()
                                       dispatch (Random/LHS/Sobol) + generate_draws_random() (legacy PRNG path, backward-compatible)
    navigation.rs                  — Navigation error profiles
    incidence.rs                   — AoA profile tables
    pilot.rs                       — Pilot dynamics parameters
    neural/                        — The NN runtime: model + layer types + codec. See src/data/neural/README.md. `mod.rs` (NeuralNetModel, Layer/LayerSpec enums incl. `LayerSpec::io()`
                                       chain-shape accessor, `Layer::from_spec` (zero layer + the ONE dim-validation site); the JSON loaders deny unknown keys at every level -- file,
                                       `LayerSpec` entry, stray tensor or stray `layer_i` entry (legacy `output_interpretation` is declared and ignored); the `LayerWeights` trait --
                                       `tensors()`/`tensors_mut()` from the per-layer table, `post_load` hook, and DEFAULT `to_flat`/`from_flat`/`n_params` -- and the generic codec: v1/v2
                                       loaders share `load_layers` (zero layer from spec -> each table tensor looked up by name + shape-checked via `json_to_flat` -> one `from_flat`, which
                                       runs `post_load`), `save_json` walks `tensors()` (keys in table order except `JSON_KEYS_LAST` = Mamba3 `a_imag`/`lambda_logit`, kept last so re-saves
                                       stay byte-identical to the retired fixed-field schema), NormSpec/OutputParam/DEFAULT_NORMALIZATION/apply_norm, stateful `forward(&mut NnState, &[f64])`)
                                       + `layers/` (`tensor.rs`: `Shape`, the `Tensor` trait over f64/Vec/Vec<Vec>/DVector/DMatrix (+ `Option` = flag-gated), and the `tensor_table!` macro
                                       that declares a layer's weights ONCE -- the listed fields in order ARE the flat order and the JSON keys; per-type submodules
                                       dense/gru/lstm/window/transformer/mamba/mamba3/cfc/slstm/mlstm each declare one table (+ `zeros` constructor); `helpers.rs` shares
                                       matvec/dot_plus_bias/gelu_exact/layer_norm_biased/build_pe_table/softplus/expm1_over_x/lecun_tanh/stabilized_exp_gates) + `tests.rs`.
                                       Frozen gates: `tests/nn_flat_order_fixtures.rs` (per-type flat vector + exact save_json bytes under `tests/fixtures/flat_order/`) and
                                       `tests/nn_model_roundtrip.rs` (every committed model JSON load->save->reload; `--ignored` resave tool for byte diffs)
    nn_state.rs                    — `NnState { layer_states: Vec<LayerState> }` per-sim mutable state for stateful NN layers; lives outside NeuralNetModel (model is Arc-shared immutable);
                                       one `LayerState` variant per stateful layer type (`None` for dense); `Clone` for RL rollout snapshots; reset at episode start via `GuidanceState::new`
                                       reconstruction
  physics/
    gravity.rs                     — J2/J3/J4 zonal harmonic gravity
    dynamics.rs                    — Equations of motion + the shared aero laws: `compute_derivatives` (the plant both integrators step), `effective_airspeed` (wind-corrected),
                                       `heat_flux` (`cq * sqrt(rho) * v_eff^3.05`, the ONE copy of the law -- EOM flux integral, peak tracker, photo rows and the guidance thermal fraction all
                                       call it) and `loads_at(state, altitude, aoa, ...)` (dispersed density + wind-corrected airspeed + heat flux / pdyn / load factor at one state; the peak
                                       tracker, the photo rows and the thermal fraction call it). Dispersions enter through `AeroDispersions`, a `Copy` view built by `RunState::aero()`, so
                                       physics stays a leaf (no simulation:: import). Every expression is pinned by the goldens, `run_grid` and the subprocess bit-identity gate -- never
                                       reassociate or `mul_add`
    atmosphere.rs                  — Density lookup (dispersed product floored at 0: a tail OU draw or >100% custom bias can push a factor below -1, and negative rho would NaN the heat flux via
                                       sqrt)
    winds.rs                       — Altitude-dependent wind model (WindTable loader, latitude-scaled zonal winds, MC dispersions)
  gnc/
    navigation/
      estimator.rs                 — Navigation orchestrator: bias mode (legacy) or EKF mode via NavigationFilter enum; phase management (capture/exit/emergency) gated by SimPhase config; density
                                       estimation via lift-corrected inverse dynamics (body-frame: Cx*cos(alpha) + Cz*sin(alpha) denominator, which must be POSITIVE — a lift-dominated negative is
                                       rejected, not `.abs()`'d into a non-physical negative density); legacy filter with rate-of-change limiting (density_gain_max_delta), skips guard-tripped steps
                                       (EKF parity), + gain saturation [0.1, 10.0] (`DENSITY_FACTOR_MIN`/`MAX`) applied unconditionally every tick; EKF mode honors exit-phase irreversibility
                                       (`exit_phase_locked`) like bias mode; shared post-density tail in helpers used verbatim by both modes (`update_bounce_and_phase` for bounce detection +
                                       capture->exit phase management, plus the energy/orbital/crash/SimPhase finalize helper); tests in `estimator_tests.rs`; NavigationOutput includes thermal
                                       fractions (heat_flux_fraction, heat_load_fraction) for guidance limiter and NN inputs
      ekf.rs                       — 13-state Extended Kalman Filter (error-state: pos/vel errors, accel/gyro biases, density correction)
      imu.rs                       — IMU sensor model (accelerometer + gyroscope with bias, scale factor, noise)
      star_tracker.rs              — Star tracker model (position updates with dynamic pressure blackout)
      coordinates.rs               — Spherical<>Cartesian, geodetic, total energy
    guidance/
      dispatch.rs                  — Central guidance dispatch (phase-aware: routes to exit guidance when guidance_phase=2), GuidanceState (carries `nn_state: Option<NnState>` for stateful NN
                                       layers), GuidanceOutput; CommandShaper (acceleration-limited S-curve rate shaping with realized-angle feedback; falls back to legacy hard-clamp when config
                                       absent); constant-bank reference mode handled via the `is_reference` flag (no separate guidance struct); the single signed-bank-scheme predicate + the shared
                                       `securize_cos_bank` (clamp+acos) helper live here
      ftc.rs                       — FTC capture-phase guidance: altitude-gain predictor-corrector (FtcCaptureState, ftc_bank_angle)
      exit.rs                      — Exit phase guidance: shared pdyn-feedback controller for ascending leg (FTC + 4 unsigned-magnitude schemes)
      lateral.rs                   — Lateral guidance (roll reversal): LateralParams, LateralState, predictive first-order inclination projection (shared by unsigned-magnitude schemes)
      neural.rs                    — NN guidance (JSON architecture v1 or v2; the input vector, the decoders and the normalization are specified in src/data/neural/README.md). Bank decoded
                                       per `output_param` via `match` in `nn_bank_angle`: `atan2_signed`/`acos_tanh`/`scaled_pi` (`wrap_to_pi(n*pi*tanh(out[0]))`)/`delta`
                                       (`wrap_to_pi(prev_realized_bank + delta_max*tanh(out[0]))`), the signed decoders reusing `angle_utils::wrap_to_pi`. `build_nn_input` extracts the 35
                                       raw candidate scalars then applies a uniform per-input `apply_norm(raw, &NormSpec{transform, scale, center})` = `transform((raw - center)/scale)`;
                                       specs resolve loaded model > `SimData::nn_normalization_override` (the TOML `[network.normalization]`, populated regardless of guidance type so the
                                       `collect_supervised` trace of a teacher scheme normalizes on the deployed NN's scales) > `DEFAULT_NORMALIZATION`. Ablation via `ablated_input` /
                                       `ablated_value`. `nn_bank_angle(nav, nn, &mut NnState, data, planet, &NnInputContext)`: `NnInputContext::from_guidance_state(&GuidanceState, sim_time,
                                       target_inclination)` is the ONE place the previous-tick telemetry (inputs 21-24, the (sin,cos) history pairs, the `delta` decoder base, the sign-flip
                                       age) is read; `build_nn_input(nav, NnModelView { input_mask, ablated_input, ablated_value }, data, planet, &ctx)` takes the same context (dispatch, the
                                       supervised trace in tick.rs, and the RL env all build it with that constructor) and threads the stateful layer state from GuidanceState.
                                       `mode = "magnitude_only"` (dispatched in dispatch.rs via NeuralNetMode) routes the NN's `.abs()`'d output through FTC's unsigned-magnitude pipeline
                                       (exit + lateral + thermal_limiter) instead of bypassing them.
      equilibrium_glide.rs         — Equilibrium glide with hdot damping + velocity bias; lift uses the nav-filtered density (`nav.density_guidance` = onboard model x estimated dispersion factor),
                                       NOT the static onboard table — the static-table variant carried the full density dispersion as bias
      energy_controller.rs         — Energy dissipation tracking via pdyn/hdot feedback
      predguid.rs                  — Apollo/Shuttle-heritage drag tracking guidance (negative feedback: drag above the reference profile drives cos_bank toward lift-up, same direction as FTC's pdyn
                                       term and the exit controller; with the sign inverted the GA pins k_drag at its floor and trains 3-4x worse than open-loop)
      fnpag.rs                     — Lu's numerical predictor-corrector (FNPAG): 3D 6-DOF forward predictor (J2/J3/J4 gravity, Coriolis/centrifugal, RK4 integration; zero lateral lift since roll
                                       sign unknown). Targets the osculating EXIT APOAPSIS RADIUS (`osc_apoapsis_radius` via `elements::from_spherical`, NOT energy — apoapsis is what the
                                       post-capture periapsis-raise + apoapsis-correction dV is paid on; energy only fixes the SMA), with the onboard atmosphere SCALED by the nav-estimated density
                                       dispersion factor (`nav.density_guidance / onboard(current_alt)`, = the clamped nav gain) so the forward model tracks the MEASURED atmosphere — density is the
                                       dominant apoapsis-error driver (corr -0.72) and a predictor blind to it cannot reject it (this scaling took FNPAG from p50 DV 200 to 138, below FTC's 166,
                                       apoapsis-err p95 38 km vs 2866). Two-phase: flies the constant candidate capture bank until the predicted handoff (post-bounce, relative speed <=
                                       `exit_velocity_threshold`), then the shared exit-phase law (`exit_law_bank`, a faithful copy of `exit::exit_guidance`) to atmosphere exit — matches the plant,
                                       though inactive at the deployed GA optimum where `exit_velocity_threshold` sits below the ~3360 m/s exit speed so the transition fires in vacuum. Replans every
                                       `replan_period` s (default 2.0, `[guidance.fnpag]`; deliberately NOT a GA gene — the wall-time-blind GA would floor it) and holds the command in between (the
                                       held command is re-clamped at the CURRENT altitude's bank limits, so a 140-deg command issued above the 50 km switch can't persist below it where bank_max_low
                                       applies); the corrector is BISECTION over [bank_min, bank_max] on the monotonic apoapsis-vs-bank curve (low bank → escape/`UNBOUND_APOAPSIS_RADIUS_M` sentinel,
                                       high bank → crash/0; ~11 forward integrations/replan) — a secant collapses capture to ~19% on the escape/crash plateaus where its gradient is zero.
                                       `prediction_dt` GA bounds floored at 2.0 s (dt=0.5 individuals ran ~3.6x slower for no fitness gain). ~166 sims/s at the GA optimum, vs ~8000 for FTC.
      piecewise_constant.rs        — N-segment bank angle profile (`bank_angles: Vec<f64>`, count tunable via TOML `n_segments` / `bank_angles = [...]`, default 10; GA-optimized, produces ref
                                       trajectory + corridor)
      thermal_limiter.rs           — Thermal safety limiter: smooth bank-to-lift-up ramp near heat flux/load limits (GA-tunable, unsigned-magnitude schemes only)
    control/
      angle_utils.rs               — `shortest_angle_diff()`: wrap-aware angular difference in [-π, π]
      pilot.rs                     — Pilot dynamics (wrap-aware via angle_utils)
  integration/
    dopri45.rs                     — Dormand-Prince 4(5) adaptive integrator (FSAL, PI step-size control, mixed atol/rtol error norm); `dopri45_step` delegates to `dopri45_step_with_stages` (single
                                       implementation); dense output (Hermite continuous extension via `dopri45_dense`)
    events.rs                      — Event detection for adaptive integration: EventDef/EventAction/EventType framework, Brent's root-finding, `check_events_and_locate` (sign-change detection +
                                       direction filtering + earliest-event arbitration on dense output), `build_aerocapture_events` (4 events: bounce/atmosphere exit/crash/phase transition); event
                                       functions use latitude-dependent ellipsoid radius for oblateness-consistent altitude; `TriggeredEvent` carries absolute time
    rk4.rs                         — Gill-variant RK4 (fixed-step, legacy default)
    sequencer.rs                   — Module cadence scheduling
  orbit/
    elements.rs                    — Orbital elements from state vector
    maneuver.rs                    — Delta-V cost computation: `compute_deltav` (the terminal-maneuver plan, only called for confirmed captures) and `predicted_dv_for_nn` (the per-tick
                                       correction-DV components fed to the NN, see src/data/neural/README.md)
  simulation/
    runner.rs                      — Main sim loop, the orchestration layer. ONE Monte Carlo fan-out, `run_core(config, data, draws, RunOptions)` (one run state per draw,
                                     Rayon-parallel for >1 draw, each result stamped with its draw); the public entry points are draw-source adapters over it:
                                     run() for CLI (draws from the config + photo/CSV output), run_for_api() for PyO3, run_for_api_with_draws() for the
                                     external-draw API, run_for_api_cell() for one grid cell (`draw_from_seed`), run_single_collect() for one undispersed default-draw run in memory (tests; the
                                     supervised/NN-input TRACE path is run_for_api_cell via collect_supervised / collect_nn_inputs).
                                     `SimStateOptions` (photo / wall timeout / single-run banner) go INTO `build_sim_state`; nothing is poked onto a
                                     `SimState` after construction; dispatches between fixed Gill RK4 and adaptive DOPRI45
                                       based on IntegrationMode; DOPRI45 mode uses `integrate_adaptive_with_events` (returns Vec<TriggeredEvent> for all events in a tick, processed chronologically)
                                       for sub-tick event detection (bounce, atmosphere exit, crash, phase transition) via dense output + Brent's root-finding (~1 ms precision); fixed RK4 uses legacy
                                       post-tick threshold checks; tracks peak heat flux, g-load, dynamic pressure; NaN/Inf state termination (prevents infinite loops from extreme GA
                                       params); optional wall-clock timeout per sim (prevents Rayon batch blocking); pending crash detection (ifinal=4); atmospheric apoapsis crash (bounce_alt > 20km +
                                       descending + still in atmosphere); virtual DV for all termination outcomes; event records interleaved into trajectory output (sorted by time).
                                       What lives here: `navigate_from_state`, `run_core` + the five adapters, `run_single`, `integrate_step`, `integrate_adaptive_with_events`,
                                       `track_peak_values`, `build_event_defs` / `build_event_ctx`. `SimResult` is `pub(crate)` so `output::write_csv_output` can take it.
                                       `src/rust/tests/entry_fan_agreement.rs` pins the fan (6 golden configs x {legacy, per_draw}, exact `to_bits` equality): (a) run_for_api ==
                                       run_for_api_with_draws on the config's draws, trajectories included, (b) run_for_api_cell(seed) == run_for_api with `monte_carlo.seed = seed`, `n_sims = 1` (ADR-0004),
                                       (c) run_single_collect == run_for_api_with_draws on one default draw
    sim_types.rs                   — Foundational sim types: `SimState`, `TermReason`, `SimError` + crash/virtual-DV consts (`CRASH_FLOOR`, `HYPERBOLIC_BASE`, `BOUNCE_ALT_UNSET`, ...); a leaf module
                                       imported by runner/finalize/run_init/tick. runner re-exports these so `runner::` paths (incl. the external aerocapture-py crate) resolve
    run_init.rs                    — `build_sim_state` (the single SimState constructor, shared by CLI `run_single` and the batch/env path)
    finalize.rs                    — `build_final_record` + `ifinal_for` + `is_pending_crash` + virtual-DV (turns a terminated SimState into the 52-element record; consumed by the CLI and the RL
                                       step API)
    tick.rs                        — `step_one_tick`: the single per-tick GNC + integration step shared by the CLI `run_single` loop and the RL `BatchedSimulation` env (+ shared helpers
                                       `promote_pending_crash_if_applicable`, `navigate_from_state`); also records the per-tick NN candidate trace and updates the NN telemetry state post-guidance from
                                       the effective command. `sim_time` is the time of `state.state`: GNC runs at the tick start, the integration step advances both by `dt`
                                       (adaptive terminal events rewind it to the event time), so peak / bounce times, the final record and the final photo row label the state they describe (#141)
    final_record.rs                — Named index constants for the 52-element final-record array; single source of truth for `fr[N]` writes in `finalize.rs` and reads in aerocapture-py
                                       (`results.rs`, `env.rs`)
    init.rs                        — Per-run initialization: `RunState` (draw-derived biases) from a `DispersionDraw`; `RunState::aero()` is the `AeroDispersions` view `physics::dynamics` reads;
                                       `RunState::noise_seed` derives the per-draw stochastic-stream seed (ADR-0006)
    photo.rs                       — The 30-column photo record: `build_photo_values` (per-tick snapshot row) + `build_event_photo_values` (sub-tick event row) share one `photo_physics`
                                       core (geodetic position, osculating orbit, inertial energy via `to_absolute_cartesian`, dispersed `aero_loads`) and write columns by `output::PHOTO_*`
                                       name; `push_photo_snapshot` (shared by `tick.rs` and `run_single`'s final row) + `append_event_photo_rows` (event rows + sort by time)
    output.rs                      — Record projections + CSV writers: the named 30-column photo-line layout (`PHOTO_*` consts + `PHOTO_LINE_LEN`, the `final_record.rs` style),
                                       `project_trajectory` (30 -> the 17-column PyO3 trajectory row, J->MJ / Pa->kPa), `extract_photo_csv_values` (22 columns) and `extract_final_csv_values`
                                       (39 of 52, read by `FR_*` name), `write_csv_output` (`final.<suffix>.csv` = `sim_number` + 39, `photo.<suffix>.csv` = 22) and the header/line writers
```

## Termination outcomes and virtual DV

Every termination produces a DV the cost function can rank. Captured runs get the real
orbital-correction DV (`orbit::maneuver::compute_deltav`); hyperbolic exits get
`HYPERBOLIC_BASE (10000) + v_excess`; crash / pending-crash / timeout get
`virtual_dv_non_capture = CRASH_FLOOR (3000) + 1000 * min(|E_orb - E_target|_MJkg, 50) - 500 * t/t_max`
with finite fallbacks on NaN/Inf inputs. The energy-proportional crash term softens the cliff near
the capture boundary so PSO/GA explore closer to it; the time-survival term keeps a cold-start
gradient. The consts live in `simulation/sim_types.rs`.

## Data files (`data/`)

- `data/atmosphere/mars.dat` — Mars density vs altitude table (tabulated MarsGram 3.8)
- `data/atmosphere/earth.dat` — Earth atmosphere table
- `data/atmosphere/mars_winds.dat` — Mars parametric wind profile (altitude vs zonal/meridional, based on Forget et al. 1999)
- `data/atmosphere/earth_winds.dat` — Earth parametric wind profile
- `data/reference_trajectory/msr_aller.dat` — MSR reference trajectory (energy vs pdyn/hdot/cos_bank)
- `data/reference_trajectory/esr_aller.dat` — ESR reference trajectory

The mission-optimized reference `training_output/mars/ref_trajectory.dat` (committed, produced by
`make_reference.py`) is what the ref-tracking training configs point at; the legacy
`data/reference_trajectory/*.dat` files stay for the nominal/test configs and the goldens.

## Tests

Three tiers: inline `#[cfg(test)]` unit tests (with proptest property tests), integration tests
under `tests/` (shared fixtures in `tests/common/`: `fixtures.rs`, `assertions.rs`), and E2E
subprocess tests. Dev-dependencies: `approx`, `rstest`, `proptest`, `tempfile`. Bit-identity gates
worth knowing by name: the six guidance goldens (`tests/reference_data/rust_golden/` at the repo
root, regenerated per the root `CLAUDE.md` lesson), `tests/entry_fan_agreement.rs`,
`tests/nn_flat_order_fixtures.rs`, `tests/nn_model_roundtrip.rs`. `cargo test --release` lists the
rest.

## Benchmarks

Two criterion benches under `benches/`; neither is a CI gate, since timings are noisy. `tick` is
the per-scheme guidance cost: every guidance call of one nominal flight, replayed from its exact
pre-call state, with the replay checked bit for bit against the flight before anything is timed.
`quant_forward` is the Mamba-962 forward pass in f64, f32 and quantized kernels. The numbers and
how to read them are in [docs/performance.md](../../docs/performance.md).

```bash
cargo bench --bench tick --manifest-path src/rust/Cargo.toml
```
