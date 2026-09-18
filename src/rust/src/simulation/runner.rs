//! Main simulation loop.
//!
//! Monte Carlo runs are parallelized with rayon (one thread per trajectory).

use crate::config::{AdaptiveConfig, PlanetConfig, SimInput};
use crate::data::SimData;
use crate::data::dispersions::DISPERSION_DRAW_LEN;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::{self, NavigationFilter};
use crate::integration::dopri45;
use crate::integration::events::{self, EventAction, EventContext, EventDef, EventRecord};
use crate::integration::rk4;
use crate::physics::atmosphere;
use crate::physics::dynamics::{AeroLoads, aero_loads, compute_derivatives, effective_airspeed};
use crate::simulation::init;
use crate::simulation::output::{self, PHOTO_LINE_LEN};
use crate::simulation::photo;
use rayon::prelude::*;
use std::time::Duration;

// Foundational simulation types and constants now live in `sim_types` (a leaf
// module). Re-exported / imported here so every existing `runner::X` path keeps
// resolving: `tick.rs`, `events.rs`, the `#[path]`-included test modules, and
// the external `aerocapture-py` crate all reach these symbols via `runner::`.
// Types are `pub` (aerocapture-py + events.rs consume them externally).
pub use super::sim_types::{SimError, SimState, SimStateOptions, TermReason};
// `DEG_TO_RAD` / `MIN_BOUNCE_ALT_FOR_CRASH_M` are both used inside this module
// AND re-consumed by `tick.rs` via `runner::`, so the re-export is `pub(crate)`.
// The remaining consts are used only inside this module's free functions.
use super::final_record::{FR_ECC, FR_ENERGY_MJKG, FR_IFINAL};
pub(crate) use super::sim_types::{DEG_TO_RAD, MIN_BOUNCE_ALT_FOR_CRASH_M};
use super::sim_types::{DOPRI45_ATOL, EVENT_TOL};
// These consts are consumed only by the `#[path]`-included `virtual_dv_tests`
// module (via `use super::*`); gate the re-export to test builds so non-test
// builds don't flag them unused (mirrors `virtual_dv_non_capture` below).
#[cfg(test)]
pub(crate) use super::sim_types::{
    CRASH_ENERGY_CAP_MJKG, CRASH_ENERGY_WEIGHT, CRASH_FLOOR, CRASH_TIME_BONUS, HYPERBOLIC_BASE,
};

// Termination classification, virtual-DV cost, and final-record assembly live in
// `finalize`. Re-exported below so existing `runner::*` paths and the
// `#[path]`-included test modules keep resolving these symbols.
pub use super::final_record::FINAL_RECORD_LEN;
pub use super::finalize::{
    build_final_record, ifinal_for, is_pending_crash, promote_pending_crash_if_applicable,
};
// Only the `#[path]`-included `virtual_dv_tests` module (via `use super::*`)
// consumes this symbol from runner's namespace; gate the re-export to test builds
// so non-test builds don't flag it unused.
#[cfg(test)]
pub(crate) use super::finalize::virtual_dv_non_capture;

// `SimState` construction lives in `run_init`. Re-exported so existing
// `runner::build_sim_state` callers (CLI path + `aerocapture-py` env) keep working.
pub use super::run_init::build_sim_state;

/// Run one navigation pass on a `SimState`, returning the `NavigationOutput`.
///
/// Shared between `build_sim_state` (primes `last_nav` so the RL env has a
/// valid initial observation) and `tick::step_one_tick` (invoked every outer
/// GNC tick). Dispatches on the state's `nav_filter` variant.
pub(crate) fn navigate_from_state(
    state: &mut SimState,
    data: &SimData,
    planet: &PlanetConfig,
) -> crate::gnc::navigation::estimator::NavigationOutput {
    let position_true = [state.state[0], state.state[1], state.state[2]];
    let velocity_true = [state.state[3], state.state[4], state.state[5]];
    match &mut state.nav_filter {
        NavigationFilter::Bias(nav_state) => estimator::navigate(
            &position_true,
            &velocity_true,
            state.guidance_state.aoa_commanded,
            state.sim_time,
            &state.nav_biases,
            nav_state,
            data,
            planet,
            state.run_state.density_bias,
            state.run_state.density_perturbation,
            state.run_state.cx_bias,
            state.run_state.cz_bias,
            state.run_state.mass_bias,
            state.run_state.incidence_bias,
            state.run_state.ref_area_bias,
            state.run_state.filter_gain_bias,
        ),
        NavigationFilter::Ekf {
            ekf,
            imu,
            star_tracker,
            st_config,
            ekf_config,
            legacy,
            ..
        } => estimator::navigate_ekf(
            &position_true,
            &velocity_true,
            state.guidance_state.aoa_commanded,
            state.sim_time,
            data.periods.navigation,
            &state.nav_biases,
            legacy,
            ekf,
            imu,
            star_tracker,
            st_config,
            ekf_config,
            data,
            planet,
            state.run_state.density_bias,
            state.run_state.density_perturbation,
            state.run_state.cx_bias,
            state.run_state.cz_bias,
            state.run_state.mass_bias,
            state.run_state.incidence_bias,
            state.run_state.ref_area_bias,
        ),
    }
}

/// Result from a single simulation run.
pub(crate) struct SimResult {
    pub(crate) sim_idx: i32,
    pub(crate) final_line: [f64; FINAL_RECORD_LEN],
    pub(crate) photo_lines: Vec<[f64; PHOTO_LINE_LEN]>,
    pub(crate) dispersions: [f64; DISPERSION_DRAW_LEN],
    pub(crate) supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,
}

/// Per-call switches for one fan-out: CLI photo output, trajectory capture, wall-clock cap.
#[derive(Clone, Copy, Debug, Default)]
struct RunOptions {
    write_photo: bool,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
}

/// Which sim gets the photo file in a Monte Carlo batch (`visualize_sim` is 1-based;
/// default the last one); the only sim when there is just one.
fn photo_sim_index(config: &SimInput, n_sims: i32) -> i32 {
    if n_sims > 1 {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    }
}

/// The draws a config asks for: `n_sims` from the dispersion config when present
/// (announced on stderr for the CLI), else `n_sims` undispersed defaults.
fn draws_from_config(
    config: &SimInput,
    data: &SimData,
    announce: bool,
) -> Vec<crate::data::dispersions::DispersionDraw> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    match data.dispersion_config.as_ref() {
        Some(dc) => {
            let draws = dc.generate_draws(n_sims as usize);
            if announce {
                let on_off = |b: bool| if b { "on" } else { "off" };
                eprintln!(
                    "Monte Carlo: {} draws from seed {}, domains: state={} atmo={} aero={} nav={} mass={} vehicle={} pilot={} nav_filter={}",
                    draws.len(),
                    dc.seed,
                    on_off(dc.initial_state.is_some()),
                    on_off(dc.atmosphere.is_some()),
                    on_off(dc.aerodynamics.is_some()),
                    on_off(dc.navigation.is_some()),
                    on_off(dc.mass.is_some()),
                    on_off(dc.vehicle.is_some()),
                    on_off(dc.pilot.is_some()),
                    on_off(dc.nav_filter.is_some()),
                );
            }
            draws
        }
        None => vec![crate::data::dispersions::DispersionDraw::default(); n_sims as usize],
    }
}

/// The one Monte Carlo fan-out: one run state per draw, Rayon-parallel when there is
/// more than one draw, sequential otherwise, each result stamped with its draw.
/// Every public entry point is a draw-source adapter over this function.
fn run_core(
    config: &SimInput,
    data: &SimData,
    draws: &[crate::data::dispersions::DispersionDraw],
    opts: RunOptions,
) -> Result<Vec<SimResult>, SimError> {
    let RunOptions {
        write_photo,
        include_trajectories,
        wall_timeout,
    } = opts;
    let n_sims = draws.len() as i32;
    if n_sims == 0 {
        return Ok(Vec::new());
    }
    let is_mc = n_sims > 1;

    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = draws
        .iter()
        .map(|draw| (init::init_run_from_draw(data, draw), draw.to_array()))
        .collect();

    let photo_sim_idx = photo_sim_index(config, n_sims);

    if is_mc {
        let start = std::time::Instant::now();
        if write_photo {
            eprintln!("Running {} simulations in parallel...", n_sims);
        }
        let results: Vec<SimResult> = run_states
            .par_iter()
            .enumerate()
            .map(|(idx, (run_state, disp_array))| {
                let do_photo = (write_photo && idx as i32 == photo_sim_idx) || include_trajectories;
                let mut result =
                    run_single(config, data, run_state, idx as i32, do_photo, wall_timeout)?;
                result.dispersions = *disp_array;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if write_photo {
            let elapsed = start.elapsed();
            eprintln!(
                "Completed {} simulations in {:.3}s ({:.1} sims/s)",
                n_sims,
                elapsed.as_secs_f64(),
                n_sims as f64 / elapsed.as_secs_f64(),
            );
        }
        Ok(results)
    } else {
        let (run_state, disp_array) = &run_states[0];
        if write_photo && config.screen_output {
            eprintln!(
                "  Entry: alt={:.3} km, vel={:.3} m/s, fpa={:.5} deg",
                run_state.entry.state.altitude / 1e3,
                run_state.entry.state.velocity,
                run_state.entry.state.flight_path.to_degrees(),
            );
        }
        let mut result = run_single(
            config,
            data,
            run_state,
            0,
            write_photo || include_trajectories,
            wall_timeout,
        )?;
        result.dispersions = *disp_array;
        Ok(vec![result])
    }
}

/// Run the full simulation (CLI): draws from the config, photo + CSV output.
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let draws = draws_from_config(config, data, true);
    let opts = RunOptions {
        write_photo: true,
        include_trajectories: false,
        wall_timeout: None,
    };
    let results = run_core(config, data, &draws, opts)?;
    output::write_csv_output(
        config,
        &results,
        photo_sim_index(config, draws.len() as i32),
    )?;
    Ok(())
}

/// Assemble a `RunOutput` from one `SimResult`: project the trajectory (only when
/// requested), extract energy/ecc, and apply the capture predicate.
fn assemble_run_output(r: SimResult, include_trajectories: bool) -> crate::RunOutput {
    let energy = r.final_line[FR_ENERGY_MJKG]; // MJ/kg
    let ecc = r.final_line[FR_ECC];
    let trajectory = if include_trajectories {
        output::project_trajectory(&r.photo_lines)
    } else {
        Vec::new()
    };
    let ifinal_val = r.final_line[FR_IFINAL] as i32;
    crate::RunOutput {
        trajectory,
        final_record: r.final_line,
        captured: ifinal_val == 3 && ecc < 1.0 && energy < 0.0,
        dispersions: r.dispersions,
        supervised_trace: r.supervised_trace,
    }
}

/// Run simulation and return structured results (no file I/O).
///
/// Same physics as `run()`, but returns `Vec<RunOutput>` instead of writing files.
/// Used by the PyO3 interface for direct Python access.
pub fn run_for_api(
    config: &SimInput,
    data: &SimData,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    let draws = draws_from_config(config, data, false);
    let opts = RunOptions {
        write_photo: false,
        include_trajectories,
        wall_timeout,
    };
    Ok(run_core(config, data, &draws, opts)?
        .into_iter()
        .map(|r| assemble_run_output(r, include_trajectories))
        .collect())
}

/// Run simulation with pre-computed dispersion draws (no file I/O).
///
/// Accepts a `Vec<DispersionDraw>` from the caller instead of generating
/// draws internally. Each draw maps to exactly one simulation run.
/// Used by the PyO3 `run_with_draws()` binding for external sampling.
pub fn run_for_api_with_draws(
    config: &SimInput,
    data: &SimData,
    external_draws: Vec<crate::data::dispersions::DispersionDraw>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    let opts = RunOptions {
        write_photo: false,
        include_trajectories,
        wall_timeout,
    };
    Ok(run_core(config, data, &external_draws, opts)?
        .into_iter()
        .map(|r| assemble_run_output(r, include_trajectories))
        .collect())
}

/// Run ONE training-grid cell, bit-identical to the per-seed `run_batch` path.
///
/// Generates the static draw from `data`'s dispersion config reseeded to `seed`
/// (mirrors the per-seed path's `monte_carlo.seed` override) and runs a single
/// trajectory with `sim_idx = 0` — so the per-sim EKF / Gauss-Markov RNG stream
/// matches the per-seed path, where `simulation.random_seed` is constant and
/// `n_sims == 1`. This is the bit-identity invariant `run_grid` relies on.
pub fn run_for_api_cell(
    config: &SimInput,
    data: &SimData,
    seed: u64,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<crate::RunOutput, SimError> {
    let draw = data.draw_from_seed(seed);
    let opts = RunOptions {
        write_photo: false,
        include_trajectories,
        wall_timeout,
    };
    let mut results = run_core(config, data, std::slice::from_ref(&draw), opts)?;
    let result = results
        .pop()
        .expect("run_core returns exactly one result for one draw");
    Ok(assemble_run_output(result, include_trajectories))
}

/// Run a single simulation, returning results.
fn run_single(
    config: &SimInput,
    data: &SimData,
    run_state: &init::RunState,
    sim_idx: i32,
    write_photo: bool,
    wall_timeout: Option<Duration>,
) -> Result<SimResult, SimError> {
    let planet = &config.planet;

    // Construct the base SimState via the shared constructor (identical seed
    // derivation, GNC init, and bias-mode last_nav priming as the RL env path);
    // `sim_idx as u64` reproduces the historical per-sim seeds exactly:
    // EKF `random_seed + sim_idx*10_000`, GM-RNG `... + 0xDE45`.
    let opts = SimStateOptions {
        write_photo,
        wall_timeout,
        is_single: config.n_sims <= 1 && config.screen_output,
    };
    let mut sim_state = build_sim_state(config, data, *run_state, sim_idx as u64, opts);
    let is_single = sim_state.is_single;

    // Event detection setup (used by adaptive integrator)
    let event_defs = build_event_defs();
    let event_ctx = build_event_ctx(config, data);

    if is_single {
        eprintln!(
            "  Init: entry.initial_bank={:.5}deg, reference_bank_angle={:.5}deg, sim.bank_angle={:.5}deg",
            run_state.entry.initial_bank.to_degrees(),
            sim_state.reference_bank_angle.to_degrees(),
            sim_state.bank_angle.to_degrees()
        );
    }

    // Main simulation loop
    while sim_state.term == TermReason::None {
        let _outcome = crate::simulation::tick::step_one_tick(
            &mut sim_state,
            config,
            data,
            planet,
            None,
            &event_defs,
            &event_ctx,
        );
    }

    // Final photo snapshot
    if sim_state.write_photo {
        photo::push_photo_snapshot(&mut sim_state, planet, data);
    }

    // === Final conditions ===
    let (alt_final, _lat_final) = geodetic_from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        planet,
    );

    if sim_state.is_single {
        eprintln!(
            "  Final: alt={:.3} km, vel={:.3} m/s, t={:.1} s, steps={}, term={:?}",
            alt_final / 1e3,
            sim_state.state[3],
            sim_state.sim_time,
            sim_state.step,
            sim_state.term,
        );
    }

    promote_pending_crash_if_applicable(&mut sim_state, planet);

    // The 52-element final record / termination classification / virtual-DV is
    // assembled by `build_final_record` (the same path the RL per-step env API
    // takes via `tick.rs`), keeping CLI and env outputs bit-identical.
    let final_record = build_final_record(&sim_state, data, planet);

    let event_records = std::mem::take(&mut sim_state.event_records);

    // Append event records as photo rows and sort by time (column 0)
    if sim_state.write_photo {
        photo::append_event_photo_rows(&mut sim_state, &event_records, planet, data);
    }

    let photo_lines = std::mem::take(&mut sim_state.photo_lines);

    let supervised_trace = if config.collect_supervised {
        std::mem::take(&mut sim_state.supervised_trace)
    } else {
        Vec::new()
    };

    Ok(SimResult {
        sim_idx,
        final_line: final_record,
        photo_lines,
        dispersions: [0.0; DISPERSION_DRAW_LEN],
        supervised_trace,
    })
}

/// Build the standard aerocapture event definitions.
///
/// Convenience wrapper around `events::build_aerocapture_events()` for tests
/// and external callers that need to drive `step_one_tick` directly.
pub fn build_event_defs() -> Vec<events::EventDef> {
    events::build_aerocapture_events()
}

/// Build the standard `EventContext` from a config + data pair.
///
/// Matches the construction in `run_single`. Use alongside `build_event_defs()`
/// when calling `step_one_tick` outside the normal runner loop.
pub fn build_event_ctx(config: &SimInput, data: &SimData) -> events::EventContext {
    let planet = &config.planet;
    let exit_altitude = data.final_conditions.altitude;
    events::EventContext {
        planet_radius: planet.equatorial_radius,
        polar_radius: planet.polar_radius,
        exit_altitude,
        exit_velocity_threshold: data.guidance.exit_velocity_threshold,
    }
}

/// Run a single simulation and return the 52-element final record in memory.
///
/// Equivalent to `run_single` but skips file I/O and returns the final record
/// directly. Intended for tests that need to compare against the step-API path.
pub fn run_single_collect(
    config: &SimInput,
    data: &SimData,
) -> Result<[f64; FINAL_RECORD_LEN], SimError> {
    let draw = crate::data::dispersions::DispersionDraw::default();
    let mut results = run_core(
        config,
        data,
        std::slice::from_ref(&draw),
        RunOptions::default(),
    )?;
    Ok(results.pop().expect("one draw, one result").final_line)
}

/// Perform one integration step using Gill's RK4.
pub(crate) fn integrate_step(
    sim: &mut SimState,
    dt: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) {
    sim.gill_toggle = 0;
    let aero = run_state.aero();

    for k in 1..=4 {
        let derivs = compute_derivatives(&sim.state, sim.bank_angle, sim.aoa, planet, data, &aero);
        rk4::rk4_increment(
            dt,
            &derivs,
            k,
            8,
            &mut sim.gill_toggle,
            &mut sim.accumulator,
            &mut sim.state,
        );
    }
}

pub(crate) struct AdaptiveEventResult {
    pub(crate) triggered: Vec<events::TriggeredEvent>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_adaptive_with_events(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    event_defs: &[EventDef],
    event_ctx: &EventContext,
    tick_start_time: f64,
) -> AdaptiveEventResult {
    const MAX_SUBSTEPS: u32 = 1000;

    let bank_angle = sim.bank_angle;
    let aoa = sim.aoa;
    let aero = run_state.aero();
    let mut t_remaining = dt_outer;
    let mut h = config.initial_dt.min(t_remaining).max(config.min_dt);
    let mut n_substeps: u32 = 0;
    let mut n_rejections: u32 = 0;

    // Cache event guard values at beginning of tick
    let mut g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);

    let mut all_triggered: Vec<events::TriggeredEvent> = Vec::new();

    while t_remaining > 1e-14 {
        h = h.min(t_remaining).min(config.max_dt).max(config.min_dt);

        // If remaining time is very small, take it in one step regardless
        if t_remaining <= config.min_dt * 1.5 {
            h = t_remaining;
        }

        let y0 = sim.state;

        let (result, stages) = dopri45::dopri45_step_with_stages(
            &mut sim.state,
            h,
            &mut sim.dopri,
            &DOPRI45_ATOL,
            config.rtol,
            &mut |state| compute_derivatives(state, bank_angle, aoa, planet, data, &aero),
        );

        if result.accepted {
            // Check for events in this accepted substep
            let k1 = &stages[0];
            let k7 = &stages[6];

            let t_base = tick_start_time + (dt_outer - t_remaining);
            if let Some(triggered) = events::check_events_and_locate(
                &y0, &sim.state, h, k1, k7, event_defs, event_ctx, &g_prev, EVENT_TOL, t_base,
            ) {
                let event = &event_defs[triggered.event_index];

                // Record this event (GNC fields populated by caller after return)
                sim.event_records.push(EventRecord {
                    time: triggered.time,
                    state: triggered.state,
                    event_type: event.event_type,
                    bank_angle_deg: 0.0,
                    aoa_deg: 0.0,
                    cumulative_bank_change_deg: 0.0,
                    guidance_phase: 0.0,
                    density_gain: 0.0,
                });

                // Rewind state to the event location
                sim.state = triggered.state;

                // Invalidate FSAL -- state was rewound, cached derivative is stale
                sim.dopri.invalidate_fsal();

                match event.action {
                    EventAction::Terminate(_) => {
                        // Terminal event: return immediately
                        all_triggered.push(triggered);
                        return AdaptiveEventResult {
                            triggered: all_triggered,
                        };
                    }
                    EventAction::Record | EventAction::PhaseTransition => {
                        // Non-terminal: adjust t_remaining for partial step consumed
                        let consumed = triggered.theta * h;
                        t_remaining -= consumed;
                        n_substeps += 1;
                        h = result.dt_next;

                        // Re-evaluate guard values at the new (event) state
                        g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);

                        // Force g_prev to exactly 0.0 for the fired event so it won't
                        // re-trigger on the next substep (the g0 == 0.0 skip in
                        // check_events_and_locate prevents re-detection at the same
                        // zero-crossing).
                        g_prev[triggered.event_index] = 0.0;

                        all_triggered.push(triggered);

                        // Check substep cap BEFORE continuing — the old `continue`
                        // bypassed the cap check at the bottom of the loop, allowing
                        // unbounded event accumulation when trajectories oscillate
                        // near an event boundary (e.g. FPA ≈ 0 at bounce).
                        if n_substeps + n_rejections >= MAX_SUBSTEPS {
                            return AdaptiveEventResult {
                                triggered: all_triggered,
                            };
                        }

                        continue;
                    }
                }
            }

            // No event: normal accepted step
            t_remaining -= h;
            n_substeps += 1;
            h = result.dt_next;

            // Update guard values for next substep
            g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);
        } else {
            // Rejected step: dopri45_step_with_stages restores state to y0 internally
            n_rejections += 1;
            h = result.dt_next;
        }

        if n_substeps + n_rejections >= MAX_SUBSTEPS {
            eprintln!(
                "WARNING: adaptive integrator hit {} step limit with t_remaining={:.2e}s ({} accepted, {} rejected)",
                MAX_SUBSTEPS, t_remaining, n_substeps, n_rejections,
            );
            return AdaptiveEventResult {
                triggered: all_triggered,
            };
        }
    }

    AdaptiveEventResult {
        triggered: all_triggered,
    }
}

/// Update peak tracking values (heat flux, load factor, dynamic pressure)
/// after each integration step.
pub(crate) fn track_peak_values(
    sim: &mut SimState,
    altitude: f64,
    sim_time: f64,
    data: &SimData,
    run_state: &init::RunState,
) {
    let v = sim.state[3];
    let gamma = sim.state[4];
    let psi = sim.state[5];
    let lat = sim.state[2];
    let aero = run_state.aero();
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        aero.density_bias,
        aero.density_perturbation,
    );

    // Wind-corrected velocity for aero-dependent quantities
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, &aero);

    // Heat flux (W/m²), dynamic pressure (Pa), load factor (m/s²): the shared laws
    // in `physics::dynamics` (same operands as dflux in compute_derivatives).
    let AeroLoads {
        heat_flux,
        pdyn,
        load_factor,
    } = aero_loads(rho, v_eff, sim.aoa, data, &aero);

    if heat_flux > sim.max_heat_flux {
        sim.max_heat_flux = heat_flux;
        sim.alt_max_flux = altitude;
        sim.time_max_flux = sim_time;
    }
    if load_factor > sim.max_load_factor {
        sim.max_load_factor = load_factor;
        sim.alt_max_load = altitude;
        sim.time_max_load = sim_time;
    }
    if pdyn > sim.max_dyn_pressure {
        sim.max_dyn_pressure = pdyn;
        sim.alt_max_pdyn = altitude;
        sim.time_max_pdyn = sim_time;
    }
}

#[cfg(test)]
#[path = "run_output_tests.rs"]
mod run_output_tests;

#[cfg(test)]
#[path = "virtual_dv_tests.rs"]
mod virtual_dv_tests;

#[cfg(test)]
#[path = "pending_crash_tests.rs"]
mod pending_crash_tests;
