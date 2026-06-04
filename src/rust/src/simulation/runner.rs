//! Main simulation loop.
//!
//! Monte Carlo runs are parallelized with rayon (one thread per trajectory).

use crate::config::{AdaptiveConfig, PlanetConfig, SimInput};
use crate::data::SimData;
use crate::data::dispersions::DISPERSION_DRAW_LEN;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::gnc::navigation::estimator::{self, NavigationFilter};
use crate::integration::dopri45;
use crate::integration::events::{self, EventAction, EventContext, EventDef, EventRecord};
use crate::integration::rk4;
use crate::orbit::elements;
use crate::physics::{atmosphere, gravity};
use crate::simulation::init;
use crate::simulation::output;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Duration;

// Foundational simulation types and constants now live in `sim_types` (a leaf
// module). Re-exported / imported here so every existing `runner::X` path keeps
// resolving: `tick.rs`, `events.rs`, the `#[path]`-included test modules, and
// the external `aerocapture-py` crate all reach these symbols via `runner::`.
// Types are `pub` (aerocapture-py + events.rs consume them externally).
pub use super::sim_types::{SimError, SimState, TermReason};
// `DEG_TO_RAD` / `MIN_BOUNCE_ALT_FOR_CRASH_M` are both used inside this module
// AND re-consumed by `tick.rs` via `runner::`, so the re-export is `pub(crate)`.
// The remaining consts are used only inside this module's free functions.
pub(crate) use super::sim_types::{DEG_TO_RAD, MIN_BOUNCE_ALT_FOR_CRASH_M};
use super::sim_types::{DOPRI45_ATOL, EVENT_TOL, G0};
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
struct SimResult {
    sim_idx: i32,
    final_line: [f64; FINAL_RECORD_LEN],
    photo_lines: Vec<[f64; 30]>,
    dispersions: [f64; DISPERSION_DRAW_LEN],
    supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,
}

/// Shared simulation orchestration: build run states, dispatch parallel/sequential runs.
fn run_core(
    config: &SimInput,
    data: &SimData,
    write_photo: bool,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<SimResult>, SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let is_mc = n_sims > 1;

    let draws = data.dispersion_config.as_ref().map(|dc| {
        let draws = dc.generate_draws(n_sims as usize);
        if write_photo {
            let on_off = |b: bool| if b { "on" } else { "off" };
            eprintln!(
                "Monte Carlo: {} draws from seed {}, domains: state={} atmo={} aero={} nav={} mass={} vehicle={} pilot={} nav_filter={}",
                draws.len(), dc.seed,
                on_off(dc.initial_state.is_some()), on_off(dc.atmosphere.is_some()),
                on_off(dc.aerodynamics.is_some()), on_off(dc.navigation.is_some()),
                on_off(dc.mass.is_some()), on_off(dc.vehicle.is_some()),
                on_off(dc.pilot.is_some()), on_off(dc.nav_filter.is_some()),
            );
        }
        draws
    });

    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = (0..n_sims)
        .map(|sim_idx| {
            let draw = if let Some(ref d) = draws {
                &d[sim_idx as usize]
            } else {
                &crate::data::dispersions::DispersionDraw::default()
            };
            (init::init_run_from_draw(data, draw), draw.to_array())
        })
        .collect();

    let photo_sim_idx = if is_mc {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

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

/// Run the full simulation.
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let photo_sim_idx = if n_sims > 1 {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

    let results = run_core(config, data, true, false, None)?;
    write_csv_output(config, &results, photo_sim_idx)?;
    Ok(())
}

/// Project each 30-element photo line onto the 17-element trajectory row exposed
/// by the PyO3 API. Index mapping and unit scaling (energy J->MJ, pdyn Pa->kPa)
/// are the contract documented on `BatchResults` trajectory columns.
fn project_trajectory(photo_lines: &[[f64; 30]]) -> Vec<[f64; 17]> {
    photo_lines
        .iter()
        .map(|p| {
            [
                p[1],        // [0]  alt_km
                p[2],        // [1]  lon_deg
                p[3],        // [2]  lat_deg
                p[4],        // [3]  vel_m_s
                p[5],        // [4]  fpa_deg
                p[6],        // [5]  heading_deg
                p[24],       // [6]  heat_flux_kw_m2
                p[0],        // [7]  time_s
                p[18] / 1e6, // [8]  energy_mj_kg
                p[19] / 1e3, // [9]  pdyn_kpa
                p[14],       // [10] bank_angle_deg
                p[9],        // [11] inclination_deg
                p[25],       // [12] g_load_g
                p[26],       // [13] nav_density_ratio
                p[27],       // [14] truth_density_kg_m3
                p[28],       // [15] heat_load_kj_m2
                p[29],       // [16] density_perturbation
            ]
        })
        .collect()
}

/// Assemble a `RunOutput` from one `SimResult`: project the trajectory (only when
/// requested), extract energy/ecc, and apply the capture predicate.
fn assemble_run_output(r: SimResult, include_trajectories: bool) -> crate::RunOutput {
    let energy = r.final_line[7]; // MJ/kg
    let ecc = r.final_line[9];
    let trajectory = if include_trajectories {
        project_trajectory(&r.photo_lines)
    } else {
        Vec::new()
    };
    let ifinal_val = r.final_line[31] as i32;
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
    let results = run_core(config, data, false, include_trajectories, wall_timeout)?;

    Ok(results
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
    let n = external_draws.len();
    let is_mc = n > 1;

    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = external_draws
        .iter()
        .map(|draw| (init::init_run_from_draw(data, draw), draw.to_array()))
        .collect();

    let results: Vec<SimResult> = if is_mc {
        run_states
            .par_iter()
            .enumerate()
            .map(|(idx, (run_state, disp_array))| {
                let mut result = run_single(
                    config,
                    data,
                    run_state,
                    idx as i32,
                    include_trajectories,
                    wall_timeout,
                )?;
                result.dispersions = *disp_array;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if n == 1 {
        let (run_state, disp_array) = &run_states[0];
        let mut result = run_single(
            config,
            data,
            run_state,
            0,
            include_trajectories,
            wall_timeout,
        )?;
        result.dispersions = *disp_array;
        vec![result]
    } else {
        return Ok(Vec::new());
    };

    Ok(results
        .into_iter()
        .map(|r| assemble_run_output(r, include_trajectories))
        .collect())
}

/// Write output in CSV format with named headers and clean schema.
fn write_csv_output(
    config: &SimInput,
    results: &[SimResult],
    photo_sim_idx: i32,
) -> Result<(), SimError> {
    let suffix = config.results_suffix.trim_start_matches('.');
    let final_path = config.output_path(&format!("final.{}.csv", suffix));
    let mut final_file = BufWriter::new(
        File::create(&final_path)
            .map_err(|e| SimError(format!("Cannot create {}: {}", final_path, e)))?,
    );

    output::write_final_csv_header(&mut final_file)
        .map_err(|e| SimError(format!("Final CSV header error: {}", e)))?;

    for result in results {
        let csv_values = extract_final_csv_values(&result.final_line);
        output::write_final_csv_line(&mut final_file, result.sim_idx + 1, &csv_values)
            .map_err(|e| SimError(format!("Final CSV write error: {}", e)))?;
    }
    final_file
        .flush()
        .map_err(|e| SimError(format!("Final CSV flush error: {}", e)))?;

    // Write photo CSV
    let photo_path = config.output_path(&format!("photo.{}.csv", suffix));
    if let Some(result) = results.iter().find(|r| r.sim_idx == photo_sim_idx) {
        let mut photo_file = BufWriter::new(
            File::create(&photo_path)
                .map_err(|e| SimError(format!("Cannot create {}: {}", photo_path, e)))?,
        );

        output::write_photo_csv_header(&mut photo_file)
            .map_err(|e| SimError(format!("Photo CSV header error: {}", e)))?;

        for line in &result.photo_lines {
            let csv_values = extract_photo_csv_values(line);
            output::write_photo_csv_line(&mut photo_file, &csv_values)
                .map_err(|e| SimError(format!("Photo CSV write error: {}", e)))?;
        }
        photo_file
            .flush()
            .map_err(|e| SimError(format!("Photo CSV flush error: {}", e)))?;
    }

    Ok(())
}

/// Extract 22 CSV values from the 30-element photo array.
/// Drops: [20] radial_velocity_2 (duplicate), [22] sim_number, [23] reserved, [24-27] trajectory-only, [29] density_perturbation.
fn extract_photo_csv_values(values: &[f64; 30]) -> [f64; 22] {
    [
        values[0],  // time_s
        values[1],  // altitude_km
        values[2],  // longitude_deg
        values[3],  // latitude_deg
        values[4],  // velocity_m_s
        values[5],  // flight_path_deg
        values[6],  // azimuth_deg
        values[7],  // semi_major_axis_km
        values[8],  // eccentricity
        values[9],  // inclination_deg
        values[10], // raan_deg
        values[11], // periapsis_alt_km
        values[12], // apoapsis_alt_km
        values[13], // phase
        values[14], // bank_angle_deg
        values[15], // radial_velocity_m_s
        values[16], // aoa_deg
        values[17], // cumulative_bank_change_deg
        values[18], // energy_j_kg
        values[19], // dynamic_pressure_pa
        values[21], // dynamic_pressure_onboard_kpa (skip [20] duplicate)
        values[28], // heat_load_kj_m2
    ]
}

/// Extract 39 CSV values from the 52-element final array.
/// Drops 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
fn extract_final_csv_values(values: &[f64; FINAL_RECORD_LEN]) -> [f64; 39] {
    [
        values[0],  // altitude_km
        values[1],  // longitude_deg
        values[2],  // latitude_deg
        values[3],  // velocity_m_s
        values[4],  // flight_path_deg
        values[5],  // azimuth_deg
        values[6],  // radial_velocity_m_s
        values[7],  // energy_mj_kg
        values[8],  // semi_major_axis_km
        values[9],  // eccentricity
        values[10], // inclination_deg
        values[11], // raan_deg
        values[12], // arg_periapsis_deg
        values[13], // true_anomaly_deg
        values[14], // periapsis_alt_km
        values[15], // apoapsis_alt_km
        values[16], // max_heat_flux_kw_m2
        values[17], // max_load_factor_g
        values[18], // max_dyn_pressure_kpa
        values[19], // alt_max_flux_km
        values[20], // alt_max_load_km
        values[21], // alt_max_pdyn_km
        values[22], // time_max_flux_s
        values[23], // time_max_load_s
        values[24], // time_max_pdyn_s
        values[25], // bounce_alt_km
        values[26], // bounce_time_s
        values[27], // sim_time_s
        values[28], // integrated_flux_mj_m2
        values[29], // periapsis_err_km
        values[30], // apoapsis_err_km
        values[31], // ifinal
        values[37], // dv1_m_s
        values[38], // dv2_m_s
        values[39], // dv3_m_s
        values[40], // dv12_m_s
        values[41], // dv_total_m_s
        values[45], // cumulative_bank_change_deg
        values[48], // n_roll_reversals
    ]
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
    let mut sim_state = build_sim_state(config, data, *run_state, sim_idx as u64);

    // CLI-specific overrides not produced by `build_sim_state` (which targets the
    // RL env defaults: no photo, no wall timeout, not the single-run banner).
    sim_state.write_photo = write_photo;
    sim_state.wall_timeout = wall_timeout;
    sim_state.is_single = config.n_sims <= 1 && config.screen_output;
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
        let sim_time = sim_state.sim_time;
        let dynamic_pressure_for_photo = sim_state.dynamic_pressure_for_photo;
        let density_estimate_for_photo = sim_state.density_estimate_for_photo;
        let sim_idx = sim_state.sim_idx;
        let cumulative_bank_change_deg = sim_state.cumulative_bank_change_deg;
        let density_gain = sim_state.nav_filter.density_gain();
        let run_state_snap = sim_state.run_state;
        let cumulative_flux = sim_state.state[6];
        let guidance_phase_for_photo = sim_state.guidance_phase_for_photo;
        let photo_line = build_photo_values(
            &sim_state,
            sim_time,
            planet,
            dynamic_pressure_for_photo,
            density_estimate_for_photo,
            sim_idx + 1,
            cumulative_bank_change_deg * DEG_TO_RAD,
            data,
            density_gain,
            &run_state_snap,
            cumulative_flux,
            guidance_phase_for_photo,
        );
        sim_state.photo_lines.push(photo_line);
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
        for record in &event_records {
            sim_state.photo_lines.push(build_event_photo_values(
                &record.state,
                record.time,
                planet,
                data,
                &sim_state.run_state,
                record.bank_angle_deg,
                record.aoa_deg,
                record.cumulative_bank_change_deg,
                record.guidance_phase,
                record.density_gain,
            ));
        }
        sim_state
            .photo_lines
            .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
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
    let run_state = init::init_run_from_draw(data, &draw);
    let result = run_single(config, data, &run_state, 0, false, None)?;
    Ok(result.final_line)
}

/// Build a photo snapshot line.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &PlanetConfig,
    dynamic_pressure: f64,
    density_estimate: f64,
    sim_index: i32,
    cumulative_bank_change: f64,
    data: &SimData,
    density_gain: f64,
    run_state: &init::RunState,
    cumulative_flux: f64,
    guidance_phase: i32,
) -> [f64; 30] {
    let (altitude, latitude) =
        geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

    let orbit = elements::from_spherical(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / sim.state[0];
    let velocity_radial = sim.state[3] * sim.state[4].sin();

    let phase = guidance_phase as f64;

    // Compute per-timestep heat flux, g-load, and truth density for trajectory output.
    // Use dispersed values (matching track_peak_values) so trajectory plots are consistent
    // with final_record peak values and constraint classification.
    let rho_truth = data.atmosphere.density_at(altitude);
    let rho_dispersed = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );
    // Wind-corrected velocity for aero-dependent quantities
    let v_eff = effective_airspeed(
        sim.state[3],
        sim.state[4],
        sim.state[5],
        sim.state[2],
        altitude,
        data,
        run_state,
    );
    let heat_flux = data.capsule.cq * rho_dispersed.sqrt() * v_eff.powf(3.05);
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho_dispersed * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    [
        sim_time,
        altitude / 1e3,
        sim.state[1] / DEG_TO_RAD,
        latitude / DEG_TO_RAD,
        sim.state[3],
        sim.state[4] / DEG_TO_RAD,
        sim.state[5] / DEG_TO_RAD,
        orbit.semi_major_axis / 1e3,
        orbit.eccentricity,
        orbit.inclination / DEG_TO_RAD,
        orbit.raan / DEG_TO_RAD,
        orbit.periapsis_alt / 1e3,
        orbit.apoapsis_alt / 1e3,
        phase,
        sim.bank_angle / DEG_TO_RAD,
        velocity_radial,
        sim.aoa / DEG_TO_RAD,
        cumulative_bank_change / DEG_TO_RAD,
        energy,
        dynamic_pressure,
        velocity_radial,
        0.5 * density_estimate * sim.state[3] * sim.state[3] / 1e3,
        sim_index as f64,
        0.0,
        heat_flux / 1e3,                // [24] heat_flux kW/m²
        load_factor / G0,               // [25] g-load in g's
        density_gain,                   // [26] nav density ratio (estimated/model)
        rho_truth,                      // [27] truth density kg/m³
        cumulative_flux / 1e3,          // [28] heat_load_kj_m2 (J/m2 -> kJ/m2)
        run_state.density_perturbation, // [29] density_perturbation (fractional GM value)
    ]
}

/// Build a photo row from an event record's state.
///
/// Computes the same physics quantities as `build_photo_values` but uses the event
/// state directly. GNC-dependent values (bank_angle, aoa, cumulative_bank_change,
/// phase, density_gain) are carried from the enclosing tick because events occur
/// mid-tick and GNC quantities are constant within a tick.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_event_photo_values(
    state: &[f64; 8],
    event_time: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    bank_angle_deg: f64,
    aoa_deg: f64,
    cumulative_bank_change_deg: f64,
    guidance_phase: f64,
    density_gain: f64,
) -> [f64; 30] {
    let (altitude, latitude) = geodetic_from_spherical(state[0], state[1], state[2], planet);

    let orbit = elements::from_spherical(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / state[0];
    let velocity_radial = state[3] * state[4].sin();

    let rho_truth = data.atmosphere.density_at(altitude);
    let rho_dispersed = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );
    let v_eff = effective_airspeed(
        state[3], state[4], state[5], state[2], altitude, data, run_state,
    );
    let heat_flux = data.capsule.cq * rho_dispersed.sqrt() * v_eff.powf(3.05);
    let pdyn = 0.5 * rho_dispersed * v_eff * v_eff;

    let aoa_dispersed = run_state.incidence_bias; // aoa=0 + bias
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho_dispersed * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    // cumulative heat load: state[6] is integrated flux in J/m²
    let cumulative_flux = state[6];

    [
        event_time,                     // [0]  time_s
        altitude / 1e3,                 // [1]  altitude_km
        state[1] / DEG_TO_RAD,          // [2]  longitude_deg
        latitude / DEG_TO_RAD,          // [3]  latitude_deg
        state[3],                       // [4]  velocity_m_s
        state[4] / DEG_TO_RAD,          // [5]  flight_path_deg
        state[5] / DEG_TO_RAD,          // [6]  azimuth_deg
        orbit.semi_major_axis / 1e3,    // [7]  semi_major_axis_km
        orbit.eccentricity,             // [8]  eccentricity
        orbit.inclination / DEG_TO_RAD, // [9]  inclination_deg
        orbit.raan / DEG_TO_RAD,        // [10] raan_deg
        orbit.periapsis_alt / 1e3,      // [11] periapsis_alt_km
        orbit.apoapsis_alt / 1e3,       // [12] apoapsis_alt_km
        guidance_phase,                 // [13] phase (from enclosing tick)
        bank_angle_deg,                 // [14] bank_angle_deg (from enclosing tick)
        velocity_radial,                // [15] radial_velocity_m_s
        aoa_deg,                        // [16] aoa_deg (from enclosing tick)
        cumulative_bank_change_deg,     // [17] cumulative_bank_change_deg (from enclosing tick)
        energy,                         // [18] energy_j_kg
        pdyn,                           // [19] dynamic_pressure_pa
        velocity_radial, // [20] radial_velocity_2 (duplicate, matches build_photo_values)
        0.0,             // [21] dynamic_pressure_onboard_kpa (no nav estimate at event time)
        0.0,             // [22] sim_index (not applicable for event rows)
        0.0,             // [23] reserved
        heat_flux / 1e3, // [24] heat_flux_kw_m2
        load_factor / G0, // [25] g_load_g
        density_gain,    // [26] nav_density_ratio (from enclosing tick)
        rho_truth,       // [27] truth_density_kg_m3
        cumulative_flux / 1e3, // [28] heat_load_kj_m2
        run_state.density_perturbation, // [29] density_perturbation
    ]
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

    for k in 1..=4 {
        let derivs =
            compute_derivatives(&sim.state, sim.bank_angle, sim.aoa, planet, data, run_state);
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
            &mut |state| compute_derivatives(state, bank_angle, aoa, planet, data, run_state),
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
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );

    // Wind-corrected velocity for aero-dependent quantities
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, run_state);

    // Heat flux (W/m²) — same formula as dflux in compute_derivatives
    let heat_flux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);

    // Dynamic pressure (Pa)
    let pdyn = 0.5 * rho * v_eff * v_eff;

    // Load factor (m/s²) — aerodynamic acceleration magnitude
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

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

/// Compute effective airspeed accounting for wind.
///
/// The state velocity `v` is relative to the planet-fixed atmosphere.
/// Wind adds a velocity perturbation: we subtract wind from the vehicle's
/// ground-relative velocity components to get the airspeed used for aero forces.
/// Returns the original `v` when wind is disabled or no wind table is loaded.
pub(crate) fn effective_airspeed(
    v: f64,
    gamma: f64,
    psi: f64,
    lat: f64,
    altitude: f64,
    data: &SimData,
    run_state: &init::RunState,
) -> f64 {
    if !data.wind_enabled {
        return v;
    }
    if let Some(ref wt) = data.wind_table {
        let w = wt.wind_at(altitude, lat);
        let scale = run_state.wind_scale;
        let rot = run_state.wind_direction_bias;
        // Apply dispersions: scale and rotate wind vector
        let we = scale * (w.east * rot.cos() - w.north * rot.sin());
        let wn = scale * (w.east * rot.sin() + w.north * rot.cos());
        // Project into trajectory frame and compute effective speed
        let cos_g = gamma.cos();
        let v_east = v * cos_g * psi.sin() - we;
        let v_north = v * cos_g * psi.cos() - wn;
        let v_vert = v * gamma.sin();
        (v_east * v_east + v_north * v_north + v_vert * v_vert).sqrt()
    } else {
        v
    }
}

/// Compute state derivatives (equations of motion).
///
/// State = [r, lon, lat, V, gamma, psi, flux, time]
pub(crate) fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64,
    aoa: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 8] {
    let r = state[0];
    let _lon = state[1];
    let lat = state[2];
    let v = state[3];
    let gamma = state[4];
    let psi = state[5];

    let (gravtl, gravtr) = gravity::gravity(r, lat, planet);
    let (altitude, _lat_geo) = geodetic_from_spherical(r, state[1], lat, planet);
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );

    let aoa_dispersed = aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);

    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);

    // Wind-corrected velocity for aero forces and heat flux.
    // Note: aero force *magnitude* uses v_eff (airspeed) but is applied along the
    // planet-relative velocity direction. This is a first-order approximation valid
    // when wind << vehicle speed. At Mars entry (100 m/s wind vs 5700 m/s), the
    // direction error is O(wind/V)² ≈ 0.03%.
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, run_state);

    let aero_factor = rho * ref_area / (2.0 * mass);
    let acdrag = aero_factor * cx * v_eff * v_eff;
    let aclift = aero_factor * cz * v_eff * v_eff;

    let cos_bank = bank_angle.cos();
    let sin_bank = bank_angle.sin();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    let cos_psi = psi.cos();
    let sin_psi = psi.sin();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;

    let omega = planet.omega;

    // Kinematic derivatives use original v (planet-relative)
    let dr = v * sin_gamma;
    let dlon = v * cos_gamma * sin_psi / (r * cos_lat);
    let dlat = v * cos_gamma * cos_psi / r;

    let dv = -acdrag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = (aclift * cos_bank / v) + (v * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v);

    let dpsi = (aclift * sin_bank / (v * cos_gamma))
        + (v * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma));

    // Heat flux uses wind-corrected velocity
    let dflux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
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
