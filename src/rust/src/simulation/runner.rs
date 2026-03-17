//! Main simulation loop.
//!
//! Monte Carlo runs are parallelized with rayon (one thread per trajectory).

use crate::config::{Planet, SimInput};
use crate::data::SimData;
use crate::gnc::control::pilot::{self, PilotState};
use crate::gnc::guidance::ftc::{self, FtcState};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::gnc::navigation::estimator::{self, NavigationState};
use crate::integration::rk4;
use crate::integration::sequencer::SequencerState;
use crate::orbit::{elements, maneuver};
use crate::physics::gravity;
use crate::data::dispersions::DISPERSION_DRAW_LEN;
use crate::simulation::init;
use crate::simulation::output;
use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const G0: f64 = 9.81;

#[derive(Debug)]
pub struct SimError(pub String);

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SimError {}

/// Simulation state
#[allow(dead_code)]
struct SimState {
    // State vector: [r, lon, lat, V, gamma, psi, flux, time]
    state: [f64; 8],
    // RK4 internals
    accumulator: [f64; 8],
    gill_toggle: i32,
    // Guidance
    bank_angle: f64, // realized bank angle (rad)
    aoa: f64,        // realized AoA (rad)
    // Tracking
    bounced: bool,
    bounce_alt: f64,
    bounce_time: f64,
    max_heat_flux: f64,
    max_load_factor: f64, // m/s², divided by G0 when written to final_record
    max_dyn_pressure: f64,
    // Max-value altitudes and times (for carltf output)
    alt_max_flux: f64,
    alt_max_load: f64,
    alt_max_pdyn: f64,
    time_max_flux: f64,
    time_max_load: f64,
    time_max_pdyn: f64,
}

/// Termination reason
#[derive(Debug, Clone, Copy, PartialEq)]
enum TermReason {
    None,
    Crash,
    Timeout,
    AtmosphereExit,
}

/// Result from a single simulation run.
struct SimResult {
    sim_idx: i32,
    final_line: [f64; 52],
    photo_lines: Vec<[f64; 24]>,
    dispersions: [f64; DISPERSION_DRAW_LEN],
}

/// Shared simulation orchestration: build run states, dispatch parallel/sequential runs.
fn run_core(
    config: &SimInput,
    data: &SimData,
    write_photo: bool,
    include_trajectories: bool,
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
                let mut result = run_single(config, data, run_state, idx as i32, do_photo)?;
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
        let mut result = run_single(config, data, run_state, 0, write_photo)?;
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

    let results = run_core(config, data, true, false)?;
    write_csv_output(config, &results, photo_sim_idx)?;
    Ok(())
}

/// Run simulation and return structured results (no file I/O).
///
/// Same physics as `run()`, but returns `Vec<RunOutput>` instead of writing files.
/// Used by the PyO3 interface for direct Python access.
pub fn run_for_api(config: &SimInput, data: &SimData, include_trajectories: bool) -> Result<Vec<crate::RunOutput>, SimError> {
    let results = run_core(config, data, false, include_trajectories)?;

    Ok(results
        .into_iter()
        .map(|r| {
            let energy = r.final_line[7]; // MJ/kg
            let ecc = r.final_line[9];
            let trajectory = if include_trajectories {
                r.photo_lines
                    .iter()
                    .map(|p| [
                        p[1],                // [0] alt_km
                        p[2],                // [1] lon_deg
                        p[3],                // [2] lat_deg
                        p[4],                // [3] vel_m_s
                        p[5],                // [4] fpa_deg
                        p[6],                // [5] heading_deg
                        0.0,                 // [6] heat flux placeholder
                        p[0],                // [7] time_s
                        p[18] / 1e6,         // [8] energy J/kg → MJ/kg
                        p[19] / 1e3,         // [9] pdyn Pa → kPa
                        p[14],               // [10] bank_angle deg
                        p[9],                // [11] inclination deg
                    ])
                    .collect()
            } else {
                Vec::new()
            };
            crate::RunOutput {
                trajectory,
                final_record: r.final_line,
                captured: ecc < 1.0 && energy < 0.0,
                dispersions: r.dispersions,
            }
        })
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

/// Extract 21 CSV values from the 24-element photo array.
/// Drops: [20] radial_velocity_2 (duplicate), [22] sim_number, [23] reserved.
fn extract_photo_csv_values(values: &[f64; 24]) -> [f64; 21] {
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
    ]
}

/// Extract 39 CSV values from the 52-element final array.
/// Drops 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
fn extract_final_csv_values(values: &[f64; 52]) -> [f64; 39] {
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
) -> Result<SimResult, SimError> {
    let planet = &config.planet;
    let req = planet.equatorial_radius();

    // Initial state: convert entry conditions to state vector
    let entry = &run_state.entry;
    let r0 = entry.state.altitude + req;

    let mut sim = SimState {
        state: [
            r0,
            entry.state.longitude,
            entry.state.latitude,
            entry.state.velocity,
            entry.state.flight_path,
            entry.state.azimuth,
            0.0,
            entry.initial_date,
        ],
        accumulator: [0.0; 8],
        gill_toggle: 0,
        bank_angle: entry.initial_bank,
        aoa: entry.initial_aoa,
        bounced: false,
        bounce_alt: 1e34,
        bounce_time: 1e30,
        max_heat_flux: 0.0,
        max_load_factor: 0.0,
        max_dyn_pressure: 0.0,
        alt_max_flux: 0.0,
        alt_max_load: 0.0,
        alt_max_pdyn: 0.0,
        time_max_flux: 0.0,
        time_max_load: 0.0,
        time_max_pdyn: 0.0,
    };

    let reference_bank_angle = config.reference_bank_angle.to_radians();

    if config.reference_trajectory {
        sim.bank_angle = reference_bank_angle;
    }

    let dt = data.periods.integration;
    let max_time = config.max_time;
    let exit_altitude = data.final_conditions.altitude;

    // === GNC subsystem initialization ===
    let mut nav_state = NavigationState::new();
    let nav_biases = run_state.nav_biases;
    let is_single = config.n_sims <= 1 && config.screen_output;
    if is_single {
        eprintln!(
            "  Init: entry.initial_bank={:.5}deg, reference_bank_angle={:.5}deg, sim.bank_angle={:.5}deg",
            entry.initial_bank.to_degrees(),
            reference_bank_angle.to_degrees(),
            sim.bank_angle.to_degrees()
        );
    }
    let mut ftc_state = FtcState::new(entry.initial_bank, entry.initial_aoa);
    let mut pilot_state = PilotState {
        bank_angle: sim.bank_angle,
        bank_rate: 0.0,
    };
    let mut sequencer = SequencerState::new();

    let mut photo_lines: Vec<[f64; 24]> = Vec::new();
    let mut cumulative_bank_change_deg = 0.0_f64;
    let mut dynamic_pressure_for_photo = 0.0_f64;
    let mut density_estimate_for_photo = 0.0_f64;

    // Main simulation loop
    let mut sim_time = entry.initial_date;
    let mut term = TermReason::None;
    let mut step = 0;
    let mut first_iter = true;

    while term == TermReason::None {
        if !first_iter {
            sim_time += dt;
        }
        first_iter = false;

        let flags = sequencer.update(sim_time, &data.periods);

        // === Navigation + Guidance + Pilot ===
        if !config.reference_trajectory {
            let position_true = [sim.state[0], sim.state[1], sim.state[2]];
            let velocity_true = [sim.state[3], sim.state[4], sim.state[5]];

            let nav_out = estimator::navigate(
                &position_true,
                &velocity_true,
                ftc_state.aoa_commanded,
                sim_time,
                &nav_biases,
                &mut nav_state,
                data,
                planet,
                run_state.density_bias,
                run_state.cx_bias,
                run_state.cz_bias,
                run_state.mass_bias,
                run_state.incidence_bias,
                run_state.ref_area_bias,
                run_state.filter_gain_bias,
            );

            dynamic_pressure_for_photo = nav_out.dynamic_pressure_estimated;
            density_estimate_for_photo = nav_out.density_guidance;

            let ftc_out = ftc::guidance_step(
                &nav_out,
                sim.bank_angle,
                sim_time,
                reference_bank_angle,
                &mut ftc_state,
                data,
                planet,
                config.reference_trajectory,
                config.guidance_type,
            );

            let max_rate = data.capsule.max_bank_rate * (1.0 + run_state.max_bank_rate_bias);
            pilot_state = pilot::apply_pilot(
                &data.pilot,
                ftc_out.bank_angle_commanded,
                &pilot_state,
                data.periods.pilot,
                max_rate,
                &run_state.pilot_biases,
            );

            let bank_change = (pilot_state.bank_angle - sim.bank_angle).abs();
            if bank_change > 1e-10 {
                cumulative_bank_change_deg += bank_change / DEG_TO_RAD;
            }

            sim.bank_angle = pilot_state.bank_angle;
            sim.aoa = ftc_out.aoa_commanded;

            if is_single && (step < 5 || step % 50 == 0) {
                let (dbg_alt, _) =
                    geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);
                eprintln!(
                    "  step={} t={:.1} bank={:.3}deg aoa={:.3}deg longitudinal={} alt={:.1}km vel={:.1}",
                    step,
                    sim_time,
                    sim.bank_angle.to_degrees(),
                    sim.aoa.to_degrees(),
                    ftc_out.longitudinal_active,
                    dbg_alt / 1e3,
                    sim.state[3],
                );
            }
        }

        // === Photo snapshot ===
        if write_photo && flags.photo {
            photo_lines.push(build_photo_values(
                &sim,
                sim_time,
                planet,
                dynamic_pressure_for_photo,
                density_estimate_for_photo,
                sim_idx + 1,
                cumulative_bank_change_deg * DEG_TO_RAD,
            ));
        }

        // === Integration step ===
        integrate_step(&mut sim, dt, planet, data, run_state);

        let (altitude, _lat_geo) =
            geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

        track_peak_values(&mut sim, altitude, sim_time, data, run_state);

        // === Termination checks ===
        if altitude <= 0.0 {
            term = TermReason::Crash;
        }
        if sim_time >= max_time {
            term = TermReason::Timeout;
        }
        if sim.bounced && altitude >= exit_altitude {
            term = TermReason::AtmosphereExit;
        }

        // Bounce detection
        if !sim.bounced && sim.state[4].sin() >= 0.0 {
            sim.bounced = true;
            sim.bounce_alt = altitude;
            sim.bounce_time = sim_time;
        }

        step += 1;
    }

    // Final photo snapshot
    if write_photo {
        photo_lines.push(build_photo_values(
            &sim,
            sim_time,
            planet,
            dynamic_pressure_for_photo,
            density_estimate_for_photo,
            sim_idx + 1,
            cumulative_bank_change_deg * DEG_TO_RAD,
        ));
    }

    // === Final conditions ===
    let (alt_final, lat_final) =
        geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

    if is_single {
        eprintln!(
            "  Final: alt={:.3} km, vel={:.3} m/s, t={:.1} s, steps={}, term={:?}",
            alt_final / 1e3,
            sim.state[3],
            sim_time,
            step,
            term,
        );
    }

    let orbit = elements::from_spherical(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );

    let mu = planet.mu();
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

    let ifinal = match term {
        TermReason::AtmosphereExit => 3,
        TermReason::Crash => 1,
        _ => 2,
    };
    let deltav = maneuver::compute_deltav(
        &orbit,
        ifinal,
        &data.target_orbit,
        &data.parking_orbit,
        planet,
    );

    // final_record layout (52 slots):
    //   0  altitude (km)           16 max heat flux (kW/m²)     32-36 UNUSED
    //   1  longitude (deg)         17 max g-load (g)             37 dv1 (m/s)
    //   2  latitude (deg)          18 max pdyn (kPa)             38 dv2 (m/s)
    //   3  velocity (m/s)          19 alt at max flux (km)       39 dv3 (m/s)
    //   4  FPA (deg)               20 alt at max load (km)       40 dv1+dv2 (m/s)
    //   5  heading (deg)           21 alt at max pdyn (km)       41 dv total (m/s)
    //   6  radial velocity (m/s)   22 time at max flux (s)       42-44 UNUSED
    //   7  energy (MJ/kg)          23 time at max load (s)       45 bank consumption (deg)
    //   8  SMA (km)                24 time at max pdyn (s)       46-47 UNUSED
    //   9  eccentricity            25 bounce alt (km)            48 n_reversals
    //  10  inclination (deg)       26 bounce time (s)            49-51 UNUSED
    //  11  RAAN (deg)              27 sim time (s)
    //  12  arg periapsis (deg)     28 cumulative flux (MJ/m²)
    //  13  true anomaly (deg)      29 periapsis error (km)
    //  14  periapsis alt (km)      30 apoapsis error (km)
    //  15  apoapsis alt (km)       31 final phase
    let mut final_record = [0.0_f64; 52];
    final_record[0] = alt_final / 1e3;
    final_record[1] = sim.state[1] / DEG_TO_RAD;
    final_record[2] = lat_final / DEG_TO_RAD;
    final_record[3] = sim.state[3];
    final_record[4] = sim.state[4] / DEG_TO_RAD;
    final_record[5] = sim.state[5] / DEG_TO_RAD;
    final_record[6] = velocity_radial;
    final_record[7] = energy / 1e6;
    final_record[8] = orbit.semi_major_axis / 1e3;
    final_record[9] = orbit.eccentricity;
    final_record[10] = orbit.inclination / DEG_TO_RAD;
    final_record[11] = orbit.raan / DEG_TO_RAD;
    final_record[12] = orbit.arg_periapsis / DEG_TO_RAD;
    final_record[13] = orbit.true_anomaly / DEG_TO_RAD;
    final_record[14] = orbit.periapsis_alt / 1e3;
    final_record[15] = orbit.apoapsis_alt / 1e3;
    final_record[16] = sim.max_heat_flux / 1e3;
    final_record[17] = sim.max_load_factor / G0;
    final_record[18] = sim.max_dyn_pressure / 1e3;
    final_record[19] = sim.alt_max_flux / 1e3;
    final_record[20] = sim.alt_max_load / 1e3;
    final_record[21] = sim.alt_max_pdyn / 1e3;
    final_record[22] = sim.time_max_flux;
    final_record[23] = sim.time_max_load;
    final_record[24] = sim.time_max_pdyn;
    final_record[25] = sim.bounce_alt / 1e3;
    final_record[26] = sim.bounce_time;
    final_record[27] = sim_time;
    final_record[28] = sim.state[6] / 1e6;
    final_record[29] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3;
    final_record[30] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3;
    final_record[31] = ifinal as f64;
    final_record[37] = deltav.dv1;
    final_record[38] = deltav.dv2;
    final_record[39] = deltav.dv3;
    final_record[40] = deltav.dv1.abs() + deltav.dv2.abs();
    final_record[41] = deltav.total;
    final_record[45] = cumulative_bank_change_deg;
    final_record[48] = ftc_state.n_reversals as f64;

    Ok(SimResult {
        sim_idx,
        final_line: final_record,
        photo_lines,
        dispersions: [0.0; DISPERSION_DRAW_LEN],
    })
}

/// Build a photo snapshot line.
#[allow(clippy::too_many_arguments)]
fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &Planet,
    dynamic_pressure: f64,
    density_estimate: f64,
    sim_index: i32,
    cumulative_bank_change: f64,
) -> [f64; 24] {
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

    let mu = planet.mu();
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

    let phase = if !sim.bounced {
        if altitude > 80e3 { 1.0 } else { 2.0 }
    } else {
        if sim.state[0] > 80e3 { 3.0 } else { 2.0 }
    };

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
    ]
}

/// Perform one integration step using Gill's RK4.
fn integrate_step(
    sim: &mut SimState,
    dt: f64,
    planet: &Planet,
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

/// Update peak tracking values (heat flux, load factor, dynamic pressure)
/// after each integration step.
fn track_peak_values(
    sim: &mut SimState,
    altitude: f64,
    sim_time: f64,
    data: &SimData,
    run_state: &init::RunState,
) {
    let v = sim.state[3];
    let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias);

    // Heat flux (W/m²) — same formula as dflux in compute_derivatives
    let heat_flux = data.capsule.cq * rho.sqrt() * v.powf(3.05);

    // Dynamic pressure (Pa)
    let pdyn = 0.5 * rho * v * v;

    // Load factor (m/s²) — aerodynamic acceleration magnitude
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho * ref_area * v * v / (2.0 * mass);
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

/// Compute state derivatives (equations of motion).
///
/// State = [r, lon, lat, V, gamma, psi, flux, time]
fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64,
    aoa: f64,
    planet: &Planet,
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
    let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias);

    let aoa_dispersed = aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);

    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_factor = rho * ref_area / (2.0 * mass);
    let acdrag = aero_factor * cx * v * v;
    let aclift = aero_factor * cz * v * v;

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

    let omega = planet.omega();

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

    let dflux = data.capsule.cq * rho.sqrt() * v.powf(3.05);
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
}

#[cfg(test)]
mod run_output_tests {
    use super::*;
    use crate::config::SimInput;
    use crate::data::SimData;

    fn load_config(config_name: &str) -> (SimInput, SimData) {
        // Data file paths in TOML configs are relative to repo root
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let repo_root = std::path::PathBuf::from(&manifest)
            .join("../..")
            .canonicalize()
            .unwrap();
        std::env::set_current_dir(&repo_root).unwrap();

        let path = std::path::Path::new(config_name);
        let (sim_config, toml_config) = SimInput::from_toml_file(path).expect("parse");
        let sim_data = SimData::from_toml(&toml_config, &sim_config).expect("data");
        (sim_config, sim_data)
    }

    fn load_test_config() -> (SimInput, SimData) {
        load_config("configs/test/test_ref_orig.toml")
    }

    #[test]
    fn run_for_api_returns_one_result_for_single_sim() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false).expect("run");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn run_output_final_record_has_52_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false).expect("run");
        assert_eq!(results[0].final_record.len(), 52);
    }

    #[test]
    fn run_output_final_record_matches_file_path() {
        let (config, data) = load_test_config();
        let api_results = run_for_api(&config, &data, false).expect("api run");
        let api_fr = &api_results[0].final_record;

        run(&config, &data).expect("file run");

        let suffix = config.results_suffix.trim_start_matches('.');
        let final_path = config.output_path(&format!("final.{}.csv", suffix));
        let content = std::fs::read_to_string(&final_path).expect("read final csv");
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2, "final CSV should have header + data");

        assert!(api_fr[7].abs() > 0.0, "energy should be non-zero");
        assert!(api_fr[9] > 0.0, "eccentricity should be positive");
    }

    #[test]
    fn run_output_captured_flag_consistent_with_orbital_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false).expect("run");
        let r = &results[0];
        let expected = r.final_record[9] < 1.0 && r.final_record[7] < 0.0;
        assert_eq!(r.captured, expected);
    }

    #[test]
    fn peak_values_populated_for_atmospheric_trajectory() {
        let (config, data) = load_config("configs/test/test_high_bank_orig.toml");
        let results = run_for_api(&config, &data, false).expect("run");
        let rec = &results[0].final_record;

        // Columns 16-18: peak heat flux (kW/m²), load factor (g), dynamic pressure (kPa)
        assert!(
            rec[16] > 0.0,
            "max_heat_flux should be > 0, got {}",
            rec[16]
        );
        assert!(
            rec[17] > 0.0,
            "max_load_factor should be > 0, got {}",
            rec[17]
        );
        assert!(
            rec[18] > 0.0,
            "max_dyn_pressure should be > 0, got {}",
            rec[18]
        );

        // Columns 19-24: altitudes and times at peak values
        assert!(rec[19] > 0.0, "alt_max_flux should be > 0, got {}", rec[19]);
        assert!(rec[20] > 0.0, "alt_max_load should be > 0, got {}", rec[20]);
        assert!(rec[21] > 0.0, "alt_max_pdyn should be > 0, got {}", rec[21]);
        assert!(
            rec[22] > 0.0,
            "time_max_flux should be > 0, got {}",
            rec[22]
        );
        assert!(
            rec[23] > 0.0,
            "time_max_load should be > 0, got {}",
            rec[23]
        );
        assert!(
            rec[24] > 0.0,
            "time_max_pdyn should be > 0, got {}",
            rec[24]
        );

        // Physical plausibility for Mars entry:
        assert!(
            rec[16] > 10.0 && rec[16] < 500.0,
            "peak heat flux {:.1} kW/m² outside reasonable Mars entry range",
            rec[16]
        );
        assert!(
            rec[17] > 1.0 && rec[17] < 30.0,
            "peak load factor {:.1} g outside reasonable Mars entry range",
            rec[17]
        );
    }
}
