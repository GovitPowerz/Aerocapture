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
use crate::simulation::init;
use crate::simulation::output;
use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};

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
    bank_angle: f64, // gitpil (rad) — realized bank angle
    aoa: f64,        // alfpil (rad) — realized AoA
    // Tracking
    bounced: bool,
    bounce_alt: f64,
    bounce_time: f64,
    max_heat_flux: f64,
    max_load_factor: f64,
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
}

/// Run the full simulation.
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let is_mc = n_sims > 1;

    // Pre-generate dispersion draws if using domain-based config
    let draws = data.dispersion_config.as_ref().map(|dc| {
        let draws = dc.generate_draws(n_sims as usize);
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
        draws
    });

    // Build run states for all simulations
    let run_states: Vec<init::RunState> = (0..n_sims)
        .map(|sim_idx| {
            if let Some(ref d) = draws {
                init::init_run_from_draw(data, &d[sim_idx as usize])
            } else {
                // No [monte_carlo] config: zero dispersions (nominal trajectory)
                init::init_run_from_draw(data, &crate::data::dispersions::DispersionDraw::default())
            }
        })
        .collect();

    // Determine which sim gets photo output
    // Single sim: always write photo. MC: write for visualize_sim (default: last sim)
    let photo_sim_idx = if is_mc {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

    // Run simulations
    let results: Vec<SimResult> = if is_mc {
        let start = std::time::Instant::now();
        eprintln!("Running {} simulations in parallel...", n_sims);

        let results: Vec<SimResult> = run_states
            .par_iter()
            .enumerate()
            .map(|(idx, run_state)| {
                let write_photo = idx as i32 == photo_sim_idx;
                run_single(config, data, run_state, idx as i32, write_photo)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let elapsed = start.elapsed();
        eprintln!(
            "Completed {} simulations in {:.3}s ({:.1} sims/s)",
            n_sims,
            elapsed.as_secs_f64(),
            n_sims as f64 / elapsed.as_secs_f64(),
        );
        results
    } else {
        // Single sim: run sequentially (no rayon overhead)
        let run_state = &run_states[0];
        if config.screen_output {
            eprintln!(
                "  Entry: alt={:.3} km, vel={:.3} m/s, fpa={:.5} deg",
                run_state.entry.state.altitude / 1e3,
                run_state.entry.state.velocity,
                run_state.entry.state.flight_path.to_degrees(),
            );
        }
        vec![run_single(config, data, run_state, 0, true)?]
    };

    // Write output files
    write_csv_output(config, &results, photo_sim_idx)?;

    Ok(())
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
    let degrad = std::f64::consts::PI / 180.0;

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
    let max_time = 5000.0;
    let exit_altitude = data.final_conditions.altitude;

    // === GNC subsystem initialization ===
    let mut nav_state = NavigationState::new();
    let nav_biases = run_state.nav_biases;
    let is_single = config.n_sims <= 1;
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
    let mut somgit_deg = 0.0_f64;
    let mut pdynan_for_photo = 0.0_f64;
    let mut romver_for_photo = 0.0_f64;

    // Main simulation loop
    let mut temsim = entry.initial_date;
    let mut term = TermReason::None;
    let mut step = 0;
    let mut first_iter = true;

    while term == TermReason::None {
        if !first_iter {
            temsim += dt;
        }
        first_iter = false;

        let flags = sequencer.update(temsim, &data.periods);

        // === Navigation + Guidance + Pilot ===
        if !config.reference_trajectory {
            let positr = [sim.state[0], sim.state[1], sim.state[2]];
            let vitesr = [sim.state[3], sim.state[4], sim.state[5]];

            let nav_out = estimator::navigate(
                &positr,
                &vitesr,
                ftc_state.aoa_commanded,
                temsim,
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

            pdynan_for_photo = nav_out.pdynan;
            romver_for_photo = nav_out.roguid;

            let ftc_out = ftc::guidance_step(
                &nav_out,
                sim.bank_angle,
                temsim,
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
                somgit_deg += bank_change / degrad;
            }

            sim.bank_angle = pilot_state.bank_angle;
            sim.aoa = ftc_out.aoa_commanded;

            if is_single && (step < 5 || step % 50 == 0) {
                let (dbg_alt, _) =
                    geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);
                eprintln!(
                    "  step={} t={:.1} bank={:.3}deg aoa={:.3}deg ilongi={} alt={:.1}km vel={:.1}",
                    step,
                    temsim,
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
                temsim,
                planet,
                degrad,
                pdynan_for_photo,
                romver_for_photo,
                sim_idx + 1,
                somgit_deg * degrad,
            ));
        }

        // === Integration step ===
        integrate_step(&mut sim, dt, planet, data, run_state);

        let (altitude, _lat_geo) =
            geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

        // === Termination checks ===
        if altitude <= 0.0 {
            term = TermReason::Crash;
        }
        if temsim >= max_time {
            term = TermReason::Timeout;
        }
        if sim.bounced && altitude >= exit_altitude {
            term = TermReason::AtmosphereExit;
        }

        // Bounce detection
        if !sim.bounced && sim.state[4].sin() >= 0.0 {
            sim.bounced = true;
            sim.bounce_alt = altitude;
            sim.bounce_time = temsim;
        }

        step += 1;
    }

    // Final photo snapshot
    if write_photo {
        photo_lines.push(build_photo_values(
            &sim,
            temsim,
            planet,
            degrad,
            pdynan_for_photo,
            romver_for_photo,
            sim_idx + 1,
            somgit_deg * degrad,
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
            temsim,
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

    let g0terr = 9.81_f64;
    let mut xsauve = [0.0_f64; 52];
    xsauve[0] = alt_final / 1e3;
    xsauve[1] = sim.state[1] / degrad;
    xsauve[2] = lat_final / degrad;
    xsauve[3] = sim.state[3];
    xsauve[4] = sim.state[4] / degrad;
    xsauve[5] = sim.state[5] / degrad;
    xsauve[6] = velocity_radial;
    xsauve[7] = energy / 1e6;
    xsauve[8] = orbit.semi_major_axis / 1e3;
    xsauve[9] = orbit.eccentricity;
    xsauve[10] = orbit.inclination / degrad;
    xsauve[11] = orbit.raan / degrad;
    xsauve[12] = orbit.arg_periapsis / degrad;
    xsauve[13] = orbit.true_anomaly / degrad;
    xsauve[14] = orbit.periapsis_alt / 1e3;
    xsauve[15] = orbit.apoapsis_alt / 1e3;
    xsauve[16] = sim.max_heat_flux / 1e3;
    xsauve[17] = sim.max_load_factor / g0terr;
    xsauve[18] = sim.max_dyn_pressure / 1e3;
    xsauve[19] = sim.alt_max_flux / 1e3;
    xsauve[20] = sim.alt_max_load / 1e3;
    xsauve[21] = sim.alt_max_pdyn / 1e3;
    xsauve[22] = sim.time_max_flux;
    xsauve[23] = sim.time_max_load;
    xsauve[24] = sim.time_max_pdyn;
    xsauve[25] = sim.bounce_alt / 1e3;
    xsauve[26] = sim.bounce_time;
    xsauve[27] = temsim;
    xsauve[28] = sim.state[6] / 1e6;
    xsauve[29] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3;
    xsauve[30] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3;
    xsauve[31] = ifinal as f64;
    xsauve[37] = deltav.dv1;
    xsauve[38] = deltav.dv2;
    xsauve[39] = deltav.dv3;
    xsauve[40] = deltav.dv1.abs() + deltav.dv2.abs();
    xsauve[41] = deltav.total;
    xsauve[45] = somgit_deg;
    xsauve[48] = ftc_state.n_reversals as f64;

    Ok(SimResult {
        sim_idx,
        final_line: xsauve,
        photo_lines,
    })
}

/// Build a photo snapshot line (24-column D12.5 format).
#[allow(clippy::too_many_arguments)]
fn build_photo_values(
    sim: &SimState,
    temsim: f64,
    planet: &Planet,
    degrad: f64,
    pdynan: f64,
    romver: f64,
    isimul: i32,
    somgit: f64,
) -> [f64; 24] {
    let (altitr, xlatit) =
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

    let iphase = if !sim.bounced {
        if altitr > 80e3 { 1.0 } else { 2.0 }
    } else {
        if sim.state[0] > 80e3 { 3.0 } else { 2.0 }
    };

    [
        temsim,
        altitr / 1e3,
        sim.state[1] / degrad,
        xlatit / degrad,
        sim.state[3],
        sim.state[4] / degrad,
        sim.state[5] / degrad,
        orbit.semi_major_axis / 1e3,
        orbit.eccentricity,
        orbit.inclination / degrad,
        orbit.raan / degrad,
        orbit.periapsis_alt / 1e3,
        orbit.apoapsis_alt / 1e3,
        iphase,
        sim.bank_angle / degrad,
        velocity_radial,
        sim.aoa / degrad,
        somgit / degrad,
        energy,
        pdynan,
        velocity_radial,
        0.5 * romver * sim.state[3] * sim.state[3] / 1e3,
        isimul as f64,
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
        rk4::rk4_increment(dt, &derivs, k, 8, &mut sim.gill_toggle, &mut sim.accumulator, &mut sim.state);
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
    let coefar = rho * ref_area / (2.0 * mass);
    let acdrag = coefar * cx * v * v;
    let aclift = coefar * cz * v * v;

    let cos_mu = bank_angle.cos();
    let sin_mu = bank_angle.sin();
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

    let dgamma = (aclift * cos_mu / v) + (v * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v);

    let dpsi = (aclift * sin_mu / (v * cos_gamma))
        + (v * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma));

    let dflux = data.capsule.cq * rho.sqrt() * v.powf(3.05);
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
}
