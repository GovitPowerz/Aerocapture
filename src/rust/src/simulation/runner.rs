//! Main simulation loop.
//!
//! Matches Fortran simmsr.f + realit.f + finmsr.f.

use crate::config::{Planet, SimInput};
use crate::data::SimData;
use crate::gnc::control::pilot::{self, PilotState};
use crate::gnc::guidance::ftc::{self, FtcState};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::gnc::navigation::estimator::{self, NavigationBiases, NavigationState};
use crate::integration::rk4;
use crate::integration::sequencer::SequencerState;
use crate::orbit::{elements, maneuver};
use crate::physics::gravity;
use crate::simulation::init;
use crate::simulation::output;
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
    qk: [f64; 8],
    ix: i32,
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

/// Run the full simulation.
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };

    // Open final file (carltf.f format) for Monte Carlo output
    let final_path = format!(
        "../sorties/final.{}",
        config.suffixes.results.trim_start_matches('.')
    );
    let mut final_file = BufWriter::new(
        File::create(&final_path)
            .map_err(|e| SimError(format!("Cannot create {}: {}", final_path, e)))?,
    );

    for sim_idx in 0..n_sims {
        let run_state = init::init_run(data, config, sim_idx, config.random_seed);

        if config.screen_output {
            eprintln!("--- Simulation {} ---", sim_idx + 1);
            eprintln!(
                "  Entry: alt={:.3} km, vel={:.3} m/s, fpa={:.5} deg",
                run_state.entry.state.altitude / 1e3,
                run_state.entry.state.velocity,
                run_state.entry.state.flight_path.to_degrees(),
            );
        }

        run_single(config, data, &run_state, sim_idx, &mut final_file)?;
    }

    final_file
        .flush()
        .map_err(|e| SimError(format!("Final file flush error: {}", e)))?;

    Ok(())
}

/// Run a single simulation.
fn run_single(
    config: &SimInput,
    data: &SimData,
    run_state: &init::RunState,
    sim_idx: i32,
    final_writer: &mut impl Write,
) -> Result<(), SimError> {
    let planet = &config.planet;
    let req = planet.equatorial_radius();
    let degrad = std::f64::consts::PI / 180.0;

    // Initial state: convert entry conditions to state vector
    // State = [r, lon, lat, V, gamma, psi, flux, time]
    let entry = &run_state.entry;
    let r0 = entry.state.altitude + req; // altitude -> radius

    let mut sim = SimState {
        state: [
            r0,
            entry.state.longitude,
            entry.state.latitude,
            entry.state.velocity,
            entry.state.flight_path,
            entry.state.azimuth,
            0.0, // integrated flux
            entry.initial_date,
        ],
        qk: [0.0; 8],
        ix: 0,
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

    // Override bank angle for reference trajectory
    let gitref = config.reference_bank_angle.to_radians();

    if config.reference_trajectory {
        sim.bank_angle = gitref;
    }

    let dt = data.periods.integration;
    let max_time = 5000.0;
    let exit_altitude = data.final_conditions.altitude;

    // === GNC subsystem initialization ===
    let mut nav_state = NavigationState::new();
    let nav_biases = NavigationBiases::default(); // TODO: apply from lottery
    eprintln!(
        "  Init: entry.initial_bank={:.5}deg, gitref={:.5}deg, sim.bank_angle={:.5}deg",
        entry.initial_bank.to_degrees(),
        gitref.to_degrees(),
        sim.bank_angle.to_degrees()
    );
    let mut ftc_state = FtcState::new(entry.initial_bank, entry.initial_aoa);
    let mut pilot_state = PilotState {
        bank_angle: sim.bank_angle,
        bank_rate: 0.0,
    };
    let mut sequencer = SequencerState::new();

    // === Open photo output file ===
    let photo_path = format!(
        "../sorties/photo.{}",
        config.suffixes.results.trim_start_matches('.')
    );
    let mut photo_file = BufWriter::new(
        File::create(&photo_path)
            .map_err(|e| SimError(format!("Cannot create {}: {}", photo_path, e)))?,
    );

    // somgit tracks cumulative bank angle changes (for photo col 18)
    let mut somgit_deg = 0.0_f64;
    let mut pdynan_for_photo = 0.0_f64;
    let mut romver_for_photo = 0.0_f64;

    // Main simulation loop
    let mut temsim = entry.initial_date;
    let mut term = TermReason::None;
    let mut step = 0;
    let mut first_iter = true;

    while term == TermReason::None {
        // === Sequencer time increment (sequen.f) ===
        if !first_iter {
            temsim += dt;
        }
        first_iter = false;

        // === Sequencer: determine which subsystems to call ===
        let flags = sequencer.update(temsim, &data.periods);

        // === Navigation (naviga.f) ===
        if !config.reference_trajectory {
            let positr = [sim.state[0], sim.state[1], sim.state[2]];
            let vitesr = [sim.state[3], sim.state[4], sim.state[5]];

            let nav_out = estimator::navigate(
                &positr,
                &vitesr,
                ftc_state.alfcom,
                temsim,
                &nav_biases,
                &mut nav_state,
                data,
                planet,
                run_state.density_bias,
                run_state.cx_bias,
                run_state.cz_bias,
                run_state.mass_bias,
            );

            pdynan_for_photo = nav_out.pdynan;
            romver_for_photo = nav_out.roguid;

            // === Guidance (guidag.f) ===
            let ftc_out = ftc::guidance_step(
                &nav_out,
                sim.bank_angle,
                temsim,
                gitref,
                &mut ftc_state,
                data,
                planet,
                config.mission_type,
                config.reference_trajectory,
                config.guidance_type,
            );

            // === Pilot (pilote.f) ===
            pilot_state = pilot::apply_pilot(
                &data.pilot,
                ftc_out.gitcom,
                &pilot_state,
                data.periods.pilot,
                data.capsule.max_bank_rate,
            );

            // Track cumulative bank angle changes
            let bank_change = (pilot_state.bank_angle - sim.bank_angle).abs();
            if bank_change > 1e-10 {
                somgit_deg += bank_change / degrad;
            }

            sim.bank_angle = pilot_state.bank_angle;
            sim.aoa = ftc_out.alfcom;

            if step < 5 || step % 50 == 0 {
                let (dbg_alt, _) =
                    geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);
                eprintln!(
                    "  step={} t={:.1} bank={:.3}deg aoa={:.3}deg ilongi={} alt={:.1}km vel={:.1}",
                    step,
                    temsim,
                    sim.bank_angle.to_degrees(),
                    sim.aoa.to_degrees(),
                    ftc_out.ilongi,
                    dbg_alt / 1e3,
                    sim.state[3],
                );
            }
        }

        // === Write photo snapshot (photra.f) ===
        if flags.photo {
            write_photo(
                &mut photo_file,
                &sim,
                temsim,
                planet,
                degrad,
                pdynan_for_photo,
                romver_for_photo,
                sim_idx + 1,
                somgit_deg * degrad,
            )
            .map_err(|e| SimError(format!("Photo write error: {}", e)))?;
        }

        // === Integration step (realit.f) ===
        integrate_step(&mut sim, dt, planet, data, run_state);

        // Compute geodetic altitude
        let (altitude, _lat_geo) =
            geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

        // === Termination checks (finmsr.f) ===
        if altitude <= 0.0 {
            term = TermReason::Crash;
            if config.screen_output {
                eprintln!("  Crash at t={:.3} s", temsim);
            }
        }

        if temsim >= max_time {
            term = TermReason::Timeout;
            if config.screen_output {
                eprintln!("  Timeout at alt={:.3} km", altitude / 1e3);
            }
        }

        if sim.bounced && altitude >= exit_altitude {
            term = TermReason::AtmosphereExit;
            if config.screen_output {
                eprintln!("  Exit at t={:.3} s", temsim);
            }
        }

        // Bounce detection
        if !sim.bounced && sim.state[4].sin() >= 0.0 {
            sim.bounced = true;
            sim.bounce_alt = altitude;
            sim.bounce_time = temsim;
        }

        step += 1;
    }

    // Write final photo snapshot
    write_photo(
        &mut photo_file,
        &sim,
        temsim,
        planet,
        degrad,
        pdynan_for_photo,
        romver_for_photo,
        sim_idx + 1,
        somgit_deg * degrad,
    )
    .map_err(|e| SimError(format!("Photo write error: {}", e)))?;

    photo_file
        .flush()
        .map_err(|e| SimError(format!("Photo flush error: {}", e)))?;

    // === Final conditions (carltf.f) ===
    let (alt_final, lat_final) =
        geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

    eprintln!(
        "  Final: alt={:.3} km, vel={:.3} m/s, t={:.1} s, steps={}, term={:?}",
        alt_final / 1e3,
        sim.state[3],
        temsim,
        step,
        term,
    );

    // Compute final orbital elements
    let orbit = elements::from_spherical(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );

    // Compute energy using absolute velocity (matches Fortran enrtot)
    let mu = planet.mu();
    let (_posita, vitesa) = to_absolute_cartesian(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );
    let vitabs = norm(&vitesa);
    let xenerg = vitabs * vitabs / 2.0 - mu / sim.state[0];
    let vitrad = sim.state[3] * sim.state[4].sin();

    // Compute delta-V (matches Fortran ergols.f)
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

    // Write final line (carltf.f format: 52 values after sim number)
    // xsauve[1..52] maps to columns 1-52 (0-indexed in Python as cols 1-52)
    let mut xsauve = [0.0_f64; 52];
    xsauve[0] = alt_final / 1e3; // xsauve(2): altitude (km)
    xsauve[1] = sim.state[1] / degrad; // xsauve(3): longitude (deg)
    xsauve[2] = lat_final / degrad; // xsauve(4): latitude (deg)
    xsauve[3] = sim.state[3]; // xsauve(5): velocity (m/s)
    xsauve[4] = sim.state[4] / degrad; // xsauve(6): FPA (deg)
    xsauve[5] = sim.state[5] / degrad; // xsauve(7): azimuth (deg)
    xsauve[6] = vitrad; // xsauve(8): radial velocity (m/s)
    xsauve[7] = xenerg / 1e6; // xsauve(9): energy (MJ/kg) — Python col 8
    xsauve[8] = orbit.semi_major_axis / 1e3; // xsauve(10): SMA (km)
    xsauve[9] = orbit.eccentricity; // xsauve(11): eccentricity — Python col 10
    xsauve[10] = orbit.inclination / degrad; // xsauve(12): inclination (deg)
    xsauve[11] = orbit.raan / degrad; // xsauve(13): RAAN (deg)
    xsauve[12] = orbit.arg_periapsis / degrad; // xsauve(14): arg periapsis (deg)
    xsauve[13] = orbit.true_anomaly / degrad; // xsauve(15): true anomaly (deg)
    xsauve[14] = orbit.periapsis_alt / 1e3; // xsauve(16): periapsis alt (km)
    xsauve[15] = orbit.apoapsis_alt / 1e3; // xsauve(17): apoapsis alt (km)
    xsauve[16] = sim.max_heat_flux / 1e3; // xsauve(18): max heat flux (kW/m2)
    xsauve[17] = sim.max_load_factor / g0terr; // xsauve(19): max load factor (g)
    xsauve[18] = sim.max_dyn_pressure / 1e3; // xsauve(20): max dyn pressure (kPa)
    xsauve[19] = sim.alt_max_flux / 1e3; // xsauve(21): alt at max flux (km)
    xsauve[20] = sim.alt_max_load / 1e3; // xsauve(22): alt at max load (km)
    xsauve[21] = sim.alt_max_pdyn / 1e3; // xsauve(23): alt at max pdyn (km)
    xsauve[22] = sim.time_max_flux; // xsauve(24): time at max flux (s)
    xsauve[23] = sim.time_max_load; // xsauve(25): time at max load (s)
    xsauve[24] = sim.time_max_pdyn; // xsauve(26): time at max pdyn (s)
    xsauve[25] = sim.bounce_alt / 1e3; // xsauve(27): bounce alt (km)
    xsauve[26] = sim.bounce_time; // xsauve(28): bounce time (s)
    xsauve[27] = temsim; // xsauve(29): total sim time — Python col 28
    xsauve[28] = sim.state[6] / 1e6; // xsauve(30): integrated flux (MW/m2)
    xsauve[29] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3; // xsauve(31): peri error — Python col 30
    xsauve[30] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3; // xsauve(32): apo error — Python col 31
    xsauve[31] = ifinal as f64; // xsauve(33): termination code
    xsauve[32] = 0.0; // xsauve(34): ecartr(1) placeholder
    xsauve[33] = 0.0; // xsauve(35): ecartr(2) placeholder
    xsauve[34] = 0.0; // xsauve(36): ecartr(3) placeholder
    xsauve[35] = 0.0; // xsauve(37): ecartr(4) placeholder
    xsauve[36] = 0.0; // xsauve(38): finesse placeholder
    xsauve[37] = deltav.dv1; // xsauve(39): deltav(1)
    xsauve[38] = deltav.dv2; // xsauve(40): deltav(2)
    xsauve[39] = deltav.dv3; // xsauve(41): deltav(3)
    xsauve[40] = deltav.dv1.abs() + deltav.dv2.abs(); // xsauve(42): |dv1|+|dv2|
    xsauve[41] = deltav.total; // xsauve(43): total DV — Python col 42
    xsauve[42] = 0.0; // xsauve(44): securization % placeholder
    xsauve[43] = 0.0; // xsauve(45): inhibition % placeholder
    xsauve[44] = 0.0; // xsauve(46): securization/active % placeholder
    xsauve[45] = somgit_deg; // xsauve(47): cumulative bank (deg)
    xsauve[46] = 0.0; // xsauve(48): v_infinity
    xsauve[47] = 0.0; // xsauve(49): true anomaly infinity (deg)
    xsauve[48] = ftc_state.nbroll as f64; // xsauve(50): roll reversals
    xsauve[49] = 0.0; // xsauve(51): v_inf error
    xsauve[50] = 0.0; // xsauve(52): anomaly error
    xsauve[51] = 0.0; // xsauve(53): isucces

    output::write_final_line(final_writer, sim_idx + 1, &xsauve)
        .map_err(|e| SimError(format!("Final write error: {}", e)))?;

    Ok(())
}

/// Write a photo snapshot line matching Fortran photra.f format.
///
/// 24 columns: format(24(1x,d12.5))
#[allow(clippy::too_many_arguments)]
fn write_photo(
    writer: &mut impl Write,
    sim: &SimState,
    temsim: f64,
    planet: &Planet,
    degrad: f64,
    pdynan: f64,
    romver: f64,
    isimul: i32,
    somgit: f64,
) -> std::io::Result<()> {
    let _req = planet.equatorial_radius();

    // Geodetic altitude
    let (altitr, xlatit) =
        geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

    // Orbital elements
    let orbit = elements::from_spherical(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );

    // Energy using absolute (inertial) velocity (matches Fortran enrtot.f → xvabsl.f)
    let mu = planet.mu();
    let (_posita, vitesa) = to_absolute_cartesian(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );
    let vitabs = norm(&vitesa);
    let enerjr = vitabs * vitabs / 2.0 - mu / sim.state[0];

    // Vertical velocity
    let vitrad = sim.state[3] * sim.state[4].sin();

    // Phase detection (matches photra.f)
    // Note: Fortran uses uninitialized xrayon (=0.0) for post-bounce check,
    // so it effectively compares positr(1) > 80km, which is always true.
    // Pre-bounce uses geodetic altitude (from frayon).
    let iphase = if !sim.bounced {
        if altitr > 80e3 { 1.0 } else { 2.0 }
    } else {
        if sim.state[0] > 80e3 { 3.0 } else { 2.0 }
    };

    let values: [f64; 24] = [
        temsim,                                           // 1: time (s)
        altitr / 1e3,                                     // 2: altitude (km)
        sim.state[1] / degrad,                            // 3: longitude (deg)
        xlatit / degrad,                                  // 4: latitude (deg)
        sim.state[3],                                     // 5: velocity (m/s)
        sim.state[4] / degrad,                            // 6: flight path angle (deg)
        sim.state[5] / degrad,                            // 7: azimuth (deg)
        orbit.semi_major_axis / 1e3,                      // 8: semi-major axis (km)
        orbit.eccentricity,                               // 9: eccentricity
        orbit.inclination / degrad,                       // 10: inclination (deg)
        orbit.raan / degrad,                              // 11: RAAN (deg)
        orbit.periapsis_alt / 1e3,                        // 12: periapsis alt (km)
        orbit.apoapsis_alt / 1e3,                         // 13: apoapsis alt (km)
        iphase,                                           // 14: flight phase
        sim.bank_angle / degrad,                          // 15: bank angle (deg)
        vitrad,                                           // 16: vertical velocity (m/s)
        sim.aoa / degrad,                                 // 17: angle of attack (deg)
        somgit / degrad,                                  // 18: cumulative bank rate (deg)
        enerjr,                                           // 19: specific orbital energy (J/kg)
        pdynan,                                           // 20: dynamic pressure (Pa)
        vitrad,                                           // 21: radial velocity (m/s)
        0.5 * romver * sim.state[3] * sim.state[3] / 1e3, // 22: dyn press (kPa)
        isimul as f64,                                    // 23: simulation number
        0.0,                                              // 24: reserved
    ];

    output::write_photo_line(writer, &values)
}

/// Perform one integration step using Gill's RK4.
///
/// Matches Fortran realit.f RK4 loop.
fn integrate_step(
    sim: &mut SimState,
    dt: f64,
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) {
    sim.ix = 0;

    for k in 1..=4 {
        // Compute derivatives at current state
        let derivs =
            compute_derivatives(&sim.state, sim.bank_angle, sim.aoa, planet, data, run_state);

        // RK4 increment (Gill's variant)
        rk4::rk4_increment(dt, &derivs, k, 8, &mut sim.ix, &mut sim.qk, &mut sim.state);
    }
}

/// Compute state derivatives (equations of motion).
///
/// Matches Fortran realit.f lines 318-353.
///
/// State = [r, lon, lat, V, gamma, psi, flux, time]
fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64, // gitpil
    aoa: f64,        // alfpil
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 8] {
    let r = state[0]; // radius
    let _lon = state[1]; // longitude
    let lat = state[2]; // latitude
    let v = state[3]; // velocity
    let gamma = state[4]; // flight path angle
    let psi = state[5]; // azimuth
    let _flux = state[6];
    let _time = state[7];

    // Gravity (Fortran fgravi.f)
    let (gravtl, gravtr) = gravity::gravity(r, lat, planet);

    // Geodetic altitude for atmosphere lookup
    let (altitude, _lat_geo) = geodetic_from_spherical(r, state[1], lat, planet);

    // Atmospheric density
    let rho = data.atmosphere.density_at(altitude);
    let rho = rho * (1.0 + run_state.density_bias);

    // Aerodynamic coefficients
    let cx = data.aero.interpolate_cx(aoa) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa) * (1.0 + run_state.cz_bias);

    // Aerodynamic specific accelerations
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let coefar = rho * data.capsule.reference_area / (2.0 * mass);
    let acdrag = coefar * cx * v * v;
    let aclift = coefar * cz * v * v;

    // Trig
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

    // Position derivatives
    let dr = v * sin_gamma;
    let dlon = v * cos_gamma * sin_psi / (r * cos_lat);
    let dlat = v * cos_gamma * cos_psi / r;

    // dV/dt
    let dv = -acdrag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    // dgamma/dt
    let dgamma = (aclift * cos_mu / v) + (v * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v);

    // dpsi/dt
    let dpsi = (aclift * sin_mu / (v * cos_gamma))
        + (v * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma));

    // Heat flux integral
    let dflux = data.capsule.cq * rho.sqrt() * v.powf(3.05);

    // Time
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
}
