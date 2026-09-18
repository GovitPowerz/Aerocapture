//! The 30-column photo record: one trajectory snapshot row per photo tick plus one
//! row per sub-tick event, both assembled here from a state vector and the tick's
//! GNC scalars. Column names live in `output::PHOTO_*` (the same style as
//! `final_record::FR_*`); the CSV and PyO3 projections in `output.rs` read them
//! by name.

use super::output::*;
use super::sim_types::{DEG_TO_RAD, G0, SimState};
use crate::config::PlanetConfig;
use crate::data::{OrbitalElements, SimData};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::integration::events::EventRecord;
use crate::orbit::elements;
use crate::physics::dynamics::{AeroLoads, loads_at};
use crate::simulation::init::RunState;

/// The state-derived quantities both row builders share: geodetic position, the
/// osculating orbit, inertial energy (CLAUDE.md "Energy Computation": absolute
/// velocity via `to_absolute_cartesian`), and the dispersed aero loads.
struct PhotoPhysics {
    altitude: f64,
    latitude: f64,
    orbit: OrbitalElements,
    energy: f64,
    velocity_radial: f64,
    rho_truth: f64,
    loads: AeroLoads,
}

/// `aoa` is the angle of attack in radians before the incidence dispersion.
fn photo_physics(
    state: &[f64; 8],
    aoa: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &RunState,
) -> PhotoPhysics {
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

    // Dispersed loads (the same `loads_at` the peak tracker uses) so trajectory
    // plots are consistent with final_record peak values and constraint classification.
    let rho_truth = data.atmosphere.density_at(altitude);
    let loads = loads_at(state, altitude, aoa, data, &run_state.aero());

    PhotoPhysics {
        altitude,
        latitude,
        orbit,
        energy,
        velocity_radial,
        rho_truth,
        loads,
    }
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
    run_state: &RunState,
    cumulative_flux: f64,
    guidance_phase: i32,
) -> [f64; PHOTO_LINE_LEN] {
    let ph = photo_physics(&sim.state, sim.aoa, planet, data, run_state);

    let mut p = [0.0; PHOTO_LINE_LEN];
    p[PHOTO_TIME_S] = sim_time;
    p[PHOTO_ALT_KM] = ph.altitude / 1e3;
    p[PHOTO_LON_DEG] = sim.state[1] / DEG_TO_RAD;
    p[PHOTO_LAT_DEG] = ph.latitude / DEG_TO_RAD;
    p[PHOTO_VEL_MS] = sim.state[3];
    p[PHOTO_FPA_DEG] = sim.state[4] / DEG_TO_RAD;
    p[PHOTO_HDG_DEG] = sim.state[5] / DEG_TO_RAD;
    p[PHOTO_SMA_KM] = ph.orbit.semi_major_axis / 1e3;
    p[PHOTO_ECC] = ph.orbit.eccentricity;
    p[PHOTO_INCL_DEG] = ph.orbit.inclination / DEG_TO_RAD;
    p[PHOTO_RAAN_DEG] = ph.orbit.raan / DEG_TO_RAD;
    p[PHOTO_PERIAPSIS_ALT_KM] = ph.orbit.periapsis_alt / 1e3;
    p[PHOTO_APOAPSIS_ALT_KM] = ph.orbit.apoapsis_alt / 1e3;
    p[PHOTO_PHASE] = guidance_phase as f64;
    p[PHOTO_BANK_DEG] = sim.bank_angle / DEG_TO_RAD;
    p[PHOTO_RADIAL_VEL_MS] = ph.velocity_radial;
    p[PHOTO_AOA_DEG] = sim.aoa / DEG_TO_RAD;
    p[PHOTO_CUMULATIVE_BANK_DEG] = cumulative_bank_change / DEG_TO_RAD;
    p[PHOTO_ENERGY_J_KG] = ph.energy;
    p[PHOTO_PDYN_PA] = dynamic_pressure;
    p[PHOTO_RADIAL_VEL_DUP] = ph.velocity_radial;
    p[PHOTO_PDYN_ONBOARD_KPA] = 0.5 * density_estimate * sim.state[3] * sim.state[3] / 1e3;
    p[PHOTO_SIM_NUMBER] = sim_index as f64;
    p[PHOTO_RESERVED] = 0.0;
    p[PHOTO_HEAT_FLUX_KW_M2] = ph.loads.heat_flux / 1e3;
    p[PHOTO_G_LOAD] = ph.loads.load_factor / G0;
    p[PHOTO_NAV_DENSITY_RATIO] = density_gain; // estimated / model
    p[PHOTO_TRUTH_DENSITY] = ph.rho_truth;
    p[PHOTO_HEAT_LOAD_KJ_M2] = cumulative_flux / 1e3; // J/m2 -> kJ/m2
    p[PHOTO_DENSITY_PERTURBATION] = run_state.density_perturbation; // fractional GM value
    p
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
    run_state: &RunState,
    bank_angle_deg: f64,
    aoa_deg: f64,
    cumulative_bank_change_deg: f64,
    guidance_phase: f64,
    density_gain: f64,
) -> [f64; PHOTO_LINE_LEN] {
    // The tick's realized AoA (passed in degrees) + the dispersed incidence
    // bias -- same operand the regular photo rows use.
    let ph = photo_physics(state, aoa_deg * DEG_TO_RAD, planet, data, run_state);

    let mut p = [0.0; PHOTO_LINE_LEN];
    p[PHOTO_TIME_S] = event_time;
    p[PHOTO_ALT_KM] = ph.altitude / 1e3;
    p[PHOTO_LON_DEG] = state[1] / DEG_TO_RAD;
    p[PHOTO_LAT_DEG] = ph.latitude / DEG_TO_RAD;
    p[PHOTO_VEL_MS] = state[3];
    p[PHOTO_FPA_DEG] = state[4] / DEG_TO_RAD;
    p[PHOTO_HDG_DEG] = state[5] / DEG_TO_RAD;
    p[PHOTO_SMA_KM] = ph.orbit.semi_major_axis / 1e3;
    p[PHOTO_ECC] = ph.orbit.eccentricity;
    p[PHOTO_INCL_DEG] = ph.orbit.inclination / DEG_TO_RAD;
    p[PHOTO_RAAN_DEG] = ph.orbit.raan / DEG_TO_RAD;
    p[PHOTO_PERIAPSIS_ALT_KM] = ph.orbit.periapsis_alt / 1e3;
    p[PHOTO_APOAPSIS_ALT_KM] = ph.orbit.apoapsis_alt / 1e3;
    p[PHOTO_PHASE] = guidance_phase; // from enclosing tick
    p[PHOTO_BANK_DEG] = bank_angle_deg; // from enclosing tick
    p[PHOTO_RADIAL_VEL_MS] = ph.velocity_radial;
    p[PHOTO_AOA_DEG] = aoa_deg; // from enclosing tick
    p[PHOTO_CUMULATIVE_BANK_DEG] = cumulative_bank_change_deg; // from enclosing tick
    p[PHOTO_ENERGY_J_KG] = ph.energy;
    p[PHOTO_PDYN_PA] = ph.loads.pdyn;
    p[PHOTO_RADIAL_VEL_DUP] = ph.velocity_radial;
    p[PHOTO_PDYN_ONBOARD_KPA] = 0.0; // no nav estimate at event time
    p[PHOTO_SIM_NUMBER] = 0.0; // not applicable for event rows
    p[PHOTO_RESERVED] = 0.0;
    p[PHOTO_HEAT_FLUX_KW_M2] = ph.loads.heat_flux / 1e3;
    p[PHOTO_G_LOAD] = ph.loads.load_factor / G0;
    p[PHOTO_NAV_DENSITY_RATIO] = density_gain; // from enclosing tick
    p[PHOTO_TRUTH_DENSITY] = ph.rho_truth;
    p[PHOTO_HEAT_LOAD_KJ_M2] = state[6] / 1e3; // state[6] is integrated flux in J/m2
    p[PHOTO_DENSITY_PERTURBATION] = run_state.density_perturbation;
    p
}

/// Append the current state's snapshot row to `state.photo_lines`: the per-tick
/// photo row in `tick::step_one_tick` and the final row in `runner::run_single`.
pub(crate) fn push_photo_snapshot(state: &mut SimState, planet: &PlanetConfig, data: &SimData) {
    let sim_time = state.sim_time;
    let dynamic_pressure_for_photo = state.dynamic_pressure_for_photo;
    let density_estimate_for_photo = state.density_estimate_for_photo;
    let sim_idx = state.sim_idx;
    let cumulative_bank_change_deg = state.cumulative_bank_change_deg;
    let density_gain = state.nav_filter.density_gain();
    let run_state_snap = state.run_state;
    let cumulative_flux = state.state[6];
    let guidance_phase_for_photo = state.guidance_phase_for_photo;
    let photo_line = build_photo_values(
        state,
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
    state.photo_lines.push(photo_line);
}

/// Append one photo row per event record, then sort every row by time (column 0).
pub(crate) fn append_event_photo_rows(
    state: &mut SimState,
    event_records: &[EventRecord],
    planet: &PlanetConfig,
    data: &SimData,
) {
    for record in event_records {
        state.photo_lines.push(build_event_photo_values(
            &record.state,
            record.time,
            planet,
            data,
            &state.run_state,
            record.bank_angle_deg,
            record.aoa_deg,
            record.cumulative_bank_change_deg,
            record.guidance_phase,
            record.density_gain,
        ));
    }
    state.photo_lines.sort_by(|a, b| {
        a[PHOTO_TIME_S]
            .partial_cmp(&b[PHOTO_TIME_S])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
