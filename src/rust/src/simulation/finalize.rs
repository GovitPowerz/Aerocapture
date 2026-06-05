//! Termination classification and final-record assembly.
//!
//! Pure transforms turning a terminated `SimState` into the 52-element final
//! record: pending-crash promotion, the `ifinal` classification, the virtual-DV
//! cost for non-capturing terminations, and the final-record builder. Shared by
//! the CLI path (`runner::run_single`) and the RL per-step path (`tick.rs`) so
//! both produce bit-identical outputs.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::orbit::maneuver::DeltaV;
use crate::orbit::{elements, maneuver};
use crate::simulation::final_record::{
    FINAL_RECORD_LEN, FR_ALT_KM, FR_ALT_MAX_FLUX_KM, FR_ALT_MAX_LOAD_KM, FR_ALT_MAX_PDYN_KM,
    FR_APOAPSIS_ALT_KM, FR_APOAPSIS_ERR_KM, FR_ARG_PERI_DEG, FR_BOUNCE_ALT_KM, FR_BOUNCE_TIME_S,
    FR_CUMULATIVE_BANK_DEG, FR_DV_PLANE_MS, FR_DV_TOTAL_MS, FR_DV1_MS, FR_DV2_MS, FR_DV3_MS,
    FR_DYN_PRESSURE_KPA, FR_ECC, FR_ENERGY_MJKG, FR_FPA_DEG, FR_G_LOAD, FR_HDG_DEG,
    FR_HEAT_FLUX_KW_M2, FR_HEAT_LOAD_MJM2, FR_IFINAL, FR_INCL_DEG, FR_INCL_ERR_DEG, FR_LAT_DEG,
    FR_LON_DEG, FR_N_REVERSALS, FR_PERIAPSIS_ALT_KM, FR_PERIAPSIS_ERR_KM, FR_RAAN_DEG,
    FR_RADIAL_VEL_MS, FR_SIM_TIME_S, FR_SMA_KM, FR_TIME_MAX_FLUX_S, FR_TIME_MAX_LOAD_S,
    FR_TIME_MAX_PDYN_S, FR_TRUE_ANOM_DEG, FR_VEL_MS,
};
use crate::simulation::sim_types::{
    CRASH_ENERGY_CAP_MJKG, CRASH_ENERGY_WEIGHT, CRASH_FLOOR, CRASH_TIME_BONUS, DEG_TO_RAD, G0,
    HYPERBOLIC_BASE, SimState, TermReason,
};

/// Virtual DV for non-capturing terminations (Crash, PendingCrash, Timeout).
///
/// Penalizes energy distance from target; softens crashes near the capture
/// boundary so PSO/GA will explore closer to the crash limit.
///
/// Non-finite inputs (NaN from degenerate-state MC dispersions) fall back
/// to the worst-case cap — caller still gets a finite, large virtual DV.
pub(crate) fn virtual_dv_non_capture(
    orbital_energy_j_kg: f64,
    target_sma_m: f64,
    mu: f64,
    sim_time: f64,
    max_time: f64,
) -> f64 {
    let target_energy_j_kg = -mu / (2.0 * target_sma_m);
    let delta_e_mj = if orbital_energy_j_kg.is_finite() && target_energy_j_kg.is_finite() {
        ((orbital_energy_j_kg - target_energy_j_kg).abs() / 1e6).min(CRASH_ENERGY_CAP_MJKG)
    } else {
        CRASH_ENERGY_CAP_MJKG
    };
    let t_ratio = if max_time.is_finite() && max_time > 0.0 && sim_time.is_finite() {
        (sim_time / max_time).clamp(0.0, 1.0)
    } else {
        0.0
    };
    CRASH_FLOOR + CRASH_ENERGY_WEIGHT * delta_e_mj - CRASH_TIME_BONUS * t_ratio
}

/// Pure predicate: would this orbit be a "pending crash" -- captured (bound + e<1)
/// but with apoapsis below the atmospheric ceiling, so guaranteed to re-enter?
///
/// Extracted so it can be unit-tested without constructing a full `SimState`.
pub fn is_pending_crash(
    eccentricity: f64,
    energy: f64,
    apoapsis_alt: f64,
    exit_altitude: f64,
) -> bool {
    let captured = eccentricity < 1.0 && energy < 0.0;
    captured && apoapsis_alt < exit_altitude
}

/// Promote `AtmosphereExit` to `PendingCrash` when the resulting orbit has
/// apoapsis below the atmospheric ceiling (captured but doomed to re-entry).
///
/// Called both by `finalize_run` (CLI path) and `tick.rs` (RL per-step path)
/// so both sources of `ifinal`/`final_record` see the same terminal classification.
pub fn promote_pending_crash_if_applicable(sim_state: &mut SimState, planet: &PlanetConfig) {
    if sim_state.term != TermReason::AtmosphereExit {
        return;
    }
    let orbit = elements::from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let (_, velocity_abs) = to_absolute_cartesian(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - planet.mu / sim_state.state[0];
    if is_pending_crash(
        orbit.eccentricity,
        energy,
        orbit.apoapsis_alt,
        sim_state.exit_altitude,
    ) {
        sim_state.term = TermReason::PendingCrash;
    }
}

/// Map a terminal `TermReason` to the `ifinal` classification code written to
/// `final_record[31]`.
///
/// Single source of truth shared by `run_single`, `build_final_record`, and the
/// RL per-step path in `tick.rs`. Genuinely unreachable on `None`: every caller
/// is reached only after the simulation has terminated.
pub fn ifinal_for(term: TermReason) -> i32 {
    match term {
        TermReason::AtmosphereExit => 3,
        TermReason::Crash => 1,
        TermReason::PendingCrash => 4,
        TermReason::Timeout => 2,
        TermReason::None => unreachable!("ifinal requested for a non-terminated state"),
    }
}

/// Assemble the 52-element final record from a terminated `SimState`.
///
/// Mirrors the block at the end of `run_single`. Requires `term != TermReason::None`.
/// Called by `BatchedSimulation::step()` on terminal steps.
pub fn build_final_record(
    sim_state: &SimState,
    data: &SimData,
    planet: &PlanetConfig,
) -> [f64; FINAL_RECORD_LEN] {
    let (alt_final, lat_final) = geodetic_from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        planet,
    );

    let orbit = elements::from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / sim_state.state[0];
    let velocity_radial = sim_state.state[3] * sim_state.state[4].sin();

    let captured = orbit.eccentricity < 1.0 && energy < 0.0;

    let ifinal = ifinal_for(sim_state.term);

    let deltav = if sim_state.term == TermReason::AtmosphereExit && captured {
        maneuver::compute_deltav(&orbit, &data.target_orbit, &data.parking_orbit, planet)
    } else if sim_state.term == TermReason::AtmosphereExit {
        let v_escape = (2.0 * mu / sim_state.state[0]).sqrt();
        let v_excess = (speed_abs - v_escape).max(0.0);
        DeltaV {
            dv1: 0.0,
            dv2: 0.0,
            dv3: 0.0,
            total: HYPERBOLIC_BASE + v_excess,
        }
    } else {
        let virtual_dv = virtual_dv_non_capture(
            energy,
            data.target_orbit.semi_major_axis,
            mu,
            sim_state.sim_time,
            sim_state.max_time,
        );
        DeltaV {
            dv1: 0.0,
            dv2: 0.0,
            dv3: 0.0,
            total: virtual_dv,
        }
    };

    let mut fr = [0.0_f64; FINAL_RECORD_LEN];
    fr[FR_ALT_KM] = alt_final / 1e3;
    fr[FR_LON_DEG] = sim_state.state[1] / DEG_TO_RAD;
    fr[FR_LAT_DEG] = lat_final / DEG_TO_RAD;
    fr[FR_VEL_MS] = sim_state.state[3];
    fr[FR_FPA_DEG] = sim_state.state[4] / DEG_TO_RAD;
    fr[FR_HDG_DEG] = sim_state.state[5] / DEG_TO_RAD;
    fr[FR_RADIAL_VEL_MS] = velocity_radial;
    fr[FR_ENERGY_MJKG] = energy / 1e6;
    fr[FR_SMA_KM] = orbit.semi_major_axis / 1e3;
    fr[FR_ECC] = orbit.eccentricity;
    fr[FR_INCL_DEG] = orbit.inclination / DEG_TO_RAD;
    fr[FR_RAAN_DEG] = orbit.raan / DEG_TO_RAD;
    fr[FR_ARG_PERI_DEG] = orbit.arg_periapsis / DEG_TO_RAD;
    fr[FR_TRUE_ANOM_DEG] = orbit.true_anomaly / DEG_TO_RAD;
    fr[FR_PERIAPSIS_ALT_KM] = orbit.periapsis_alt / 1e3;
    fr[FR_APOAPSIS_ALT_KM] = orbit.apoapsis_alt / 1e3;
    fr[FR_HEAT_FLUX_KW_M2] = sim_state.max_heat_flux / 1e3;
    fr[FR_G_LOAD] = sim_state.max_load_factor / G0;
    fr[FR_DYN_PRESSURE_KPA] = sim_state.max_dyn_pressure / 1e3;
    fr[FR_ALT_MAX_FLUX_KM] = sim_state.alt_max_flux / 1e3;
    fr[FR_ALT_MAX_LOAD_KM] = sim_state.alt_max_load / 1e3;
    fr[FR_ALT_MAX_PDYN_KM] = sim_state.alt_max_pdyn / 1e3;
    fr[FR_TIME_MAX_FLUX_S] = sim_state.time_max_flux;
    fr[FR_TIME_MAX_LOAD_S] = sim_state.time_max_load;
    fr[FR_TIME_MAX_PDYN_S] = sim_state.time_max_pdyn;
    fr[FR_BOUNCE_ALT_KM] = sim_state.bounce_alt / 1e3;
    fr[FR_BOUNCE_TIME_S] = sim_state.bounce_time;
    fr[FR_SIM_TIME_S] = sim_state.sim_time;
    fr[FR_HEAT_LOAD_MJM2] = sim_state.state[6] / 1e6;
    fr[FR_PERIAPSIS_ERR_KM] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3;
    fr[FR_APOAPSIS_ERR_KM] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3;
    fr[FR_IFINAL] = ifinal as f64;
    fr[FR_DV1_MS] = deltav.dv1;
    fr[FR_DV2_MS] = deltav.dv2;
    fr[FR_DV3_MS] = deltav.dv3;
    fr[FR_DV_PLANE_MS] = deltav.dv1.abs() + deltav.dv2.abs();
    fr[FR_DV_TOTAL_MS] = deltav.total;
    fr[FR_CUMULATIVE_BANK_DEG] = sim_state.cumulative_bank_change_deg;
    fr[FR_INCL_ERR_DEG] =
        orbit.inclination / DEG_TO_RAD - data.target_orbit.inclination / DEG_TO_RAD;
    fr[FR_N_REVERSALS] = sim_state.guidance_state.lateral_state.n_reversals as f64;
    fr
}
