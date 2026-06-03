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
use crate::simulation::runner::{
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
) -> [f64; 52] {
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

    let mut fr = [0.0_f64; 52];
    fr[0] = alt_final / 1e3;
    fr[1] = sim_state.state[1] / DEG_TO_RAD;
    fr[2] = lat_final / DEG_TO_RAD;
    fr[3] = sim_state.state[3];
    fr[4] = sim_state.state[4] / DEG_TO_RAD;
    fr[5] = sim_state.state[5] / DEG_TO_RAD;
    fr[6] = velocity_radial;
    fr[7] = energy / 1e6;
    fr[8] = orbit.semi_major_axis / 1e3;
    fr[9] = orbit.eccentricity;
    fr[10] = orbit.inclination / DEG_TO_RAD;
    fr[11] = orbit.raan / DEG_TO_RAD;
    fr[12] = orbit.arg_periapsis / DEG_TO_RAD;
    fr[13] = orbit.true_anomaly / DEG_TO_RAD;
    fr[14] = orbit.periapsis_alt / 1e3;
    fr[15] = orbit.apoapsis_alt / 1e3;
    fr[16] = sim_state.max_heat_flux / 1e3;
    fr[17] = sim_state.max_load_factor / G0;
    fr[18] = sim_state.max_dyn_pressure / 1e3;
    fr[19] = sim_state.alt_max_flux / 1e3;
    fr[20] = sim_state.alt_max_load / 1e3;
    fr[21] = sim_state.alt_max_pdyn / 1e3;
    fr[22] = sim_state.time_max_flux;
    fr[23] = sim_state.time_max_load;
    fr[24] = sim_state.time_max_pdyn;
    fr[25] = sim_state.bounce_alt / 1e3;
    fr[26] = sim_state.bounce_time;
    fr[27] = sim_state.sim_time;
    fr[28] = sim_state.state[6] / 1e6;
    fr[29] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3;
    fr[30] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3;
    fr[31] = ifinal as f64;
    fr[37] = deltav.dv1;
    fr[38] = deltav.dv2;
    fr[39] = deltav.dv3;
    fr[40] = deltav.dv1.abs() + deltav.dv2.abs();
    fr[41] = deltav.total;
    fr[45] = sim_state.cumulative_bank_change_deg;
    fr[46] = orbit.inclination / DEG_TO_RAD - data.target_orbit.inclination / DEG_TO_RAD;
    fr[48] = sim_state.guidance_state.lateral_state.n_reversals as f64;
    fr
}
