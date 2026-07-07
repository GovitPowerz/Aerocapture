//! FNPAG -- Fully Numerical Predictor-corrector Aerocapture Guidance.
//!
//! Based on Ping Lu's algorithm (Journal of Guidance, Control, and Dynamics,
//! 2015). This is a modern predictor-corrector specifically designed for
//! aerocapture, using numerical forward prediction of the trajectory to
//! find the bank angle that achieves a target exit apoapsis radius.
//!
//! Algorithm overview:
//! 1. Predict forward trajectory with current bank angle using 3D equations
//!    of motion (J2 gravity, planet rotation, onboard atmosphere model)
//! 2. Compute the predicted exit orbit's osculating apoapsis radius (inertial)
//! 3. Bisect the bank angle (monotonic apoapsis-vs-bank) to hit target apoapsis
//! 4. Hand off to the shared exit-phase controller after the bounce
//!
//! The predictor uses the same EOM as the main simulator, with the onboard
//! atmosphere SCALED by the nav-estimated density dispersion factor (so the
//! forward model tracks the measured atmosphere -- the dominant apoapsis-error
//! driver), no winds, and zero lateral lift (roll sign unknown). RK4 integration.
//!
//! Apoapsis (not energy) is the target because the post-capture orbit-correction
//! dV -- a periapsis-raise burn at apoapsis plus an apoapsis-correction burn --
//! is paid on the apoapsis radius. Energy fixes only the semi-major axis and
//! leaves the apoapsis (the dV-dominant, dispersion-sensitive quantity) free.
//! Targeting it directly is what aligns the corrector with the mission cost.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;
use crate::physics::gravity;

/// Altitude breakpoint (m) below which the tighter bank-angle limit applies.
const BANK_LIMIT_SWITCH_ALTITUDE_M: f64 = 50e3;

/// Bisection steps for the apoapsis corrector. Halves the bank bracket each
/// step; 8 takes the ~110-deg bank range to ~0.4-deg resolution, well below the
/// noise floor of the dt~=2 s onboard predictor. Plus 2 endpoint evals + 1
/// initial midpoint, the corrector costs ~11 forward integrations per replan.
const N_BISECT_STEPS: usize = 8;

/// Apoapsis-radius sentinel (m) returned when the predicted exit orbit is
/// unbound (eccentricity >= 1) or degenerate. Far above any captured apoapsis
/// so the corrector reads it as "apoapsis at infinity" and commands more bank
/// (more dissipation) -- the correct direction for an under-dissipated pass.
const UNBOUND_APOAPSIS_RADIUS_M: f64 = 1e9;

/// FNPAG persistent state (mutable runtime state only).
#[derive(Debug, Clone)]
pub struct FnpagState {
    /// Last commanded bank (held between replans; re-clamped to current limits)
    pub bank_prev: f64,
    /// Last bisection residual = predicted exit apoapsis radius − target (m); diagnostic
    pub resid_prev: f64,
    /// Whether predictor has been initialized
    pub initialized: bool,
    /// Sim time of the last forward-prediction replan (s)
    pub last_replan_time: f64,
}

impl FnpagState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            bank_prev: initial_bank,
            resid_prev: 0.0,
            initialized: false,
            last_replan_time: f64::NEG_INFINITY,
        }
    }
}

/// State for forward prediction (matches main sim's 6 translational DOFs).
#[derive(Clone, Copy)]
struct PredState {
    r: f64,     // radius (m)
    lon: f64,   // longitude (rad)
    lat: f64,   // latitude (rad)
    v: f64,     // relative velocity (m/s)
    gamma: f64, // flight path angle (rad)
    psi: f64,   // heading/azimuth (rad)
}

/// Compute 3D trajectory derivatives for the onboard predictor.
///
/// Matches the main simulator EOM (runner.rs `compute_derivatives`) with:
/// - Onboard atmosphere model (no dispersions)
/// - J2/J3/J4 gravity via `gravity::gravity()`
/// - Planet rotation (Coriolis + centrifugal)
/// - Zero lateral lift (sin_bank = 0): predictor doesn't know roll sign
/// - Nominal aero coefficients at initial AoA
/// - No winds
fn pred_derivatives(
    s: &PredState,
    bank_angle: f64,
    planet: &PlanetConfig,
    data: &SimData,
    density_factor: f64,
) -> [f64; 6] {
    let (altitude, _) = geodetic_from_spherical(s.r, s.lon, s.lat, planet);
    // Scale the nominal onboard density by the nav-estimated dispersion factor so
    // the forward prediction tracks the MEASURED atmosphere, not the nominal one.
    // Atmosphere density is the dominant apoapsis-error driver; a predictor blind
    // to it cannot reject the dispersion the corrector exists to reject.
    let rho = density_factor
        * data
            .atmosphere_onboard
            .density_at(altitude, &data.atmosphere);

    let cx = data.aero.interpolate_cx(data.entry.initial_aoa);
    let cz = data.aero.interpolate_cz(data.entry.initial_aoa).abs();

    let mass = data.capsule.mass;
    let sref = data.capsule.reference_area;
    let aero_factor = rho * sref / (2.0 * mass);
    let drag = aero_factor * cx * s.v * s.v;
    let lift = aero_factor * cz * s.v * s.v;

    let (gravtl, gravtr) = gravity::gravity(s.r, s.lat, planet);

    let cos_bank = bank_angle.cos();
    // sin_bank = 0: predictor assumes no lateral lift (roll sign unknown)
    let cos_gamma = s.gamma.cos();
    let sin_gamma = s.gamma.sin();
    let cos_psi = s.psi.cos();
    let sin_psi = s.psi.sin();
    let cos_lat = s.lat.cos();
    let sin_lat = s.lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;

    let omega = planet.omega;

    let dr = s.v * sin_gamma;
    let dlon = s.v * cos_gamma * sin_psi / (s.r * cos_lat);
    let dlat = s.v * cos_gamma * cos_psi / s.r;

    let dv = -drag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * s.r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = if s.v.abs() > 1.0 {
        (lift * cos_bank / s.v) + (s.v * cos_gamma / s.r)
            - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / s.v)
            + (2.0 * omega * sin_psi * cos_lat)
            + (omega
                * omega
                * s.r
                * cos_lat
                * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma)
                / s.v)
    } else {
        0.0
    };

    // Lateral lift term is zero (sin_bank = 0), but gravity/Coriolis/centrifugal
    // still drive heading evolution.
    let dpsi = if s.v.abs() > 1.0 && cos_gamma.abs() > 1e-10 {
        (s.v * cos_gamma * sin_psi * tan_lat / s.r)
            + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
            + (gravtl * sin_psi / (s.v * cos_gamma))
            + (omega * omega * s.r * cos_lat * sin_lat * sin_psi / (s.v * cos_gamma))
    } else {
        0.0
    };

    [dr, dlon, dlat, dv, dgamma, dpsi]
}

/// Osculating apoapsis radius (m) of the current predicted state.
///
/// Uses inertial velocity (via `from_spherical` -> `to_absolute_cartesian`).
/// An unbound (eccentricity >= 1) or degenerate orbit has no apoapsis, so it
/// returns `UNBOUND_APOAPSIS_RADIUS_M` -- the corrector then reads it as
/// "apoapsis at infinity" and adds bank (dissipation), the right direction.
fn osc_apoapsis_radius(s: &PredState, planet: &PlanetConfig) -> f64 {
    let elem = elements::from_spherical(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet);
    // e >= 1.0 (or NaN) is unbound; apoapsis_alt is +inf for a parabolic orbit.
    if elem.eccentricity >= 1.0 || elem.eccentricity.is_nan() || !elem.apoapsis_alt.is_finite() {
        return UNBOUND_APOAPSIS_RADIUS_M;
    }
    elem.apoapsis_alt + planet.equatorial_radius
}

/// Exit-phase bank magnitude (rad) inside the predictor -- a faithful copy of
/// `gnc::guidance::exit::exit_guidance` so the forward model matches the plant on
/// the ascending leg (which the vehicle flies under the shared exit controller,
/// NOT the FNPAG capture bank). No dispersion: densities come straight from the
/// onboard model (the plant's `density_gain` ~= 1 in nominal prediction).
fn exit_law_bank(
    s: &PredState,
    planet: &PlanetConfig,
    data: &SimData,
    rho_exit: f64,
    v_r_ref: f64,
    density_factor: f64,
) -> f64 {
    let v = s.v;
    let v_radial = v * s.gamma.sin();
    let (altitude, _) = geodetic_from_spherical(s.r, s.lon, s.lat, planet);
    let rho_cur = density_factor
        * data
            .atmosphere_onboard
            .density_at(altitude, &data.atmosphere);
    let pdyn_target = 0.5 * rho_exit * v * v * data.guidance.exit_pdyn_margin;
    let pdyn_current = 0.5 * rho_cur * v * v;
    let pdyn_safe = if pdyn_current.abs() > 1e-10 {
        pdyn_current
    } else {
        1e-10
    };
    let cos_bank = (pdyn_current - pdyn_target) / pdyn_safe
        + data.guidance.exit_radial_vel_gain * (v_radial - v_r_ref) / pdyn_safe;
    cos_bank.clamp(-1.0, 1.0).acos()
}

/// Predict the realized exit apoapsis radius by integrating the EOM forward
/// through BOTH guidance phases the vehicle actually flies:
/// - capture phase: the constant `capture_bank` under evaluation, and
/// - exit phase: the shared exit-phase controller (`exit_law_bank`), engaged at
///   the predicted handoff (post-bounce, relative speed <= exit_velocity_threshold),
///   matching estimator.rs phase management.
///
/// Modeling the handoff is the point: ~half the energy is dissipated on the
/// ascending leg under the exit law, so a constant-bank-to-exit prediction
/// mis-estimates the apoapsis and the corrector cannot reject dispersions.
///
/// Same EOM as the main simulator (J2 gravity, rotation, Coriolis/centrifugal),
/// onboard atmosphere, no dispersions/winds, zero lateral lift. RK4 integration.
/// A crash / negative-velocity blow-up returns 0.0 (apoapsis below the surface)
/// so the corrector reads over-dissipation and backs the bank off.
fn predict_exit_apoapsis(
    initial: PredState,
    capture_bank: f64,
    planet: &PlanetConfig,
    data: &SimData,
    exit_alt: f64,
    dt: f64,
    density_factor: f64,
) -> f64 {
    let req = planet.equatorial_radius;
    let max_steps = 2000;
    let vphase = data.guidance.exit_velocity_threshold;
    let rho_exit = density_factor
        * data
            .atmosphere_onboard
            .density_at(data.guidance.exit_altitude_threshold, &data.atmosphere);

    let mut s = initial;
    // Replicate the plant's phase state: bounce latches once ascending, then the
    // handoff to the exit controller fires when the relative speed drops to the
    // exit-velocity threshold (estimator.rs). `v_r_ref` is latched there.
    let mut bounced = s.gamma.sin() > 0.0;
    let mut in_exit = false;
    let mut v_r_ref = 0.0;

    for _ in 0..max_steps {
        let alt = s.r - req;

        // Termination: crash (over-dissipated -> apoapsis below surface)
        if alt <= 0.0 {
            return 0.0;
        }
        // Termination: atmosphere exit (ascending)
        if alt >= exit_alt && s.gamma.sin() > 0.0 {
            return osc_apoapsis_radius(&s, planet);
        }

        // Phase bookkeeping + bank selection for this step.
        if !bounced && s.gamma.sin() > 0.0 {
            bounced = true;
        }
        if !in_exit && bounced && s.v <= vphase {
            in_exit = true;
            v_r_ref = s.v * s.gamma.sin();
        }
        let bank = if in_exit {
            exit_law_bank(&s, planet, data, rho_exit, v_r_ref, density_factor)
        } else {
            capture_bank
        };

        // Classic RK4 (bank held across the four stages, mirroring tick cadence)
        let k1 = pred_derivatives(&s, bank, planet, data, density_factor);

        let s2 = PredState {
            r: s.r + 0.5 * dt * k1[0],
            lon: s.lon + 0.5 * dt * k1[1],
            lat: s.lat + 0.5 * dt * k1[2],
            v: s.v + 0.5 * dt * k1[3],
            gamma: s.gamma + 0.5 * dt * k1[4],
            psi: s.psi + 0.5 * dt * k1[5],
        };
        let k2 = pred_derivatives(&s2, bank, planet, data, density_factor);

        let s3 = PredState {
            r: s.r + 0.5 * dt * k2[0],
            lon: s.lon + 0.5 * dt * k2[1],
            lat: s.lat + 0.5 * dt * k2[2],
            v: s.v + 0.5 * dt * k2[3],
            gamma: s.gamma + 0.5 * dt * k2[4],
            psi: s.psi + 0.5 * dt * k2[5],
        };
        let k3 = pred_derivatives(&s3, bank, planet, data, density_factor);

        let s4 = PredState {
            r: s.r + dt * k3[0],
            lon: s.lon + dt * k3[1],
            lat: s.lat + dt * k3[2],
            v: s.v + dt * k3[3],
            gamma: s.gamma + dt * k3[4],
            psi: s.psi + dt * k3[5],
        };
        let k4 = pred_derivatives(&s4, bank, planet, data, density_factor);

        s.r += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        s.lon += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        s.lat += dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
        s.v += dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
        s.gamma += dt / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]);
        s.psi += dt / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]);

        // Safety: velocity can't go negative (over-dissipated)
        if s.v <= 0.0 {
            return 0.0;
        }
    }

    // Timeout -- didn't exit atmosphere; report the current osculating apoapsis.
    osc_apoapsis_radius(&s, planet)
}

/// Compute FNPAG bank angle command.
///
/// Bisects over two-phase forward predictions (capture bank + exit-law ascent)
/// to find the capture bank whose realized exit apoapsis hits the target.
///
/// Returns bank angle magnitude in radians.
pub fn fnpag_bank(
    nav: &NavigationOutput,
    state: &mut FnpagState,
    data: &SimData,
    planet: &PlanetConfig,
    sim_time: f64,
) -> f64 {
    // Target: apoapsis radius of the desired orbit. FNPAG targets apoapsis
    // (not energy) because the post-capture periapsis-raise dV is paid at
    // apoapsis; energy fixes only the semi-major axis and leaves apoapsis --
    // the dV-dominant, dispersion-sensitive quantity -- uncontrolled.
    let target_apo = planet.equatorial_radius + data.target_orbit.apoapsis;

    let exit_alt = data.final_conditions.altitude;

    // Current state for prediction (full 6-DOF from navigation)
    let current = PredState {
        r: nav.position_estimated[0],
        lon: nav.position_estimated[1],
        lat: nav.position_estimated[2],
        v: nav.velocity_estimated[0],
        gamma: nav.velocity_estimated[1],
        psi: nav.velocity_estimated[2],
    };

    // Check if we're in the sensible atmosphere (density > threshold)
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    let rho = data
        .atmosphere_onboard
        .density_at(altitude, &data.atmosphere);
    if rho < 1e-10 {
        // Outside sensible atmosphere — hold current bank angle
        return state.bank_prev.abs();
    }

    // Estimated atmospheric dispersion factor: nav.density_guidance is the
    // onboard density at the current altitude scaled by the nav-estimated
    // dispersion. Propagate this multiplicative bias through the predictor so the
    // forward model tracks the measured atmosphere (rho > 1e-10 here). Fall back
    // to nominal (1.0) if the estimate is degenerate (e.g. guard-tripped to 0).
    let density_factor = {
        let f = nav.density_guidance / rho;
        if f.is_finite() && f > 0.0 { f } else { 1.0 }
    };

    let params = &data.guidance.fnpag;

    // Bank angle limits from params
    let bank_min = params.bank_min_deg.to_radians();
    let bank_max = if altitude < BANK_LIMIT_SWITCH_ALTITUDE_M {
        params.bank_max_low_deg.to_radians()
    } else {
        params.bank_max_high_deg.to_radians()
    };

    // Replan throttle: between replans, hold the previous command. The command
    // evolves slowly mid-pass and each replan costs ~11 forward integrations.
    // Re-clamp at the CURRENT altitude's limits: a command issued above
    // BANK_LIMIT_SWITCH_ALTITUDE_M (bank_max_high) and held across the descent
    // would otherwise violate bank_max_low for up to replan_period seconds.
    if state.initialized && sim_time - state.last_replan_time < params.replan_period {
        return state.bank_prev.abs().clamp(bank_min, bank_max);
    }
    state.initialized = true;
    state.last_replan_time = sim_time;

    // Bisection on the apoapsis residual over [bank_min, bank_max]. The
    // apoapsis-vs-bank curve is monotonic decreasing -- low bank under-dissipates
    // (apoapsis high / escape sentinel, residual > 0), high bank over-dissipates
    // (apoapsis low / crash sentinel = 0, residual < 0). A secant fails here
    // because the escape/crash plateaus give it zero gradient; bisection only
    // needs the sign and converges on the monotonic curve regardless. Closed-loop
    // robustness comes from re-solving this each replan on the measured state.
    let resid = |bank: f64| -> f64 {
        predict_exit_apoapsis(
            current,
            bank,
            planet,
            data,
            exit_alt,
            params.prediction_dt,
            density_factor,
        ) - target_apo
    };

    let f_lo = resid(bank_min);
    let f_hi = resid(bank_max);

    let (bank_cmd, resid_cmd) = if f_lo <= 0.0 {
        // Even the least bank over-dissipates: command least dissipation.
        (bank_min, f_lo)
    } else if f_hi >= 0.0 {
        // Even the most bank can't dissipate enough (still escaping): command
        // most dissipation.
        (bank_max, f_hi)
    } else {
        // f_lo > 0 > f_hi: the bracket straddles the target. Bisect.
        let mut lo = bank_min;
        let mut hi = bank_max;
        let mut mid = 0.5 * (lo + hi);
        let mut f_mid = resid(mid);
        for _ in 0..N_BISECT_STEPS {
            if f_mid.abs() < params.energy_tol {
                break;
            }
            if f_mid > 0.0 {
                lo = mid; // apoapsis still too high -> need more bank
            } else {
                hi = mid; // apoapsis too low -> need less bank
            }
            mid = 0.5 * (lo + hi);
            f_mid = resid(mid);
        }
        (mid, f_mid)
    };

    state.bank_prev = bank_cmd;
    state.resid_prev = resid_cmd;
    bank_cmd.clamp(bank_min, bank_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::GuidanceParams;
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };
    use std::sync::Arc;

    fn test_nav(velocity: f64) -> NavigationOutput {
        let r = 3_396_200.0 + 50_000.0; // Mars radius + 50 km
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [velocity, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            density_exit: 1e-6,
            dynamic_pressure_estimated: 0.5 * 0.001 * velocity * velocity,
            energy_estimated: -1e6,
            ..Default::default()
        }
    }

    fn test_sim_data() -> SimData {
        SimData {
            capsule: Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.00008242,
                max_bank_rate: 15.0_f64.to_radians(),
                periods: TimePeriods::default(),
            },
            aero: AeroTables {
                n_points: 2,
                incidence: vec![-0.5, 0.0],
                cx: vec![1.269, 1.269],
                cz: vec![-0.205, -0.205],
                equilibrium_aoa: -0.48,
                ..Default::default()
            },
            atmosphere: Arc::new(AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            }),
            atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    velocity: 5687.0,
                    flight_path: -10.8_f64.to_radians(),
                    ..Default::default()
                },
                initial_bank: 64.77_f64.to_radians(),
                initial_aoa: -27.5_f64.to_radians(),
                initial_date: 0.0,
            },
            guidance: GuidanceParams {
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                ..Default::default()
            },
            incidence: IncidenceProfile {
                n_points: 2,
                altitudes: vec![-10_000.0, 150_000.0],
                incidences: vec![-0.48, -0.48],
            },
            periods: TimePeriods::default(),
            pilot: PilotModel {
                pilot_type: PilotType::Perfect,
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            target_orbit: OrbitalTarget {
                semi_major_axis: 3_649_622.0,
                eccentricity: 0.067,
                inclination: 50.0_f64.to_radians(),
                raan: -7.612_f64.to_radians(),
                apoapsis: 500_130.0,
                periapsis: 11_233.0,
            },
            final_conditions: FinalConditions {
                altitude: 60_000.0,
                ..Default::default()
            },
            parking_orbit: ParkingOrbit::default(),
            constraints: Constraints::default(),
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
            integration_mode: crate::config::IntegrationMode::FixedGill,
            sim_phase: crate::config::SimPhase::Full,
            density_perturbation: None,
            nn_normalization_override: None,
        }
    }

    // ── Deterministic tests ──────────────────────────────────────────────────

    /// When the spacecraft is above the sensible atmosphere (density < 1e-10),
    /// FNPAG must return the previous bank angle unchanged.
    #[test]
    fn low_density_returns_previous_bank() {
        // Place spacecraft at 200 km — exponential tail gives ≈9e-12 kg/m³ < 1e-10
        let mut nav = test_nav(5000.0);
        nav.position_estimated[0] = PlanetConfig::mars().equatorial_radius + 200_000.0;

        let prev_bank = 55.0_f64.to_radians();
        let mut state = FnpagState::new(prev_bank);
        state.initialized = true; // doesn't matter — early exit fires first

        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = fnpag_bank(&nav, &mut state, &data, &planet, 0.0);

        assert_relative_eq!(bank, prev_bank, epsilon = 1e-12);
    }

    /// A fresh (uninitialized) FnpagState must be marked initialized after the
    /// first call and the stored bank_prev must be updated.
    #[test]
    fn first_call_initializes_state() {
        let nav = test_nav(5687.0);
        let initial_bank = 0.5_f64; // arbitrary seed; will be overwritten
        let mut state = FnpagState::new(initial_bank);
        assert!(!state.initialized, "state should start uninitialized");

        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let _ = fnpag_bank(&nav, &mut state, &data, &planet, 0.0);

        assert!(
            state.initialized,
            "state must be initialized after first call"
        );
        // bank_prev is now the bisection result: finite and within the bank
        // limits resolved for this altitude (>50 km -> bank_max_high).
        let bank_min = data.guidance.fnpag.bank_min_deg.to_radians();
        let bank_max = data.guidance.fnpag.bank_max_high_deg.to_radians();
        assert!(state.bank_prev.is_finite(), "bank_prev not finite");
        assert!(
            (bank_min - 1e-9..=bank_max + 1e-9).contains(&state.bank_prev),
            "bank_prev {:.4} rad outside [{:.4}, {:.4}] after init",
            state.bank_prev,
            bank_min,
            bank_max
        );
    }

    /// Typical MSR entry state — bank angle must be finite and within [0, π].
    #[rstest]
    #[case(3000.0)]
    #[case(4500.0)]
    #[case(5687.0)]
    fn output_finite_for_typical_state(#[case] velocity: f64) {
        let nav = test_nav(velocity);
        let mut state = FnpagState::new(64.77_f64.to_radians());
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = fnpag_bank(&nav, &mut state, &data, &planet, 0.0);

        assert!(
            bank.is_finite(),
            "bank not finite for V={velocity} m/s: {bank}"
        );
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "bank {:.4} rad outside [0, π] for V={velocity} m/s",
            bank
        );
    }

    /// Within `replan_period` of the last replan, FNPAG must hold the previous
    /// bank command without re-running the forward predictor (no state update).
    #[test]
    fn replanning_throttled_within_replan_period() {
        let data = test_sim_data(); // default replan_period = 2.0 s
        let planet = PlanetConfig::mars();
        let mut state = FnpagState::new(64.77_f64.to_radians());

        let nav_entry = test_nav(5687.0);
        let bank0 = fnpag_bank(&nav_entry, &mut state, &data, &planet, 0.0);
        let resid_prev0 = state.resid_prev;
        assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);

        // 1 s later with a substantially different nav state: must hold the
        // command and leave the secant state untouched.
        let nav_later = test_nav(4000.0);
        let bank_held = fnpag_bank(&nav_later, &mut state, &data, &planet, 1.0);
        assert_relative_eq!(bank_held, bank0, epsilon = 1e-12);
        assert_relative_eq!(state.resid_prev, resid_prev0, epsilon = 1e-12);
        assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);

        // Past the period: replans (timestamp advances).
        let _ = fnpag_bank(&nav_later, &mut state, &data, &planet, 2.5);
        assert_relative_eq!(state.last_replan_time, 2.5, epsilon = 1e-12);
    }

    /// A command issued at high altitude (where bank_max_high applies) and held
    /// across the descent through BANK_LIMIT_SWITCH_ALTITUDE_M must be re-clamped
    /// to the low-altitude limit — the hold path is the only return that skipped
    /// the altitude-dependent clamp.
    #[test]
    fn held_command_reclamps_to_low_altitude_bank_limit() {
        let data = test_sim_data(); // default replan_period = 2.0 s
        let planet = PlanetConfig::mars();

        let mut state = FnpagState::new(0.0);
        state.initialized = true;
        state.last_replan_time = 0.0;
        // Legal above 50 km (bank_max_high_deg = 140), illegal below (100).
        state.bank_prev = 140.0_f64.to_radians();

        // 40 km altitude: below the switch, within the hold window (t = 1.0 s).
        let mut nav = test_nav(5000.0);
        nav.position_estimated[0] = planet.equatorial_radius + 40_000.0;

        let bank = fnpag_bank(&nav, &mut state, &data, &planet, 1.0);

        let bank_max_low = data.guidance.fnpag.bank_max_low_deg.to_radians();
        assert!(
            bank <= bank_max_low + 1e-12,
            "held bank {:.4} rad exceeds low-altitude limit {:.4} rad",
            bank,
            bank_max_low
        );
        // Still a hold: no replan happened.
        assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);
    }

    /// Subsequent calls (initialized state) also produce finite, bounded output.
    #[test]
    fn second_call_produces_finite_output() {
        let nav = test_nav(5000.0);
        let mut state = FnpagState::new(64.77_f64.to_radians());
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        // Prime the state
        let _ = fnpag_bank(&nav, &mut state, &data, &planet, 0.0);
        assert!(state.initialized);

        // Second call — exercises secant method path (past replan_period)
        let bank = fnpag_bank(&nav, &mut state, &data, &planet, 10.0);

        assert!(bank.is_finite(), "second-call bank not finite: {bank}");
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "second-call bank {:.4} rad outside [0, π]",
            bank
        );
    }

    /// A bound (elliptical) predicted exit state for the 3D-physics sensitivity
    /// tests: 110 km / shallow FPA, where the single-pass prediction captures
    /// (finite apoapsis) rather than escaping (the unbound sentinel) or crashing
    /// at BOTH equatorial and 60-deg latitude, and with/without planet rotation.
    fn bound_pred_state(lat: f64) -> PredState {
        PredState {
            r: PlanetConfig::mars().equatorial_radius + 110_000.0,
            lon: 0.0,
            lat,
            v: 5687.0,
            gamma: -0.055,
            psi: 0.6,
        }
    }

    /// The predictor must use inertial (absolute) velocity, not relative.
    /// For a prograde entry the inertial speed is higher with rotation, so the
    /// predicted exit apoapsis is higher. Asserted on the predictor directly:
    /// under the apoapsis metric the secant saturates on these shallow fixtures,
    /// so the physics is checked where it lives -- in `predict_exit_apoapsis`.
    #[test]
    fn predict_apoapsis_uses_inertial_velocity() {
        let data = test_sim_data();
        let exit_alt = data.final_conditions.altitude;
        let bank = 40.0_f64.to_radians();

        let planet_rotating = PlanetConfig::mars();
        let planet_static = PlanetConfig {
            omega: 0.0,
            ..PlanetConfig::mars()
        };

        let s = bound_pred_state(0.0);
        let apo_rot = predict_exit_apoapsis(s, bank, &planet_rotating, &data, exit_alt, 2.0, 1.0);
        let apo_stat = predict_exit_apoapsis(s, bank, &planet_static, &data, exit_alt, 2.0, 1.0);

        // Both must be bound (below the unbound sentinel), and rotation must
        // raise the apoapsis by a large, unambiguous margin.
        assert!(apo_rot < UNBOUND_APOAPSIS_RADIUS_M, "rotating exit escaped");
        assert!(apo_stat < UNBOUND_APOAPSIS_RADIUS_M, "static exit escaped");
        assert!(
            apo_rot - apo_stat > 1e5,
            "rotation should raise apoapsis: rot={apo_rot:.0} stat={apo_stat:.0}"
        );
    }

    /// J2 gravity depends on latitude: the predicted exit apoapsis must differ
    /// measurably between an equatorial and a high-latitude entry at the same
    /// speed/FPA. Asserted on the predictor directly (see note above).
    #[test]
    fn j2_sensitivity_with_latitude() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let exit_alt = data.final_conditions.altitude;
        let bank = 40.0_f64.to_radians();

        let apo_eq = predict_exit_apoapsis(
            bound_pred_state(0.0),
            bank,
            &planet,
            &data,
            exit_alt,
            2.0,
            1.0,
        );
        let apo_hl = predict_exit_apoapsis(
            bound_pred_state(60.0_f64.to_radians()),
            bank,
            &planet,
            &data,
            exit_alt,
            2.0,
            1.0,
        );

        assert!(
            apo_eq < UNBOUND_APOAPSIS_RADIUS_M,
            "equatorial exit escaped"
        );
        assert!(apo_hl < UNBOUND_APOAPSIS_RADIUS_M, "high-lat exit escaped");
        assert!(
            (apo_eq - apo_hl).abs() > 1e5,
            "J2 latitude effect too small: eq={apo_eq:.0} hl={apo_hl:.0}"
        );
    }

    // ── Proptest ─────────────────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For valid atmospheric entry conditions, FNPAG must always return a
            /// finite bank angle within [0, pi].
            #[test]
            fn output_always_finite_and_bounded(
                alt in 20_000.0..100_000.0_f64,
                vel in 3_000.0..6_000.0_f64,
                fpa in -0.15..0.0_f64,
                lat in -1.0..1.0_f64,
                psi in -3.0..3.0_f64,
            ) {
                let mut nav = test_nav(vel);
                let r = PlanetConfig::mars().equatorial_radius + alt;
                nav.position_estimated[0] = r;
                nav.position_estimated[2] = lat;
                nav.velocity_estimated[1] = fpa;
                nav.velocity_estimated[2] = psi;
                nav.density_guidance = 0.001;
                nav.dynamic_pressure_estimated = 0.5 * 0.001 * vel * vel;

                let mut state = FnpagState::new(64.77_f64.to_radians());
                let data = test_sim_data();
                let planet = PlanetConfig::mars();

                let bank = fnpag_bank(&nav, &mut state, &data, &planet, 0.0);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= 0.0 - 1e-10, "bank negative: {}", bank);
                prop_assert!(bank <= std::f64::consts::PI + 1e-10, "bank > pi: {}", bank);
            }
        }
    }
}
