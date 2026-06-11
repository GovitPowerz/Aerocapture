//! FNPAG -- Fully Numerical Predictor-corrector Aerocapture Guidance.
//!
//! Based on Ping Lu's algorithm (Journal of Guidance, Control, and Dynamics,
//! 2015). This is a modern predictor-corrector specifically designed for
//! aerocapture, using numerical forward prediction of the trajectory to
//! find the bank angle that achieves a target exit energy.
//!
//! Algorithm overview:
//! 1. Predict forward trajectory with current bank angle using 3D equations
//!    of motion (J2 gravity, planet rotation, onboard atmosphere model)
//! 2. Compute predicted exit orbital energy (inertial velocity)
//! 3. Use secant method to find the bank angle that achieves target energy
//! 4. Blend with equilibrium glide near atmosphere boundaries
//!
//! The predictor uses the same EOM as the main simulator but with onboard
//! atmosphere (no dispersions/winds) and zero lateral lift (roll sign unknown).
//! RK4 integration.
//!
//! The key insight vs FTC: FNPAG directly targets the exit orbital energy
//! rather than tracking a pre-computed reference trajectory. This makes it
//! inherently more robust to dispersions since it continuously re-plans.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::physics::gravity;

/// Altitude breakpoint (m) below which the tighter bank-angle limit applies.
const BANK_LIMIT_SWITCH_ALTITUDE_M: f64 = 50e3;

/// FNPAG persistent state (mutable runtime state only).
#[derive(Debug, Clone)]
pub struct FnpagState {
    /// Previous bank angle command (for secant method seeding)
    pub bank_prev: f64,
    /// Previous predicted exit energy (for secant method)
    pub energy_prev: f64,
    /// Whether predictor has been initialized
    pub initialized: bool,
    /// Sim time of the last forward-prediction replan (s)
    pub last_replan_time: f64,
}

impl FnpagState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            bank_prev: initial_bank,
            energy_prev: 0.0,
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
) -> [f64; 6] {
    let (altitude, _) = geodetic_from_spherical(s.r, s.lon, s.lat, planet);
    let rho = data
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

/// Predict exit energy by integrating 3D equations of motion forward.
///
/// Uses the same EOM as the main simulator (J2 gravity, planet rotation,
/// Coriolis/centrifugal) but with onboard atmosphere, no dispersions, no winds,
/// and zero lateral lift (sin_bank = 0). RK4 integration.
///
/// Integrates until atmosphere exit or crash.
fn predict_exit_energy(
    initial: PredState,
    bank_angle: f64,
    planet: &PlanetConfig,
    data: &SimData,
    exit_alt: f64,
    dt: f64,
) -> f64 {
    let req = planet.equatorial_radius;
    let max_steps = 2000;
    let mut s = initial;

    for _ in 0..max_steps {
        let alt = s.r - req;

        // Termination: crash
        if alt <= 0.0 {
            return 1e8;
        }
        // Termination: atmosphere exit (ascending)
        if alt >= exit_alt && s.gamma.sin() > 0.0 {
            return total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet);
        }

        // Classic RK4
        let k1 = pred_derivatives(&s, bank_angle, planet, data);

        let s2 = PredState {
            r: s.r + 0.5 * dt * k1[0],
            lon: s.lon + 0.5 * dt * k1[1],
            lat: s.lat + 0.5 * dt * k1[2],
            v: s.v + 0.5 * dt * k1[3],
            gamma: s.gamma + 0.5 * dt * k1[4],
            psi: s.psi + 0.5 * dt * k1[5],
        };
        let k2 = pred_derivatives(&s2, bank_angle, planet, data);

        let s3 = PredState {
            r: s.r + 0.5 * dt * k2[0],
            lon: s.lon + 0.5 * dt * k2[1],
            lat: s.lat + 0.5 * dt * k2[2],
            v: s.v + 0.5 * dt * k2[3],
            gamma: s.gamma + 0.5 * dt * k2[4],
            psi: s.psi + 0.5 * dt * k2[5],
        };
        let k3 = pred_derivatives(&s3, bank_angle, planet, data);

        let s4 = PredState {
            r: s.r + dt * k3[0],
            lon: s.lon + dt * k3[1],
            lat: s.lat + dt * k3[2],
            v: s.v + dt * k3[3],
            gamma: s.gamma + dt * k3[4],
            psi: s.psi + dt * k3[5],
        };
        let k4 = pred_derivatives(&s4, bank_angle, planet, data);

        s.r += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        s.lon += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        s.lat += dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
        s.v += dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
        s.gamma += dt / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]);
        s.psi += dt / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]);

        // Safety: velocity can't go negative
        if s.v <= 0.0 {
            return 1e8;
        }
    }

    // Timeout -- didn't exit atmosphere
    total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet)
}

/// Compute FNPAG bank angle command.
///
/// Uses secant method over forward trajectory predictions to find
/// the bank angle that achieves the target exit energy.
///
/// Returns bank angle magnitude in radians.
pub fn fnpag_bank(
    nav: &NavigationOutput,
    state: &mut FnpagState,
    data: &SimData,
    planet: &PlanetConfig,
    sim_time: f64,
) -> f64 {
    let mu = planet.mu;

    // Target exit energy: E = -mu / (2a) for the target orbit
    let target_sma =
        (data.target_orbit.apoapsis + data.target_orbit.periapsis) / 2.0 + planet.equatorial_radius;
    let target_energy = -mu / (2.0 * target_sma);

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

    let params = &data.guidance.fnpag;

    // Bank angle limits from params
    let bank_min = params.bank_min_deg.to_radians();
    let bank_max = if altitude < BANK_LIMIT_SWITCH_ALTITUDE_M {
        params.bank_max_low_deg.to_radians()
    } else {
        params.bank_max_high_deg.to_radians()
    };

    // Initialize with a bisection-style search over a wide bracket
    if !state.initialized {
        let bank1 = 40.0_f64.to_radians();
        let bank2 = 90.0_f64.to_radians();

        let e1 = predict_exit_energy(current, bank1, planet, data, exit_alt, params.prediction_dt);
        let e2 = predict_exit_energy(current, bank2, planet, data, exit_alt, params.prediction_dt);

        let err1 = e1 - target_energy;
        let err2 = e2 - target_energy;

        state.initialized = true;
        state.last_replan_time = sim_time;

        // Use the one closer to target
        if err1.abs() < err2.abs() {
            state.bank_prev = bank1;
            state.energy_prev = err1;
            return bank1;
        } else {
            state.bank_prev = bank2;
            state.energy_prev = err2;
            return bank2;
        }
    }

    // Replan throttle: between replans, hold the previous command. The
    // command evolves slowly mid-pass, and each replan costs up to 5 full
    // forward integrations.
    if sim_time - state.last_replan_time < params.replan_period {
        return state.bank_prev.abs();
    }

    // Secant method iterations
    state.last_replan_time = sim_time;
    let mut bank_k = state.bank_prev;
    let mut err_k = state.energy_prev;

    // Perturb for secant step (small delta to estimate local gradient)
    let delta_bank = 3.0_f64.to_radians();
    let mut bank_trial = (bank_k + delta_bank).clamp(bank_min, bank_max);

    let mut best_bank = bank_k;
    let mut best_err = err_k.abs();
    // Signed error of best_bank, evaluated at the CURRENT state. None while
    // best_bank is still the carried-over bank_prev (whose err_k is stale —
    // it was evaluated at the previous replan's state).
    let mut best_err_signed: Option<f64> = None;

    for _iter in 0..5 {
        let e_trial = predict_exit_energy(
            current,
            bank_trial,
            planet,
            data,
            exit_alt,
            params.prediction_dt,
        );
        let err_trial = e_trial - target_energy;

        // Track best solution
        if err_trial.abs() < best_err {
            best_err = err_trial.abs();
            best_bank = bank_trial;
            best_err_signed = Some(err_trial);
        }

        // Check convergence
        if err_trial.abs() < params.energy_tol {
            state.bank_prev = bank_trial;
            state.energy_prev = err_trial;
            return bank_trial.clamp(bank_min, bank_max);
        }

        // Secant update
        let d_err = err_trial - err_k;
        if d_err.abs() < 1e-20 {
            break;
        }

        let bank_new = bank_trial - err_trial * (bank_trial - bank_k) / d_err;

        // Update for next iteration
        bank_k = bank_trial;
        err_k = err_trial;
        bank_trial = bank_new.clamp(bank_min, bank_max);

        // Stall: the next trial equals the point just evaluated (typically
        // both clamped at the same bound) — re-predicting it is pure waste.
        if (bank_trial - bank_k).abs() < 1e-12 {
            break;
        }
    }

    // Use best result found. Its signed error was already computed in the
    // loop; re-predict only when no trial beat the carried-over error, so the
    // next replan's secant doesn't reuse a stale residual.
    state.bank_prev = best_bank;
    state.energy_prev = match best_err_signed {
        Some(err) => err,
        None => {
            let e_final = predict_exit_energy(
                current,
                best_bank,
                planet,
                data,
                exit_alt,
                params.prediction_dt,
            );
            e_final - target_energy
        }
    };

    best_bank.clamp(bank_min, bank_max)
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
        // bank_prev should now be one of the two bisection candidates (40° or 90°)
        let bank40 = 40.0_f64.to_radians();
        let bank90 = 90.0_f64.to_radians();
        assert!(
            (state.bank_prev - bank40).abs() < 1e-9 || (state.bank_prev - bank90).abs() < 1e-9,
            "bank_prev {:.4} rad should be either 40° or 90° after init",
            state.bank_prev
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
        let energy_prev0 = state.energy_prev;
        assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);

        // 1 s later with a substantially different nav state: must hold the
        // command and leave the secant state untouched.
        let nav_later = test_nav(4000.0);
        let bank_held = fnpag_bank(&nav_later, &mut state, &data, &planet, 1.0);
        assert_relative_eq!(bank_held, bank0, epsilon = 1e-12);
        assert_relative_eq!(state.energy_prev, energy_prev0, epsilon = 1e-12);
        assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);

        // Past the period: replans (timestamp advances).
        let _ = fnpag_bank(&nav_later, &mut state, &data, &planet, 2.5);
        assert_relative_eq!(state.last_replan_time, 2.5, epsilon = 1e-12);
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

    /// The predictor must use inertial (absolute) velocity for exit energy,
    /// not relative velocity. For a planet with nonzero omega, these differ.
    #[test]
    fn exit_energy_uses_inertial_velocity() {
        // Two predictions: one on a planet with rotation, one without.
        // With rotation, the inertial velocity is higher (prograde entry),
        // so the predicted exit energy should be higher (less negative).
        //
        // Use a high-altitude shallow-FPA state so the predictor can resolve
        // trajectories that exit the atmosphere (not crash).
        let mut nav = test_nav(5687.0);
        nav.position_estimated[0] = PlanetConfig::mars().equatorial_radius + 100_000.0;
        nav.velocity_estimated[1] = -0.05; // shallow FPA ~-2.9 deg
        let data = test_sim_data();

        let planet_rotating = PlanetConfig::mars();
        let planet_static = PlanetConfig {
            omega: 0.0,
            ..PlanetConfig::mars()
        };

        let mut state_rot = FnpagState::new(64.77_f64.to_radians());
        let mut state_stat = FnpagState::new(64.77_f64.to_radians());

        // First call initializes (picks from fixed candidates) -- may be identical
        let _ = fnpag_bank(&nav, &mut state_rot, &data, &planet_rotating, 0.0);
        let _ = fnpag_bank(&nav, &mut state_stat, &data, &planet_static, 0.0);

        // Second call exercises the secant method where 3D effects differentiate
        let bank_rot = fnpag_bank(&nav, &mut state_rot, &data, &planet_rotating, 10.0);
        let bank_stat = fnpag_bank(&nav, &mut state_stat, &data, &planet_static, 10.0);

        assert!(bank_rot.is_finite(), "rotating bank not finite: {bank_rot}");
        assert!(bank_stat.is_finite(), "static bank not finite: {bank_stat}");

        // The bank angles should differ because the energy model differs
        assert!(
            (bank_rot - bank_stat).abs() > 1e-6,
            "rotation should affect bank angle: rot={bank_rot:.6} stat={bank_stat:.6}"
        );
    }

    /// J2 gravity depends on latitude. The predictor should produce different
    /// bank angles for high-latitude vs equatorial entries (same speed/FPA).
    #[test]
    fn j2_sensitivity_with_latitude() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        // Use high-altitude shallow-FPA state so trajectories exit (not crash)
        // Equatorial entry (lat = 0)
        let mut nav_equator = test_nav(5687.0);
        nav_equator.position_estimated[0] = PlanetConfig::mars().equatorial_radius + 100_000.0;
        nav_equator.velocity_estimated[1] = -0.05; // shallow FPA

        // High-latitude entry (lat = 60 deg)
        let mut nav_high_lat = nav_equator;
        nav_high_lat.position_estimated[2] = 60.0_f64.to_radians();

        let mut state_eq = FnpagState::new(64.77_f64.to_radians());
        let mut state_hl = FnpagState::new(64.77_f64.to_radians());

        // First call initializes (picks from fixed candidates) -- may be identical
        let _ = fnpag_bank(&nav_equator, &mut state_eq, &data, &planet, 0.0);
        let _ = fnpag_bank(&nav_high_lat, &mut state_hl, &data, &planet, 0.0);

        // Second call exercises the secant method where 3D effects differentiate
        let bank_eq = fnpag_bank(&nav_equator, &mut state_eq, &data, &planet, 10.0);
        let bank_hl = fnpag_bank(&nav_high_lat, &mut state_hl, &data, &planet, 10.0);

        assert!(bank_eq.is_finite(), "equatorial bank not finite");
        assert!(bank_hl.is_finite(), "high-lat bank not finite");

        // J2 + 3D effects should produce measurably different bank commands
        assert!(
            (bank_eq - bank_hl).abs() > 1e-4,
            "J2 latitude effect too small: eq={bank_eq:.6} hl={bank_hl:.6}"
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
