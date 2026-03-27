//! FNPAG — Fully Numerical Predictor-corrector Aerocapture Guidance.
//!
//! Based on Ping Lu's algorithm (Journal of Guidance, Control, and Dynamics,
//! 2015). This is a modern predictor-corrector specifically designed for
//! aerocapture, using numerical forward prediction of the trajectory to
//! find the bank angle that achieves a target exit energy.
//!
//! Algorithm overview:
//! 1. Predict forward trajectory with current bank angle using simplified
//!    equations of motion (no J2, constant bank, onboard atmosphere model)
//! 2. Compute predicted exit energy
//! 3. Use secant method to find the bank angle that achieves target energy
//! 4. Blend with equilibrium glide near atmosphere boundaries
//!
//! The key insight vs FTC: FNPAG directly targets the exit orbital energy
//! rather than tracking a pre-computed reference trajectory. This makes it
//! inherently more robust to dispersions since it continuously re-plans.

use crate::config::Planet;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;

/// FNPAG persistent state (mutable runtime state only).
#[derive(Debug, Clone)]
pub struct FnpagState {
    /// Previous bank angle command (for secant method seeding)
    pub bank_prev: f64,
    /// Previous predicted exit energy (for secant method)
    pub energy_prev: f64,
    /// Whether predictor has been initialized
    pub initialized: bool,
}

impl FnpagState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            bank_prev: initial_bank,
            energy_prev: 0.0,
            initialized: false,
        }
    }
}

/// Simplified state for forward prediction.
#[derive(Clone, Copy)]
struct PredState {
    r: f64,     // radius (m)
    v: f64,     // velocity (m/s)
    gamma: f64, // flight path angle (rad)
}

/// Predict exit energy by integrating simplified equations of motion forward.
///
/// Uses a planar, non-rotating model with exponential atmosphere:
///   dr/dt = V sin(gamma)
///   dV/dt = -D/m - g sin(gamma)
///   dgamma/dt = (L cos(bank)/m - (g - V²/r) cos(gamma)) / V
///
/// Integrates until atmosphere exit or crash.
fn predict_exit_energy(
    initial: PredState,
    bank_angle: f64,
    planet: &Planet,
    data: &SimData,
    exit_alt: f64,
    dt: f64,
) -> f64 {
    let mu = planet.mu();
    let req = planet.equatorial_radius();
    let max_steps = 2000;
    let cos_bank = bank_angle.cos();

    let sref = data.capsule.reference_area;
    let mass = data.capsule.mass;

    let mut s = initial;

    for _ in 0..max_steps {
        let alt = s.r - req;

        // Termination: crash or atmosphere exit
        if alt <= 0.0 {
            return 1e8; // crash penalty — very high energy
        }
        if alt >= exit_alt && s.gamma.sin() > 0.0 {
            // Exited atmosphere — compute orbital energy
            let energy = s.v * s.v / 2.0 - mu / s.r;
            return energy;
        }

        // Atmospheric density (using the simulator's tabulated model)
        let rho = data.atmosphere_onboard.density_at(alt, &data.atmosphere);

        // Aero forces
        let cx = data.aero.interpolate_cx(data.entry.initial_aoa);
        let cz = data.aero.interpolate_cz(data.entry.initial_aoa).abs();
        let q = 0.5 * rho * s.v * s.v;
        let drag = q * sref * cx / mass;
        let lift = q * sref * cz / mass;

        // Gravity
        let g = mu / (s.r * s.r);

        // Derivatives (planar, non-rotating)
        let sin_g = s.gamma.sin();
        let cos_g = s.gamma.cos();

        let dr = s.v * sin_g;
        let dv = -drag - g * sin_g;
        let dgamma = if s.v.abs() > 1.0 {
            (lift * cos_bank - (g - s.v * s.v / s.r) * cos_g) / s.v
        } else {
            0.0
        };

        // Euler integration (RK4 would be better but this is fast enough for prediction)
        s.r += dr * dt;
        s.v += dv * dt;
        s.gamma += dgamma * dt;

        // Safety: velocity can't go negative
        if s.v <= 0.0 {
            return 1e8;
        }
    }

    // Timeout — didn't exit atmosphere
    s.v * s.v / 2.0 - mu / s.r
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
    planet: &Planet,
) -> f64 {
    let mu = planet.mu();

    // Target exit energy: E = -mu / (2a) for the target orbit
    let target_sma = (data.target_orbit.apoapsis + data.target_orbit.periapsis) / 2.0
        + planet.equatorial_radius();
    let target_energy = -mu / (2.0 * target_sma);

    let exit_alt = data.final_conditions.altitude;

    // Current state for prediction
    let current = PredState {
        r: nav.position_estimated[0],
        v: nav.velocity_estimated[0],
        gamma: nav.velocity_estimated[1],
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
    let bank_max = if altitude < 50e3 {
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

    // Secant method iterations
    let mut bank_k = state.bank_prev;
    let mut err_k = state.energy_prev;

    // Perturb for secant step (small delta to estimate local gradient)
    let delta_bank = 3.0_f64.to_radians();
    let mut bank_trial = (bank_k + delta_bank).clamp(bank_min, bank_max);

    let mut best_bank = bank_k;
    let mut best_err = err_k.abs();

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
    }

    // Use best result found
    state.bank_prev = best_bank;
    let e_final = predict_exit_energy(
        current,
        best_bank,
        planet,
        data,
        exit_alt,
        params.prediction_dt,
    );
    state.energy_prev = e_final - target_energy;

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
            atmosphere: AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            },
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
        }
    }

    // ── Deterministic tests ──────────────────────────────────────────────────

    /// When the spacecraft is above the sensible atmosphere (density < 1e-10),
    /// FNPAG must return the previous bank angle unchanged.
    #[test]
    fn low_density_returns_previous_bank() {
        // Place spacecraft at 200 km — exponential tail gives ≈9e-12 kg/m³ < 1e-10
        let mut nav = test_nav(5000.0);
        nav.position_estimated[0] = Planet::Mars.equatorial_radius() + 200_000.0;

        let prev_bank = 55.0_f64.to_radians();
        let mut state = FnpagState::new(prev_bank);
        state.initialized = true; // doesn't matter — early exit fires first

        let data = test_sim_data();
        let planet = Planet::Mars;

        let bank = fnpag_bank(&nav, &mut state, &data, &planet);

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
        let planet = Planet::Mars;

        let _ = fnpag_bank(&nav, &mut state, &data, &planet);

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
        let planet = Planet::Mars;

        let bank = fnpag_bank(&nav, &mut state, &data, &planet);

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

    /// Subsequent calls (initialized state) also produce finite, bounded output.
    #[test]
    fn second_call_produces_finite_output() {
        let nav = test_nav(5000.0);
        let mut state = FnpagState::new(64.77_f64.to_radians());
        let data = test_sim_data();
        let planet = Planet::Mars;

        // Prime the state
        let _ = fnpag_bank(&nav, &mut state, &data, &planet);
        assert!(state.initialized);

        // Second call — exercises secant method path
        let bank = fnpag_bank(&nav, &mut state, &data, &planet);

        assert!(bank.is_finite(), "second-call bank not finite: {bank}");
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "second-call bank {:.4} rad outside [0, π]",
            bank
        );
    }

    // ── Proptest ─────────────────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For valid atmospheric entry conditions, FNPAG must always return a
            /// finite bank angle within [0, π].
            #[test]
            fn output_always_finite_and_bounded(
                alt in 20_000.0..100_000.0_f64,
                vel in 3_000.0..6_000.0_f64,
                fpa in -0.15..0.0_f64,
                rho in 1e-5..0.01_f64,
            ) {
                let mut nav = test_nav(vel);
                let r = Planet::Mars.equatorial_radius() + alt;
                nav.position_estimated[0] = r;
                nav.velocity_estimated[1] = fpa;
                nav.density_guidance = rho;
                nav.dynamic_pressure_estimated = 0.5 * rho * vel * vel;

                let mut state = FnpagState::new(64.77_f64.to_radians());
                let data = test_sim_data();
                let planet = Planet::Mars;

                let bank = fnpag_bank(&nav, &mut state, &data, &planet);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= 0.0 - 1e-10, "bank negative: {}", bank);
                prop_assert!(bank <= std::f64::consts::PI + 1e-10, "bank > π: {}", bank);
            }
        }
    }
}
