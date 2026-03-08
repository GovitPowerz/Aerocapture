//! FNPAG — Fully Numerical Predictor-corrector Aerocapture Guidance.
//!
//! Based on Ping Lu's algorithm (Journal of Guidance, Control, and Dynamics,
//! 2015). This is a modern predictor-corrector specifically designed for
//! aerocapture, using numerical forward prediction of the trajectory to
//! find the bank angle that achieves a target exit energy.
//!
//! Algorithm overview:
//! 1. Predict forward trajectory with current bank angle using simplified
//!    equations of motion (no J2, constant bank, analytical density model)
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
        let rho = data.atmosphere.density_at(alt);

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
    let energy = s.v * s.v / 2.0 - mu / s.r;
    energy
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
        r: nav.positn[0],
        v: nav.vitesn[0],
        gamma: nav.vitesn[1],
    };

    // Check if we're in the sensible atmosphere (density > threshold)
    let (altitude, _) = geodetic_from_spherical(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        planet,
    );
    let rho = data.atmosphere.density_at(altitude);
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
        let e_trial = predict_exit_energy(current, bank_trial, planet, data, exit_alt, params.prediction_dt);
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
    let e_final = predict_exit_energy(current, best_bank, planet, data, exit_alt, params.prediction_dt);
    state.energy_prev = e_final - target_energy;

    best_bank.clamp(bank_min, bank_max)
}
