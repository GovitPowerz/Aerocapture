//! Equilibrium Glide guidance.
//!
//! Computes bank angle to maintain equilibrium between gravitational,
//! centrifugal, and aerodynamic lift forces. The vehicle "glides" along
//! a path where the vertical acceleration is zero (d²h/dt² ≈ 0).
//!
//! The equilibrium condition gives:
//!   cos(bank) = (g - V²/r) * m / L
//!
//! where g is local gravity, V is velocity, r is radius, m is mass,
//! and L is lift force. This is the simplest closed-form aerocapture
//! guidance law — no reference trajectory or prediction needed.
//!
//! For aerocapture, pure equilibrium glide tends to dissipate too much
//! energy (crashes) or too little (skip-out). We augment it with a
//! radial velocity feedback term to dampen altitude oscillations and
//! a velocity-dependent bias to control energy dissipation rate.

use crate::config::Planet;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Compute equilibrium glide bank angle.
///
/// Returns the bank angle magnitude (always positive) in radians.
/// The caller handles roll sign via lateral guidance.
pub fn equilibrium_glide_bank(nav: &NavigationOutput, data: &SimData, planet: &Planet) -> f64 {
    let r = nav.positn[0];
    let v = nav.vitesn[0];

    // Local gravity (simplified — use mu/r² for the dominant term)
    let mu = planet.mu();
    let g = mu / (r * r);

    // Centrifugal acceleration
    let v2_over_r = v * v / r;

    // Lift force per unit mass: L/m = 0.5 * rho * V² * S * Cz / m
    let (altitude, _) = geodetic_from_spherical(r, nav.positn[1], nav.positn[2], planet);
    let rho = data.atmosphere.density_at(altitude);
    let cz = nav.coefan[1]; // lift coefficient from navigation
    let sref = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let lift_accel = 0.5 * rho * v * v * sref * cz.abs() / mass;

    if lift_accel.abs() < 1e-10 {
        // No lift available — hold moderate bank angle
        return 60.0_f64.to_radians();
    }

    // Base equilibrium: cos(bank) = (g - V²/r) / lift_accel
    let cos_eq = (g - v2_over_r) / lift_accel;

    let params = &data.guidance.eq_glide;

    // Radial velocity damping: if sinking, reduce bank (more lift-up);
    // if rising, increase bank (more lift-down toward equilibrium)
    let hdot = v * nav.vitesn[1].sin();
    let k_hdot = params.k_hdot_scale / v.max(100.0);
    let cos_hdot_correction = -k_hdot * hdot;

    // Velocity-dependent bias: at hyperbolic velocities, we want more drag
    // (higher bank angle) to dissipate energy. As velocity drops toward
    // circular, reduce bank to stop dissipation.
    let v_circular = (mu / r).sqrt();
    let v_ratio = v / v_circular;
    let velocity_bias = if v_ratio > params.v_ratio_threshold {
        -params.velocity_bias_high * (v_ratio - params.v_ratio_threshold).min(1.0)
    } else {
        params.velocity_bias_low * (params.v_ratio_threshold - v_ratio).min(0.5)
    };

    // Altitude-dependent correction: if too low, bias toward lift-up
    let alt_km = altitude / 1e3;
    let alt_bias = if alt_km < params.alt_bias_threshold {
        params.velocity_bias_low * (1.0 - alt_km / params.alt_bias_threshold)
    } else {
        0.0
    };

    let cos_bank = (cos_eq + cos_hdot_correction + velocity_bias + alt_bias)
        .clamp(params.cos_bank_min, params.cos_bank_max);
    let bank = cos_bank.acos();

    // Safety clamp: never go below 15° (skip-out) or above 120° (crash risk)
    bank.clamp(15.0_f64.to_radians(), 120.0_f64.to_radians())
}
