//! Aerodynamic force computation.
//!
//! Matches Fortran faeros.f.

use crate::data::aerodynamics::AeroTables;
use crate::data::capsule::Capsule;

/// Aerodynamic forces result
pub struct AeroForces {
    pub drag: f64,    // N (along velocity, opposing motion)
    pub lift: f64,    // N (perpendicular to velocity)
    pub heat_flux: f64, // W/m^2
}

/// Compute aerodynamic forces given flight conditions.
///
/// - `density`: atmospheric density (kg/m^3)
/// - `velocity`: relative velocity (m/s)
/// - `alpha`: angle of attack (radians)
/// - `cx_bias`: drag coefficient bias (fractional, for MC)
/// - `cz_bias`: lift coefficient bias (fractional, for MC)
pub fn aero_forces(
    aero: &AeroTables,
    capsule: &Capsule,
    density: f64,
    velocity: f64,
    alpha: f64,
    cx_bias: f64,
    cz_bias: f64,
) -> AeroForces {
    let cx = aero.interpolate_cx(alpha) * (1.0 + cx_bias);
    let cz = aero.interpolate_cz(alpha) * (1.0 + cz_bias);

    let q = 0.5 * density * velocity * velocity; // dynamic pressure

    let drag = q * capsule.reference_area * cx;
    let lift = q * capsule.reference_area * cz;

    // Convective heat flux: q_dot = Cq * sqrt(rho) * V^3
    let heat_flux = capsule.cq * density.sqrt() * velocity.powi(3);

    AeroForces {
        drag,
        lift,
        heat_flux,
    }
}
