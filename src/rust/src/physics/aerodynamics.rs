//! Aerodynamic force computation.
//!
//! Matches Fortran faeros.f.

use crate::data::aerodynamics::AeroTables;
use crate::data::capsule::Capsule;

/// Aerodynamic forces result
#[allow(dead_code)]
pub struct AeroForces {
    pub drag: f64,      // N (along velocity, opposing motion)
    pub lift: f64,      // N (perpendicular to velocity)
    pub heat_flux: f64, // W/m^2
}

/// Compute aerodynamic forces given flight conditions.
///
/// - `density`: atmospheric density (kg/m^3)
/// - `velocity`: relative velocity (m/s)
/// - `alpha`: angle of attack (radians)
/// - `cx_bias`: drag coefficient bias (fractional, for MC)
/// - `cz_bias`: lift coefficient bias (fractional, for MC)
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TimePeriods;
    use approx::assert_abs_diff_eq;

    /// Build a simple 3-point AeroTables fixture.
    /// Incidence: [0, 10, 20] deg → [0.0, 0.1745…, 0.3491…] rad
    /// Cx:       [1.5, 1.6, 1.7]
    /// Cz:       [0.0, -0.2, -0.4]
    fn make_aero() -> AeroTables {
        let deg2rad = std::f64::consts::PI / 180.0;
        AeroTables {
            equilibrium_aoa: 10.0 * deg2rad,
            n_points: 3,
            incidence: vec![0.0 * deg2rad, 10.0 * deg2rad, 20.0 * deg2rad],
            cx: vec![1.5, 1.6, 1.7],
            cz: vec![0.0, -0.2, -0.4],
            nominal_cx: 1.6,
            nominal_cz: -0.2,
            nominal_finesse: -0.2 / 1.6,
            ballistic_coeff: 0.0,
        }
    }

    fn make_capsule() -> Capsule {
        Capsule {
            mass: 2400.0,
            reference_area: 4.52,
            cq: 1.75e-4,
            max_bank_rate: 0.1,
            periods: TimePeriods::default(),
        }
    }

    #[test]
    fn zero_velocity_zero_forces() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa;
        let f = aero_forces(&aero, &cap, 0.01, 0.0, alpha, 0.0, 0.0);
        assert_eq!(f.drag, 0.0);
        assert_eq!(f.lift, 0.0);
        assert_eq!(f.heat_flux, 0.0);
    }

    #[test]
    fn drag_is_cx_q_s() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa; // Cx = 1.6
        let rho = 0.02;
        let v = 5000.0;
        let f = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);

        let q = 0.5 * rho * v * v;
        let expected_drag = q * cap.reference_area * 1.6;
        assert_abs_diff_eq!(f.drag, expected_drag, epsilon = 1e-6);
    }

    #[test]
    fn lift_is_cz_q_s() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa; // Cz = -0.2
        let rho = 0.02;
        let v = 5000.0;
        let f = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);

        let q = 0.5 * rho * v * v;
        let expected_lift = q * cap.reference_area * (-0.2);
        assert_abs_diff_eq!(f.lift, expected_lift, epsilon = 1e-6);
    }

    #[test]
    fn cx_bias_scales_drag() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa;
        let rho = 0.02;
        let v = 5000.0;

        let f_nom = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);
        let f_biased = aero_forces(&aero, &cap, rho, v, alpha, 0.1, 0.0);

        assert_abs_diff_eq!(f_biased.drag, f_nom.drag * 1.1, epsilon = 1e-6);
    }

    #[test]
    fn cz_bias_scales_lift() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa;
        let rho = 0.02;
        let v = 5000.0;

        let f_nom = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);
        let f_biased = aero_forces(&aero, &cap, rho, v, alpha, 0.0, -0.15);

        assert_abs_diff_eq!(f_biased.lift, f_nom.lift * 0.85, epsilon = 1e-6);
    }

    #[test]
    fn heat_flux_formula() {
        let aero = make_aero();
        let cap = make_capsule();
        let alpha = aero.equilibrium_aoa;
        let rho = 0.02;
        let v = 5000.0;

        let f = aero_forces(&aero, &cap, rho, v, alpha, 0.0, 0.0);
        let expected = cap.cq * rho.sqrt() * v.powi(3);
        assert_abs_diff_eq!(f.heat_flux, expected, epsilon = 1e-6);
    }

    #[test]
    fn interpolation_at_boundary() {
        let aero = make_aero();
        let cap = make_capsule();
        let rho = 0.02;
        let v = 5000.0;
        // Alpha well below table min (0 deg) → should clamp to first Cx = 1.5
        let alpha_below = -0.5; // radians, ~-28.6 deg

        let f = aero_forces(&aero, &cap, rho, v, alpha_below, 0.0, 0.0);
        let q = 0.5 * rho * v * v;
        let expected_drag = q * cap.reference_area * 1.5; // first Cx value
        assert_abs_diff_eq!(f.drag, expected_drag, epsilon = 1e-6);
        // Lift should also clamp to first Cz = 0.0
        assert_abs_diff_eq!(f.lift, 0.0, epsilon = 1e-6);
    }
}
