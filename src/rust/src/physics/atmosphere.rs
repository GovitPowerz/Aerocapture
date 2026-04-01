//! Atmospheric density computation.
//!
//! Wraps the atmosphere data model for use in the simulation loop.

use crate::data::atmosphere::AtmosphereModel;

/// Compute atmospheric density at a given geodetic altitude.
///
/// Applies optional density bias (MC static) and density perturbation (GM time-varying).
pub fn density(
    atm: &AtmosphereModel,
    altitude: f64,
    density_bias: f64,
    density_perturbation: f64,
) -> f64 {
    let rho = atm.density_at(altitude);
    rho * (1.0 + density_bias) * (1.0 + density_perturbation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atmosphere::DensityProfile;
    use approx::assert_abs_diff_eq;

    /// Build a small 3-point atmosphere table for testing.
    fn test_atm() -> AtmosphereModel {
        AtmosphereModel {
            n_points: 3,
            altitudes: vec![10_000.0, 20_000.0, 30_000.0],
            densities: vec![1.0, 0.5, 0.1],
            ref_density: 0.1,
            scale_factor: 1e-4,
            ref_altitude: 30_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        }
    }

    #[test]
    fn exact_table_hit() {
        let atm = test_atm();
        assert_abs_diff_eq!(atm.density_at(10_000.0), 1.0);
        assert_abs_diff_eq!(atm.density_at(20_000.0), 0.5);
        assert_abs_diff_eq!(atm.density_at(30_000.0), 0.1);
    }

    #[test]
    fn interpolation_midpoint() {
        let atm = test_atm();
        assert_abs_diff_eq!(atm.density_at(15_000.0), 0.75, epsilon = 1e-12);
    }

    #[test]
    fn below_table_clamps() {
        let atm = test_atm();
        assert_abs_diff_eq!(atm.density_at(0.0), 1.0);
        assert_abs_diff_eq!(atm.density_at(-5_000.0), 1.0);
    }

    #[test]
    fn above_table_uses_exponential() {
        let atm = test_atm();
        let alt = 40_000.0;
        let expected = 0.1_f64 * (-1e-4_f64 * (alt - 30_000.0)).exp();
        assert_abs_diff_eq!(atm.density_at(alt), expected, epsilon = 1e-15);
    }

    #[test]
    fn density_bias_positive() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, 0.1, 0.0);
        assert_abs_diff_eq!(rho_biased, rho_nominal * 1.1, epsilon = 1e-12);
    }

    #[test]
    fn density_bias_zero_is_nominal() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, 0.0, 0.0);
        assert_abs_diff_eq!(rho_biased, rho_nominal);
    }

    #[test]
    fn density_bias_negative() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, -0.2, 0.0);
        assert_abs_diff_eq!(rho_biased, rho_nominal * 0.8, epsilon = 1e-12);
    }

    #[test]
    fn density_with_perturbation() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho = density(&atm, 15_000.0, 0.1, 0.05);
        // (1 + 0.1) * (1 + 0.05) = 1.155
        assert_abs_diff_eq!(rho, rho_nominal * 1.155, epsilon = 1e-12);
    }

    #[test]
    fn density_bias_and_perturbation_are_multiplicative() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_bias_only = density(&atm, 15_000.0, 0.2, 0.0);
        let rho_pert_only = density(&atm, 15_000.0, 0.0, 0.15);
        let rho_both = density(&atm, 15_000.0, 0.2, 0.15);
        // Multiplicative: rho_both = rho_nominal * 1.2 * 1.15, not rho_nominal * 1.35
        assert_abs_diff_eq!(rho_both, rho_nominal * 1.2 * 1.15, epsilon = 1e-12);
        assert!(
            (rho_both - rho_nominal * 1.35).abs() > 1e-6,
            "bias and perturbation must be multiplicative, not additive"
        );
    }
}
