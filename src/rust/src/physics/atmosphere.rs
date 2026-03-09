//! Atmospheric density computation.
//!
//! Wraps the atmosphere data model for use in the simulation loop.
//! Matches Fortran fatmos.f.

use crate::data::atmosphere::AtmosphereModel;

/// Compute atmospheric density at a given geodetic altitude.
///
/// Applies optional density bias for Monte Carlo dispersions.
#[allow(dead_code)]
pub fn density(atm: &AtmosphereModel, altitude: f64, density_bias: f64) -> f64 {
    let rho = atm.density_at(altitude);
    rho * (1.0 + density_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::atmosphere::DensityProfile;

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
        assert_eq!(atm.density_at(10_000.0), 1.0);
        assert_eq!(atm.density_at(20_000.0), 0.5);
        assert_eq!(atm.density_at(30_000.0), 0.1);
    }

    #[test]
    fn interpolation_midpoint() {
        let atm = test_atm();
        // Midpoint between 10km (rho=1.0) and 20km (rho=0.5) -> 0.75
        let rho = atm.density_at(15_000.0);
        assert!((rho - 0.75).abs() < 1e-12, "expected 0.75, got {rho}");
    }

    #[test]
    fn below_table_clamps() {
        let atm = test_atm();
        assert_eq!(atm.density_at(0.0), 1.0);
        assert_eq!(atm.density_at(-5_000.0), 1.0);
    }

    #[test]
    fn above_table_uses_exponential() {
        let atm = test_atm();
        let alt = 40_000.0;
        let expected = 0.1_f64 * (-1e-4_f64 * (alt - 30_000.0)).exp();
        let rho = atm.density_at(alt);
        assert!(
            (rho - expected).abs() < 1e-15,
            "expected {expected}, got {rho}"
        );
    }

    #[test]
    fn density_bias_positive() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, 0.1);
        assert!(
            (rho_biased - rho_nominal * 1.1).abs() < 1e-12,
            "bias=0.1 should multiply by 1.1"
        );
    }

    #[test]
    fn density_bias_zero_is_nominal() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, 0.0);
        assert_eq!(rho_biased, rho_nominal);
    }

    #[test]
    fn density_bias_negative() {
        let atm = test_atm();
        let rho_nominal = atm.density_at(15_000.0);
        let rho_biased = density(&atm, 15_000.0, -0.2);
        assert!(
            (rho_biased - rho_nominal * 0.8).abs() < 1e-12,
            "bias=-0.2 should multiply by 0.8"
        );
    }
}
