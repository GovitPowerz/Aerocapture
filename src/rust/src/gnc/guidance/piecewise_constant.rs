//! Piecewise-constant bank angle guidance.
//!
//! Divides the orbital energy range into 10 uniform segments, each with
//! a constant bank angle in [-180deg, +180deg]. The bank angle sign is part
//! of the profile (negative = implicit roll reversal). No navigation
//! feedback, no lateral guidance — pure open-loop bank profile.
//!
//! GA-optimized to produce reference trajectories and corridor envelopes.

use crate::config::Planet;
use crate::data::guidance_params::PiecewiseConstantParams;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Number of segments in the piecewise-constant bank profile.
const N_SEGMENTS: usize = 10;

/// Compute piecewise-constant bank angle from current orbital energy.
///
/// Returns the **signed** bank angle in radians. Unlike other schemes
/// that return magnitude only (with roll sign applied by lateral guidance),
/// piecewise_constant encodes the sign directly in the bank profile.
pub fn piecewise_constant_bank(
    nav: &NavigationOutput,
    params: &PiecewiseConstantParams,
    planet: &Planet,
) -> f64 {
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    segment_bank_angle(energy, params)
}

/// Pure lookup: energy -> segment -> bank angle.
/// Exposed for unit testing without needing a full NavigationOutput.
pub fn segment_bank_angle(energy: f64, params: &PiecewiseConstantParams) -> f64 {
    let e_min = params.energy_min;
    let e_max = params.energy_max;

    if e_max <= e_min {
        return params.bank_angles[0];
    }

    // Segment 0 = highest energy (entry), segment 9 = lowest energy (deep capture)
    // Energy DECREASES during flight, so segment index increases as energy drops
    let frac = (e_max - energy) / (e_max - e_min);
    let seg = (frac * N_SEGMENTS as f64).floor() as i64;
    let seg = seg.clamp(0, (N_SEGMENTS - 1) as i64) as usize;

    params.bank_angles[seg]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn test_params() -> PiecewiseConstantParams {
        PiecewiseConstantParams {
            bank_angles: [
                60.0_f64.to_radians(),
                50.0_f64.to_radians(),
                40.0_f64.to_radians(),
                30.0_f64.to_radians(),
                20.0_f64.to_radians(),
                -20.0_f64.to_radians(),
                -30.0_f64.to_radians(),
                -40.0_f64.to_radians(),
                -50.0_f64.to_radians(),
                -60.0_f64.to_radians(),
            ],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        }
    }

    #[test]
    fn entry_energy_gives_segment_0() {
        let params = test_params();
        let bank = segment_bank_angle(4.5e6, &params);
        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn deep_capture_gives_last_segment() {
        let params = test_params();
        let bank = segment_bank_angle(-5.5e6, &params);
        assert_relative_eq!(bank, -60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn mid_energy_gives_middle_segment() {
        let params = test_params();
        // frac = (5.0e6 - (-0.5e6)) / (5.0e6 - (-6.0e6)) = 5.5e6 / 11.0e6 = 0.5
        // seg = floor(0.5 * 10) = 5
        let bank = segment_bank_angle(-0.5e6, &params);
        assert_relative_eq!(bank, -20.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn above_range_clamps_to_segment_0() {
        let params = test_params();
        let bank = segment_bank_angle(10.0e6, &params);
        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn below_range_clamps_to_last_segment() {
        let params = test_params();
        let bank = segment_bank_angle(-20.0e6, &params);
        assert_relative_eq!(bank, -60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn negative_bank_angle_preserved() {
        let params = test_params();
        let bank = segment_bank_angle(-3.0e6, &params);
        assert!(bank < 0.0, "expected negative bank, got {}", bank);
    }

    #[test]
    fn returns_signed_value() {
        let mut params = test_params();
        params.bank_angles[0] = -PI / 3.0;
        let bank = segment_bank_angle(4.9e6, &params);
        assert!(bank < 0.0, "bank should be negative: {}", bank);
        assert_relative_eq!(bank.abs(), PI / 3.0, epsilon = 1e-10);
    }
}
