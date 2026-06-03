//! Piecewise-constant bank angle guidance.
//!
//! Divides the orbital energy range into `bank_angles.len()` uniform
//! segments, each with a constant bank angle in [-180deg, +180deg]. The
//! bank angle sign is part of the profile (negative = implicit roll
//! reversal). No navigation feedback, no lateral guidance -- pure
//! open-loop bank profile.
//!
//! GA-optimized to produce reference trajectories and corridor envelopes.

use crate::data::guidance_params::PiecewiseConstantParams;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Compute piecewise-constant bank angle from current orbital energy.
///
/// Returns the **signed** bank angle in radians. Unlike other schemes
/// that return magnitude only (with roll sign applied by lateral guidance),
/// piecewise_constant encodes the sign directly in the bank profile.
pub fn piecewise_constant_bank(
    _nav: &NavigationOutput,
    params: &PiecewiseConstantParams,
    energy: f64,
) -> f64 {
    segment_bank_angle(energy, params)
}

/// Pure lookup: energy -> segment -> bank angle.
/// Exposed for unit testing without needing a full NavigationOutput.
pub fn segment_bank_angle(energy: f64, params: &PiecewiseConstantParams) -> f64 {
    let n = params.bank_angles.len();
    assert!(
        n > 0,
        "PiecewiseConstantParams.bank_angles must be non-empty"
    );

    let e_min = params.energy_min;
    let e_max = params.energy_max;

    if e_max <= e_min {
        return params.bank_angles[0];
    }

    // Segment 0 = highest energy (entry), segment n-1 = lowest energy (deep capture)
    // Energy DECREASES during flight, so segment index increases as energy drops
    let frac = (e_max - energy) / (e_max - e_min);
    let seg = (frac * n as f64).floor() as i64;
    let seg = seg.clamp(0, (n - 1) as i64) as usize;

    params.bank_angles[seg]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn test_params() -> PiecewiseConstantParams {
        PiecewiseConstantParams {
            bank_angles: vec![
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

    #[test]
    fn arbitrary_segment_count_partitions_energy_range() {
        // 5-segment profile, distinct bank per segment.
        let params = PiecewiseConstantParams {
            bank_angles: vec![
                10.0_f64.to_radians(),
                20.0_f64.to_radians(),
                30.0_f64.to_radians(),
                40.0_f64.to_radians(),
                50.0_f64.to_radians(),
            ],
            energy_min: 0.0,
            energy_max: 5.0,
        };
        // frac=0 -> seg 0, frac=0.5 -> seg 2, frac~=1 -> seg 4
        assert_relative_eq!(
            segment_bank_angle(5.0, &params),
            10.0_f64.to_radians(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            segment_bank_angle(2.5, &params),
            30.0_f64.to_radians(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            segment_bank_angle(0.01, &params),
            50.0_f64.to_radians(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn single_segment_returns_constant() {
        let params = PiecewiseConstantParams {
            bank_angles: vec![PI / 4.0],
            energy_min: -1.0,
            energy_max: 1.0,
        };
        assert_relative_eq!(segment_bank_angle(0.5, &params), PI / 4.0, epsilon = 1e-10);
        assert_relative_eq!(segment_bank_angle(-0.5, &params), PI / 4.0, epsilon = 1e-10);
    }
}
