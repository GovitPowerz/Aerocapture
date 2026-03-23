//! Angular utility functions for wrap-aware bank angle control.

use std::f64::consts::{PI, TAU};

/// Shortest signed angular difference from `from` to `to`, in [-PI, PI].
///
/// Returns the smallest rotation needed to get from `from` to `to`,
/// with positive meaning counterclockwise and negative meaning clockwise.
/// Inputs must be finite; propagates NaN for non-finite inputs.
#[inline]
pub fn shortest_angle_diff(from: f64, to: f64) -> f64 {
    debug_assert!(
        from.is_finite() && to.is_finite(),
        "shortest_angle_diff: inputs must be finite"
    );
    let mut d = (to - from) % TAU;
    if d > PI {
        d -= TAU;
    }
    if d < -PI {
        d += TAU;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    const DEG: f64 = PI / 180.0;

    #[test]
    fn wrap_through_plus_pi() {
        let d = shortest_angle_diff(170.0 * DEG, -170.0 * DEG);
        assert_abs_diff_eq!(d, 20.0 * DEG, epsilon = 1e-12);
    }

    #[test]
    fn wrap_through_minus_pi() {
        let d = shortest_angle_diff(-170.0 * DEG, 170.0 * DEG);
        assert_abs_diff_eq!(d, -20.0 * DEG, epsilon = 1e-12);
    }

    #[test]
    fn zero_to_pi_is_exactly_pi() {
        let d = shortest_angle_diff(0.0, PI);
        assert_abs_diff_eq!(d, PI, epsilon = 1e-15);
    }

    #[test]
    fn identical_angles_give_zero() {
        assert_abs_diff_eq!(shortest_angle_diff(0.0, 0.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(shortest_angle_diff(1.0, 1.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn pi_and_minus_pi_are_same_angle() {
        let d1 = shortest_angle_diff(PI, -PI);
        let d2 = shortest_angle_diff(-PI, PI);
        assert!(d1.abs() < 1e-12, "PI to -PI should be ~0, got {d1}");
        assert!(d2.abs() < 1e-12, "-PI to PI should be ~0, got {d2}");
    }

    #[test]
    fn normal_no_wrap_case() {
        let d = shortest_angle_diff(30.0 * DEG, 60.0 * DEG);
        assert_abs_diff_eq!(d, 30.0 * DEG, epsilon = 1e-12);
    }

    proptest! {
        #[test]
        fn result_in_range(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
            let d = shortest_angle_diff(a, b);
            prop_assert!(d >= -PI && d <= PI, "diff={d} outside [-PI, PI]");
        }

        #[test]
        fn approximate_antisymmetry(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
            let d1 = shortest_angle_diff(a, b);
            let d2 = shortest_angle_diff(b, a);
            prop_assert!((d1 + d2).abs() < 1e-10, "antisymmetry violated: d(a,b)={d1}, d(b,a)={d2}");
        }

        #[test]
        fn magnitude_at_most_pi(a in -100.0_f64..100.0, b in -100.0_f64..100.0) {
            let d = shortest_angle_diff(a, b);
            prop_assert!(d.abs() <= PI + 1e-15, "|diff|={} > PI", d.abs());
        }
    }
}
