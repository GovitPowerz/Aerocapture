//! Attitude command realization.
//!
//! Applies bank angle rate limits and computes the realized attitude.

/// Limit bank angle rate to maximum allowed rate.
#[allow(dead_code)]
pub fn rate_limited_bank(current: f64, commanded: f64, max_rate: f64, dt: f64) -> f64 {
    let error = commanded - current;
    let max_change = max_rate * dt;

    if error.abs() <= max_change {
        commanded
    } else {
        current + error.signum() * max_change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn within_limit_reaches_target() {
        // error = 0.01, max_change = 0.5*0.1 = 0.05 → within limit
        let result = rate_limited_bank(1.0, 1.01, 0.5, 0.1);
        assert!((result - 1.01).abs() < EPS);
    }

    #[test]
    fn exceeds_limit_is_clamped() {
        // error = 10.0, max_change = 0.5*0.1 = 0.05 → clamped
        let result = rate_limited_bank(0.0, 10.0, 0.5, 0.1);
        assert!((result - 0.05).abs() < EPS);
    }

    #[test]
    fn negative_direction() {
        // error = -10.0, max_change = 0.5*0.1 = 0.05 → moves down
        let result = rate_limited_bank(1.0, -9.0, 0.5, 0.1);
        assert!((result - 0.95).abs() < EPS);
    }

    #[test]
    fn zero_error_stays_put() {
        let result = rate_limited_bank(1.0, 1.0, 0.5, 0.1);
        assert!((result - 1.0).abs() < EPS);
    }
}
