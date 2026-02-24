//! Attitude command realization.
//!
//! Applies bank angle rate limits and computes the realized attitude.

/// Limit bank angle rate to maximum allowed rate.
pub fn rate_limited_bank(
    current: f64,
    commanded: f64,
    max_rate: f64,
    dt: f64,
) -> f64 {
    let error = commanded - current;
    let max_change = max_rate * dt;

    if error.abs() <= max_change {
        commanded
    } else {
        current + error.signum() * max_change
    }
}
