//! Domain-specific test assertions for aerocapture simulation.

/// Assert a value is finite and within [min, max].
pub fn assert_finite_bounded(value: f64, min: f64, max: f64, context: &str) {
    assert!(value.is_finite(), "{context}: expected finite, got {value}");
    assert!(
        value >= min && value <= max,
        "{context}: expected [{min}, {max}], got {value}"
    );
}

/// Assert a bank angle is finite and within [0, pi] radians.
pub fn assert_bank_angle_valid(angle_rad: f64, context: &str) {
    assert_finite_bounded(
        angle_rad,
        0.0,
        std::f64::consts::PI,
        &format!("{context} (bank angle)"),
    );
}

/// Assert all components of a 3-vector are finite.
pub fn assert_vector_finite(v: &[f64; 3], context: &str) {
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "{context}[{i}]: expected finite, got {x}");
    }
}
