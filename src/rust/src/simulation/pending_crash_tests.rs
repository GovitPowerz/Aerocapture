
use super::is_pending_crash;

// exit_altitude in meters matches the field's unit.
const EXIT_ALT: f64 = 125_000.0;

#[test]
fn hyperbolic_orbit_is_not_pending_crash() {
    // e >= 1 -> not captured -> not pending crash regardless of apoapsis.
    assert!(!is_pending_crash(1.1, 1.0e6, 0.0, EXIT_ALT));
}

#[test]
fn positive_energy_is_not_pending_crash() {
    // energy > 0 -> unbound -> not captured even if e < 1.
    assert!(!is_pending_crash(0.5, 1.0e6, 100_000.0, EXIT_ALT));
}

#[test]
fn captured_with_high_apoapsis_is_not_pending_crash() {
    // Bound + apoapsis well above exit altitude -> clean capture.
    assert!(!is_pending_crash(
        0.5,
        -1.0e6,
        EXIT_ALT + 10_000.0,
        EXIT_ALT
    ));
}

#[test]
fn captured_with_apoapsis_below_ceiling_is_pending_crash() {
    // Bound but apoapsis under the atmosphere -> guaranteed re-entry.
    assert!(is_pending_crash(0.5, -1.0e6, EXIT_ALT - 10_000.0, EXIT_ALT));
}

#[test]
fn boundary_apoapsis_equal_exit_is_not_pending_crash() {
    // Strict inequality -> apoapsis == exit is a clean edge.
    assert!(!is_pending_crash(0.5, -1.0e6, EXIT_ALT, EXIT_ALT));
}

#[test]
fn nan_inputs_do_not_promote() {
    // NaN comparisons are false -> no spurious promotion on numerical blow-up.
    assert!(!is_pending_crash(f64::NAN, -1.0e6, 0.0, EXIT_ALT));
    assert!(!is_pending_crash(0.5, f64::NAN, 0.0, EXIT_ALT));
    assert!(!is_pending_crash(0.5, -1.0e6, f64::NAN, EXIT_ALT));
}
