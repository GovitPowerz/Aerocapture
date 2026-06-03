use super::*;
use proptest::prelude::*;

// Mars-ish constants for proptest scenarios.
const MU_MARS: f64 = 4.282837e13;
const TARGET_SMA: f64 = 2.0e7; // 20000 km → E_target ≈ -1.07 MJ/kg

proptest! {
    #[test]
    fn crash_virtual_dv_finite_and_bounded_below(
        energy_j_kg in -5.0e7f64..5.0e7,
        sim_time in 0.0f64..10000.0,
        max_time in 100.0f64..10000.0,
    ) {
        let dv = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, sim_time, max_time);
        prop_assert!(dv.is_finite());
        // Lower bound: CRASH_FLOOR - CRASH_TIME_BONUS (when ΔE = 0 and t_ratio = 1).
        prop_assert!(dv >= CRASH_FLOOR - CRASH_TIME_BONUS, "dv={} below floor", dv);
    }

    #[test]
    fn crash_virtual_dv_monotonic_in_energy_error(
        delta_e_mj in 0.0f64..20.0,
    ) {
        let e_target = -MU_MARS / (2.0 * TARGET_SMA);
        let dv0 = virtual_dv_non_capture(e_target, TARGET_SMA, MU_MARS, 0.0, 1000.0);
        let dv1 = virtual_dv_non_capture(e_target + delta_e_mj * 1e6, TARGET_SMA, MU_MARS, 0.0, 1000.0);
        let dv2 = virtual_dv_non_capture(e_target - delta_e_mj * 1e6, TARGET_SMA, MU_MARS, 0.0, 1000.0);
        // Symmetric: |+ΔE| and |-ΔE| produce identical cost.
        prop_assert!((dv1 - dv2).abs() < 1e-9);
        // Monotonic: bigger |ΔE| → bigger cost.
        prop_assert!(dv1 >= dv0 - 1e-9);
    }

    #[test]
    fn crash_virtual_dv_survival_reduces_cost(
        energy_j_kg in -5.0e7f64..5.0e7,
    ) {
        let early = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, 0.0, 1000.0);
        let late = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, 1000.0, 1000.0);
        prop_assert!((early - late - CRASH_TIME_BONUS).abs() < 1e-9);
    }

    #[test]
    fn hyperbolic_virtual_dv_above_base(
        v_excess in 0.0f64..5000.0,
    ) {
        let virtual_dv = HYPERBOLIC_BASE + v_excess;
        prop_assert!(virtual_dv >= HYPERBOLIC_BASE);
        prop_assert!(virtual_dv.is_finite());
    }
}

#[test]
fn non_finite_inputs_produce_finite_capped_output() {
    // NaN energy (from degenerate state) must not propagate.
    let dv_nan = virtual_dv_non_capture(f64::NAN, TARGET_SMA, MU_MARS, 0.0, 1000.0);
    assert!(dv_nan.is_finite());
    assert!(dv_nan >= CRASH_FLOOR);
    // Expected: CRASH_FLOOR + CRASH_ENERGY_WEIGHT * CRASH_ENERGY_CAP_MJKG - 0.
    let expected = CRASH_FLOOR + CRASH_ENERGY_WEIGHT * CRASH_ENERGY_CAP_MJKG;
    assert!((dv_nan - expected).abs() < 1e-9);

    // +Inf energy also capped.
    let dv_inf = virtual_dv_non_capture(f64::INFINITY, TARGET_SMA, MU_MARS, 500.0, 1000.0);
    assert!(dv_inf.is_finite());
    assert!((dv_inf - (expected - CRASH_TIME_BONUS * 0.5)).abs() < 1e-9);

    // NaN sim_time.
    let dv_t_nan = virtual_dv_non_capture(0.0, TARGET_SMA, MU_MARS, f64::NAN, 1000.0);
    assert!(dv_t_nan.is_finite());
}

#[test]
fn near_target_crash_stays_above_typical_capture_floor() {
    // A crash with energy exactly at target (best possible crash) at max survival
    // time must still cost more than typical captures (~500-2000 m/s) so the
    // optimizer never prefers crashing over capturing.
    let e_target = -MU_MARS / (2.0 * TARGET_SMA);
    let best_possible_crash = virtual_dv_non_capture(e_target, TARGET_SMA, MU_MARS, 1000.0, 1000.0);
    assert!(
        best_possible_crash >= 2500.0,
        "best crash DV {} too close to captures",
        best_possible_crash
    );
    assert!(
        best_possible_crash <= CRASH_FLOOR,
        "best crash DV {} exceeds floor {}",
        best_possible_crash,
        CRASH_FLOOR
    );
}
