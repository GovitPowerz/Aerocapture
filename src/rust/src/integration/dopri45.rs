//! Dormand-Prince 4(5) embedded Runge-Kutta integrator with adaptive step sizing.
//!
//! Provides local error estimation via embedded 4th/5th order solutions and
//! PI step-size control (Gustafsson). Uses FSAL optimization — accepted steps
//! cost 6 derivative evaluations instead of 7.

/// Dormand-Prince 4(5) Butcher tableau coefficients.
/// 7 stages, FSAL: k7 of step n = k1 of step n+1.
///
/// Source: Dormand, J.R.; Prince, P.J. (1980), "A family of embedded
/// Runge-Kutta formulae", Journal of Computational and Applied Mathematics.
mod tableau {
    /// Stage time offsets (c_i): fraction of dt at which each stage is evaluated.
    /// Not used directly in integration (stages are computed inline), but required for
    /// tableau consistency verification in tests.
    #[cfg(test)]
    pub const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

    /// Stage coupling coefficients (a_ij): how each stage depends on previous stages.
    /// a[i] contains coefficients for stage i+1 (stage 0 has no dependencies).
    pub const A: [[f64; 6]; 6] = [
        // Stage 2 (i=1)
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // Stage 3 (i=2)
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        // Stage 4 (i=3)
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        // Stage 5 (i=4)
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
        ],
        // Stage 6 (i=5)
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
        ],
        // Stage 7 (i=6) — FSAL stage
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];

    /// 5th-order solution weights (b_i): same as A[5] (the FSAL row).
    pub const B5: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];

    /// 4th-order solution weights (b*_i): used for error estimation.
    pub const B4: [f64; 7] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];
}

const N: usize = 8; // state vector dimension

/// Persistent state for FSAL (First Same As Last) optimization.
/// Between accepted steps, k7 from step n becomes k1 of step n+1.
#[derive(Debug, Clone)]
pub struct Dopri45State {
    /// Last stage derivative from previous accepted step (becomes k1 of next step).
    k_last: [f64; N],
    /// Whether k_last is valid (false on first step or after rejection).
    fsal_valid: bool,
    /// Previous error norm for PI controller (0.0 before first accepted step).
    err_prev: f64,
}

impl Default for Dopri45State {
    fn default() -> Self {
        Self {
            k_last: [0.0; N],
            fsal_valid: false,
            err_prev: 0.0,
        }
    }
}

impl Dopri45State {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result of a single DOPRI45 step attempt.
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// Whether the step was accepted (error_norm <= 1.0).
    pub accepted: bool,
    /// Scaled error norm. Values <= 1.0 mean the step meets tolerance.
    pub error_norm: f64,
    /// Suggested step size for the next attempt (smaller if rejected, possibly larger if accepted).
    pub dt_next: f64,
}

/// Compute the scaled error norm: sqrt(mean((err_i / scale_i)^2)).
/// scale_i = atol[i] + rtol * |y_i|.
fn error_norm(y: &[f64; N], y4: &[f64; N], y5: &[f64; N], atol: &[f64; N], rtol: f64) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..N {
        let scale = atol[i] + rtol * y[i].abs();
        let err = (y4[i] - y5[i]) / scale;
        sum_sq += err * err;
    }
    (sum_sq / N as f64).sqrt()
}

/// PI step-size controller (Gustafsson).
///
/// On first accepted step or after rejection, uses elementary controller (beta2=0).
/// Otherwise uses PI controller to smooth step-size changes.
fn compute_dt_next(dt: f64, err: f64, err_prev: f64, is_first_or_rejected: bool) -> f64 {
    const FAC: f64 = 0.9; // safety factor
    const FAC_MIN: f64 = 0.2; // max shrink: dt * 0.2
    const FAC_MAX: f64 = 5.0; // max grow: dt * 5.0
    const BETA1: f64 = 0.7 / 5.0; // PI controller exponent (proportional)
    const BETA2: f64 = 0.4 / 5.0; // PI controller exponent (integral)

    let err_safe = err.max(1e-10); // avoid division by zero

    let factor = if is_first_or_rejected || err_prev <= 0.0 {
        // Elementary controller: dt_new = dt * fac * (1/err)^(1/5)
        FAC * (1.0 / err_safe).powf(1.0 / 5.0)
    } else {
        // PI controller: dt_new = dt * fac * (1/err)^beta1 * (err_prev/err)^beta2
        FAC * (1.0 / err_safe).powf(BETA1) * (err_prev / err_safe).powf(BETA2)
    };

    dt * factor.clamp(FAC_MIN, FAC_MAX)
}

/// Attempt one Dormand-Prince 4(5) step of size `dt`.
///
/// On acceptance: `state` is updated to the 5th-order solution, `dopri.k_last` is set
/// for FSAL reuse, and `dopri.err_prev` is updated for the PI controller.
///
/// On rejection: `state` is restored to its value before the call. The caller should
/// retry with `result.dt_next`.
///
/// The `deriv_fn` closure computes state derivatives given the current state.
/// It must not have side effects — it may be called up to 7 times per step attempt.
pub fn dopri45_step(
    state: &mut [f64; N],
    dt: f64,
    dopri: &mut Dopri45State,
    atol: &[f64; N],
    rtol: f64,
    deriv_fn: &mut impl FnMut(&[f64; N]) -> [f64; N],
) -> StepResult {
    let y0 = *state; // save for restoration on rejection

    // Stage 1: reuse from FSAL if available, otherwise evaluate
    let k1 = if dopri.fsal_valid {
        dopri.k_last
    } else {
        deriv_fn(state)
    };

    // Stage 2
    let mut y_stage = [0.0; N];
    for i in 0..N {
        y_stage[i] = y0[i] + dt * tableau::A[0][0] * k1[i];
    }
    let k2 = deriv_fn(&y_stage);

    // Stage 3
    for i in 0..N {
        y_stage[i] = y0[i] + dt * (tableau::A[1][0] * k1[i] + tableau::A[1][1] * k2[i]);
    }
    let k3 = deriv_fn(&y_stage);

    // Stage 4
    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[2][0] * k1[i] + tableau::A[2][1] * k2[i] + tableau::A[2][2] * k3[i]);
    }
    let k4 = deriv_fn(&y_stage);

    // Stage 5
    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[3][0] * k1[i]
                + tableau::A[3][1] * k2[i]
                + tableau::A[3][2] * k3[i]
                + tableau::A[3][3] * k4[i]);
    }
    let k5 = deriv_fn(&y_stage);

    // Stage 6
    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[4][0] * k1[i]
                + tableau::A[4][1] * k2[i]
                + tableau::A[4][2] * k3[i]
                + tableau::A[4][3] * k4[i]
                + tableau::A[4][4] * k5[i]);
    }
    let k6 = deriv_fn(&y_stage);

    // 5th-order solution (used as the accepted state)
    let mut y5 = [0.0; N];
    for i in 0..N {
        y5[i] = y0[i]
            + dt * (tableau::B5[0] * k1[i]
                + tableau::B5[2] * k3[i]
                + tableau::B5[3] * k4[i]
                + tableau::B5[4] * k5[i]
                + tableau::B5[5] * k6[i]);
        // B5[1] = 0, B5[6] = 0 — skipped
    }

    // Stage 7 (FSAL — evaluated at the 5th-order solution)
    let k7 = deriv_fn(&y5);

    // 4th-order solution (for error estimation only)
    let mut y4 = [0.0; N];
    for i in 0..N {
        y4[i] = y0[i]
            + dt * (tableau::B4[0] * k1[i]
                + tableau::B4[2] * k3[i]
                + tableau::B4[3] * k4[i]
                + tableau::B4[4] * k5[i]
                + tableau::B4[5] * k6[i]
                + tableau::B4[6] * k7[i]);
        // B4[1] = 0 — skipped
    }

    let err = error_norm(&y0, &y4, &y5, atol, rtol);
    let accepted = err <= 1.0;

    let dt_next = compute_dt_next(dt, err, dopri.err_prev, !dopri.fsal_valid);

    if accepted {
        *state = y5;
        dopri.k_last = k7;
        dopri.fsal_valid = true;
        dopri.err_prev = err;
    } else {
        *state = y0; // restore
        dopri.fsal_valid = false;
        // err_prev NOT updated on rejection — PI controller uses last accepted error
    }

    StepResult {
        accepted,
        error_norm: err,
        dt_next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DOPRI45 is order 5 — it must integrate t^4 exactly (up to float precision).
    /// dy/dt = 4*t^3, y(0) = 0 => y(1) = 1.0
    #[test]
    fn exact_for_degree4_polynomial() {
        let atol = [1e-12; 8];
        let rtol = 1e-10;
        let mut state = [0.0; 8];
        let mut dopri = Dopri45State::new();

        let result = dopri45_step(&mut state, 1.0, &mut dopri, &atol, rtol, &mut |s| {
            let t = s[7]; // time is state[7]
            let mut d = [0.0; 8];
            d[0] = 4.0 * t * t * t; // dy/dt = 4t^3
            d[7] = 1.0; // dt/dt = 1
            d
        });

        assert!(
            result.accepted,
            "Step should be accepted for smooth polynomial"
        );
        assert!(
            (state[0] - 1.0).abs() < 1e-10,
            "Expected y(1) = 1.0, got {}",
            state[0]
        );
    }

    /// Butcher tableau row-sum consistency: sum of a[i][j] should equal c[i+1].
    #[test]
    fn tableau_row_sums_match_c() {
        for (row_idx, row) in tableau::A.iter().enumerate() {
            let row_sum: f64 = row.iter().sum();
            let expected = tableau::C[row_idx + 1];
            assert!(
                (row_sum - expected).abs() < 1e-14,
                "Row {} sum = {}, expected c[{}] = {}",
                row_idx + 1,
                row_sum,
                row_idx + 1,
                expected,
            );
        }
    }

    /// 5th-order weights must sum to 1.0.
    #[test]
    fn b5_weights_sum_to_one() {
        let sum: f64 = tableau::B5.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "B5 sum = {}, expected 1.0", sum);
    }

    /// 4th-order weights must sum to 1.0.
    #[test]
    fn b4_weights_sum_to_one() {
        let sum: f64 = tableau::B4.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14, "B4 sum = {}, expected 1.0", sum);
    }

    /// FSAL: k7 from an accepted step must equal k1 recomputed at the new state.
    #[test]
    fn fsal_continuity() {
        let atol = [1e-10; 8];
        let rtol = 1e-8;
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut dopri = Dopri45State::new();

        // Exponential ODE: dy/dt = y
        let mut deriv = |s: &[f64; 8]| -> [f64; 8] {
            let mut d = [0.0; 8];
            d[0] = s[0];
            d[7] = 1.0;
            d
        };

        let r1 = dopri45_step(&mut state, 0.1, &mut dopri, &atol, rtol, &mut deriv);
        assert!(r1.accepted);
        assert!(dopri.fsal_valid);

        // k_last should match a fresh evaluation at the current state
        let k_fresh = deriv(&state);
        for (i, &kf) in k_fresh.iter().enumerate() {
            assert!(
                (dopri.k_last[i] - kf).abs() < 1e-14,
                "FSAL mismatch at component {}: k_last={}, k_fresh={}",
                i,
                dopri.k_last[i],
                kf,
            );
        }
    }

    /// Large step on a decaying exponential should be rejected, then accepted at smaller dt.
    #[test]
    fn rejection_and_recovery() {
        let atol = [1e-6; 8];
        let rtol = 1e-6;
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let state_before = state;
        let mut dopri = Dopri45State::new();

        // dy/dt = -1000*y — stiff-ish, large dt should fail
        let mut deriv = |s: &[f64; 8]| -> [f64; 8] {
            let mut d = [0.0; 8];
            d[0] = -1000.0 * s[0];
            d[7] = 1.0;
            d
        };

        let r1 = dopri45_step(&mut state, 1.0, &mut dopri, &atol, rtol, &mut deriv);
        assert!(!r1.accepted, "Large step on stiff ODE should be rejected");
        assert!(
            r1.dt_next < 1.0,
            "dt_next should be smaller than attempted dt"
        );
        // State must be restored on rejection
        for i in 0..8 {
            assert_eq!(
                state[i], state_before[i],
                "State must be restored on rejection"
            );
        }

        // Retry with suggested dt — should eventually accept
        let r2 = dopri45_step(&mut state, r1.dt_next, &mut dopri, &atol, rtol, &mut deriv);
        // May need multiple retries for very stiff problems, but dt_next should keep shrinking
        assert!(
            r2.accepted || r2.dt_next < r1.dt_next,
            "Should either accept or keep shrinking"
        );
    }

    /// PI controller must respect facmin=0.2 and facmax=5.0 bounds.
    #[test]
    fn pi_controller_bounds() {
        // Very small error => want to grow a lot, but capped at facmax=5.0
        let dt_next = compute_dt_next(1.0, 1e-12, 1e-12, false);
        assert!(
            dt_next <= 5.0 + 1e-10,
            "dt_next={} should not exceed dt * facmax = 5.0",
            dt_next,
        );

        // Very large error => want to shrink a lot, but floored at facmin=0.2
        let dt_next = compute_dt_next(1.0, 1e6, 1.0, false);
        assert!(
            dt_next >= 0.2 - 1e-10,
            "dt_next={} should not go below dt * facmin = 0.2",
            dt_next,
        );
    }

    /// atol dominates when y is near zero; rtol dominates when y is large.
    #[test]
    fn error_norm_scaling() {
        let atol = [1.0; 8];
        let rtol = 1e-6;

        // Near zero: scale ≈ atol = 1.0, so err of 0.5 gives norm ≈ 0.5
        let y_small = [0.0; 8];
        let y4 = [0.0; 8];
        let y5 = [0.5; 8];
        let norm_small = error_norm(&y_small, &y4, &y5, &atol, rtol);
        assert!(
            (norm_small - 0.5).abs() < 0.01,
            "Near zero, atol should dominate: norm={}",
            norm_small,
        );

        // Large y: scale = atol + rtol * |y| = 1.0 + 1e-6 * 1e6 = 2.0, so err of 0.5 gives norm ≈ 0.25
        let y_large = [1e6; 8];
        let y4_l = [0.0; 8];
        let y5_l = [0.5; 8];
        let norm_large = error_norm(&y_large, &y4_l, &y5_l, &atol, rtol);
        assert!(
            (norm_large - 0.25).abs() < 0.01,
            "Large y, mixed atol+rtol scaling: norm={}",
            norm_large,
        );

        // Very large y with tiny atol: rtol must dominate
        let atol_tiny = [1e-20; 8];
        let y_huge = [1e10; 8];
        let y4_h = [0.0; 8];
        let y5_h = [1e4; 8]; // err = 1e4, scale = 1e-20 + 1e-6 * 1e10 = 1e4, so norm ≈ 1.0
        let norm_rtol = error_norm(&y_huge, &y4_h, &y5_h, &atol_tiny, rtol);
        assert!(
            (norm_rtol - 1.0).abs() < 0.1,
            "rtol should dominate for large y: norm={}",
            norm_rtol,
        );
    }

    /// Harmonic oscillator: dx/dt = v, dv/dt = -x over one full period.
    /// DOPRI45 should return to initial conditions more accurately than fixed-step Gill.
    #[test]
    fn harmonic_oscillator_one_period() {
        let atol = [1e-10; 8];
        let rtol = 1e-10;
        // state[0] = x, state[1] = v, state[7] = time
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut dopri = Dopri45State::new();
        let period = 2.0 * std::f64::consts::PI;
        let mut t = 0.0;
        let mut dt: f64 = 0.1;
        let mut steps = 0;

        while t < period {
            let h = dt.min(period - t);
            let result = dopri45_step(&mut state, h, &mut dopri, &atol, rtol, &mut |s| {
                let mut d = [0.0; 8];
                d[0] = s[1]; // dx/dt = v
                d[1] = -s[0]; // dv/dt = -x
                d[7] = 1.0;
                d
            });
            if result.accepted {
                t += h;
                dt = result.dt_next;
                steps += 1;
            } else {
                dt = result.dt_next;
            }
        }

        assert!(
            (state[0] - 1.0).abs() < 1e-7,
            "Expected x ≈ 1.0 after one period, got {} ({} steps)",
            state[0],
            steps,
        );
        assert!(
            state[1].abs() < 1e-7,
            "Expected v ≈ 0.0 after one period, got {} ({} steps)",
            state[1],
            steps,
        );
    }

    use proptest::prelude::*;

    proptest! {
        /// For any reasonable initial state, a DOPRI45 step should:
        /// 1. Always produce finite state values (no NaN/Inf)
        /// 2. Return a positive dt_next
        /// 3. Restore state exactly on rejection
        #[test]
        fn step_produces_finite_output(
            r in 3.3e6_f64..3.5e6,
            v in 3000.0_f64..7000.0,
            gamma in -0.15_f64..0.05,
            dt in 0.001_f64..2.0,
        ) {
            let atol = [1.0, 1e-8, 1e-8, 1e-3, 1e-8, 1e-8, 1e-2, 1e-6];
            let rtol = 1e-6;
            let mut state = [r, 0.0, 0.0, v, gamma, 0.0, 0.0, 0.0];
            let state_before = state;
            let mut dopri = Dopri45State::new();

            // Simple gravity + drag ODE (no tables needed)
            let mu = 4.2828e13_f64; // Mars GM
            let result = dopri45_step(
                &mut state,
                dt,
                &mut dopri,
                &atol,
                rtol,
                &mut |s| {
                    let mut d = [0.0; 8];
                    d[0] = s[3] * s[4].sin();                    // dr/dt = V * sin(gamma)
                    d[3] = -mu / (s[0] * s[0]) * s[4].sin();     // dV/dt (gravity drag)
                    d[4] = (s[3] / s[0] - mu / (s[0] * s[0] * s[3])) * s[4].cos(); // dgamma/dt
                    d[7] = 1.0;
                    d
                },
            );

            // dt_next must be positive and finite
            prop_assert!(result.dt_next > 0.0, "dt_next must be positive: {}", result.dt_next);
            prop_assert!(result.dt_next.is_finite(), "dt_next must be finite: {}", result.dt_next);
            prop_assert!(result.error_norm.is_finite(), "error_norm must be finite");

            if result.accepted {
                // All state components must be finite
                for (i, &val) in state.iter().enumerate() {
                    prop_assert!(val.is_finite(), "state[{}] = {} is not finite", i, val);
                }
            } else {
                // State must be exactly restored on rejection
                for i in 0..8 {
                    prop_assert_eq!(state[i], state_before[i], "state[{}] not restored", i);
                }
            }
        }
    }
}
