# Adaptive DOPRI45 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Dormand-Prince 4(5) adaptive integrator as an opt-in alternative to the existing fixed-step Gill RK4, with local error estimation and PI step-size control.

**Architecture:** New `dopri45.rs` module in `src/rust/src/integration/` implements the embedded RK method with FSAL optimization. The runner dispatches between fixed Gill and adaptive DOPRI45 based on a new `[integration]` TOML section. GNC cadences are unchanged — adaptivity is purely within each outer integration tick.

**Tech Stack:** Rust (nalgebra not needed — fixed-size `[f64; 8]` arrays), TOML config via serde, PyO3 dot-path overrides, proptest for property-based testing.

**Spec:** `docs/superpowers/specs/2026-03-27-adaptive-dopri45-integration-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/rust/src/integration/dopri45.rs` | DOPRI45 integrator: Butcher tableau, step function, error norm, PI controller |
| Modify | `src/rust/src/integration/mod.rs` | Add `pub mod dopri45;` |
| Modify | `src/rust/src/config.rs` | Add `TomlIntegration` struct, `IntegrationMode` enum, `AdaptiveConfig` struct, parse `[integration]` section |
| Modify | `src/rust/src/data/mod.rs` | Add `integration_mode: IntegrationMode` field to `SimData`, wire from `TomlConfig` |
| Modify | `src/rust/src/simulation/runner.rs` | Add `Dopri45State` to `SimState`, `integrate_adaptive()` function, dispatch logic |
| Create | `src/rust/tests/dopri45_integration.rs` | Integration tests for adaptive mode (tick coverage, safety cap, Gill agreement) |
| Create | `configs/test/test_ref_adaptive.toml` | Test config with `[integration] mode = "adaptive"` |

---

### Task 1: DOPRI45 Core — Butcher Tableau and Single Step

**Files:**
- Modify: `src/rust/src/integration/mod.rs`
- Create: `src/rust/src/integration/dopri45.rs`

- [ ] **Step 1: Add module declaration**

In `src/rust/src/integration/mod.rs`, add the new module:

```rust
pub mod dopri45;
pub mod rk4;
pub mod sequencer;
```

- [ ] **Step 2: Write the failing test for a degree-5 polynomial ODE**

Create `src/rust/src/integration/dopri45.rs` with the test first:

```rust
//! Dormand-Prince 4(5) embedded Runge-Kutta integrator with adaptive step sizing.
//!
//! Provides local error estimation via embedded 4th/5th order solutions and
//! PI step-size control (Gustafsson). Uses FSAL optimization — accepted steps
//! cost 6 derivative evaluations instead of 7.

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

        let result = dopri45_step(
            &mut state,
            1.0,
            &mut dopri,
            &atol,
            rtol,
            &mut |s| {
                let t = s[7]; // time is state[7]
                let mut d = [0.0; 8];
                d[0] = 4.0 * t * t * t; // dy/dt = 4t^3
                d[7] = 1.0;             // dt/dt = 1
                d
            },
        );

        assert!(result.accepted, "Step should be accepted for smooth polynomial");
        assert!(
            (state[0] - 1.0).abs() < 1e-10,
            "Expected y(1) = 1.0, got {}",
            state[0]
        );
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test dopri45 -- --nocapture 2>&1 | head -30`
Expected: FAIL — `Dopri45State`, `dopri45_step`, `StepResult` not defined.

- [ ] **Step 4: Implement the Butcher tableau, types, and step function**

Add above the `#[cfg(test)]` block in `dopri45.rs`:

```rust
/// Dormand-Prince 4(5) Butcher tableau coefficients.
/// 7 stages, FSAL: k7 of step n = k1 of step n+1.
///
/// Source: Dormand, J.R.; Prince, P.J. (1980), "A family of embedded
/// Runge-Kutta formulae", Journal of Computational and Applied Mathematics.
mod tableau {
    /// Stage time offsets (c_i): fraction of dt at which each stage is evaluated.
    pub const C: [f64; 7] = [
        0.0,
        1.0 / 5.0,
        3.0 / 10.0,
        4.0 / 5.0,
        8.0 / 9.0,
        1.0,
        1.0,
    ];

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

impl Dopri45State {
    pub fn new() -> Self {
        Self {
            k_last: [0.0; N],
            fsal_valid: false,
            err_prev: 0.0,
        }
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
    const FAC: f64 = 0.9;     // safety factor
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test dopri45::tests::exact_for_degree4_polynomial -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/integration/mod.rs src/rust/src/integration/dopri45.rs
git commit -m "feat(integration): add DOPRI45 core — Butcher tableau, step function, error norm, PI controller"
```

---

### Task 2: DOPRI45 Unit Tests — Tableau, FSAL, Rejection, Controller Bounds

**Files:**
- Modify: `src/rust/src/integration/dopri45.rs` (add tests to existing `#[cfg(test)]` block)

- [ ] **Step 1: Add Butcher tableau consistency test**

Add to the `tests` module in `dopri45.rs`:

```rust
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
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "B5 sum = {}, expected 1.0",
            sum
        );
    }

    /// 4th-order weights must sum to 1.0.
    #[test]
    fn b4_weights_sum_to_one() {
        let sum: f64 = tableau::B4.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "B4 sum = {}, expected 1.0",
            sum
        );
    }
```

- [ ] **Step 2: Add FSAL continuity test**

```rust
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
        for i in 0..8 {
            assert!(
                (dopri.k_last[i] - k_fresh[i]).abs() < 1e-14,
                "FSAL mismatch at component {}: k_last={}, k_fresh={}",
                i,
                dopri.k_last[i],
                k_fresh[i],
            );
        }
    }
```

- [ ] **Step 3: Add rejection and recovery test**

```rust
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
            assert_eq!(state[i], state_before[i], "State must be restored on rejection");
        }

        // Retry with suggested dt — should eventually accept
        let r2 = dopri45_step(&mut state, r1.dt_next, &mut dopri, &atol, rtol, &mut deriv);
        // May need multiple retries for very stiff problems, but dt_next should keep shrinking
        assert!(
            r2.accepted || r2.dt_next < r1.dt_next,
            "Should either accept or keep shrinking"
        );
    }
```

- [ ] **Step 4: Add PI controller bounds test**

```rust
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
```

- [ ] **Step 5: Add error norm scaling test**

```rust
    /// atol dominates when y is near zero; rtol dominates when y is large.
    #[test]
    fn error_norm_scaling() {
        let atol = [1.0; 8];
        let rtol = 1e-6;

        // Near zero: scale ≈ atol = 1.0, so err of 0.5 gives norm ≈ 0.5
        let y_small = [0.0; 8];
        let mut y4 = [0.0; 8];
        let mut y5 = [0.5; 8];
        let norm_small = error_norm(&y_small, &y4, &y5, &atol, rtol);
        assert!(
            (norm_small - 0.5).abs() < 0.01,
            "Near zero, atol should dominate: norm={}",
            norm_small,
        );

        // Large y: scale ≈ rtol * |y| = 1e-6 * 1e6 = 1.0, so err of 0.5 also gives ~0.5
        let y_large = [1e6; 8];
        y4 = [0.0; 8];
        y5 = [0.5; 8];
        let norm_large = error_norm(&y_large, &y4, &y5, &atol, rtol);
        assert!(
            (norm_large - 0.5).abs() < 0.01,
            "Large y, rtol should dominate: norm={}",
            norm_large,
        );

        // Very large y with tiny atol: rtol must dominate
        let atol_tiny = [1e-20; 8];
        let y_huge = [1e10; 8];
        y4 = [0.0; 8];
        y5 = [1e4; 8]; // err = 1e4, scale = 1e-20 + 1e-6 * 1e10 = 1e4, so norm ≈ 1.0
        let norm_rtol = error_norm(&y_huge, &y4, &y5, &atol_tiny, rtol);
        assert!(
            (norm_rtol - 1.0).abs() < 0.1,
            "rtol should dominate for large y: norm={}",
            norm_rtol,
        );
    }
```

- [ ] **Step 6: Add harmonic oscillator accuracy test**

```rust
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
        let mut dt = 0.1;
        let mut steps = 0;

        while t < period {
            let h = dt.min(period - t);
            let result = dopri45_step(
                &mut state,
                h,
                &mut dopri,
                &atol,
                rtol,
                &mut |s| {
                    let mut d = [0.0; 8];
                    d[0] = s[1];      // dx/dt = v
                    d[1] = -s[0];     // dv/dt = -x
                    d[7] = 1.0;
                    d
                },
            );
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
```

- [ ] **Step 7: Run all unit tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test dopri45 -- --nocapture`
Expected: All 8 tests PASS (degree4_polynomial, tableau row sums, b5 sum, b4 sum, fsal, rejection, pi bounds, error norm scaling, harmonic oscillator).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/integration/dopri45.rs
git commit -m "test(integration): add DOPRI45 unit tests — tableau, FSAL, rejection, PI bounds, error norm"
```

---

### Task 3: TOML Configuration — `[integration]` Section

**Files:**
- Modify: `src/rust/src/config.rs`

- [ ] **Step 1: Write failing test for parsing `[integration]` section**

Add to the `tests` module at the bottom of `config.rs`:

```rust
    #[test]
    fn parse_integration_section_adaptive() {
        let toml_str = r#"
            [mission]
            mission_type = "aerocapture"
            planet = "mars"
            phase = "full"

            [guidance]
            type = "ftc"

            [data]
            base_dir = "."
            output_dir = "."

            [integration]
            mode = "adaptive"
            rtol = 1e-8
            initial_dt = 0.05
            min_dt = 1e-8
            max_dt = 1.5
        "#;
        let (_, toml) = SimInput::from_toml(toml_str).expect("parse");
        let integ = toml.integration.unwrap();
        assert_eq!(integ.mode, "adaptive");
        assert!((integ.rtol.unwrap() - 1e-8).abs() < 1e-15);
        assert!((integ.initial_dt.unwrap() - 0.05).abs() < 1e-15);
        assert!((integ.min_dt.unwrap() - 1e-8).abs() < 1e-15);
        assert!((integ.max_dt.unwrap() - 1.5).abs() < 1e-15);
    }

    #[test]
    fn parse_integration_section_absent_defaults_to_none() {
        let toml_str = r#"
            [mission]
            mission_type = "aerocapture"
            planet = "mars"
            phase = "full"

            [guidance]
            type = "ftc"

            [data]
            base_dir = "."
            output_dir = "."
        "#;
        let (_, toml) = SimInput::from_toml(toml_str).expect("parse");
        assert!(toml.integration.is_none());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test parse_integration -- --nocapture 2>&1 | head -20`
Expected: FAIL — `integration` field not found on `TomlConfig`.

- [ ] **Step 3: Add `TomlIntegration` struct and field to `TomlConfig`**

In `config.rs`, add after the `TomlAtmosphereOnboard` block (around line 161):

```rust
// ─── Integration TOML structs ───

/// TOML config for the integration method.
#[derive(Debug, Clone, Deserialize)]
pub struct TomlIntegration {
    pub mode: String,              // "fixed" or "adaptive"
    pub rtol: Option<f64>,         // relative tolerance (default 1e-6)
    pub initial_dt: Option<f64>,   // initial sub-step guess in seconds (default 0.1)
    pub min_dt: Option<f64>,       // floor to prevent sub-step collapse (default 1e-6)
    pub max_dt: Option<f64>,       // ceiling in seconds (default = periods.integration)
}
```

In the `TomlConfig` struct, add after the `onboard_atmosphere` field:

```rust
    /// Integration method config (adaptive DOPRI45 vs fixed Gill RK4)
    pub integration: Option<TomlIntegration>,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test parse_integration -- --nocapture`
Expected: Both tests PASS.

- [ ] **Step 5: Add `IntegrationMode` and `AdaptiveConfig` to config.rs**

Add after the `GuidanceType` enum (around line 94):

```rust
/// Adaptive integration configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    pub rtol: f64,       // relative tolerance
    pub initial_dt: f64, // initial sub-step guess (seconds)
    pub min_dt: f64,     // floor (seconds)
    pub max_dt: f64,     // ceiling (seconds)
}

/// Integration method selection.
#[derive(Debug, Clone, Copy)]
pub enum IntegrationMode {
    /// Fixed-step Gill-variant RK4 (legacy, default).
    FixedGill,
    /// Adaptive Dormand-Prince 4(5) with error control.
    AdaptiveDopri45(AdaptiveConfig),
}

impl Default for IntegrationMode {
    fn default() -> Self {
        Self::FixedGill
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat(config): add [integration] TOML section with IntegrationMode and AdaptiveConfig"
```

---

### Task 4: Wire Integration Mode Through SimData

**Files:**
- Modify: `src/rust/src/data/mod.rs`
- Modify: `src/rust/src/config.rs` (add builder function)

- [ ] **Step 1: Add `integration_mode` field to `SimData`**

In `src/rust/src/data/mod.rs`, add to the `SimData` struct (after `nav_config`):

```rust
    /// Integration method: fixed Gill RK4 (default) or adaptive DOPRI45
    pub integration_mode: IntegrationMode,
```

Add the import at the top of `mod.rs`:

```rust
use crate::config::IntegrationMode;
```

- [ ] **Step 2: Add builder function in config.rs**

Add a helper function in `config.rs` (after the `IntegrationMode` impl):

```rust
impl IntegrationMode {
    /// Build from TOML config. `integration_period` is the outer tick dt from [vehicle.periods].
    pub fn from_toml(toml: &Option<TomlIntegration>, integration_period: f64) -> Self {
        let Some(cfg) = toml else {
            return Self::FixedGill;
        };
        match cfg.mode.as_str() {
            "adaptive" => Self::AdaptiveDopri45(AdaptiveConfig {
                rtol: cfg.rtol.unwrap_or(1e-6),
                initial_dt: cfg.initial_dt.unwrap_or(0.1),
                min_dt: cfg.min_dt.unwrap_or(1e-6),
                max_dt: cfg.max_dt.unwrap_or(integration_period),
            }),
            _ => Self::FixedGill, // "fixed" or unrecognized => default
        }
    }
}
```

- [ ] **Step 3: Wire in `SimData::from_toml`**

In `src/rust/src/data/mod.rs`, inside the `from_toml` method, add after the `nav_config` field assignment (near the end of the method, where the `SimData` struct is constructed):

```rust
            integration_mode: IntegrationMode::from_toml(
                &toml.integration,
                v.periods.integration,
            ),
```

- [ ] **Step 4: Update all test fixtures that construct `SimData` directly**

In `src/rust/tests/common/fixtures.rs`, add the field to `minimal_sim_data()`:

```rust
            integration_mode: aerocapture::config::IntegrationMode::FixedGill,
```

Search for any other places that construct `SimData` directly and add the field:

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && grep -rn "SimData {" --include="*.rs" | grep -v "pub struct"`
Fix each occurrence by adding `integration_mode: IntegrationMode::FixedGill,`.

- [ ] **Step 5: Run full test suite to verify nothing broke**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -5`
Expected: All existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/mod.rs src/rust/src/config.rs src/rust/tests/
git commit -m "feat(data): wire IntegrationMode through SimData from TOML config"
```

---

### Task 5: Runner — Adaptive Sub-stepping Within Outer Tick

**Files:**
- Modify: `src/rust/src/simulation/runner.rs`

- [ ] **Step 1: Add imports and default tolerances**

At the top of `runner.rs`, add:

```rust
use crate::config::{AdaptiveConfig, IntegrationMode};
use crate::integration::dopri45::{self, Dopri45State};
```

Add the default absolute tolerances as a module constant:

```rust
/// Default absolute tolerances for DOPRI45, one per state component.
/// State = [r(m), lon(rad), lat(rad), V(m/s), gamma(rad), psi(rad), flux(kJ/m²), time(s)]
const DOPRI45_ATOL: [f64; 8] = [
    1.0,    // r: 1 m on ~3.4e6 m
    1e-8,   // lon: ~0.03 m at Mars equator
    1e-8,   // lat: ~0.03 m
    1e-3,   // V: 1 mm/s on ~5700 m/s
    1e-8,   // gamma: ~0.03 m position equiv
    1e-8,   // psi: ~0.03 m
    1e-2,   // flux: 0.01 kJ/m² on O(1000) total
    1e-6,   // time: machine-level for identity derivative
];
```

- [ ] **Step 2: Add `Dopri45State` to `SimState`**

In the `SimState` struct, add after `gill_toggle`:

```rust
    // DOPRI45 adaptive integrator state (only used in adaptive mode)
    dopri: Dopri45State,
```

Update the `SimState` initialization in `run_single` (where `SimState { ... }` is constructed) to include:

```rust
        dopri: Dopri45State::new(),
```

- [ ] **Step 3: Implement `integrate_adaptive` function**

Add after the existing `integrate_step` function:

```rust
/// Advance the state by `dt_outer` using adaptive DOPRI45 sub-stepping.
///
/// The integrator takes variable-size sub-steps within the outer tick,
/// controlled by local error estimation. The outer tick duration is covered
/// exactly (final sub-step is clamped to land on the tick boundary).
///
/// Returns the number of sub-steps taken and rejections for diagnostics.
fn integrate_adaptive(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) -> (u32, u32) {
    const MAX_SUBSTEPS: u32 = 1000;

    let mut t_remaining = dt_outer;
    let mut h = config.initial_dt.min(t_remaining).max(config.min_dt);
    let mut n_substeps: u32 = 0;
    let mut n_rejections: u32 = 0;

    while t_remaining > 1e-14 {
        // Clamp to remaining time and configured bounds
        h = h.min(t_remaining).min(config.max_dt).max(config.min_dt);

        // If remaining time is very small, take it in one step regardless
        if t_remaining <= config.min_dt * 1.5 {
            h = t_remaining;
        }

        let result = dopri45::dopri45_step(
            &mut sim.state,
            h,
            &mut sim.dopri,
            &DOPRI45_ATOL,
            config.rtol,
            &mut |state| {
                compute_derivatives(
                    state,
                    sim.bank_angle,
                    sim.aoa,
                    planet,
                    data,
                    run_state,
                )
            },
        );

        if result.accepted {
            t_remaining -= h;
            n_substeps += 1;
            h = result.dt_next;
        } else {
            n_rejections += 1;
            h = result.dt_next;
        }

        if n_substeps + n_rejections >= MAX_SUBSTEPS {
            eprintln!(
                "WARNING: adaptive integrator hit {} step limit with t_remaining={:.2e}s",
                MAX_SUBSTEPS, t_remaining,
            );
            break;
        }
    }

    (n_substeps, n_rejections)
}
```

- [ ] **Step 4: Add dispatch logic in the main loop**

In `run_single`, replace the integration call site. Find the line:

```rust
        integrate_step(&mut sim, dt, planet, data, run_state);
```

Replace with:

```rust
        match &data.integration_mode {
            IntegrationMode::FixedGill => {
                integrate_step(&mut sim, dt, planet, data, run_state);
            }
            IntegrationMode::AdaptiveDopri45(adaptive_config) => {
                integrate_adaptive(&mut sim, dt, adaptive_config, &config.planet, data, run_state);
            }
        }
```

Note: `dt` is already `data.periods.integration` (set at line 434), so it serves as `dt_outer`.

- [ ] **Step 5: Fix the borrow issue with `compute_derivatives`**

The closure in `integrate_adaptive` captures `sim.bank_angle` and `sim.aoa` but also needs `&mut sim.state` to be passed into `dopri45_step`. Since `compute_derivatives` takes `state: &[f64; 8]` (not `&SimState`), we need to extract bank_angle and aoa before the closure:

The function signature already handles this — `sim.bank_angle` and `sim.aoa` are read inside the closure, but `dopri45_step` takes `&mut sim.state` separately. However, Rust won't allow borrowing `sim` mutably (for `sim.state`) and immutably (for `sim.bank_angle`) at the same time inside the closure.

Fix by extracting the values before the loop in `integrate_adaptive`. Change the function to capture them:

```rust
fn integrate_adaptive(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) -> (u32, u32) {
    const MAX_SUBSTEPS: u32 = 1000;

    let bank_angle = sim.bank_angle;
    let aoa = sim.aoa;
    let mut t_remaining = dt_outer;
    let mut h = config.initial_dt.min(t_remaining).max(config.min_dt);
    let mut n_substeps: u32 = 0;
    let mut n_rejections: u32 = 0;

    while t_remaining > 1e-14 {
        h = h.min(t_remaining).min(config.max_dt).max(config.min_dt);

        if t_remaining <= config.min_dt * 1.5 {
            h = t_remaining;
        }

        let result = dopri45::dopri45_step(
            &mut sim.state,
            h,
            &mut sim.dopri,
            &DOPRI45_ATOL,
            config.rtol,
            &mut |state| compute_derivatives(state, bank_angle, aoa, planet, data, run_state),
        );

        if result.accepted {
            t_remaining -= h;
            n_substeps += 1;
            h = result.dt_next;
        } else {
            n_rejections += 1;
            h = result.dt_next;
        }

        if n_substeps + n_rejections >= MAX_SUBSTEPS {
            eprintln!(
                "WARNING: adaptive integrator hit {} step limit with t_remaining={:.2e}s",
                MAX_SUBSTEPS, t_remaining,
            );
            break;
        }
    }

    (n_substeps, n_rejections)
}
```

- [ ] **Step 6: Verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release 2>&1 | tail -5`
Expected: Compiles without errors.

- [ ] **Step 7: Run existing tests to verify nothing broke**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -5`
Expected: All existing tests PASS (all use `FixedGill` by default).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "feat(runner): add adaptive DOPRI45 sub-stepping with mode dispatch"
```

---

### Task 6: Integration Tests — Adaptive Mode End-to-End

**Files:**
- Create: `configs/test/test_ref_adaptive.toml`
- Create: `src/rust/tests/dopri45_integration.rs`

- [ ] **Step 1: Create test config with adaptive integration**

Create `configs/test/test_ref_adaptive.toml`:

```toml
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_trajectory = true
reference_bank_angle = 0.1

[simulation]
n_sims = 1
random_seed = 0.6866

[data]
results_suffix = ".test_ref_adaptive"

[onboard_atmosphere]
mode = "identical"

[integration]
mode = "adaptive"
rtol = 1e-6
initial_dt = 0.1
```

- [ ] **Step 2: Write integration test — adaptive produces valid capture**

Create `src/rust/tests/dopri45_integration.rs`:

```rust
//! Integration tests for adaptive DOPRI45 integration mode.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner;
use std::path::Path;

#[test]
fn adaptive_produces_valid_capture() {
    let path = common::config_path("test/test_ref_adaptive.toml");
    let (config, toml) = SimInput::from_toml_file(Path::new(&path)).expect("parse config");
    let data = SimData::from_toml(&toml, &config).expect("load data");

    let results = runner::run_for_api(&config, &data, false).expect("run simulation");
    assert_eq!(results.len(), 1, "Should produce exactly one result");

    let r = &results[0];
    assert!(r.captured, "Adaptive mode should produce a captured trajectory");
    assert!(r.ecc < 1.0, "Eccentricity should be < 1.0 for capture, got {}", r.ecc);
    assert!(r.final_record[0] > 0.0, "Final time should be positive");
}
```

- [ ] **Step 3: Write integration test — adaptive and fixed agree on smooth trajectory**

Add to `dopri45_integration.rs`:

```rust
#[test]
fn adaptive_agrees_with_fixed_on_reference_trajectory() {
    // Run with fixed Gill (the golden reference config)
    let path_fixed = common::config_path("test/test_ref_orig.toml");
    let (config_fixed, toml_fixed) =
        SimInput::from_toml_file(Path::new(&path_fixed)).expect("parse fixed config");
    let data_fixed = SimData::from_toml(&toml_fixed, &config_fixed).expect("load fixed data");
    let results_fixed =
        runner::run_for_api(&config_fixed, &data_fixed, false).expect("run fixed");

    // Run with adaptive DOPRI45
    let path_adaptive = common::config_path("test/test_ref_adaptive.toml");
    let (config_adaptive, toml_adaptive) =
        SimInput::from_toml_file(Path::new(&path_adaptive)).expect("parse adaptive config");
    let data_adaptive =
        SimData::from_toml(&toml_adaptive, &config_adaptive).expect("load adaptive data");
    let results_adaptive =
        runner::run_for_api(&config_adaptive, &data_adaptive, false).expect("run adaptive");

    let rf = &results_fixed[0];
    let ra = &results_adaptive[0];

    // Both should capture
    assert!(rf.captured, "Fixed should capture");
    assert!(ra.captured, "Adaptive should capture");

    // Final energy should agree within 1% (both integrate the same physics)
    let energy_rel_err = ((rf.energy - ra.energy) / rf.energy).abs();
    assert!(
        energy_rel_err < 0.01,
        "Energy mismatch: fixed={:.6}, adaptive={:.6}, rel_err={:.4}",
        rf.energy,
        ra.energy,
        energy_rel_err,
    );

    // Eccentricity should agree within 1%
    let ecc_abs_err = (rf.ecc - ra.ecc).abs();
    assert!(
        ecc_abs_err < 0.01,
        "Eccentricity mismatch: fixed={:.6}, adaptive={:.6}",
        rf.ecc,
        ra.ecc,
    );
}
```

- [ ] **Step 4: Run integration tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test dopri45_integration -- --nocapture`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/test/test_ref_adaptive.toml src/rust/tests/dopri45_integration.rs
git commit -m "test(e2e): add adaptive DOPRI45 integration tests — capture validity and Gill agreement"
```

---

### Task 7: Proptest — Robustness Properties

**Files:**
- Modify: `src/rust/src/integration/dopri45.rs` (add proptest to unit tests)

- [ ] **Step 1: Add proptest for step coverage and finiteness**

Add to the `tests` module in `dopri45.rs` (requires `proptest` which is already a dev-dependency):

```rust
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
```

- [ ] **Step 2: Run proptest**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test dopri45::tests::step_produces_finite_output -- --nocapture`
Expected: PASS (256 cases by default).

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/integration/dopri45.rs
git commit -m "test(integration): add proptest for DOPRI45 robustness — finiteness and state restoration"
```

---

### Task 8: PyO3 Override Support and Test

**Files:**
- Modify: `tests/test_pyo3.py`

The PyO3 override system already handles arbitrary dot-paths and creates intermediate tables, so no Rust-side changes are needed for `aerocapture-py/`. We just need a test to confirm it works end-to-end.

- [ ] **Step 1: Rebuild PyO3 bindings**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust/aerocapture-py && maturin develop --release`
Expected: Builds successfully.

- [ ] **Step 2: Add PyO3 test for adaptive override**

Add a new test class to `tests/test_pyo3.py`:

```python
class TestAdaptiveIntegration:
    """Test adaptive DOPRI45 integration via PyO3 overrides."""

    def test_adaptive_override_produces_valid_result(self):
        """Setting integration.mode = 'adaptive' via overrides should work."""
        result = aero.run(
            GOLDEN_TOML,
            overrides={"integration.mode": "adaptive", "integration.rtol": 1e-6},
        )
        assert result.captured, "Adaptive mode should produce a captured trajectory"
        assert result.final_record.shape == (52,)

    def test_adaptive_agrees_with_fixed(self):
        """Adaptive and fixed modes should produce similar results on the same config."""
        r_fixed = aero.run(GOLDEN_TOML)
        r_adaptive = aero.run(
            GOLDEN_TOML,
            overrides={"integration.mode": "adaptive"},
        )
        assert r_fixed.captured
        assert r_adaptive.captured
        # Energy agreement within 1%
        energy_err = abs(r_fixed.energy - r_adaptive.energy) / abs(r_fixed.energy)
        assert energy_err < 0.01, f"Energy mismatch: {energy_err:.4f}"
```

- [ ] **Step 3: Run PyO3 tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_pyo3.py -v -k "TestAdaptiveIntegration" 2>&1 | tail -20`
Expected: Both tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pyo3.py
git commit -m "test(pyo3): add adaptive integration override tests"
```

---

### Task 9: Smart Commit — Final Documentation Sync

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole `feature/Improve_simulation` branch into account. This will update CLAUDE.md and README.md to reflect the new adaptive integration capability, then commit everything.
