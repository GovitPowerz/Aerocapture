# Shortest-Path Bank Angle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix bank angle wrap-around so the control chain always takes the shortest angular path through ±π, correcting rate saturation, pilot dynamics, and bank consumption tracking.

**Architecture:** Add a `shortest_angle_diff(from, to)` utility in a new `angle_utils.rs` module, then apply it at the three call sites (ftc.rs rate saturation, pilot.rs dynamics, runner.rs consumption). TDD with proptest property tests + specific edge cases.

**Tech Stack:** Rust (nalgebra, proptest), cargo test

**Spec:** `docs/superpowers/specs/2026-03-24-shortest-path-bank-angle-design.md`

---

### Task 1: Create `angle_utils.rs` with tests

**Files:**
- Create: `src/rust/src/gnc/control/angle_utils.rs`
- Modify: `src/rust/src/gnc/control/mod.rs`

- [ ] **Step 1: Write the failing tests**

Create `src/rust/src/gnc/control/angle_utils.rs` with the test module first:

```rust
//! Angular utility functions for wrap-aware bank angle control.

use std::f64::consts::{PI, TAU};

/// Shortest signed angular difference from `from` to `to`, in [-PI, PI].
///
/// Returns the smallest rotation needed to get from `from` to `to`,
/// with positive meaning counterclockwise and negative meaning clockwise.
/// Inputs must be finite; propagates NaN for non-finite inputs.
#[inline]
pub fn shortest_angle_diff(from: f64, to: f64) -> f64 {
    debug_assert!(from.is_finite() && to.is_finite(), "shortest_angle_diff: inputs must be finite");
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
        // 170° to -170° should be +20° (short path through +180°)
        let d = shortest_angle_diff(170.0 * DEG, -170.0 * DEG);
        assert_abs_diff_eq!(d, 20.0 * DEG, epsilon = 1e-12);
    }

    #[test]
    fn wrap_through_minus_pi() {
        // -170° to 170° should be -20° (short path through -180°)
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
        // |diff| should be 0 or very close (they're the same angle)
        let d1 = shortest_angle_diff(PI, -PI);
        let d2 = shortest_angle_diff(-PI, PI);
        assert!(d1.abs() < 1e-12, "PI to -PI should be ~0, got {d1}");
        assert!(d2.abs() < 1e-12, "-PI to PI should be ~0, got {d2}");
    }

    #[test]
    fn normal_no_wrap_case() {
        // 30° to 60° should be +30°
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
```

- [ ] **Step 2: Register the module in `mod.rs`**

In `src/rust/src/gnc/control/mod.rs`, add:

```rust
pub mod angle_utils;
```

After the existing `pub mod attitude;` and `pub mod pilot;` lines.

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd src/rust && cargo test gnc::control::angle_utils -v`
Expected: All 9 tests pass (6 specific + 3 proptest).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/control/angle_utils.rs src/rust/src/gnc/control/mod.rs
git commit -m "feat: add shortest_angle_diff utility with proptest coverage"
```

---

### Task 2: Apply to rate saturation and cumulative tracking in `ftc.rs`

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs:257-278`

- [ ] **Step 1: Add import**

At the top of `ftc.rs`, add to the existing imports:

```rust
use crate::gnc::control::angle_utils::shortest_angle_diff;
```

- [ ] **Step 2: Replace rate saturation block**

Replace lines 257-278 (the `// === Roll rate saturation ===` block through the cumulative tracking):

Old code:
```rust
    // === Roll rate saturation ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    let bank_rate = (state.bank_angle_commanded - state.bank_angle_previous) / guidance_period;
    let mut rate_saturated = 0;

    if bank_rate.abs() - max_bank_rate > 1e-10 {
        rate_saturated = 1;
        if state.bank_angle_commanded > state.bank_angle_previous {
            state.bank_angle_commanded =
                state.bank_angle_previous + max_bank_rate * guidance_period;
        } else {
            state.bank_angle_commanded =
                state.bank_angle_previous - max_bank_rate * guidance_period;
        }
    }

    // Cumulative bank angle tracking
    if bank_rate.abs() > 1e-10 {
        state.cumulative_bank_change +=
            (state.bank_angle_commanded - state.bank_angle_previous).abs();
    }
```

New code:
```rust
    // === Roll rate saturation (wrap-aware) ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    let angle_diff = shortest_angle_diff(state.bank_angle_previous, state.bank_angle_commanded);
    let bank_rate = angle_diff / guidance_period;
    let mut rate_saturated = 0;

    if bank_rate.abs() - max_bank_rate > 1e-10 {
        rate_saturated = 1;
        state.bank_angle_commanded =
            state.bank_angle_previous + max_bank_rate.copysign(angle_diff) * guidance_period;
    }

    // Cumulative bank angle tracking (shortest path)
    if bank_rate.abs() > 1e-10 {
        state.cumulative_bank_change += angle_diff.abs();
    }
```

- [ ] **Step 3: Run all Rust tests**

Run: `cd src/rust && cargo test`
Expected: All ~201 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "fix: use shortest_angle_diff in rate saturation and bank tracking"
```

---

### Task 3: Apply to pilot dynamics in `pilot.rs`

**Files:**
- Modify: `src/rust/src/gnc/control/pilot.rs:5,41,52`

- [ ] **Step 1: Add import and fix error computations**

Add import at top of `pilot.rs`:

```rust
use crate::gnc::control::angle_utils::shortest_angle_diff;
```

Replace line 41 (`let error = commanded - state.bank_angle;`):
```rust
            let error = shortest_angle_diff(state.bank_angle, commanded);
```

Replace line 52 (`let error = state.bank_angle - commanded;`):
```rust
            let error = shortest_angle_diff(commanded, state.bank_angle);
```

- [ ] **Step 2: Add wrap-around test**

Add to the existing `mod tests` block in `pilot.rs`:

```rust
    #[test]
    fn first_order_wraps_through_pi_shortest_path() {
        // Current at +170°, commanded at -170° — should go through +180° (positive rate)
        use std::f64::consts::PI;
        let deg = PI / 180.0;
        let model = make_model(PilotType::FirstOrder, 1.0, 0.0, 0.0);
        let state = PilotState {
            bank_angle: 170.0 * deg,
            bank_rate: 0.0,
        };
        let result = apply_pilot(&model, -170.0 * deg, &state, 0.1, 10.0, &PilotBiases::default());
        // Shortest diff is +20° (0.349 rad), rate = 0.349/1.0 = 0.349, new = 170° + 0.349*0.1 = 170.035°
        // Key: rate should be POSITIVE (going through +180°), not negative (going through 0°)
        assert!(result.bank_rate > 0.0, "rate should be positive (through +180°), got {}", result.bank_rate);
        assert!(result.bank_angle > 170.0 * deg, "should move toward +180°, got {} deg", result.bank_angle / deg);
    }
```

- [ ] **Step 3: Run pilot tests**

Run: `cd src/rust && cargo test gnc::control::pilot -v`
Expected: All 6 tests pass (5 existing + 1 new).

- [ ] **Step 4: Run full test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/control/pilot.rs
git commit -m "fix: use shortest_angle_diff in pilot dynamics error computation"
```

---

### Task 4: Apply to bank consumption in `runner.rs`

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:518`

- [ ] **Step 1: Add import**

At the top of `runner.rs`, add to existing imports:

```rust
use crate::gnc::control::angle_utils::shortest_angle_diff;
```

- [ ] **Step 2: Replace bank change computation**

Replace line 518:
```rust
            let bank_change = (pilot_state.bank_angle - sim.bank_angle).abs();
```

With:
```rust
            let bank_change = shortest_angle_diff(sim.bank_angle, pilot_state.bank_angle).abs();
```

- [ ] **Step 3: Run full test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass.

- [ ] **Step 4: Run E2E regression test**

Run: `cd src/rust && cargo test e2e -v`
Expected: All E2E tests pass. The FTC guided regression test should be unchanged (bank angles don't cross ±π in standard FTC trajectories with perfect pilot).

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "fix: use shortest_angle_diff in bank consumption tracking"
```

---

### Task 5: Lint, verify, and final checks

**Files:**
- All modified files

- [ ] **Step 1: Run clippy and fmt**

Run: `cd src/rust && cargo fmt --check && cargo clippy -- -D warnings`
Expected: No errors.

- [ ] **Step 2: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass.

- [ ] **Step 3: Fix any lint issues and commit**

If clippy/fmt produced issues:
```bash
cd src/rust && cargo fmt
git add -u
git commit -m "style: apply cargo fmt"
```

---

### Task 6: Update TODO.md and smart-commit

- [ ] **Step 1: Remove the completed TODO item**

In `TODO.md`, remove the line:
```
- [ ] improve control to follow shortest path (mod[2pi])
```

- [ ] **Step 2: Invoke the `smart-commit` skill**

Use `/smart-commit` to sync docs and commit the whole branch.
