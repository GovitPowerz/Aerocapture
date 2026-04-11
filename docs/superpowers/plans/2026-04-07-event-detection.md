# DOPRI45 Event Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add precise event detection (bounce, atmosphere exit, crash, phase transition) to the DOPRI45 adaptive integrator using dense output interpolation and Brent's root-finding, achieving ~1 ms event location accuracy.

**Architecture:** Dense output (4th-order continuous extension of DOPRI45) provides cheap interpolation within accepted substeps. After each substep, event functions are checked for sign changes; Brent's method locates zero-crossings on the dense output polynomial. Events are handled inside the adaptive integration loop, replacing the post-tick threshold checks for DOPRI45 mode. Fixed RK4 is unchanged.

**Tech Stack:** Rust (nalgebra not needed -- pure scalar/array math), proptest for property-based testing.

**Spec:** `docs/superpowers/specs/2026-04-07-event-detection-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/rust/src/integration/dopri45.rs` | Dense output coefficients + `dopri45_dense()` + `dopri45_step_with_stages()` |
| `src/rust/src/integration/events.rs` | Event framework types + Brent's root-finder + `check_events_and_locate()` |
| `src/rust/src/integration/mod.rs` | Add `pub mod events;` |
| `src/rust/src/simulation/runner.rs` | `integrate_adaptive_with_events()`, `SimState.event_records`, main loop changes, trajectory interleaving |
| `src/rust/tests/event_detection.rs` | Integration tests for event precision, guards, multi-event, RK4 non-regression |

---

### Task 1: Dense Output Coefficients and Interpolation

**Files:**
- Modify: `src/rust/src/integration/dopri45.rs`

This task adds the DOPRI45 continuous extension polynomial and the interpolation function. The dense output coefficients are from Dormand & Prince (1986) / Hairer-Norsett-Wanner "Solving ODEs I" Table 6.2.

- [ ] **Step 1: Write failing test for dense output at theta=0 and theta=1 boundaries**

Add to the `#[cfg(test)] mod tests` block at the bottom of `dopri45.rs`:

```rust
/// Dense output at theta=0 must return y0 exactly.
/// Dense output at theta=1 must match the accepted 5th-order solution.
#[test]
fn dense_output_boundary_conditions() {
    let atol = [1e-12; 8];
    let rtol = 1e-10;
    let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y0 = state;
    let mut dopri = Dopri45State::new();
    let h = 0.1;

    let (result, stages) = dopri45_step_with_stages(
        &mut state,
        h,
        &mut dopri,
        &atol,
        rtol,
        &mut |s| {
            let mut d = [0.0; 8];
            d[0] = s[0]; // dy/dt = y (exponential)
            d[7] = 1.0;
            d
        },
    );
    assert!(result.accepted);

    let y_at_0 = dopri45_dense(&y0, h, 0.0, &stages);
    let y_at_1 = dopri45_dense(&y0, h, 1.0, &stages);

    for i in 0..8 {
        assert!(
            (y_at_0[i] - y0[i]).abs() < 1e-15,
            "theta=0: component {} should match y0: got {}, expected {}",
            i, y_at_0[i], y0[i],
        );
        assert!(
            (y_at_1[i] - state[i]).abs() < 1e-12,
            "theta=1: component {} should match y5: got {}, expected {}",
            i, y_at_1[i], state[i],
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test --lib dense_output_boundary_conditions 2>&1 | tail -5`
Expected: compilation error -- `dopri45_step_with_stages` and `dopri45_dense` don't exist yet.

- [ ] **Step 3: Implement `dopri45_step_with_stages`**

Add this function to `dopri45.rs`, below the existing `dopri45_step` function (before the `#[cfg(test)]` block). It's identical to `dopri45_step` but also returns the 7 stage derivative arrays:

```rust
/// Same as `dopri45_step`, but also returns the 7 stage derivative arrays
/// needed for dense output interpolation.
///
/// Returns `(StepResult, [[f64; N]; 7])`. The stages are valid only when
/// `result.accepted` is true.
pub fn dopri45_step_with_stages(
    state: &mut [f64; N],
    dt: f64,
    dopri: &mut Dopri45State,
    atol: &[f64; N],
    rtol: f64,
    deriv_fn: &mut impl FnMut(&[f64; N]) -> [f64; N],
) -> (StepResult, [[f64; N]; 7]) {
    let y0 = *state;

    let k1 = if dopri.fsal_valid {
        dopri.k_last
    } else {
        deriv_fn(state)
    };

    let mut y_stage = [0.0; N];
    for i in 0..N {
        y_stage[i] = y0[i] + dt * tableau::A[0][0] * k1[i];
    }
    let k2 = deriv_fn(&y_stage);

    for i in 0..N {
        y_stage[i] = y0[i] + dt * (tableau::A[1][0] * k1[i] + tableau::A[1][1] * k2[i]);
    }
    let k3 = deriv_fn(&y_stage);

    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[2][0] * k1[i] + tableau::A[2][1] * k2[i] + tableau::A[2][2] * k3[i]);
    }
    let k4 = deriv_fn(&y_stage);

    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[3][0] * k1[i]
                + tableau::A[3][1] * k2[i]
                + tableau::A[3][2] * k3[i]
                + tableau::A[3][3] * k4[i]);
    }
    let k5 = deriv_fn(&y_stage);

    for i in 0..N {
        y_stage[i] = y0[i]
            + dt * (tableau::A[4][0] * k1[i]
                + tableau::A[4][1] * k2[i]
                + tableau::A[4][2] * k3[i]
                + tableau::A[4][3] * k4[i]
                + tableau::A[4][4] * k5[i]);
    }
    let k6 = deriv_fn(&y_stage);

    let mut y5 = [0.0; N];
    for i in 0..N {
        y5[i] = y0[i]
            + dt * (tableau::B5[0] * k1[i]
                + tableau::B5[2] * k3[i]
                + tableau::B5[3] * k4[i]
                + tableau::B5[4] * k5[i]
                + tableau::B5[5] * k6[i]);
    }

    let k7 = deriv_fn(&y5);

    let mut y4 = [0.0; N];
    for i in 0..N {
        y4[i] = y0[i]
            + dt * (tableau::B4[0] * k1[i]
                + tableau::B4[2] * k3[i]
                + tableau::B4[3] * k4[i]
                + tableau::B4[4] * k5[i]
                + tableau::B4[5] * k6[i]
                + tableau::B4[6] * k7[i]);
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
        *state = y0;
        dopri.fsal_valid = false;
    }

    let stages = [k1, k2, k3, k4, k5, k6, k7];
    (StepResult { accepted, error_norm: err, dt_next }, stages)
}
```

- [ ] **Step 4: Implement dense output coefficients and `dopri45_dense`**

Add the dense output coefficients to the `tableau` module (inside the existing `mod tableau` block, after the `B4` constant), and the interpolation function after `dopri45_step_with_stages`:

Dense output coefficients inside `mod tableau`:

```rust
    /// Dense output coefficients for the 4th-order continuous extension.
    /// Each row corresponds to one stage (k1..k7). Each column is a polynomial
    /// coefficient: b_i(theta) = theta * (D[i][0] + theta * (D[i][1] + theta * (D[i][2] + theta * D[i][3])))
    ///
    /// Source: Dormand & Prince (1986), Hairer/Norsett/Wanner "Solving ODEs I" Table 6.2.
    pub const DENSE: [[f64; 4]; 7] = [
        // k1
        [1.0, -4034104133.0 / 1410260304.0, 105330401.0 / 33982176.0, -13107642775.0 / 11282082432.0],
        // k2 (zero -- stage 2 has zero weight in both B5 and the dense extension)
        [0.0, 0.0, 0.0, 0.0],
        // k3
        [0.0, 132343189600.0 / 32700410799.0, -833316000.0 / 131326951.0, 91394880000.0 / 32700410799.0],
        // k4
        [0.0, -115792950.0 / 29387358.0, 185270875.0 / 16991088.0, -12653452750.0 / 1880347072.0],
        // k5
        [0.0, 70805911779.0 / 67845853200.0, -2691285.0 / 1264628.0, 30210245.0 / 21467984.0],
        // k6
        [0.0, -236174517.0 / 199450200.0, 10557375.0 / 3992504.0, -22430157.0 / 19952520.0],
        // k7
        [0.0, 0.0, -100.0 / 63.0, 100.0 / 63.0],
    ];
```

Then the interpolation function (after `dopri45_step_with_stages`, before `#[cfg(test)]`):

```rust
/// Evaluate the 4th-order continuous extension of DOPRI45 at fractional offset
/// `theta` in [0, 1] within an accepted step of size `h`.
///
/// `y0` is the state at the start of the step, `k` contains the 7 stage derivatives.
/// Returns the interpolated state at time `t_n + theta * h`.
///
/// The interpolant is 4th-order accurate: error = O(h^5), matching the local
/// truncation error of the integrator itself.
pub fn dopri45_dense(
    y0: &[f64; N],
    h: f64,
    theta: f64,
    k: &[[f64; N]; 7],
) -> [f64; N] {
    let mut y = *y0;
    for i in 0..N {
        let mut sum = 0.0;
        for s in 0..7 {
            let d = &tableau::DENSE[s];
            let bi = theta * (d[0] + theta * (d[1] + theta * (d[2] + theta * d[3])));
            sum += bi * k[s][i];
        }
        y[i] += h * sum;
    }
    y
}
```

- [ ] **Step 5: Run the boundary condition test**

Run: `cd src/rust && cargo test --lib dense_output_boundary_conditions -- --nocapture 2>&1 | tail -10`
Expected: PASS. theta=0 returns y0, theta=1 matches y5.

- [ ] **Step 6: Write and run accuracy test for dense output at midpoints**

Add to the test module:

```rust
/// Dense output on a harmonic oscillator must match the analytical solution
/// to 4th-order accuracy at intermediate theta values.
#[test]
fn dense_output_midpoint_accuracy() {
    let atol = [1e-12; 8];
    let rtol = 1e-12;
    // state[0] = x, state[1] = v, state[7] = time
    let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y0 = state;
    let mut dopri = Dopri45State::new();
    let h = 0.5;

    let (result, stages) = dopri45_step_with_stages(
        &mut state,
        h,
        &mut dopri,
        &atol,
        rtol,
        &mut |s| {
            let mut d = [0.0; 8];
            d[0] = s[1];     // dx/dt = v
            d[1] = -s[0];    // dv/dt = -x
            d[7] = 1.0;
            d
        },
    );
    assert!(result.accepted, "Step should be accepted");

    for &theta in &[0.25, 0.5, 0.75] {
        let t = theta * h;
        let y_interp = dopri45_dense(&y0, h, theta, &stages);
        let x_exact = t.cos();
        let v_exact = -t.sin();
        // 4th-order dense output on h=0.5: error ~ h^5 = 0.5^5 = 0.03
        // In practice much better for smooth ODEs
        assert!(
            (y_interp[0] - x_exact).abs() < 1e-6,
            "theta={}: x interpolated={}, exact={}, err={}",
            theta, y_interp[0], x_exact, (y_interp[0] - x_exact).abs(),
        );
        assert!(
            (y_interp[1] - v_exact).abs() < 1e-6,
            "theta={}: v interpolated={}, exact={}, err={}",
            theta, y_interp[1], v_exact, (y_interp[1] - v_exact).abs(),
        );
    }
}
```

Run: `cd src/rust && cargo test --lib dense_output_midpoint_accuracy -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 7: Add proptest for dense output finiteness**

Add to the proptest block in the test module:

```rust
/// Dense output must produce finite values for any theta in [0, 1]
/// given a valid accepted step.
#[test]
fn dense_output_always_finite(
    theta in 0.0_f64..=1.0,
    r in 3.3e6_f64..3.5e6,
    v in 3000.0_f64..7000.0,
    gamma in -0.15_f64..0.05,
) {
    let atol = [1.0, 1e-8, 1e-8, 1e-3, 1e-8, 1e-8, 1e-2, 1e-6];
    let rtol = 1e-6;
    let mut state = [r, 0.0, 0.0, v, gamma, 0.0, 0.0, 0.0];
    let y0 = state;
    let mut dopri = Dopri45State::new();
    let mu = 4.2828e13_f64;

    let (result, stages) = dopri45_step_with_stages(
        &mut state,
        0.1,
        &mut dopri,
        &atol,
        rtol,
        &mut |s| {
            let mut d = [0.0; 8];
            d[0] = s[3] * s[4].sin();
            d[3] = -mu / (s[0] * s[0]) * s[4].sin();
            d[4] = (s[3] / s[0] - mu / (s[0] * s[0] * s[3])) * s[4].cos();
            d[7] = 1.0;
            d
        },
    );

    if result.accepted {
        let y_interp = dopri45_dense(&y0, 0.1, theta, &stages);
        for (i, &val) in y_interp.iter().enumerate() {
            prop_assert!(val.is_finite(), "dense output[{}] = {} is not finite at theta={}", i, val, theta);
        }
    }
}
```

Run: `cd src/rust && cargo test --lib dense_output_always_finite 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/integration/dopri45.rs
git commit -m "feat: add DOPRI45 dense output interpolation

4th-order continuous extension with standard Dormand-Prince
coefficients. dopri45_step_with_stages returns stage derivatives
for interpolation. dopri45_dense evaluates the polynomial at any
theta in [0,1] within an accepted step."
```

---

### Task 2: Brent's Root-Finder

**Files:**
- Create: `src/rust/src/integration/events.rs`
- Modify: `src/rust/src/integration/mod.rs`

This task implements Brent's method for locating zero-crossings of a scalar function on a bracketed interval. It's a standalone numerical utility used later by the event detection loop.

- [ ] **Step 1: Write failing test for Brent's method on sin(x)**

Create `src/rust/src/integration/events.rs`:

```rust
//! Event detection for adaptive ODE integration.
//!
//! Provides event function evaluation, Brent's root-finding on dense output,
//! and multi-event arbitration within DOPRI45 substeps.

const N: usize = 8;

/// Find a root of `f` on the interval `[a, b]` using Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs. Returns the root
/// location to within `tol` absolute tolerance. Panics if the bracket
/// is invalid (same sign at both endpoints).
///
/// Maximum iterations is capped at 50 (sufficient for double precision
/// on any reasonable bracket width down to ~1e-15).
pub fn brent(
    mut a: f64,
    mut b: f64,
    tol: f64,
    f: &mut impl FnMut(f64) -> f64,
) -> f64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_finds_sin_root() {
        // sin(x) has a root at pi, bracket [3.0, 3.5]
        let root = brent(3.0, 3.5, 1e-12, &mut |x| x.sin());
        assert!(
            (root - std::f64::consts::PI).abs() < 1e-11,
            "Expected root near pi, got {}",
            root,
        );
    }
}
```

- [ ] **Step 2: Register the module**

Add to `src/rust/src/integration/mod.rs`:

```rust
pub mod events;
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/rust && cargo test --lib brent_finds_sin_root 2>&1 | tail -5`
Expected: FAIL -- `todo!()` panics.

- [ ] **Step 4: Implement Brent's method**

Replace the `todo!()` body of `brent` in `events.rs`:

```rust
pub fn brent(
    mut a: f64,
    mut b: f64,
    tol: f64,
    f: &mut impl FnMut(f64) -> f64,
) -> f64 {
    const MAX_ITER: usize = 50;

    let mut fa = f(a);
    let mut fb = f(b);
    assert!(
        fa * fb <= 0.0,
        "brent: f(a)={} and f(b)={} must have opposite signs",
        fa,
        fb,
    );

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for _ in 0..MAX_ITER {
        if fb.abs() < tol * 0.1 {
            return b;
        }
        if (b - a).abs() < tol {
            return b;
        }

        // Inverse quadratic interpolation or secant
        let s = if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant method
            b - fb * (b - a) / (fb - fa)
        };

        // Conditions for rejecting interpolation and using bisection instead
        let between = if a < b {
            s > (3.0 * a + b) / 4.0 && s < b
        } else {
            s > b && s < (3.0 * a + b) / 4.0
        };
        let use_bisection = !between
            || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
            || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol);

        let s = if use_bisection {
            mflag = true;
            (a + b) / 2.0
        } else {
            mflag = false;
            s
        };

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    b
}
```

- [ ] **Step 5: Run the test**

Run: `cd src/rust && cargo test --lib brent_finds_sin_root -- --nocapture 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Add edge case tests**

Add to the test module in `events.rs`:

```rust
#[test]
fn brent_finds_linear_root() {
    // f(x) = x - 2.5, root at 2.5
    let root = brent(0.0, 5.0, 1e-12, &mut |x| x - 2.5);
    assert!((root - 2.5).abs() < 1e-11, "got {}", root);
}

#[test]
fn brent_root_at_endpoint_a() {
    let root = brent(0.0, 1.0, 1e-12, &mut |x| x);
    assert!(root.abs() < 1e-11, "got {}", root);
}

#[test]
fn brent_root_at_endpoint_b() {
    let root = brent(-1.0, 0.0, 1e-12, &mut |x| x);
    assert!(root.abs() < 1e-11, "got {}", root);
}

#[test]
fn brent_converges_on_tight_bracket() {
    // Bracket width 1e-4, should converge in very few iterations
    let pi = std::f64::consts::PI;
    let root = brent(pi - 5e-5, pi + 5e-5, 1e-12, &mut |x| x.sin());
    assert!((root - pi).abs() < 1e-11, "got {}", root);
}

#[test]
#[should_panic(expected = "opposite signs")]
fn brent_panics_on_same_sign() {
    brent(1.0, 2.0, 1e-12, &mut |x| x * x + 1.0);
}
```

Run: `cd src/rust && cargo test --lib integration::events::tests 2>&1 | tail -10`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/integration/events.rs src/rust/src/integration/mod.rs
git commit -m "feat: add Brent's root-finding method for event detection

Standalone Brent's method implementation with inverse quadratic
interpolation fallback to bisection. 50-iteration cap, bracket
validation. Tests cover sin, linear, endpoints, tight bracket,
and invalid bracket panic."
```

---

### Task 3: Event Framework Types

**Files:**
- Modify: `src/rust/src/integration/events.rs`

This task adds the type definitions for events, context, and results.

- [ ] **Step 1: Add event types**

Add these types at the top of `events.rs` (after the `const N` line, before the `brent` function):

```rust
use crate::simulation::runner::TermReason;

/// What happens when an event fires.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventAction {
    /// Stop the simulation with the given reason.
    Terminate(TermReason),
    /// Record the event state but continue integration (e.g., bounce).
    Record,
    /// Trigger capture-to-exit phase transition, continue integration.
    PhaseTransition,
}

/// Which type of event occurred (for output recording).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    Bounce,
    AtmosphereExit,
    Crash,
    PhaseTransition,
}

/// Read-only context for event function evaluation.
#[derive(Debug, Clone)]
pub struct EventContext {
    /// Planet equatorial radius (m).
    pub planet_radius: f64,
    /// Altitude above which the vehicle has exited the atmosphere (m).
    pub exit_altitude: f64,
    /// Relative velocity threshold for capture-to-exit phase transition (m/s).
    pub exit_velocity_threshold: f64,
}

/// Definition of a single event to monitor.
pub struct EventDef {
    /// Scalar function evaluated on state + context. Sign change = event.
    pub eval: fn(&[f64; N], &EventContext) -> f64,
    /// Direction filter: +1 = rising only, -1 = falling only, 0 = both.
    pub direction: i8,
    /// Action to take when this event fires.
    pub action: EventAction,
    /// Event type tag for output recording.
    pub event_type: EventType,
}

/// A recorded event with precise time and interpolated state.
#[derive(Debug, Clone)]
pub struct EventRecord {
    /// Absolute simulation time at the event (s).
    pub time: f64,
    /// Full 8-component state vector at the event.
    pub state: [f64; N],
    /// Which event type occurred.
    pub event_type: EventType,
}

/// Result of event checking within a substep.
pub struct TriggeredEvent {
    /// Index into the events slice.
    pub event_index: usize,
    /// Fractional offset within the substep [0, 1].
    pub theta: f64,
    /// Interpolated state at the event.
    pub state: [f64; N],
}
```

- [ ] **Step 2: Make `TermReason` public**

Currently `TermReason` is a private enum in `runner.rs`. The events module needs to reference it. Change its visibility in `src/rust/src/simulation/runner.rs`:

Find:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum TermReason {
```
Replace with:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TermReason {
```

- [ ] **Step 3: Verify compilation**

Run: `cd src/rust && cargo check 2>&1 | tail -5`
Expected: compiles with no errors.

- [ ] **Step 4: Implement the four event functions**

Add below the type definitions in `events.rs`:

```rust
/// Event function: bounce detection.
/// g(state) = sin(gamma). Rising through zero means FPA crosses from negative to positive.
fn event_bounce(state: &[f64; N], _ctx: &EventContext) -> f64 {
    state[4].sin()
}

/// Event function: atmosphere exit.
/// g(state) = altitude - exit_altitude. Rising through zero means exiting.
fn event_atmosphere_exit(state: &[f64; N], ctx: &EventContext) -> f64 {
    (state[0] - ctx.planet_radius) - ctx.exit_altitude
}

/// Event function: ground crash.
/// g(state) = altitude. Falling through zero means impact.
fn event_crash(state: &[f64; N], ctx: &EventContext) -> f64 {
    state[0] - ctx.planet_radius
}

/// Event function: phase transition.
/// g(state) = exit_velocity_threshold - V_relative. Rising through zero means
/// velocity has dropped below the threshold.
fn event_phase_transition(state: &[f64; N], ctx: &EventContext) -> f64 {
    ctx.exit_velocity_threshold - state[3]
}

/// Build the standard set of event definitions for aerocapture.
///
/// Returns 4 events: bounce, atmosphere exit, crash, phase transition.
/// The caller is responsible for applying guard conditions after root-finding.
pub fn build_aerocapture_events() -> Vec<EventDef> {
    vec![
        EventDef {
            eval: event_bounce,
            direction: 1,  // rising: sin(gamma) crosses 0 upward
            action: EventAction::Record,
            event_type: EventType::Bounce,
        },
        EventDef {
            eval: event_atmosphere_exit,
            direction: 1,  // rising: altitude crosses exit_altitude upward
            action: EventAction::Terminate(TermReason::AtmosphereExit),
            event_type: EventType::AtmosphereExit,
        },
        EventDef {
            eval: event_crash,
            direction: -1, // falling: altitude crosses 0 downward
            action: EventAction::Terminate(TermReason::Crash),
            event_type: EventType::Crash,
        },
        EventDef {
            eval: event_phase_transition,
            direction: 1,  // rising: threshold - V crosses 0 upward (V drops below threshold)
            action: EventAction::PhaseTransition,
            event_type: EventType::PhaseTransition,
        },
    ]
}
```

- [ ] **Step 5: Add a test for event function evaluation**

Add to the test module:

```rust
#[test]
fn event_functions_sign_correctness() {
    let ctx = EventContext {
        planet_radius: 3.4e6,
        exit_altitude: 60e3,
        exit_velocity_threshold: 4400.0,
    };

    // State: r=3.46e6 (alt=60km), lon=0, lat=0, V=5000, gamma=-0.1, psi=0, flux=0, t=0
    let state_descending = [3.46e6, 0.0, 0.0, 5000.0, -0.1, 0.0, 0.0, 0.0];
    // Descending: sin(gamma) < 0
    assert!(event_bounce(&state_descending, &ctx) < 0.0);
    // At exit altitude: alt - exit_alt = 0
    assert!((event_atmosphere_exit(&state_descending, &ctx)).abs() < 1.0);
    // Well above ground
    assert!(event_crash(&state_descending, &ctx) > 0.0);
    // V=5000 > 4400: threshold - V < 0
    assert!(event_phase_transition(&state_descending, &ctx) < 0.0);

    // State: ascending, slow, low
    let state_ascending = [3.41e6, 0.0, 0.0, 4000.0, 0.1, 0.0, 0.0, 0.0];
    // Ascending: sin(gamma) > 0
    assert!(event_bounce(&state_ascending, &ctx) > 0.0);
    // Below exit altitude
    assert!(event_atmosphere_exit(&state_ascending, &ctx) < 0.0);
    // Above ground
    assert!(event_crash(&state_ascending, &ctx) > 0.0);
    // V=4000 < 4400: threshold - V > 0
    assert!(event_phase_transition(&state_ascending, &ctx) > 0.0);
}
```

Run: `cd src/rust && cargo test --lib event_functions_sign_correctness 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/integration/events.rs src/rust/src/simulation/runner.rs
git commit -m "feat: add event detection type framework

EventDef, EventAction, EventType, EventContext, EventRecord,
TriggeredEvent types. Four event functions (bounce, atmosphere
exit, crash, phase transition) with direction filters. TermReason
made public for cross-module use."
```

---

### Task 4: Event Checking and Locating Within Substeps

**Files:**
- Modify: `src/rust/src/integration/events.rs`

This task implements the core logic: checking event functions for sign changes after each substep and locating the earliest crossing via Brent's method on the dense output.

- [ ] **Step 1: Write failing test for event location on a synthetic trajectory**

Add to the test module in `events.rs`:

```rust
use crate::integration::dopri45::{dopri45_dense, dopri45_step_with_stages, Dopri45State};

/// Simulate a simple ascending trajectory and detect the zero-crossing of
/// a linear "altitude" function. Verify the located theta is accurate.
#[test]
fn check_events_locates_zero_crossing() {
    let ctx = EventContext {
        planet_radius: 0.0,    // simplified: altitude = state[0]
        exit_altitude: 1.5,    // event at state[0] = 1.5
        exit_velocity_threshold: 0.0,
    };
    // Single event: state[0] crosses 1.5 upward
    let events = vec![EventDef {
        eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_altitude,
        direction: 1,
        action: EventAction::Terminate(TermReason::AtmosphereExit),
        event_type: EventType::AtmosphereExit,
    }];

    // ODE: dx/dt = 1 (linear ramp from 1.0 to 2.0 over h=1.0)
    let atol = [1e-12; 8];
    let rtol = 1e-12;
    let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y0 = state;
    let h = 1.0;

    let (result, stages) = dopri45_step_with_stages(
        &mut state,
        h,
        &mut Dopri45State::new(),
        &atol,
        rtol,
        &mut |_s| {
            let mut d = [0.0; 8];
            d[0] = 1.0; // dx/dt = 1
            d[7] = 1.0;
            d
        },
    );
    assert!(result.accepted);

    // g(y0) = 1.0 - 1.5 = -0.5 (before), g(y1) = 2.0 - 1.5 = 0.5 (after)
    let g_start = [(events[0].eval)(&y0, &ctx)];
    let triggered = check_events_and_locate(&y0, h, &stages, &events, &ctx, &g_start, 1e-3);

    assert!(triggered.is_some(), "Should detect the crossing");
    let t = triggered.unwrap();
    assert_eq!(t.event_index, 0);
    // Exact crossing at theta = 0.5 (state[0] goes from 1.0 to 2.0, crosses 1.5 at midpoint)
    assert!(
        (t.theta - 0.5).abs() < 0.01,
        "theta should be ~0.5, got {}",
        t.theta,
    );
    assert!(
        (t.state[0] - 1.5).abs() < 0.01,
        "state[0] at event should be ~1.5, got {}",
        t.state[0],
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test --lib check_events_locates_zero_crossing 2>&1 | tail -5`
Expected: FAIL -- `check_events_and_locate` doesn't exist yet.

- [ ] **Step 3: Implement `check_events_and_locate`**

Add to `events.rs` (after `build_aerocapture_events`, before `#[cfg(test)]`):

```rust
use crate::integration::dopri45::dopri45_dense;

/// Check all event functions for sign changes within a substep, and locate
/// the earliest zero-crossing via Brent's method on the dense output.
///
/// - `y0`: state at the start of the substep
/// - `h`: substep size (seconds)
/// - `stages`: the 7 stage derivatives from `dopri45_step_with_stages`
/// - `events`: the event definitions to check
/// - `ctx`: read-only event context
/// - `g_start`: cached event function values at `y0` (one per event)
/// - `tol`: root-finding tolerance in seconds (e.g., 1e-3 for 1 ms)
///
/// Returns `Some(TriggeredEvent)` if any event's sign changed with the correct
/// direction, `None` otherwise. When multiple events trigger in the same substep,
/// returns the one with the smallest theta (earliest in time).
pub fn check_events_and_locate(
    y0: &[f64; N],
    h: f64,
    stages: &[[f64; N]; 7],
    events: &[EventDef],
    ctx: &EventContext,
    g_start: &[f64],
    tol: f64,
) -> Option<TriggeredEvent> {
    // Evaluate all event functions at the end of the substep
    let y_end = dopri45_dense(y0, h, 1.0, stages);

    let mut earliest: Option<TriggeredEvent> = None;

    for (i, event) in events.iter().enumerate() {
        let g0 = g_start[i];
        let g1 = (event.eval)(&y_end, ctx);

        // Check for sign change
        if g0 * g1 > 0.0 {
            continue; // no sign change
        }
        // If g0 == 0 exactly, skip (event was at previous substep boundary)
        if g0 == 0.0 {
            continue;
        }

        // Check direction filter
        let rising = g0 < 0.0 && g1 >= 0.0;
        let falling = g0 > 0.0 && g1 <= 0.0;
        match event.direction {
            1 if !rising => continue,
            -1 if !falling => continue,
            _ => {}
        }

        // Locate the crossing via Brent's method on theta in [0, 1]
        let tol_theta = (tol / h).min(0.5); // convert seconds to theta-space
        let theta = brent(0.0, 1.0, tol_theta, &mut |theta| {
            let y_theta = dopri45_dense(y0, h, theta, stages);
            (event.eval)(&y_theta, ctx)
        });

        // Keep the earliest event
        let dominated = earliest.as_ref().is_some_and(|e| e.theta <= theta);
        if !dominated {
            let state = dopri45_dense(y0, h, theta, stages);
            earliest = Some(TriggeredEvent {
                event_index: i,
                theta,
                state,
            });
        }
    }

    earliest
}

/// Evaluate all event functions at a given state, returning one value per event.
pub fn evaluate_events(
    state: &[f64; N],
    events: &[EventDef],
    ctx: &EventContext,
) -> Vec<f64> {
    events.iter().map(|e| (e.eval)(state, ctx)).collect()
}
```

- [ ] **Step 4: Run the test**

Run: `cd src/rust && cargo test --lib check_events_locates_zero_crossing -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Add test for direction filtering**

```rust
#[test]
fn check_events_respects_direction_filter() {
    let ctx = EventContext {
        planet_radius: 0.0,
        exit_altitude: 0.5,
        exit_velocity_threshold: 0.0,
    };

    // Event: state[0] crosses 0.5, but only on FALLING (direction = -1)
    let events = vec![EventDef {
        eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_altitude,
        direction: -1, // only falling
        action: EventAction::Terminate(TermReason::Crash),
        event_type: EventType::Crash,
    }];

    let atol = [1e-12; 8];
    let rtol = 1e-12;
    // Rising trajectory: state[0] goes from 0.0 to 1.0 (crosses 0.5 upward)
    let mut state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y0 = state;
    let h = 1.0;

    let (result, stages) = dopri45_step_with_stages(
        &mut state, h, &mut Dopri45State::new(), &atol, rtol,
        &mut |_| { let mut d = [0.0; 8]; d[0] = 1.0; d[7] = 1.0; d },
    );
    assert!(result.accepted);

    let g_start = [(&events[0].eval)(&y0, &ctx)];
    let triggered = check_events_and_locate(&y0, h, &stages, &events, &ctx, &g_start, 1e-3);

    assert!(triggered.is_none(), "Rising crossing should be filtered out by direction=-1");
}
```

Run: `cd src/rust && cargo test --lib check_events_respects_direction 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 6: Add test for earliest-event arbitration**

```rust
#[test]
fn check_events_picks_earliest_of_two() {
    let ctx = EventContext {
        planet_radius: 0.0,
        exit_altitude: 0.0,
        exit_velocity_threshold: 0.0,
    };

    // Two events, both triggered by state[0] crossing a threshold:
    // Event A: state[0] crosses 0.3 (earlier, at theta ~0.3)
    // Event B: state[0] crosses 0.7 (later, at theta ~0.7)
    let events = vec![
        EventDef {
            eval: |state: &[f64; 8], _ctx: &EventContext| state[0] - 0.3,
            direction: 1,
            action: EventAction::Record,
            event_type: EventType::Bounce,
        },
        EventDef {
            eval: |state: &[f64; 8], _ctx: &EventContext| state[0] - 0.7,
            direction: 1,
            action: EventAction::Terminate(TermReason::AtmosphereExit),
            event_type: EventType::AtmosphereExit,
        },
    ];

    let atol = [1e-12; 8];
    let rtol = 1e-12;
    let mut state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y0 = state;
    let h = 1.0;

    let (result, stages) = dopri45_step_with_stages(
        &mut state, h, &mut Dopri45State::new(), &atol, rtol,
        &mut |_| { let mut d = [0.0; 8]; d[0] = 1.0; d[7] = 1.0; d },
    );
    assert!(result.accepted);

    let g_start = evaluate_events(&y0, &events, &ctx);
    let triggered = check_events_and_locate(&y0, h, &stages, &events, &ctx, &g_start, 1e-3);

    assert!(triggered.is_some());
    let t = triggered.unwrap();
    assert_eq!(t.event_index, 0, "Should pick event A (earlier, at theta ~0.3)");
    assert!(t.theta < 0.5, "theta should be ~0.3, got {}", t.theta);
}
```

Run: `cd src/rust && cargo test --lib check_events_picks_earliest 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/integration/events.rs
git commit -m "feat: add event checking and zero-crossing location

check_events_and_locate evaluates event functions after each
DOPRI45 substep, detects sign changes with direction filtering,
locates the earliest crossing via Brent's method on dense output.
evaluate_events helper for caching g values."
```

---

### Task 5: Wire Event Detection Into the Adaptive Integrator

**Files:**
- Modify: `src/rust/src/simulation/runner.rs`

This is the main integration task: replace the event-unaware `integrate_adaptive` with `integrate_adaptive_with_events`, and update the main loop to use it.

- [ ] **Step 1: Add `event_records` to `SimState`**

In `runner.rs`, add the import and modify `SimState`:

Add to the imports at the top of `runner.rs`:

```rust
use crate::integration::events::{
    self, EventAction, EventContext, EventDef, EventRecord, EventType, TriggeredEvent,
};
```

Add a new field to the `SimState` struct:

```rust
    // Event detection records (DOPRI45 only)
    event_records: Vec<EventRecord>,
```

And initialize it in `run_single` where `SimState` is constructed:

```rust
        event_records: Vec::new(),
```

- [ ] **Step 2: Implement `integrate_adaptive_with_events`**

Add this function after the existing `integrate_adaptive` function (keep `integrate_adaptive` -- it will be dead code until step 4 removes the old call site):

```rust
/// Result from adaptive integration with event detection.
struct AdaptiveEventResult {
    stats: AdaptiveStepStats,
    /// If an event fired, its details.
    triggered: Option<TriggeredEvent>,
}

/// Event location tolerance in seconds (~1 ms).
const EVENT_TOL: f64 = 1e-3;

/// Advance the state by `dt_outer` using adaptive DOPRI45 sub-stepping
/// with event detection via dense output interpolation.
///
/// After each accepted substep, checks all event functions for sign changes.
/// If a crossing is found, locates it to within EVENT_TOL seconds via Brent's
/// method on the dense output polynomial. Returns the earliest triggered event.
///
/// For `Record` events: the state is set to the event point and integration
/// resumes for the remainder of the outer tick.
/// For `Terminate` events: integration stops immediately.
/// For `PhaseTransition` events: the event is recorded and integration resumes.
fn integrate_adaptive_with_events(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    event_defs: &[EventDef],
    event_ctx: &EventContext,
    tick_start_time: f64,
) -> AdaptiveEventResult {
    const MAX_SUBSTEPS: u32 = 1000;

    let bank_angle = sim.bank_angle;
    let aoa = sim.aoa;
    let mut t_remaining = dt_outer;
    let mut h = config.initial_dt.min(t_remaining).max(config.min_dt);
    let mut n_substeps: u32 = 0;
    let mut n_rejections: u32 = 0;
    let mut term_event: Option<TriggeredEvent> = None;

    // Cache event function values at the current state
    let mut g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);

    while t_remaining > 1e-14 {
        h = h.min(t_remaining).min(config.max_dt).max(config.min_dt);

        if t_remaining <= config.min_dt * 1.5 {
            h = t_remaining;
        }

        let y0 = sim.state;

        let (result, stages) = dopri45::dopri45_step_with_stages(
            &mut sim.state,
            h,
            &mut sim.dopri,
            &DOPRI45_ATOL,
            config.rtol,
            &mut |state| compute_derivatives(state, bank_angle, aoa, planet, data, run_state),
        );

        if result.accepted {
            // Check for events within this substep
            if let Some(triggered) = events::check_events_and_locate(
                &y0, h, &stages, event_defs, event_ctx, &g_prev, EVENT_TOL,
            ) {
                let event_time = tick_start_time + (dt_outer - t_remaining) + triggered.theta * h;

                match event_defs[triggered.event_index].action {
                    EventAction::Terminate(_) => {
                        // Set state to the event point, stop integration
                        sim.state = triggered.state;
                        sim.dopri.fsal_valid = false;
                        sim.event_records.push(EventRecord {
                            time: event_time,
                            state: triggered.state,
                            event_type: event_defs[triggered.event_index].event_type,
                        });
                        return AdaptiveEventResult {
                            stats: AdaptiveStepStats {
                                n_substeps,
                                n_rejections,
                                hit_limit: false,
                            },
                            triggered: Some(triggered),
                        };
                    }
                    EventAction::Record | EventAction::PhaseTransition => {
                        // Record the event, set state to event point, resume from there
                        sim.state = triggered.state;
                        sim.dopri.fsal_valid = false;
                        sim.event_records.push(EventRecord {
                            time: event_time,
                            state: triggered.state,
                            event_type: event_defs[triggered.event_index].event_type,
                        });
                        // Adjust remaining time: consumed up to the event point
                        let t_consumed = triggered.theta * h;
                        t_remaining -= t_consumed;
                        // Reset step size for the remainder
                        h = config.initial_dt.min(t_remaining).max(config.min_dt);
                        // Re-evaluate event functions at the new state
                        g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);
                        n_substeps += 1;

                        // Return the triggered event info so the caller can apply guards
                        // But don't stop integration -- continue the loop
                        term_event = Some(triggered);
                        continue;
                    }
                }
            }

            // No event in this substep -- normal advancement
            t_remaining -= h;
            n_substeps += 1;
            h = result.dt_next;
            g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);
        } else {
            n_rejections += 1;
            h = result.dt_next;
        }

        if n_substeps + n_rejections >= MAX_SUBSTEPS {
            eprintln!(
                "WARNING: adaptive integrator hit {} step limit with t_remaining={:.2e}s ({} accepted, {} rejected)",
                MAX_SUBSTEPS, t_remaining, n_substeps, n_rejections,
            );
            return AdaptiveEventResult {
                stats: AdaptiveStepStats {
                    n_substeps,
                    n_rejections,
                    hit_limit: true,
                },
                triggered: term_event,
            };
        }
    }

    AdaptiveEventResult {
        stats: AdaptiveStepStats {
            n_substeps,
            n_rejections,
            hit_limit: false,
        },
        triggered: term_event,
    }
}
```

- [ ] **Step 3: Update the main loop to use `integrate_adaptive_with_events`**

This step modifies the main loop in `run_single`. The changes are:

1. Before the main loop, build the event context and event definitions.
2. Replace the `IntegrationMode::AdaptiveDopri45` arm to call `integrate_adaptive_with_events`.
3. After integration, process triggered events (apply guard conditions, update sim state).
4. Remove the 4 root-found event checks from the post-tick section for DOPRI45 mode; keep them for fixed RK4.

Before the `while term == TermReason::None {` loop, add:

```rust
    // Event detection setup (DOPRI45 only)
    let event_defs = events::build_aerocapture_events();
    let event_ctx = EventContext {
        planet_radius: planet.equatorial_radius,
        exit_altitude: exit_altitude,
        exit_velocity_threshold: data.guidance.exit_velocity_threshold,
    };
```

Replace the integration match block (currently lines ~819-826):

```rust
        // === Integration step ===
        let mut adaptive_event: Option<TriggeredEvent> = None;
        match &data.integration_mode {
            IntegrationMode::FixedGill => {
                integrate_step(&mut sim, dt, planet, data, &run_state);
            }
            IntegrationMode::AdaptiveDopri45(adaptive_config) => {
                let result = integrate_adaptive_with_events(
                    &mut sim,
                    dt,
                    adaptive_config,
                    planet,
                    data,
                    &run_state,
                    &event_defs,
                    &event_ctx,
                    sim_time - dt, // tick_start_time (sim_time already advanced)
                );
                adaptive_event = result.triggered;
            }
        }
```

After `track_peak_values`, process adaptive events with guard conditions:

```rust
        // Process events detected by the adaptive integrator
        if let Some(ref triggered) = adaptive_event {
            let event = &event_defs[triggered.event_index];
            match event.event_type {
                EventType::Bounce => {
                    // Guard: only fire if not already bounced
                    if !sim.bounced {
                        let evt_record = sim.event_records.last().unwrap();
                        let evt_alt = evt_record.state[0] - planet.equatorial_radius;
                        sim.bounced = true;
                        sim.bounce_alt = evt_alt;
                        sim.bounce_time = evt_record.time;
                    }
                }
                EventType::AtmosphereExit => {
                    // Guard: only fire if bounced
                    if sim.bounced {
                        let evt_record = sim.event_records.last().unwrap();
                        sim_time = evt_record.time;
                        term = TermReason::AtmosphereExit;
                    }
                }
                EventType::Crash => {
                    let evt_record = sim.event_records.last().unwrap();
                    sim_time = evt_record.time;
                    term = TermReason::Crash;
                }
                EventType::PhaseTransition => {
                    // Guard: only fire if bounced and not already locked
                    // Actual phase transition happens in the navigation layer
                    // on its next tick -- we just record the precise time.
                }
            }
        }
```

Finally, wrap the existing post-tick termination checks so they only run for fixed RK4:

```rust
        // === Termination checks (fixed RK4 only -- DOPRI45 uses event detection) ===
        if matches!(data.integration_mode, IntegrationMode::FixedGill) {
            if altitude <= 0.0 {
                term = TermReason::Crash;
            }
            if sim.bounced && altitude >= exit_altitude {
                term = TermReason::AtmosphereExit;
            }

            // Bounce detection
            if !sim.bounced && sim.state[4].sin() >= 0.0 {
                sim.bounced = true;
                sim.bounce_alt = altitude;
                sim.bounce_time = sim_time;
            }
        }

        if sim_time >= max_time {
            term = TermReason::Timeout;
        }
```

Note: the timeout check, atmospheric apoapsis crash, and trapped orbit detection remain for both modes (they're composite checks, not zero-crossings).

- [ ] **Step 4: Verify compilation and existing tests pass**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all existing tests pass. The adaptive integration test (`adaptive_produces_valid_capture`) should still pass since the event detection is now active but the trajectory behavior is the same -- just with more precise event timing.

- [ ] **Step 5: Remove the now-unused `integrate_adaptive` function**

The old `integrate_adaptive` is no longer called. Remove it.

Run: `cd src/rust && cargo test 2>&1 | tail -10`
Expected: still passes, no unused warnings.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "feat: wire event detection into DOPRI45 adaptive integrator

integrate_adaptive_with_events replaces integrate_adaptive for
DOPRI45 mode. Events (bounce, exit, crash, phase transition) are
detected within substeps via dense output + Brent's method.
Guard conditions applied after root-finding. Fixed RK4 path
unchanged. Post-tick checks split by integration mode."
```

---

### Task 6: Trajectory Output Interleaving

**Files:**
- Modify: `src/rust/src/simulation/runner.rs`

Event records need to be interleaved into the trajectory output at their correct time positions.

- [ ] **Step 1: Add event rows to trajectory output**

In `run_for_api` and `run_for_api_with_draws`, the trajectory is built from `r.photo_lines`. We need to interleave event records. However, event records are stored on `SimState` (which doesn't survive past `run_single`), so we need to pass them through `SimResult`.

Add a new field to `SimResult`:

```rust
struct SimResult {
    sim_idx: i32,
    final_line: [f64; 52],
    photo_lines: Vec<[f64; 30]>,
    dispersions: [f64; DISPERSION_DRAW_LEN],
    event_records: Vec<EventRecord>,
}
```

At the end of `run_single`, before the `Ok(SimResult { ... })`, move event records out of SimState:

```rust
    let event_records = std::mem::take(&mut sim.event_records);

    Ok(SimResult {
        sim_idx,
        final_line: final_record,
        photo_lines,
        dispersions: [0.0; DISPERSION_DRAW_LEN],
        event_records,
    })
```

- [ ] **Step 2: Build trajectory rows from event records**

Add a helper function that converts an `EventRecord` into the 17-element trajectory row format:

```rust
/// Convert an EventRecord into the 17-column trajectory row format.
/// Uses the raw state vector -- no GNC-dependent quantities (nav density ratio = 1.0,
/// density perturbation = 0.0) since events occur between GNC ticks.
fn event_record_to_trajectory_row(
    record: &EventRecord,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 17] {
    let state = &record.state;
    let (altitude, _lat_geo) =
        geodetic_from_spherical(state[0], state[1], state[2], planet);
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        0.0, // no perturbation info at sub-tick resolution
    );
    let v_eff = effective_airspeed(
        state[3], state[4], state[5], state[2], altitude, data, run_state,
    );
    let heat_flux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    let pdyn = 0.5 * rho * v_eff * v_eff;
    let (_pos_abs, vel_abs) = to_absolute_cartesian(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );
    let speed_abs = norm(&vel_abs);
    let energy = speed_abs * speed_abs / 2.0 - planet.mu / state[0];
    let orbit = elements::from_spherical(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );

    [
        altitude / 1e3,                  // [0]  alt_km
        state[1] / DEG_TO_RAD,           // [1]  lon_deg
        state[2] / DEG_TO_RAD,           // [2]  lat_deg (geocentric, consistent with photo)
        state[3],                         // [3]  vel_m_s
        state[4] / DEG_TO_RAD,           // [4]  fpa_deg
        state[5] / DEG_TO_RAD,           // [5]  heading_deg
        heat_flux / 1e3,                  // [6]  heat_flux_kw_m2
        record.time,                      // [7]  time_s
        energy / 1e6,                     // [8]  energy_mj_kg
        pdyn / 1e3,                       // [9]  pdyn_kpa
        0.0,                              // [10] bank_angle_deg (not available at sub-tick)
        orbit.inclination / DEG_TO_RAD,   // [11] inclination_deg
        0.0,                              // [12] g_load_g (would need bank angle)
        1.0,                              // [13] nav_density_ratio (no nav at sub-tick)
        rho,                              // [14] truth_density_kg_m3
        state[6] / 1e3,                   // [15] heat_load_kj_m2
        0.0,                              // [16] density_perturbation
    ]
}
```

- [ ] **Step 3: Interleave event rows in `run_for_api`**

In both `run_for_api` and `run_for_api_with_draws`, after building the trajectory from photo_lines, interleave event records and sort by time. Replace the trajectory building logic in `run_for_api`:

```rust
            let mut trajectory: Vec<[f64; 17]> = if include_trajectories {
                let mut rows: Vec<[f64; 17]> = r.photo_lines
                    .iter()
                    .map(|p| {
                        [
                            p[1], p[2], p[3], p[4], p[5], p[6], p[24], p[0],
                            p[18] / 1e6, p[19] / 1e3, p[14], p[9], p[25],
                            p[26], p[27], p[28], p[29],
                        ]
                    })
                    .collect();
                // Interleave event records at their correct time positions
                for record in &r.event_records {
                    rows.push(event_record_to_trajectory_row(record, &config.planet, data, &run_state_for_events));
                }
                rows.sort_by(|a, b| a[7].partial_cmp(&b[7]).unwrap_or(std::cmp::Ordering::Equal));
                rows
            } else {
                Vec::new()
            };
```

Note: `run_state_for_events` is not available in `run_for_api` because run states are constructed inside `run_core`. This is a problem. The simplest fix: have `event_record_to_trajectory_row` take just the planet config and compute minimal fields, or store the trajectory row directly in `SimResult` instead of the raw `EventRecord`.

Actually, the cleaner approach: convert event records to trajectory rows inside `run_single` (where `run_state` is available) and store them as additional `photo_lines` entries already in the right format. This avoids threading `run_state` through `run_for_api`.

Revised approach -- at the end of `run_single`, before returning, append event records as photo lines:

```rust
    // Append event records as trajectory rows (interleaved by time during API extraction)
    if write_photo {
        for record in &sim.event_records {
            let row = build_event_photo_values(&record.state, record.time, planet, data, &run_state);
            photo_lines.push(row);
        }
        // Sort by time (column 0) to maintain chronological order
        photo_lines.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    }
```

Where `build_event_photo_values` produces a `[f64; 30]` in the same format as `build_photo_values`, using the state from the event record. This keeps the existing trajectory extraction code in `run_for_api` completely unchanged.

Add this helper:

```rust
/// Build a photo-format row from an event record's state.
/// Uses raw physics quantities -- no GNC-dependent values (bank angle from pilot,
/// nav density ratio) since events occur between GNC ticks.
fn build_event_photo_values(
    state: &[f64; 8],
    time: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 30] {
    let (altitude, latitude) = geodetic_from_spherical(state[0], state[1], state[2], planet);
    let orbit = elements::from_spherical(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );
    let (_pos_abs, vel_abs) = to_absolute_cartesian(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );
    let speed_abs = norm(&vel_abs);
    let energy = speed_abs * speed_abs / 2.0 - planet.mu / state[0];
    let velocity_radial = state[3] * state[4].sin();
    let rho = atmosphere::density(
        &data.atmosphere, altitude, run_state.density_bias, run_state.density_perturbation,
    );
    let v_eff = effective_airspeed(state[3], state[4], state[5], state[2], altitude, data, run_state);
    let heat_flux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    let pdyn = 0.5 * rho * v_eff * v_eff;
    let aoa_dispersed = run_state.incidence_bias; // no commanded AoA available
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    [
        time,                           // [0]  time_s
        altitude / 1e3,                 // [1]  altitude_km
        state[1] / DEG_TO_RAD,         // [2]  longitude_deg
        latitude / DEG_TO_RAD,          // [3]  latitude_deg
        state[3],                       // [4]  velocity_m_s
        state[4] / DEG_TO_RAD,         // [5]  fpa_deg
        state[5] / DEG_TO_RAD,         // [6]  heading_deg
        orbit.semi_major_axis / 1e3,    // [7]  sma_km
        orbit.eccentricity,             // [8]  ecc
        orbit.inclination / DEG_TO_RAD, // [9]  incl_deg
        orbit.raan / DEG_TO_RAD,        // [10] raan_deg
        orbit.periapsis_alt / 1e3,      // [11] periapsis_km
        orbit.apoapsis_alt / 1e3,       // [12] apoapsis_km
        0.0,                            // [13] phase (unknown at sub-tick)
        0.0,                            // [14] bank_angle_deg (unknown at sub-tick)
        velocity_radial,                // [15] radial_velocity_m_s
        0.0,                            // [16] aoa_deg
        0.0,                            // [17] cumulative_bank_change_deg
        energy,                         // [18] energy_j_kg
        pdyn,                           // [19] dynamic_pressure_pa
        velocity_radial,                // [20] radial_velocity_2 (duplicate)
        0.5 * rho * state[3] * state[3] / 1e3, // [21] pdyn_onboard_kpa
        0.0,                            // [22] sim_number
        0.0,                            // [23] reserved
        heat_flux / 1e3,                // [24] heat_flux_kw_m2
        load_factor / G0,               // [25] g_load_g
        1.0,                            // [26] nav_density_ratio
        rho,                            // [27] truth_density_kg_m3
        state[6] / 1e3,                 // [28] heat_load_kj_m2
        run_state.density_perturbation, // [29] density_perturbation
    ]
}
```

- [ ] **Step 4: Verify compilation and tests**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "feat: interleave event records into trajectory output

Event records converted to photo-format rows and sorted by time
into trajectory output. Keeps run_for_api extraction unchanged.
Sub-tick rows have zeroed GNC-dependent fields (bank angle, nav
density ratio) since events occur between guidance ticks."
```

---

### Task 7: Integration Tests

**Files:**
- Create: `src/rust/tests/event_detection.rs`

- [ ] **Step 1: Create test file with bounce precision test**

```rust
//! Integration tests for DOPRI45 event detection.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner::run_for_api;

fn load_config(name: &str) -> (SimInput, SimData) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd");
    let path = repo.join("configs").join(name);
    let (si, tc) = SimInput::from_toml_file(&path).unwrap();
    let sd = SimData::from_toml(&tc, &si).unwrap();
    (si, sd)
}

/// Bounce time with event detection should be more precise than the
/// 1-second tick resolution. The bounce time should fall between two
/// consecutive tick boundaries.
#[test]
fn bounce_time_is_sub_tick() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, false, None).expect("run");
    let r = &results[0];

    let bounce_time = r.final_record[26];
    let dt = 1.0; // outer tick period

    // Bounce time should NOT be an exact multiple of dt (that would mean
    // it was detected at a tick boundary, i.e., no sub-tick precision)
    let remainder = bounce_time % dt;
    assert!(
        remainder > 0.01 && remainder < dt - 0.01,
        "Bounce time {:.6} should be sub-tick (remainder from dt={}: {:.6})",
        bounce_time, dt, remainder,
    );
}

/// Atmosphere exit time should also be sub-tick precise.
#[test]
fn exit_time_is_sub_tick() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, false, None).expect("run");
    let r = &results[0];

    let sim_time = r.final_record[27];
    let dt = 1.0;

    let remainder = sim_time % dt;
    // Exit time should have sub-tick precision
    assert!(
        remainder.abs() > 0.001 || remainder.abs() < dt - 0.001,
        "Exit time {:.6} should be sub-tick (remainder {:.6})",
        sim_time, remainder,
    );
}
```

- [ ] **Step 2: Run bounce precision test**

Run: `cd src/rust && cargo test --test event_detection bounce_time_is_sub_tick -- --nocapture 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 3: Add fixed RK4 non-regression test**

```rust
/// Fixed RK4 output must be bit-identical to the baseline (event detection
/// is DOPRI45-only and must not affect the RK4 path).
#[test]
fn fixed_rk4_unchanged() {
    let (config, data) = load_config("test/test_ref_orig.toml");
    let results = run_for_api(&config, &data, false, None).expect("run");
    let r = &results[0];

    // These are the known baseline values from the existing golden test.
    // If this test fails, event detection has leaked into the RK4 path.
    let ifinal = r.final_record[31] as i32;
    assert_eq!(ifinal, 3, "RK4 reference should still capture (ifinal=3)");
    assert!(r.captured, "RK4 reference should still be captured");
}

/// Adaptive DOPRI45 with events should still produce a valid capture.
#[test]
fn adaptive_with_events_still_captures() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, false, None).expect("run");
    let r = &results[0];

    assert!(r.captured, "Adaptive should still capture with event detection");
    let ecc = r.final_record[9];
    assert!(ecc < 1.0, "Eccentricity should be < 1.0, got {}", ecc);
}
```

- [ ] **Step 4: Add trajectory event interleaving test**

```rust
/// When trajectories are requested, event records should appear as
/// additional rows interleaved at the correct time positions.
#[test]
fn trajectory_includes_event_rows() {
    let (config, data) = load_config("test/test_ref_adaptive.toml");
    let results = run_for_api(&config, &data, true, None).expect("run");
    let r = &results[0];

    assert!(!r.trajectory.is_empty(), "Should have trajectory data");

    // Check that trajectory times are monotonically non-decreasing
    for window in r.trajectory.windows(2) {
        assert!(
            window[1][7] >= window[0][7],
            "Trajectory times should be sorted: {} >= {}",
            window[1][7], window[0][7],
        );
    }

    // The bounce time should appear in the trajectory data
    let bounce_time = r.final_record[26];
    let has_bounce_row = r.trajectory.iter().any(|row| (row[7] - bounce_time).abs() < 0.01);
    assert!(has_bounce_row, "Trajectory should include a row near bounce time {:.3}", bounce_time);
}
```

- [ ] **Step 5: Run all event detection tests**

Run: `cd src/rust && cargo test --test event_detection -- --nocapture 2>&1 | tail -20`
Expected: all PASS.

- [ ] **Step 6: Run the full test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass. Some DOPRI45-related tests may have slightly different numerical results due to more precise event timing, but all assertions should still hold.

- [ ] **Step 7: Commit**

```bash
git add src/rust/tests/event_detection.rs
git commit -m "test: add integration tests for DOPRI45 event detection

Bounce/exit sub-tick precision, fixed RK4 non-regression,
adaptive capture validity, trajectory event interleaving
with monotonic time ordering."
```

---

### Task 8: Proptest for Event Detection

**Files:**
- Modify: `src/rust/tests/event_detection.rs`

- [ ] **Step 1: Add proptest for event state finiteness**

```rust
use proptest::prelude::*;

proptest! {
    /// For random seeds, adaptive integration with events should produce
    /// finite bounce_time and bounce_alt when a bounce occurs.
    #[test]
    fn event_bounce_values_finite(seed in 0.0_f64..1.0) {
        let repo = common::repo_root();
        std::env::set_current_dir(&repo).expect("set cwd");
        let path = repo.join("configs/test/test_ref_adaptive.toml");
        let (mut config, data) = {
            let (si, tc) = SimInput::from_toml_file(&path).unwrap();
            let sd = SimData::from_toml(&tc, &si).unwrap();
            (si, sd)
        };
        config.random_seed = seed;

        let results = run_for_api(&config, &data, false, None).expect("run");
        let r = &results[0];

        let bounce_time = r.final_record[26];
        let bounce_alt = r.final_record[25];
        let sim_time = r.final_record[27];

        prop_assert!(bounce_time.is_finite(), "bounce_time not finite: {}", bounce_time);
        prop_assert!(bounce_alt.is_finite(), "bounce_alt not finite: {}", bounce_alt);
        prop_assert!(sim_time.is_finite(), "sim_time not finite: {}", sim_time);
        prop_assert!(sim_time > 0.0, "sim_time should be positive: {}", sim_time);
    }
}
```

- [ ] **Step 2: Run proptest**

Run: `cd src/rust && cargo test --test event_detection event_bounce_values_finite 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/rust/tests/event_detection.rs
git commit -m "test: add proptest for event detection state finiteness

Property-based test verifies bounce_time, bounce_alt, and
sim_time are finite for random seeds in adaptive mode."
```

---

### Task 9: Run Full CI Checks and Clean Up

**Files:**
- Various (clippy fixes, format)

- [ ] **Step 1: Run clippy**

Run: `cd src/rust && cargo clippy -- -D warnings 2>&1 | tail -20`
Expected: no warnings. Fix any that appear.

- [ ] **Step 2: Run rustfmt**

Run: `cd src/rust && cargo fmt --check 2>&1 | tail -10`
Expected: no formatting issues. If any, run `cargo fmt`.

- [ ] **Step 3: Run the full Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 4: Run Python linting and tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -20`
Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -20`
Expected: all pass (Python code is untouched, but verify no regressions from PyO3 changes).

- [ ] **Step 5: Commit any cleanup**

If any clippy/fmt fixes were needed:

```bash
git add -A
git commit -m "fix: clippy and rustfmt cleanup for event detection"
```

---

### Task 10: Smart Commit (Final)

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole git branch into account. This will sync CLAUDE.md and README.md with the new event detection feature.
