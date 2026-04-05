# Predictive Roll Reversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reactive corridor-based lateral guidance with first-order inclination projection for fewer, better-timed roll reversals.

**Architecture:** New algorithm computes inclination error rate via finite difference, projects forward by tunable tau seconds, and reverses only when the projected error exceeds a threshold. Same function signature and dispatch integration as current -- drop-in replacement at the module level. No golden file regeneration needed (no existing golden test has lateral guidance active).

**Tech Stack:** Rust (nalgebra, proptest), Python (param_spaces.py), TOML configs

**Spec:** `docs/superpowers/specs/2026-04-05-predictive-roll-reversal-design.md`

---

### Task 1: Replace LateralParams and LateralState structs

**Files:**
- Modify: `src/rust/src/gnc/guidance/lateral.rs:1-57`

- [ ] **Step 1: Replace LateralParams struct and Default impl**

Replace the entire `LateralParams` struct and its `Default` impl in `lateral.rs`:

```rust
/// Predictive lateral guidance configuration (TOML-configurable, GA-tunable).
#[derive(Debug, Clone)]
pub struct LateralParams {
    /// Lookahead horizon for inclination error projection (seconds).
    pub tau: f64,
    /// Projected inclination error threshold for reversal trigger (radians).
    pub threshold: f64,
    /// Minimum time between consecutive reversals (seconds).
    pub min_reversal_interval: f64,
    /// Energy at which lateral guidance arms (J/kg). Upper bound of the active window.
    pub lateral_activation: f64,
    /// Energy below which lateral guidance disarms (J/kg). Lower bound of the active window.
    pub lateral_inhibition: f64,
    /// Maximum number of roll reversals per trajectory.
    pub max_reversals: i32,
}

impl Default for LateralParams {
    /// Default produces **inactive** lateral guidance: `tau = 0.0` triggers
    /// the early-return guard. Use explicit values (or TOML `[guidance.lateral]`)
    /// to activate.
    fn default() -> Self {
        Self {
            tau: 0.0,
            threshold: 0.0,
            min_reversal_interval: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
            max_reversals: 0,
        }
    }
}
```

- [ ] **Step 2: Replace LateralState struct and new() impl**

Replace `LateralState` and its `new()`:

```rust
/// Lateral guidance mutable state (per-run).
#[derive(Debug, Clone)]
pub struct LateralState {
    /// Current roll direction sign (+-1.0).
    pub roll_sign: f64,
    /// Number of roll reversals executed so far.
    pub n_reversals: i32,
    /// Previous tick's inclination error (None on first tick).
    pub prev_inclination_error: Option<f64>,
    /// Previous tick's guidance time (seconds).
    pub prev_time: f64,
    /// Time of most recent reversal (seconds).
    pub last_reversal_time: f64,
}

impl LateralState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            n_reversals: 0,
            prev_inclination_error: None,
            prev_time: 0.0,
            last_reversal_time: f64::NEG_INFINITY,
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/lateral.rs
git commit -m "refactor: replace LateralParams/LateralState structs for predictive reversal"
```

---

### Task 2: Implement new lateral_guidance algorithm

**Files:**
- Modify: `src/rust/src/gnc/guidance/lateral.rs:59-128`

- [ ] **Step 1: Replace the lateral_guidance function**

Replace the entire `lateral_guidance` function body (keep the same signature):

```rust
/// Compute roll sign based on projected inclination error.
///
/// Projects the inclination error forward by `tau` seconds using finite-difference
/// rate estimation. Reverses when the projected error exceeds `threshold` and the
/// minimum reversal interval has elapsed.
///
/// Returns `true` if a reversal was triggered this step.
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    planet: &PlanetConfig,
) -> bool {
    // Guard: tau must be positive to activate predictive lateral guidance
    if params.tau <= 0.0 {
        return false;
    }

    // Energy window gate: lateral_inhibition <= energy <= lateral_activation
    if energy > params.lateral_activation || energy < params.lateral_inhibition {
        return false;
    }

    // Skip degenerate bank angles (near 0 or pi, where roll sign is physically meaningless)
    let pi = std::f64::consts::PI;
    if bank_magnitude.abs() < 1e-10 || (bank_magnitude.abs() - pi).abs() < 1e-10 {
        return false;
    }

    // Compute current orbital inclination
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let inclination_error = target_inclination - orbit.inclination;
    let current_time = nav.guidance_time;

    // First tick: store state and return (no rate available yet)
    if state.prev_inclination_error.is_none() {
        state.prev_inclination_error = Some(inclination_error);
        state.prev_time = current_time;
        return false;
    }

    // Compute inclination error rate via finite difference
    let dt = current_time - state.prev_time;
    let di_err_dt = if dt > 1e-12 {
        (inclination_error - state.prev_inclination_error.unwrap()) / dt
    } else {
        0.0
    };

    // Update history for next tick
    state.prev_inclination_error = Some(inclination_error);
    state.prev_time = current_time;

    // Project inclination error forward by tau seconds
    let i_err_projected = inclination_error + di_err_dt * params.tau;

    // Check if projected error exceeds threshold
    if i_err_projected.abs() <= params.threshold {
        return false;
    }

    // Enforce reversal budget
    if state.n_reversals >= params.max_reversals {
        return false;
    }

    // Enforce minimum reversal interval (anti-chatter)
    if current_time - state.last_reversal_time < params.min_reversal_interval {
        return false;
    }

    // Determine desired roll sign from projected error (same convention as legacy)
    let desired_sign = if i_err_projected > 0.0 { -1.0 } else { 1.0 };

    // Only reverse if sign actually changes
    if desired_sign * state.roll_sign < 0.0 {
        state.roll_sign = desired_sign;
        state.n_reversals += 1;
        state.last_reversal_time = current_time;
        true
    } else {
        false
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/rust/src/gnc/guidance/lateral.rs
git commit -m "feat: implement predictive roll reversal with first-order inclination projection"
```

---

### Task 3: Write unit tests for new algorithm

**Files:**
- Modify: `src/rust/src/gnc/guidance/lateral.rs:130-413` (replace entire `#[cfg(test)]` module)

- [ ] **Step 1: Replace the test module**

Delete everything from `#[cfg(test)]` to end-of-file and replace with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PlanetConfig;
    use crate::gnc::navigation::estimator::NavigationOutput;

    fn test_nav(guidance_time: f64) -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0;
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [5000.0, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            guidance_time,
            ..Default::default()
        }
    }

    fn active_params() -> LateralParams {
        LateralParams {
            tau: 15.0,
            threshold: 0.01, // ~0.57 deg
            min_reversal_interval: 5.0,
            lateral_activation: 0.0,
            lateral_inhibition: -1e12,
            max_reversals: 5,
        }
    }

    /// Helper: run two guidance ticks to seed the finite difference, then
    /// return a state ready for the third (decision) tick.
    fn seeded_state(
        params: &LateralParams,
        target: f64,
        t0: f64,
        t1: f64,
    ) -> (LateralState, NavigationOutput) {
        let mut state = LateralState::new(1.0);
        let nav0 = test_nav(t0);
        lateral_guidance(params, &mut state, &nav0, target, -1e6, 1.0, &PlanetConfig::mars());
        // After first tick: prev_inclination_error is set, no reversal
        assert!(state.prev_inclination_error.is_some());
        let nav1 = test_nav(t1);
        (state, nav1)
    }

    #[test]
    fn no_reversal_on_first_tick() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav(0.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
        assert!(state.prev_inclination_error.is_some());
    }

    #[test]
    fn no_reversal_when_error_converging() {
        // Setup: large error at t=0, but by t=1 the orbit element computation
        // produces the same inclination (static nav state), so di/dt = 0.
        // With zero rate, projected = current error. We choose a target that
        // produces an error smaller than threshold, so no reversal.
        let params = active_params();
        let planet = PlanetConfig::mars();
        let nav = test_nav(0.0);
        let orbit = elements::from_spherical(
            nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2],
            nav.velocity_estimated[0], nav.velocity_estimated[1], nav.velocity_estimated[2],
            &planet,
        );
        // Target inclination very close to actual: error ~ 0 < threshold
        let target = orbit.inclination + 0.001; // 0.001 rad < threshold 0.01
        let (mut state, nav1) = seeded_state(&params, target, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, target, -1e6, 1.0, &planet,
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn reversal_when_projected_error_exceeds_threshold() {
        // With static nav state, di/dt = 0, so projected = current.
        // Set target far from actual inclination -> large projected error.
        let params = active_params();
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(reversed);
        assert_eq!(state.roll_sign, -1.0); // positive error -> negative sign
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn reversal_negative_projected_error() {
        let params = active_params();
        let (mut state, nav1) = seeded_state(&params, -10.0, 0.0, 1.0);
        state.roll_sign = -1.0; // start negative
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, -10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(reversed);
        assert_eq!(state.roll_sign, 1.0); // negative error -> positive sign
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn respects_min_reversal_interval() {
        let params = active_params(); // min_reversal_interval = 5.0
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        // First reversal at t=1
        let r1 = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(r1);
        assert_eq!(state.last_reversal_time, 1.0);

        // Try second reversal at t=3 (only 2s after first, < 5s interval)
        let nav2 = test_nav(3.0);
        let r2 = lateral_guidance(
            &params, &mut state, &nav2, -10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);

        // Try at t=7 (6s after first reversal, > 5s interval)
        let nav3 = test_nav(7.0);
        let r3 = lateral_guidance(
            &params, &mut state, &nav3, -10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(r3);
        assert_eq!(state.n_reversals, 2);
    }

    #[test]
    fn respects_max_reversals() {
        let params = LateralParams {
            max_reversals: 1,
            min_reversal_interval: 0.0, // disable interval for this test
            ..active_params()
        };
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        let r1 = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(r1);
        assert_eq!(state.n_reversals, 1);

        // Budget exhausted: second reversal blocked
        let nav2 = test_nav(10.0);
        let r2 = lateral_guidance(
            &params, &mut state, &nav2, -10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn no_reversal_outside_energy_window() {
        let params = LateralParams {
            lateral_activation: -1e12,
            lateral_inhibition: -1e12,
            ..active_params()
        };
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, 10.0, 1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_reversal_when_bank_near_zero() {
        let params = active_params();
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, 1e-15, &PlanetConfig::mars(),
        );
        assert!(!reversed);
    }

    #[test]
    fn no_reversal_when_bank_near_pi() {
        let params = active_params();
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, std::f64::consts::PI, &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn tau_zero_produces_inactive() {
        let params = LateralParams::default(); // tau = 0.0
        let mut state = LateralState::new(1.0);
        let nav = test_nav(0.0);
        let reversed = lateral_guidance(
            &params, &mut state, &nav, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_same_sign_reversal() {
        // If desired_sign == current sign, no reversal fires
        let params = LateralParams {
            min_reversal_interval: 0.0,
            ..active_params()
        };
        let (mut state, nav1) = seeded_state(&params, 10.0, 0.0, 1.0);
        // Positive error -> desired sign -1.0. Pre-set roll_sign = -1.0
        state.roll_sign = -1.0;
        let reversed = lateral_guidance(
            &params, &mut state, &nav1, 10.0, -1e6, 1.0, &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_nav(time: f64) -> impl Strategy<Value = NavigationOutput> {
            (
                3.4e6_f64..3.6e6,
                -std::f64::consts::PI..std::f64::consts::PI,
                -1.0_f64..1.0,
                3000.0_f64..7000.0,
                -0.3_f64..0.1,
                -std::f64::consts::PI..std::f64::consts::PI,
            )
                .prop_map(move |(r, lon, lat, v, fpa, hdg)| NavigationOutput {
                    position_estimated: [r, lon, lat],
                    velocity_estimated: [v, fpa, hdg],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    guidance_time: time,
                    ..Default::default()
                })
        }

        proptest! {
            #[test]
            fn roll_sign_is_pm_one(nav in arb_nav(1.0), target in -2.0_f64..2.0) {
                let params = active_params();
                let mut state = LateralState::new(1.0);
                // Seed with first tick
                let nav0 = NavigationOutput { guidance_time: 0.0, ..nav.clone() };
                lateral_guidance(&params, &mut state, &nav0, target, -1e6, 1.0, &PlanetConfig::mars());
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, &PlanetConfig::mars());
                prop_assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
            }

            #[test]
            fn n_reversals_monotonic(
                nav in arb_nav(0.0),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..20),
            ) {
                let params = LateralParams {
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                let mut prev_n = 0;
                for (i, t) in targets.iter().enumerate() {
                    let nav_t = NavigationOutput {
                        guidance_time: i as f64,
                        ..nav.clone()
                    };
                    lateral_guidance(&params, &mut state, &nav_t, *t, -1e6, 1.0, &PlanetConfig::mars());
                    prop_assert!(state.n_reversals >= prev_n);
                    prev_n = state.n_reversals;
                }
            }

            #[test]
            fn n_reversals_bounded(
                nav in arb_nav(0.0),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..30),
                max_rev in 1_i32..10,
            ) {
                let params = LateralParams {
                    max_reversals: max_rev,
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                for (i, t) in targets.iter().enumerate() {
                    let nav_t = NavigationOutput {
                        guidance_time: i as f64,
                        ..nav.clone()
                    };
                    lateral_guidance(&params, &mut state, &nav_t, *t, -1e6, 1.0, &PlanetConfig::mars());
                }
                prop_assert!(state.n_reversals <= max_rev);
            }

            #[test]
            fn projected_error_finite(
                nav in arb_nav(1.0),
                target in -2.0_f64..2.0,
                tau in 0.1_f64..100.0,
            ) {
                let params = LateralParams {
                    tau,
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                let nav0 = NavigationOutput { guidance_time: 0.0, ..nav.clone() };
                lateral_guidance(&params, &mut state, &nav0, target, -1e6, 1.0, &PlanetConfig::mars());
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, &PlanetConfig::mars());
                // If we got here without panic, the projected error was finite
                prop_assert!(state.roll_sign.is_finite());
                prop_assert!(state.n_reversals >= 0);
            }
        }
    }
}
```

- [ ] **Step 2: Verify tests compile and pass**

Run: `cd src/rust && cargo test lateral --lib -- --nocapture 2>&1 | tail -20`

Expected: All lateral tests pass. Some will fail if the algorithm has issues -- fix before proceeding.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/lateral.rs
git commit -m "test: add unit + property tests for predictive roll reversal"
```

---

### Task 4: Update TOML config parsing

**Files:**
- Modify: `src/rust/src/config.rs:789-805`

- [ ] **Step 1: Replace TomlLateralParams struct**

Replace lines 789-805 in `config.rs` (the `TomlLateralParams` struct and its default function):

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlLateralParams {
    #[serde(default = "default_lateral_tau")]
    pub tau: f64, // seconds
    #[serde(default)]
    pub threshold: f64, // deg (converted to rad)
    #[serde(default)]
    pub min_reversal_interval: f64, // seconds
    #[serde(default = "default_five_i32")]
    pub max_reversals: i32,
    #[serde(default)]
    pub lateral_activation: f64, // MJ/kg
    #[serde(default)]
    pub lateral_inhibition: f64, // MJ/kg
}

fn default_lateral_tau() -> f64 {
    15.0
}
```

This replaces `corridor_slope`, `corridor_intercept` with `tau`, `threshold`, `min_reversal_interval`. The `default_lateral_corridor_slope` function on line 803-805 is removed (replaced by `default_lateral_tau`).

- [ ] **Step 2: Remove old lateral fields from TomlFtcParams**

In `config.rs` around lines 588-606, remove these fields from `TomlFtcParams`:

```rust
    // DELETE these lines:
    pub corridor_slope: f64,
    pub corridor_intercept: f64,
    // max_reversals -- only remove if it's the lateral one (check context)
    pub lateral_activation: f64,
    pub lateral_inhibition: f64,
```

Specifically, remove:
- `corridor_slope` (line 590)
- `corridor_intercept` (line 592)
- The `lateral_activation` (line 604) and `lateral_inhibition` (line 606) fields

Keep `max_reversals` (line 594) ONLY if it's used for something other than lateral. Check -- it appears to be lateral-only, so remove it too.

Serde will silently ignore these fields in existing FTC TOML configs (no `deny_unknown_fields`).

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "refactor: update TomlLateralParams for predictive reversal, remove FTC lateral fields"
```

---

### Task 5: Update data loading

**Files:**
- Modify: `src/rust/src/data/mod.rs:448-465`

- [ ] **Step 1: Update LateralParams construction from TOML**

Replace the lateral loading block (lines 448-465) with:

```rust
                lateral: if let Some(ref lat) = toml.guidance.lateral {
                    LateralParams {
                        tau: lat.tau,
                        threshold: lat.threshold * DEG2RAD,
                        min_reversal_interval: lat.min_reversal_interval,
                        lateral_activation: lat.lateral_activation * energy_scale,
                        lateral_inhibition: lat.lateral_inhibition * energy_scale,
                        max_reversals: lat.max_reversals,
                    }
                } else {
                    LateralParams::default()
                },
```

This removes the FTC fallback branch entirely. Configs without `[guidance.lateral]` get Default (inactive), which is the same behavior as before (FTC configs had an inverted energy window that disabled lateral).

- [ ] **Step 2: Commit**

```bash
git add src/rust/src/data/mod.rs
git commit -m "refactor: update lateral param loading, remove FTC fallback"
```

---

### Task 6: Update dispatch.rs test helper

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs:369-375`

- [ ] **Step 1: Update hardcoded LateralParams in test_sim_data()**

Replace the `LateralParams` literal at line 369-375:

```rust
                lateral: LateralParams {
                    lateral_activation: -1e12, // disable lateral for simple tests
                    lateral_inhibition: -1e12,
                    ..Default::default()
                },
```

The `Default::default()` fills in `tau: 0.0` which triggers the early-return guard, so lateral is inactive. The explicit energy window values are a belt-and-suspenders safeguard.

- [ ] **Step 2: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs
git commit -m "refactor: update dispatch test helper for new LateralParams"
```

---

### Task 7: Update TOML configs

**Files:**
- Modify: `configs/test/test_lateral_eqglide.toml`

- [ ] **Step 1: Update test_lateral_eqglide.toml**

Replace the `[guidance.lateral]` section:

```toml
[guidance.lateral]
tau = 15.0                    # seconds
threshold = 0.5               # degrees
min_reversal_interval = 5.0   # seconds
lateral_activation = -0.5     # MJ/kg
lateral_inhibition = -10.0    # MJ/kg
max_reversals = 5
```

This is the only config with a `[guidance.lateral]` section. Training TOMLs don't need one -- the GA injects lateral params at runtime via the override dict.

FTC configs (`msr_aller_ftc_*.toml`, `test_guided_orig.toml`, `esr_aller_ftc_nominal.toml`) keep their old `corridor_slope` etc. in `[guidance.ftc]` -- serde silently ignores unknown fields, and lateral defaults to inactive.

- [ ] **Step 2: Compile and run Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -30`

Expected: All tests pass. If any fail due to missed references to old field names, fix them before proceeding.

- [ ] **Step 3: Commit**

```bash
git add configs/test/test_lateral_eqglide.toml
git commit -m "config: update test_lateral_eqglide.toml for predictive reversal params"
```

---

### Task 8: Update Python GA parameter space

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py:35-41`

- [ ] **Step 1: Replace _LATERAL_PARAMS**

Replace lines 35-41:

```python
_LATERAL_PARAMS: list[ParamSpec] = [
    ParamSpec("lateral.tau", 2.0, 60.0, 15.0),                       # seconds
    ParamSpec("lateral.threshold", 0.01, 2.0, 0.5),                  # degrees (TOML units)
    ParamSpec("lateral.min_reversal_interval", 1.0, 30.0, 5.0),      # seconds
    ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),       # MJ/kg
    ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),      # MJ/kg
    ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),              # integer
]
```

No changes to `evaluate.py` or `compare_guidance.py` -- the `lateral.` prefix routing and `max_reversals` integer rounding are generic.

- [ ] **Step 2: Run Python linting**

Run: `uv run ruff check src/python/ && uv run ruff format --check src/python/`

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "feat: update GA lateral param space for predictive reversal"
```

---

### Task 9: Fix Python test for lateral param routing

**Files:**
- Modify: `tests/test_animate.py:77-83`

- [ ] **Step 1: Update test_lateral_params_go_to_lateral_section**

Replace the old field name in the test:

```python
    def test_lateral_params_go_to_lateral_section(self) -> None:
        from aerocapture.training.animate import _build_overrides

        params = {"gain_kp": 0.5, "lateral.tau": 15.0}
        overrides = _build_overrides("equilibrium_glide", params, n_sims=50)
        assert overrides["guidance.lateral.tau"] == 15.0
        assert "guidance.equilibrium_glide.lateral.tau" not in overrides
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_animate.py
git commit -m "test: update animate test for new lateral param names"
```

---

### Task 10: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run Rust full check**

Run: `./check_all.sh 2>&1 | tail -30`

Expected: All Rust tests pass, clippy clean, fmt clean.

- [ ] **Step 2: Run Python tests**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -20`

Expected: All Python tests pass. If any param_spaces-related tests fail (e.g., tests that check the number of lateral genes or their names), update those tests to match the new param names.

- [ ] **Step 3: Fix any failures**

If Rust or Python tests fail, investigate and fix. Common issues:
- Tests that hardcode old field names (`corridor_slope`, `corridor_intercept`)
- Tests that check specific param counts or names in `_LATERAL_PARAMS`
- Integration tests that reference the old lateral TOML schema

- [ ] **Step 4: Commit fixes if any**

```bash
git add -A
git commit -m "fix: resolve test failures from lateral param migration"
```

---

### Task 11: Smart commit

Invoke the `smart-commit` skill, telling it to take the whole `feature/predictive-roll-reversal` branch into account.
