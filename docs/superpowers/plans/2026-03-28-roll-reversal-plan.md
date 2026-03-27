# Roll Reversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract lateral guidance (roll reversal) into a clean shared module, fix bugs, add per-scheme TOML-configurable lateral parameters, and expose them to the GA optimizer.

**Architecture:** New `lateral.rs` module under `gnc/guidance/` with `LateralParams`, `LateralState`, and a pure `lateral_guidance()` function. TOML gets a new `[guidance.lateral]` section parsed via `TomlLateralParams`. The four unsigned-magnitude schemes (EqGlide, EnergyController, PredGuid, FNPAG) use this; NN and PiecewiseConstant continue to bypass it. Python GA parameter spaces gain 5 lateral genes for each of the four schemes.

**Tech Stack:** Rust (serde, nalgebra), Python (param_spaces.py, evaluate.py), TOML config

**Spec:** `docs/superpowers/specs/2026-03-28-roll-reversal-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/rust/src/gnc/guidance/lateral.rs` | `LateralParams`, `LateralState`, `lateral_guidance()` + unit tests |
| Modify | `src/rust/src/gnc/guidance/mod.rs` | Add `pub mod lateral;` |
| Modify | `src/rust/src/data/guidance_params.rs:122-171` | Add `lateral: LateralParams` field to `GuidanceParams` |
| Modify | `src/rust/src/config.rs:307-327` | Add `TomlLateralParams` struct, `lateral` field on `TomlGuidance` |
| Modify | `src/rust/src/data/mod.rs:428-457` | Wire `TomlLateralParams` → `LateralParams` in conversion |
| Modify | `src/rust/src/gnc/guidance/ftc.rs:16-46` | Replace roll fields on `FtcState` with `LateralState` |
| Modify | `src/rust/src/gnc/guidance/ftc.rs:86-284` | Use new `lateral_guidance()` in `guidance_step`, delete old function |
| Modify | `src/python/aerocapture/training/param_spaces.py` | Add 5 lateral `ParamSpec` entries to 4 schemes |
| Modify | `src/python/aerocapture/training/evaluate.py:336-378` | Split lateral params into `[guidance.lateral]` section |

---

### Task 1: Create `lateral.rs` with `LateralParams` and `LateralState`

**Files:**
- Create: `src/rust/src/gnc/guidance/lateral.rs`
- Modify: `src/rust/src/gnc/guidance/mod.rs`

- [ ] **Step 1: Create `lateral.rs` with structs**

```rust
//! Lateral guidance — inclination corridor roll reversal logic.
//!
//! Shared by all unsigned-magnitude guidance schemes (EqGlide, EnergyController,
//! PredGuid, FNPAG). Schemes that produce signed bank angles (NeuralNetwork,
//! PiecewiseConstant) bypass this entirely.

use crate::config::Planet;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Lateral guidance configuration (TOML-configurable, per-scheme tunable).
#[derive(Debug, Clone)]
pub struct LateralParams {
    /// Velocity scaling for corridor width (m/s).
    pub corridor_slope: f64,
    /// Baseline corridor width at low velocity (rad).
    pub corridor_intercept: f64,
    /// Energy at which lateral guidance arms (J/kg). Upper bound of the active window.
    pub lateral_activation: f64,
    /// Energy below which lateral guidance disarms (J/kg). Lower bound of the active window.
    pub lateral_inhibition: f64,
    /// Maximum number of roll reversals per trajectory.
    pub max_reversals: i32,
}

impl Default for LateralParams {
    fn default() -> Self {
        Self {
            corridor_slope: 0.0,
            corridor_intercept: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
            max_reversals: 0,
        }
    }
}

/// Lateral guidance mutable state (per-run).
#[derive(Debug, Clone)]
pub struct LateralState {
    /// Current roll direction sign (±1.0).
    pub roll_sign: f64,
    /// Number of roll reversals executed so far.
    pub n_reversals: i32,
}

impl LateralState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            n_reversals: 0,
        }
    }
}
```

- [ ] **Step 2: Register the module in `mod.rs`**

Add `pub mod lateral;` to `src/rust/src/gnc/guidance/mod.rs` after the `ftc` line:

```rust
pub mod ftc;
pub mod lateral;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -20`
Expected: compiles with no errors (unused warnings are fine)

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/lateral.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "feat: add lateral guidance module with LateralParams and LateralState structs"
```

---

### Task 2: Implement `lateral_guidance()` with TDD

**Files:**
- Modify: `src/rust/src/gnc/guidance/lateral.rs`

- [ ] **Step 1: Write unit tests for `lateral_guidance()`**

Append to `lateral.rs`:

```rust
/// Compute roll sign based on inclination error and corridor boundary.
///
/// Returns `true` if a reversal was triggered this step.
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    planet: &Planet,
) -> bool {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Planet;
    use crate::gnc::navigation::estimator::NavigationOutput;

    fn test_nav() -> NavigationOutput {
        let r = Planet::Mars.equatorial_radius() + 50_000.0;
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [5000.0, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
        }
    }

    fn active_params() -> LateralParams {
        LateralParams {
            corridor_slope: 13080.458,
            corridor_intercept: 0.0,
            lateral_activation: 0.0,     // 0 J/kg (upper bound)
            lateral_inhibition: -1e12,   // very negative (lower bound)
            max_reversals: 5,
        }
    }

    #[test]
    fn no_reversal_when_outside_energy_window() {
        let params = LateralParams {
            lateral_activation: -1e12,   // very negative upper bound
            lateral_inhibition: -1e12,   // very negative lower bound (zero-width window)
            ..active_params()
        };
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        // energy will be large positive, outside the [-1e12, -1e12] window
        let reversed = lateral_guidance(&params, &mut state, &nav, 1.0, 1e6, 1.0, &Planet::Mars);
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_reversal_when_inclination_within_corridor() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let orbit = elements::from_spherical(
            nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2],
            nav.velocity_estimated[0], nav.velocity_estimated[1], nav.velocity_estimated[2],
            &Planet::Mars,
        );
        // Set target = current inclination → error = 0
        let reversed = lateral_guidance(&params, &mut state, &nav, orbit.inclination, -1e6, 1.0, &Planet::Mars);
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn reversal_when_inclination_exceeds_corridor() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        assert_eq!(state.roll_sign, 1.0);
        let nav = test_nav();
        // Set target far from current → large positive error → roll_sign should become -1
        let reversed = lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(reversed);
        assert_eq!(state.roll_sign, -1.0);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn reversal_negative_error() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        state.roll_sign = -1.0; // start negative
        let nav = test_nav();
        // Large negative error → roll_sign should become +1
        let reversed = lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(reversed);
        assert_eq!(state.roll_sign, 1.0);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn respects_max_reversals() {
        let params = LateralParams {
            max_reversals: 1,
            ..active_params()
        };
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        // First reversal succeeds
        let r1 = lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(r1);
        assert_eq!(state.n_reversals, 1);
        assert_eq!(state.roll_sign, -1.0);
        // Second reversal blocked
        let r2 = lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);
        assert_eq!(state.roll_sign, -1.0); // unchanged
    }

    #[test]
    fn no_reversal_when_bank_near_zero() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        // bank_magnitude ~0 → no reversal
        let reversed = lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1e-15, &Planet::Mars);
        assert!(!reversed);
    }

    #[test]
    fn roll_sign_always_pm_one() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
        lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
    }
}
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd src/rust && cargo test lateral --lib 2>&1 | tail -20`
Expected: FAIL with "not yet implemented"

- [ ] **Step 3: Implement `lateral_guidance()`**

Replace the `todo!()` body:

```rust
/// Compute roll sign based on inclination error and corridor boundary.
///
/// Returns `true` if a reversal was triggered this step.
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    planet: &Planet,
) -> bool {
    // Energy window gate: lateral_inhibition <= energy <= lateral_activation
    if energy > params.lateral_activation || energy < params.lateral_inhibition {
        return false;
    }

    // Skip degenerate bank angles
    if bank_magnitude.abs() < 1e-10 {
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
    let velocity = nav.velocity_estimated[0];

    // Corridor boundary: narrows with decreasing velocity
    let corridor_width = (velocity / params.corridor_slope).powi(4) + params.corridor_intercept;

    // Check reversal conditions
    if inclination_error.abs() < corridor_width {
        return false;
    }
    if state.n_reversals >= params.max_reversals {
        return false;
    }

    let previous_sign = state.roll_sign;

    if inclination_error > corridor_width {
        state.roll_sign = -1.0;
    } else if inclination_error < -corridor_width {
        state.roll_sign = 1.0;
    }

    // Check if sign actually changed
    if state.roll_sign * previous_sign < 0.0 {
        state.n_reversals += 1;
        true
    } else {
        false
    }
}
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd src/rust && cargo test lateral --lib 2>&1 | tail -20`
Expected: all 7 tests pass

- [ ] **Step 5: Add proptest property tests**

Add to the `tests` module in `lateral.rs`:

```rust
    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_nav() -> impl Strategy<Value = NavigationOutput> {
            (
                3.4e6_f64..3.6e6,     // radius near Mars
                -std::f64::consts::PI..std::f64::consts::PI, // longitude
                -1.0_f64..1.0,         // latitude
                3000.0_f64..7000.0,    // velocity
                -0.3_f64..0.1,         // fpa
                -std::f64::consts::PI..std::f64::consts::PI, // heading
            ).prop_map(|(r, lon, lat, v, fpa, hdg)| NavigationOutput {
                position_estimated: [r, lon, lat],
                velocity_estimated: [v, fpa, hdg],
                acceleration_estimated: [50.0, -8.0],
                aero_coefficients: [1.269, -0.205],
                density_guidance: 0.001,
            })
        }

        proptest! {
            #[test]
            fn roll_sign_is_pm_one(nav in arb_nav(), target in -2.0_f64..2.0) {
                let params = LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: 0.0,
                    lateral_inhibition: -1e12,
                    max_reversals: 5,
                };
                let mut state = LateralState::new(1.0);
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, &Planet::Mars);
                prop_assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
            }

            #[test]
            fn n_reversals_monotonic(
                nav in arb_nav(),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..20),
            ) {
                let params = LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: 0.0,
                    lateral_inhibition: -1e12,
                    max_reversals: 100,
                };
                let mut state = LateralState::new(1.0);
                let mut prev_n = 0;
                for t in &targets {
                    lateral_guidance(&params, &mut state, &nav, *t, -1e6, 1.0, &Planet::Mars);
                    prop_assert!(state.n_reversals >= prev_n);
                    prev_n = state.n_reversals;
                }
            }

            #[test]
            fn corridor_width_positive(v in 1000.0_f64..8000.0) {
                let slope = 13080.458_f64;
                let intercept = 0.01_f64;
                let width = (v / slope).powi(4) + intercept;
                prop_assert!(width > 0.0);
            }
        }
    }
```

- [ ] **Step 6: Run all lateral tests**

Run: `cd src/rust && cargo test lateral --lib 2>&1 | tail -20`
Expected: all tests pass (7 unit + 3 proptest)

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/gnc/guidance/lateral.rs
git commit -m "feat: implement lateral_guidance() with unit + proptest coverage"
```

---

### Task 3: Add `TomlLateralParams` to TOML config parsing

**Files:**
- Modify: `src/rust/src/config.rs:307-327`

- [ ] **Step 1: Add `TomlLateralParams` struct**

Add after the `TomlPiecewiseConstantParams` block (around line 730 in `config.rs`), before the defaults section. Find the line `fn default_bank_65` and add above it:

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlLateralParams {
    #[serde(default = "default_lateral_corridor_slope")]
    pub corridor_slope: f64, // m/s
    #[serde(default)]
    pub corridor_intercept: f64, // deg (converted to rad)
    #[serde(default = "default_five_i32")]
    pub max_reversals: i32,
    #[serde(default)]
    pub lateral_activation: f64, // MJ/kg
    #[serde(default)]
    pub lateral_inhibition: f64, // MJ/kg
}

fn default_lateral_corridor_slope() -> f64 {
    13080.458
}
```

- [ ] **Step 2: Add `lateral` field to `TomlGuidance`**

In `TomlGuidance` (config.rs:307-327), add after the `piecewise_constant` field:

```rust
    /// Lateral guidance parameters (shared by unsigned-magnitude schemes)
    #[serde(default)]
    pub lateral: Option<TomlLateralParams>,
```

- [ ] **Step 3: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -20`
Expected: compiles (with warnings about unused fields)

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat: add TomlLateralParams struct and lateral field on TomlGuidance"
```

---

### Task 4: Wire `LateralParams` into `GuidanceParams` and `SimData` conversion

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs:122-171, 294-327`
- Modify: `src/rust/src/data/mod.rs:428-498`

- [ ] **Step 1: Add `lateral` field to `GuidanceParams`**

In `guidance_params.rs`, add a use statement at the top:

```rust
use crate::gnc::guidance::lateral::LateralParams;
```

In the `GuidanceParams` struct (line ~140), replace the three lateral fields:

```rust
    // Lateral guidance
    pub corridor_slope: f64,     // inclination corridor slope (m/s)
    pub corridor_intercept: f64, // inclination corridor intercept (rad, converted from deg)
    pub max_reversals: i32,      // max number of bank reversals
```

with:

```rust
    // Lateral guidance
    pub lateral: LateralParams,
```

Also remove the `lateral_activation` and `lateral_inhibition` fields (lines 155-156):

```rust
    pub lateral_activation: f64, // lateral guidance activation threshold (J/kg)
    pub lateral_inhibition: f64, // lateral guidance inhibition threshold (J/kg)
```

- [ ] **Step 2: Update `GuidanceParams::default()`**

In the `Default` impl (line ~294), replace:

```rust
            corridor_slope: 0.0,
            corridor_intercept: 0.0,
            max_reversals: 0,
```

with:

```rust
            lateral: LateralParams::default(),
```

And remove:

```rust
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
```

- [ ] **Step 3: Update `SimData` conversion in `data/mod.rs`**

In the conversion at `data/mod.rs:428-457`, find where `corridor_slope`, `corridor_intercept`, `max_reversals`, `lateral_activation`, `lateral_inhibition` are set. Replace those 5 lines with:

```rust
                lateral: if let Some(ref lat) = config.guidance.lateral {
                    LateralParams {
                        corridor_slope: lat.corridor_slope,
                        corridor_intercept: lat.corridor_intercept * DEG2RAD,
                        lateral_activation: lat.lateral_activation * energy_scale,
                        lateral_inhibition: lat.lateral_inhibition * energy_scale,
                        max_reversals: lat.max_reversals,
                    }
                } else {
                    // Backward compat: read from FTC section if present
                    LateralParams {
                        corridor_slope: ftc.corridor_slope,
                        corridor_intercept: ftc.corridor_intercept * DEG2RAD,
                        lateral_activation: ftc.lateral_activation * energy_scale,
                        lateral_inhibition: ftc.lateral_inhibition * energy_scale,
                        max_reversals: ftc.max_reversals,
                    }
                },
```

Add the import at the top of `data/mod.rs`:

```rust
use crate::gnc::guidance::lateral::LateralParams;
```

Do the same for the fallback `GuidanceParams` block at `data/mod.rs:469-498` — replace the 5 lateral fields with:

```rust
                lateral: LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: 1.311e6,
                    lateral_inhibition: 1e9,
                    max_reversals: 5,
                },
```

- [ ] **Step 4: Fix all compile errors from the field removal**

The old `data.guidance.corridor_slope`, `data.guidance.max_reversals`, `data.guidance.lateral_activation`, `data.guidance.lateral_inhibition`, and `data.guidance.corridor_intercept` references in `ftc.rs` now need `data.guidance.lateral.` prefix. Update `ftc.rs`:

At line 193, change:
```rust
    if energy <= data.guidance.lateral_activation && energy >= data.guidance.lateral_inhibition {
```
to:
```rust
    if energy <= data.guidance.lateral.lateral_activation && energy >= data.guidance.lateral.lateral_inhibition {
```

At line 432-434, change:
```rust
    let corridor_slope = data.guidance.corridor_slope;
    let corridor_intercept = data.guidance.corridor_intercept;
```
to:
```rust
    let corridor_slope = data.guidance.lateral.corridor_slope;
    let corridor_intercept = data.guidance.lateral.corridor_intercept;
```

At line 439, change:
```rust
        && state.n_reversals < data.guidance.max_reversals
```
to:
```rust
        && state.n_reversals < data.guidance.lateral.max_reversals
```

Also update the test fixtures in `ftc.rs` — the `test_data()` function builds `GuidanceParams` directly. Replace the old lateral fields with the new `lateral: LateralParams { ... }` struct.

- [ ] **Step 5: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -30`
Expected: compiles

- [ ] **Step 6: Run all Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/guidance_params.rs src/rust/src/data/mod.rs src/rust/src/config.rs src/rust/src/gnc/guidance/ftc.rs
git commit -m "refactor: move lateral params into LateralParams struct, wire TOML → GuidanceParams"
```

---

### Task 5: Replace lateral logic in `guidance_step` with new `lateral_guidance()`

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs:16-46, 86-284, 397-466`

- [ ] **Step 1: Add `LateralState` to `FtcState`**

In `FtcState` (ftc.rs:16-46), add the import at the top of the file:

```rust
use crate::gnc::guidance::lateral::{self, LateralState};
```

Replace these fields:

```rust
    // Roll sign and reversal tracking
    pub roll_sign: f64,              // roll polarity sign (-1, 0, +1)
    pub cumulative_bank_change: f64, // cumulative bank angle changes (rad)
    pub n_reversals: i32,            // number of roll reversals
    pub reversal_active: i32,        // roll reversal active flag
    pub roll_path: i32,              // roll reversal path (+1=short, -1=long)
    pub reversal_duration: f64,      // roll reversal duration (s)
```

with:

```rust
    // Roll sign and reversal tracking
    pub lateral_state: LateralState,
    pub cumulative_bank_change: f64, // cumulative bank angle changes (rad)
```

- [ ] **Step 2: Update `FtcState::new()`**

Replace the old field initializations:

```rust
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            cumulative_bank_change: 0.0,
            n_reversals: 0,
            reversal_active: 0,
            roll_path: 1,
            reversal_duration: 0.0,
```

with:

```rust
            lateral_state: LateralState::new(initial_bank),
            cumulative_bank_change: 0.0,
```

- [ ] **Step 3: Update `guidance_step` to use new lateral module**

In `guidance_step`, make these changes:

**Line 100** — change `let previous_roll_sign = state.roll_sign;` to:
```rust
    let previous_roll_sign = state.lateral_state.roll_sign;
```

**Lines 182-189** (skip_lateral block) — change `state.roll_sign` to `state.lateral_state.roll_sign`:
```rust
    if skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal;
        state.lateral_state.roll_sign = if bank_angle_longitudinal >= 0.0 {
            1.0
        } else {
            -1.0
        };
    }
```

**Lines 191-222** (lateral guidance section) — replace the entire block with:

```rust
    // === Lateral guidance ===
    let mut roll_reversal_active = false;
    if !skip_lateral {
        roll_reversal_active = lateral::lateral_guidance(
            &data.guidance.lateral,
            &mut state.lateral_state,
            nav,
            data.target_orbit.inclination,
            energy,
            bank_angle_longitudinal,
            planet,
        );
    }
    if !skip_lateral && !roll_reversal_active {
        // Hold previous roll sign when lateral is inactive or no reversal triggered
        if energy > data.guidance.lateral.lateral_activation
            || energy < data.guidance.lateral.lateral_inhibition
        {
            state.lateral_state.roll_sign = previous_roll_sign;
        }
    }
```

**Lines 224-256** (combine block) — simplify by removing the dead `reversal_active` sweep logic:

```rust
    // === Combine longitudinal and lateral commands ===
    if !is_reference && !skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal * state.lateral_state.roll_sign;
    }
```

**Line 281** — update the output field:
```rust
    out.roll_reversal_active = if roll_reversal_active { 1 } else { 0 };
```

- [ ] **Step 4: Delete the old `lateral_guidance()` function**

Remove the entire function at ftc.rs:397-466 (the old `fn lateral_guidance(...)` with its `FtcState`-based logic). The new `lateral::lateral_guidance()` replaces it.

- [ ] **Step 5: Fix remaining references to old fields**

Search for any remaining `state.roll_sign`, `state.n_reversals`, `state.reversal_active`, `state.roll_path`, `state.reversal_duration` and update them:

- `state.roll_sign` → `state.lateral_state.roll_sign` (in tests too)
- `state.n_reversals` → `state.lateral_state.n_reversals`
- `state.reversal_active` → remove (no longer exists)
- `state.roll_path` → remove (no longer exists)
- `state.reversal_duration` → remove (no longer exists)

Also update line 221 (`state.roll_sign = previous_roll_sign`) if it still exists after the rewrite.

- [ ] **Step 6: Update test fixtures in `ftc.rs`**

The `test_data()` function in the test module builds `GuidanceParams` manually. It references old fields like `lateral_activation`, `lateral_inhibition`, `corridor_slope`, `max_reversals`, `corridor_intercept`. Update to use the new `lateral: LateralParams { ... }` struct. Example:

```rust
                lateral: LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: -1e12, // disable lateral for simple tests
                    lateral_inhibition: -1e12,
                    max_reversals: 5,
                },
```

- [ ] **Step 7: Verify it compiles and all tests pass**

Run: `cd src/rust && cargo test 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 8: Run clippy**

Run: `cd src/rust && cargo clippy 2>&1 | tail -20`
Expected: no errors (warnings about dead_code are fine)

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs src/rust/src/gnc/guidance/lateral.rs
git commit -m "refactor: replace inline lateral logic in guidance_step with lateral::lateral_guidance()"
```

---

### Task 6: Add lateral params to GA parameter spaces (Python)

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py`
- Modify: `src/python/aerocapture/training/evaluate.py:336-378`

- [ ] **Step 1: Add lateral `ParamSpec` entries to the four schemes**

In `param_spaces.py`, append 5 lateral params to each of the four unsigned-magnitude schemes. For `equilibrium_glide` (line 26-34), change the list to:

```python
    "equilibrium_glide": [
        ParamSpec("k_hdot_scale", 0.05, 1.0, 0.3),
        ParamSpec("v_ratio_threshold", 0.9, 1.5, 1.1),
        ParamSpec("velocity_bias_high", 0.0, 0.5, 0.15),
        ParamSpec("velocity_bias_low", 0.0, 1.0, 0.3),
        ParamSpec("alt_bias_threshold", 20.0, 80.0, 40.0),
        ParamSpec("cos_bank_min", -1.0, 0.0, -0.5),
        ParamSpec("cos_bank_max", 0.5, 1.0, 0.95),
        # Lateral guidance
        ParamSpec("lateral.corridor_slope", 5000.0, 20000.0, 13080.458),
        ParamSpec("lateral.corridor_intercept", 0.0, 0.1, 0.0),
        ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),
        ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),
        ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),
    ],
```

Do the same for `energy_controller` (append after `kd`):

```python
    "energy_controller": [
        ParamSpec("gain", 1e-8, 1e-5, 5e-7, log_scale=True),
        ParamSpec("kp", 0.1, 5.0, 1.0),
        ParamSpec("kd", 0.0, 3.0, 0.5),
        # Lateral guidance
        ParamSpec("lateral.corridor_slope", 5000.0, 20000.0, 13080.458),
        ParamSpec("lateral.corridor_intercept", 0.0, 0.1, 0.0),
        ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),
        ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),
        ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),
    ],
```

For `pred_guid` (append after `pdyn_threshold`):

```python
    "pred_guid": [
        ParamSpec("k_drag_high", 0.1, 3.0, 0.8),
        ParamSpec("k_drag_low", 0.05, 2.0, 0.3),
        ParamSpec("pdyn_threshold", 10.0, 500.0, 100.0),
        # Lateral guidance
        ParamSpec("lateral.corridor_slope", 5000.0, 20000.0, 13080.458),
        ParamSpec("lateral.corridor_intercept", 0.0, 0.1, 0.0),
        ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),
        ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),
        ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),
    ],
```

For `fnpag` (append after `bank_max_low_deg`):

```python
    "fnpag": [
        ParamSpec("energy_tol", 1e2, 1e5, 1e4, log_scale=True),
        ParamSpec("prediction_dt", 0.5, 5.0, 2.0),
        ParamSpec("bank_min_deg", 10.0, 40.0, 20.0),
        ParamSpec("bank_max_high_deg", 100.0, 170.0, 140.0),
        ParamSpec("bank_max_low_deg", 70.0, 130.0, 100.0),
        # Lateral guidance
        ParamSpec("lateral.corridor_slope", 5000.0, 20000.0, 13080.458),
        ParamSpec("lateral.corridor_intercept", 0.0, 0.1, 0.0),
        ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),
        ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),
        ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),
    ],
```

**Note:** `max_reversals` is encoded as float [1.0, 10.0] in the chromosome and rounded to int during TOML write.

- [ ] **Step 2: Update `write_guidance_toml()` to split lateral params**

In `evaluate.py`, modify `write_guidance_toml()` (around line 360-362). Replace:

```python
    # Merge GA params into existing section (preserves non-GA fields like energy_min/max)
    section_name = GUIDANCE_TOML_SECTIONS[guidance_type]
    toml_data["guidance"].setdefault(section_name, {}).update(params)
```

with:

```python
    # Split lateral params from scheme-specific params
    lateral_params = {k.removeprefix("lateral."): v for k, v in params.items() if k.startswith("lateral.")}
    scheme_params = {k: v for k, v in params.items() if not k.startswith("lateral.")}

    # Round max_reversals to integer
    if "max_reversals" in lateral_params:
        lateral_params["max_reversals"] = int(round(lateral_params["max_reversals"]))

    # Merge scheme params into [guidance.<scheme>]
    section_name = GUIDANCE_TOML_SECTIONS[guidance_type]
    toml_data["guidance"].setdefault(section_name, {}).update(scheme_params)

    # Merge lateral params into [guidance.lateral]
    if lateral_params:
        toml_data["guidance"].setdefault("lateral", {}).update(lateral_params)
```

- [ ] **Step 3: Verify Python tests pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Run linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -20`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py src/python/aerocapture/training/evaluate.py
git commit -m "feat: add lateral guidance params to GA search space for 4 schemes"
```

---

### Task 7: Add integration test for lateral reversal in simulation

**Files:**
- Create: `src/rust/tests/lateral_reversal.rs`

- [ ] **Step 1: Write an integration test that runs a simulation with lateral guidance active**

```rust
//! Integration test: verify lateral guidance triggers roll reversals during simulation.

use std::path::Path;

mod common;

#[test]
fn lateral_reversal_fires_during_eqglide_sim() {
    // Build a TOML config string with equilibrium_glide + active lateral guidance
    let toml = r#"
base = ["configs/missions/mars.toml"]

[guidance]
type = "equilibrium_glide"

[guidance.lateral]
corridor_slope = 13080.458
corridor_intercept = 0.0
lateral_activation = -0.5
lateral_inhibition = -10.0
max_reversals = 5

[simulation]
n_sims = 1
max_time = 3000.0
"#;

    // Write to a temp file
    let dir = tempfile::tempdir().unwrap();
    let toml_path = dir.path().join("test_lateral.toml");
    std::fs::write(&toml_path, toml).unwrap();

    // Run simulation
    let output = aerocapture::run_for_api(&toml_path);
    assert!(output.is_ok(), "Simulation should not crash with lateral guidance active");

    let result = output.unwrap();
    // Verify the simulation completed (the bank angle should have changed sign at some point)
    // This is a smoke test — we don't assert on specific reversal count, just that it runs
    assert!(result.n_steps > 10, "Simulation should run for multiple steps");
}
```

**Note:** The exact test structure depends on what `run_for_api` returns. Adjust field names to match `RunOutput`. If integration tests in `src/rust/tests/` don't have access to a `common` module, use the pattern from existing integration tests in that directory.

- [ ] **Step 2: Run the integration test**

Run: `cd src/rust && cargo test lateral_reversal -- --nocapture 2>&1 | tail -20`
Expected: test passes (simulation completes without panic)

- [ ] **Step 3: Commit**

```bash
git add src/rust/tests/lateral_reversal.rs
git commit -m "test: add integration test for lateral roll reversal during simulation"
```

---

### Task 8: Regression — verify default configs are unchanged

**Files:** (no new files)

- [ ] **Step 1: Run the full Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -30`
Expected: all tests pass, including existing golden reference tests

- [ ] **Step 2: Run check_all.sh**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh 2>&1 | tail -30`
Expected: all checks pass (fmt, clippy, test, build)

- [ ] **Step 3: Run full Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 4: Run a default config to verify bit-identical output**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml`
Expected: same output as before (lateral guidance inactive with default params)

---

### Task 9: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
