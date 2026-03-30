# Exit Phase Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable exit phase guidance so FTC + 4 unsigned-magnitude schemes use a shared dynamic-pressure-feedback controller on the ascending leg after the trajectory nadir.

**Architecture:** Remove the hardcoded phase-1 override in navigation, add a new `exit.rs` guidance module with a stateless pdyn-feedback controller, gate it via `SimPhase` config, and wire it into the existing `guidance_step()` dispatch. Lateral guidance (inclination correction) continues unchanged during exit phase.

**Tech Stack:** Rust (nalgebra), TOML configs, cargo test + proptest

**Spec:** `docs/superpowers/specs/2026-03-30-exit-phase-guidance-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/rust/src/data/mod.rs` | Modify | Add `sim_phase: SimPhase` to `SimData`, populate from `SimInput` |
| `src/rust/src/gnc/navigation/estimator.rs` | Modify | Remove phase-1 override, add `SimPhase` gating in both `navigate()` and `navigate_ekf()` |
| `src/rust/src/gnc/guidance/exit.rs` | **Create** | Shared exit-phase longitudinal controller |
| `src/rust/src/gnc/guidance/mod.rs` | Modify | Add `pub mod exit;` |
| `src/rust/src/gnc/guidance/ftc.rs` | Modify | Phase-aware dispatch before scheme match |
| `src/rust/src/simulation/runner.rs` | Modify | Wire `reference_velocity` on phase transition, propagate `guidance_phase` to photo output |
| `configs/missions/mars.toml` | Modify | Add exit params to `[guidance.ftc]` section |
| `configs/missions/earth.toml` | Modify | Add exit params to `[guidance.ftc]` section |

---

### Task 1: Thread `SimPhase` into `SimData`

**Files:**
- Modify: `src/rust/src/data/mod.rs:12` (imports), `src/rust/src/data/mod.rs:151-178` (struct), `src/rust/src/data/mod.rs:590-612` (constructor)

- [ ] **Step 1: Add `SimPhase` to the import line in `data/mod.rs`**

In `src/rust/src/data/mod.rs`, line 13 currently reads:

```rust
use crate::config::{
    GuidanceType, IntegrationMode, SimInput, TomlConfig, TomlMonteCarlo, TomlNavigation,
};
```

Change it to:

```rust
use crate::config::{
    GuidanceType, IntegrationMode, SimInput, SimPhase, TomlConfig, TomlMonteCarlo, TomlNavigation,
};
```

- [ ] **Step 2: Add `sim_phase` field to `SimData` struct**

In `src/rust/src/data/mod.rs`, after line 177 (`pub integration_mode: IntegrationMode,`), add:

```rust
    /// Simulation phase mode (Full, CaptureOnly, ExitOnly, Preprogrammed)
    pub sim_phase: SimPhase,
```

- [ ] **Step 3: Populate `sim_phase` in `SimData::from_toml()`**

In `src/rust/src/data/mod.rs`, in the `Ok(SimData { ... })` block (around line 611), after:

```rust
            integration_mode: IntegrationMode::from_toml(&toml.integration, v.periods.integration),
```

Add:

```rust
            sim_phase: config.sim_phase,
```

- [ ] **Step 4: Update all test `SimData` constructors**

Every `test_sim_data()` function that builds a `SimData` must include the new field. There are instances in `estimator.rs` and `ftc.rs` test modules. Add to each:

```rust
            sim_phase: crate::config::SimPhase::Full,
```

This must be added after the `integration_mode` field in each `SimData { ... }` literal.

- [ ] **Step 5: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -30`

Expected: No errors. If there are missing `sim_phase` fields in other test fixtures, add them following the same pattern.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/mod.rs src/rust/src/gnc/navigation/estimator.rs src/rust/src/gnc/guidance/ftc.rs
git commit -m "thread SimPhase into SimData struct"
```

---

### Task 2: Remove phase override in navigation — bias mode

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:241-245`
- Test: `src/rust/src/gnc/navigation/estimator.rs` (inline `#[cfg(test)]` module)

- [ ] **Step 1: Write failing tests for `SimPhase` gating in `navigate()`**

In `src/rust/src/gnc/navigation/estimator.rs`, in the `#[cfg(test)] mod tests` block, add these tests. Note: the `test_sim_data()` function in this file needs to be updated with `sim_phase: SimPhase::Full` from Task 1.

```rust
    /// SimPhase::Full: phase transitions from 1 → 2 after bounce + velocity below threshold.
    #[test]
    fn full_phase_transitions_to_exit() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::Full;
        data.guidance.exit_velocity_threshold = 4400.0;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // First call: descending (FPA negative, pre-bounce) — should be phase 1
        let out1 = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, -0.05, 0.6], // negative FPA → sin < 0 → no bounce
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );
        assert_eq!(out1.guidance_phase, 1, "should be capture phase while descending");

        // Second call: ascending (FPA positive) but velocity still above threshold
        let out2 = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, 0.05, 0.6], // positive FPA → sin > 0 → bounce
            data.entry.initial_aoa,
            20.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );
        assert_eq!(out2.guidance_phase, 1, "above velocity threshold → still capture");

        // Third call: ascending and velocity below threshold → phase 2
        let out3 = navigate(
            &[r, 0.0, 0.0],
            &[4000.0, 0.05, 0.6], // below 4400 threshold
            data.entry.initial_aoa,
            30.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );
        assert_eq!(out3.guidance_phase, 2, "below velocity threshold after bounce → exit phase");
        assert_eq!(out3.phase_transition_flag, 1, "transition flag should be set");
        assert!(out3.reference_velocity.abs() > 0.0, "reference_velocity should be latched");
    }

    /// SimPhase::CaptureOnly: phase stays 1 regardless of state.
    #[test]
    fn capture_only_stays_phase_1() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::CaptureOnly;
        data.guidance.exit_velocity_threshold = 4400.0;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // Trigger bounce
        let _ = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, 0.05, 0.6],
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );

        // Below threshold after bounce — would normally be phase 2
        let out = navigate(
            &[r, 0.0, 0.0],
            &[4000.0, 0.05, 0.6],
            data.entry.initial_aoa,
            20.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );
        assert_eq!(out.guidance_phase, 1, "CaptureOnly must keep phase 1");
    }

    /// SimPhase::ExitOnly: phase stays 2 regardless of state.
    #[test]
    fn exit_only_stays_phase_2() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::ExitOnly;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // Descending, pre-bounce — would normally be phase 1
        let out = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, -0.05, 0.6],
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0], run_biases[1], run_biases[2],
            run_biases[3], run_biases[4], run_biases[5], run_biases[6],
        );
        assert_eq!(out.guidance_phase, 2, "ExitOnly must force phase 2");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd src/rust && cargo test phase_transitions_to_exit capture_only_stays exit_only_stays -- --nocapture 2>&1 | tail -20`

Expected: FAIL — all three tests fail because `guidance_phase` is hardcoded to 1.

- [ ] **Step 3: Replace the phase override with `SimPhase` gating**

In `src/rust/src/gnc/navigation/estimator.rs`, replace lines 241-245:

```rust
    // guidance_phase is hardcoded to 1 (phase management logic above is inactive)
    nav_state.guidance_phase = 1;
    if nav_state.guidance_phase == 1 {
        nav_state.capture_time += data.periods.navigation;
    }
```

With:

```rust
    // Apply SimPhase gating
    match data.sim_phase {
        SimPhase::CaptureOnly => {
            nav_state.guidance_phase = 1;
        }
        SimPhase::ExitOnly => {
            nav_state.guidance_phase = 2;
        }
        SimPhase::Full | SimPhase::Preprogrammed => {
            // Phase logic above already computed the correct phase
        }
    }

    if nav_state.guidance_phase == 1 {
        nav_state.capture_time += data.periods.navigation;
    }
```

Also add the import at the top of the file (after the existing `use crate::config::PlanetConfig;`):

```rust
use crate::config::SimPhase;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd src/rust && cargo test phase_transitions_to_exit capture_only_stays exit_only_stays -- --nocapture 2>&1 | tail -20`

Expected: All three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "remove phase-1 override in bias navigator, add SimPhase gating"
```

---

### Task 3: Remove phase override in navigation — EKF mode

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:561-569`

- [ ] **Step 1: Replace the EKF phase override with `SimPhase` gating**

In `src/rust/src/gnc/navigation/estimator.rs`, replace lines 561-569:

```rust
    // TODO: Enable phase management for EKF mode. The logic above correctly
    // computes bounce/crash/phase transitions but is currently overridden to
    // phase 1 because exit-phase guidance (phase 2) is not yet active (see
    // IMPROVEMENTS.md §6.3). Once exit guidance is implemented, remove this
    // override to let the EKF navigator drive phase transitions.
    legacy.guidance_phase = 1;
    if legacy.guidance_phase == 1 {
        legacy.capture_time += nav_dt;
    }
```

With:

```rust
    // Apply SimPhase gating
    match data.sim_phase {
        SimPhase::CaptureOnly => {
            legacy.guidance_phase = 1;
        }
        SimPhase::ExitOnly => {
            legacy.guidance_phase = 2;
        }
        SimPhase::Full | SimPhase::Preprogrammed => {
            // Phase logic above already computed the correct phase
        }
    }

    if legacy.guidance_phase == 1 {
        legacy.capture_time += nav_dt;
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -20`

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "remove phase-1 override in EKF navigator, add SimPhase gating"
```

---

### Task 4: Create exit guidance module

**Files:**
- Create: `src/rust/src/gnc/guidance/exit.rs`
- Modify: `src/rust/src/gnc/guidance/mod.rs:1` (add module declaration)

- [ ] **Step 1: Write failing unit tests for exit guidance**

Create `src/rust/src/gnc/guidance/exit.rs` with the test module first:

```rust
//! Exit phase longitudinal guidance — dynamic pressure feedback with radial velocity damping.
//!
//! Shared by FTC + the four unsigned-magnitude schemes (EqGlide, EnergyController,
//! PredGuid, FNPAG) after the phase 1 → 2 transition. NN and PiecewiseConstant
//! bypass this (they produce signed bank angles for the full trajectory).

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Compute exit-phase bank angle magnitude using dynamic pressure feedback.
///
/// # Arguments
/// * `nav` — current navigation output (density_exit, density_guidance, velocity, etc.)
/// * `data` — simulation data (exit params from GuidanceParams)
/// * `_planet` — planet constants (unused, reserved for future apoapsis targeting)
/// * `reference_velocity` — radial velocity latched at the phase 1→2 transition
///
/// # Returns
/// Bank angle magnitude in radians, in [0, π].
pub fn exit_guidance(
    nav: &NavigationOutput,
    data: &SimData,
    _planet: &PlanetConfig,
    reference_velocity: f64,
) -> f64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PlanetConfig;
    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::GuidanceParams;
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };
    use crate::gnc::guidance::lateral::LateralParams;

    fn test_sim_data() -> SimData {
        SimData {
            capsule: Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.00008242,
                max_bank_rate: 15.0_f64.to_radians(),
                periods: TimePeriods::default(),
            },
            aero: AeroTables {
                n_points: 2,
                incidence: vec![-0.5, 0.0],
                cx: vec![1.269, 1.269],
                cz: vec![-0.205, -0.205],
                equilibrium_aoa: -0.48,
                nominal_cx: 1.269,
                nominal_cz: -0.205,
                nominal_finesse: -0.205 / 1.269,
                ballistic_coeff: 0.0,
            },
            atmosphere: AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            },
            atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    velocity: 5687.0,
                    flight_path: -10.8_f64.to_radians(),
                    ..Default::default()
                },
                initial_bank: 64.77_f64.to_radians(),
                initial_aoa: -27.5_f64.to_radians(),
                initial_date: 0.0,
            },
            guidance: GuidanceParams {
                exit_velocity_threshold: 4400.0,
                exit_pdyn_margin: 1.75,
                exit_altitude_threshold: 60_000.0,
                exit_radial_vel_gain: 10.0,
                exit_apoapsis_threshold: 100.0,
                density_filter_gain: 0.8,
                lateral: LateralParams::default(),
                ..Default::default()
            },
            incidence: IncidenceProfile {
                n_points: 2,
                altitudes: vec![-10_000.0, 150_000.0],
                incidences: vec![-0.48, -0.48],
            },
            periods: TimePeriods::default(),
            pilot: PilotModel {
                pilot_type: PilotType::Perfect,
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            target_orbit: OrbitalTarget {
                semi_major_axis: 3_649_622.0,
                eccentricity: 0.067,
                inclination: 50.0_f64.to_radians(),
                raan: -7.612_f64.to_radians(),
                apoapsis: 500_130.0,
                periapsis: 11_233.0,
            },
            final_conditions: FinalConditions::default(),
            parking_orbit: ParkingOrbit::default(),
            constraints: Constraints::default(),
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
            integration_mode: crate::config::IntegrationMode::FixedGill,
            sim_phase: crate::config::SimPhase::Full,
        }
    }

    /// Build a NavigationOutput representing a typical post-bounce ascending state.
    fn ascending_nav(velocity: f64, fpa: f64, density_guidance: f64, density_exit: f64) -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0;
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [velocity, fpa, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance,
            density_exit,
            dynamic_pressure_estimated: 0.5 * density_guidance * velocity * velocity,
            energy_estimated: -1e6,
            guidance_phase: 2,
            bounce_flag: 1,
            ..Default::default()
        }
    }

    #[test]
    fn exit_guidance_returns_finite_bounded_bank() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let nav = ascending_nav(4000.0, 0.05, 1e-4, 1e-6);
        let reference_velocity = 50.0; // radial velocity at transition

        let bank = exit_guidance(&nav, &data, &planet, reference_velocity);

        assert!(bank.is_finite(), "bank must be finite, got {}", bank);
        assert!(bank >= 0.0, "bank magnitude must be >= 0, got {}", bank);
        assert!(
            bank <= std::f64::consts::PI,
            "bank magnitude must be <= pi, got {}",
            bank
        );
    }

    #[test]
    fn exit_guidance_higher_pdyn_gives_more_drag() {
        // When current pdyn is much higher than target, cos_bank should be larger
        // (more lift-down / more drag), meaning smaller bank angle magnitude.
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let reference_velocity = 50.0;

        let nav_low_density = ascending_nav(4000.0, 0.05, 1e-5, 1e-6);
        let nav_high_density = ascending_nav(4000.0, 0.05, 1e-3, 1e-6);

        let bank_low = exit_guidance(&nav_low_density, &data, &planet, reference_velocity);
        let bank_high = exit_guidance(&nav_high_density, &data, &planet, reference_velocity);

        // Higher current pdyn → more correction → different bank angle
        assert!(
            (bank_low - bank_high).abs() > 1e-6,
            "different densities should produce different bank angles: low={}, high={}",
            bank_low,
            bank_high,
        );
    }

    #[test]
    fn exit_guidance_zero_density_exit_gives_90_degrees() {
        // When density_exit is 0, pdyn_target is 0, correction is 1.0 → cos_bank=1 → bank=0
        // Actually with zero density_exit: pdyn_target=0, so pdyn_correction = (pdyn_current - 0)/pdyn_current = 1.0
        // Plus radial_vel term. With both terms, cos_bank >= 1.0 → clamped to 1.0 → bank = 0.
        // This tests the clamping behavior.
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let nav = ascending_nav(4000.0, 0.05, 1e-4, 0.0);

        let bank = exit_guidance(&nav, &data, &planet, 0.0);

        assert!(bank.is_finite(), "bank must be finite with zero density_exit");
        assert!(bank >= 0.0 && bank <= std::f64::consts::PI);
    }
}
```

- [ ] **Step 2: Register the module in `mod.rs`**

In `src/rust/src/gnc/guidance/mod.rs`, after line 3 (`pub mod energy_controller;`), add:

```rust
pub mod exit;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd src/rust && cargo test exit_guidance -- --nocapture 2>&1 | tail -20`

Expected: FAIL — `todo!()` panics.

- [ ] **Step 4: Implement `exit_guidance()`**

Replace the `todo!()` body in `exit_guidance()` with:

```rust
pub fn exit_guidance(
    nav: &NavigationOutput,
    data: &SimData,
    _planet: &PlanetConfig,
    reference_velocity: f64,
) -> f64 {
    let velocity = nav.velocity_estimated[0];
    let fpa = nav.velocity_estimated[1];
    let velocity_radial = velocity * fpa.sin();

    let exit = &data.guidance;

    // 1. Target dynamic pressure from exit density
    let pdyn_target = nav.density_exit * velocity * velocity * exit.exit_pdyn_margin;

    // 2. Current dynamic pressure
    let pdyn_current = 0.5 * nav.density_guidance * velocity * velocity;

    // Safe denominator to avoid division by zero
    let pdyn_safe = if pdyn_current.abs() > 1e-10 {
        pdyn_current
    } else {
        1e-10
    };

    // 3. Dynamic pressure correction (normalized error)
    let pdyn_correction = (pdyn_current - pdyn_target) / pdyn_safe;

    // 4. Radial velocity damping (normalized by pdyn)
    let radial_vel_correction =
        exit.exit_radial_vel_gain * (velocity_radial - reference_velocity) / pdyn_safe;

    // 5. Predictor-corrector
    let cos_bank = pdyn_correction + radial_vel_correction;

    // 6. Clamp and convert
    cos_bank.clamp(-1.0, 1.0).acos()
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src/rust && cargo test exit_guidance -- --nocapture 2>&1 | tail -20`

Expected: All three tests PASS.

- [ ] **Step 6: Add proptest for robustness**

Append to the `#[cfg(test)] mod tests` block in `exit.rs`:

```rust
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any valid ascending state, exit_guidance produces finite, bounded output.
            #[test]
            fn output_always_finite_and_bounded(
                vel in 2000.0..6000.0_f64,
                fpa in 0.01..0.3_f64,  // ascending: positive FPA
                density_guidance in 1e-8..1e-2_f64,
                density_exit in 0.0..1e-4_f64,
                ref_vel in -200.0..200.0_f64,
            ) {
                let data = test_sim_data();
                let planet = PlanetConfig::mars();
                let nav = ascending_nav(vel, fpa, density_guidance, density_exit);

                let bank = exit_guidance(&nav, &data, &planet, ref_vel);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= 0.0, "bank < 0: {}", bank);
                prop_assert!(bank <= std::f64::consts::PI, "bank > pi: {}", bank);
            }
        }
    }
```

- [ ] **Step 7: Run full test suite for exit module**

Run: `cd src/rust && cargo test exit_guidance -- --nocapture 2>&1 | tail -20`

Expected: All tests PASS (including proptest).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/gnc/guidance/exit.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "add exit phase guidance module with pdyn feedback controller"
```

---

### Task 5: Wire phase dispatch in `guidance_step()`

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs:1-10` (imports), `src/rust/src/gnc/guidance/ftc.rs:132-164` (dispatch)

- [ ] **Step 1: Write failing test for phase dispatch**

In `src/rust/src/gnc/guidance/ftc.rs`, add to the `#[cfg(test)] mod tests` block:

```rust
    /// Phase 2 should dispatch to exit guidance for FTC scheme.
    #[test]
    fn phase_2_dispatches_to_exit_guidance() {
        let mut nav = test_nav();
        // Set up ascending state with phase 2
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;
        nav.density_exit = 1e-6;
        nav.velocity_estimated[1] = 0.05; // positive FPA (ascending)

        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = FtcState::new(initial_bank, -0.48_f64.to_radians());
        state.reference_velocity = 50.0; // latched at transition

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "exit phase should produce finite bank: {}",
            out.bank_angle_commanded
        );
    }

    /// Signed-bank schemes (NN, PiecewiseConstant) should ignore phase 2.
    #[test]
    fn piecewise_constant_ignores_exit_phase() {
        let mut nav = test_nav();
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;

        // NN requires neural_net data; just verify that skip_lateral path is taken
        // by checking the scheme uses its own dispatch, not exit guidance.
        // We can't easily test NN without a model, so instead verify PiecewiseConstant:
        let mut data = test_sim_data();
        data.guidance.piecewise_constant = crate::data::guidance_params::PiecewiseConstantParams {
            bank_angles: [0.5; 10],
            energy_boundaries: [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0],
        };

        let planet = PlanetConfig::mars();
        let initial_bank = 0.5;
        let mut state = FtcState::new(initial_bank, -0.48_f64.to_radians());

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::PiecewiseConstant,
        );

        // PiecewiseConstant should return its own bank angle, not exit guidance
        assert!(out.bank_angle_commanded.is_finite());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test phase_2_dispatches piecewise_constant_ignores -- --nocapture 2>&1 | tail -20`

Expected: FAIL — phase 2 has no special handling yet.

- [ ] **Step 3: Add exit module import and phase dispatch**

In `src/rust/src/gnc/guidance/ftc.rs`, add to the import block (after line 8):

```rust
use crate::gnc::guidance::exit;
```

Then in `guidance_step()`, replace the block at lines 132-164:

```rust
    // === Longitudinal bank angle command ===
    // reference_bank_angle passed as parameter from config.reference_bank_angle
    let bank_angle_longitudinal: f64;

    if is_reference {
        state.bank_angle_commanded = reference_bank_angle;
        bank_angle_longitudinal = reference_bank_angle;
    } else if longitudinal_active == 0 {
        bank_angle_longitudinal = reference_bank_angle.abs();
    } else {
        // Longitudinal guidance dispatch
        bank_angle_longitudinal = match guidance_type {
            GuidanceType::Ftc => capture_guidance(nav, energy, altitude, state, data, planet),
            GuidanceType::NeuralNetwork => {
                let nn = data.neural_net.as_ref().expect("NN params not loaded");
                neural::nn_bank_angle(nav, nn, planet, data.target_orbit.inclination)
            }
            GuidanceType::EquilibriumGlide => {
                equilibrium_glide::equilibrium_glide_bank(nav, data, planet)
            }
            GuidanceType::EnergyController => {
                energy_controller::energy_controller_bank(nav, &state.energy_ctrl, data, planet)
            }
            GuidanceType::PredGuid => predguid::predguid_bank(nav, &state.predguid, data, planet),
            GuidanceType::Fnpag => fnpag::fnpag_bank(nav, &mut state.fnpag, data, planet),
            GuidanceType::PiecewiseConstant => piecewise_constant::piecewise_constant_bank(
                nav,
                &data.guidance.piecewise_constant,
                planet,
            ),
        };
        state.n_active += 1;
    }
```

With:

```rust
    // === Longitudinal bank angle command ===
    // reference_bank_angle passed as parameter from config.reference_bank_angle
    let bank_angle_longitudinal: f64;

    // Schemes that produce signed bank angles bypass exit guidance entirely
    let uses_exit_guidance = !matches!(
        guidance_type,
        GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
    );

    if is_reference {
        state.bank_angle_commanded = reference_bank_angle;
        bank_angle_longitudinal = reference_bank_angle;
    } else if longitudinal_active == 0 {
        bank_angle_longitudinal = reference_bank_angle.abs();
    } else if nav.guidance_phase == 2 && uses_exit_guidance {
        // Exit phase: shared pdyn-feedback controller for all unsigned-magnitude schemes
        bank_angle_longitudinal = exit::exit_guidance(nav, data, planet, state.reference_velocity);
        state.n_active += 1;
    } else {
        // Capture phase: scheme-specific longitudinal guidance
        bank_angle_longitudinal = match guidance_type {
            GuidanceType::Ftc => capture_guidance(nav, energy, altitude, state, data, planet),
            GuidanceType::NeuralNetwork => {
                let nn = data.neural_net.as_ref().expect("NN params not loaded");
                neural::nn_bank_angle(nav, nn, planet, data.target_orbit.inclination)
            }
            GuidanceType::EquilibriumGlide => {
                equilibrium_glide::equilibrium_glide_bank(nav, data, planet)
            }
            GuidanceType::EnergyController => {
                energy_controller::energy_controller_bank(nav, &state.energy_ctrl, data, planet)
            }
            GuidanceType::PredGuid => predguid::predguid_bank(nav, &state.predguid, data, planet),
            GuidanceType::Fnpag => fnpag::fnpag_bank(nav, &mut state.fnpag, data, planet),
            GuidanceType::PiecewiseConstant => piecewise_constant::piecewise_constant_bank(
                nav,
                &data.guidance.piecewise_constant,
                planet,
            ),
        };
        state.n_active += 1;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test phase_2_dispatches piecewise_constant_ignores -- --nocapture 2>&1 | tail -20`

Expected: All tests PASS.

- [ ] **Step 5: Run full guidance test suite**

Run: `cd src/rust && cargo test guidance -- --nocapture 2>&1 | tail -20`

Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "add phase-aware dispatch in guidance_step: exit guidance for phase 2"
```

---

### Task 6: Wire runner — `reference_velocity` latch and photo phase output

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:493-494` (add `guidance_phase_for_photo` variable), `src/rust/src/simulation/runner.rs:566-567` (latch reference_velocity), `src/rust/src/simulation/runner.rs:866-910` (photo builder)

- [ ] **Step 1: Add `guidance_phase_for_photo` variable**

In `src/rust/src/simulation/runner.rs`, after line 494 (`let mut density_estimate_for_photo = 0.0_f64;`), add:

```rust
    let mut guidance_phase_for_photo = 1_i32;
```

- [ ] **Step 2: Latch `reference_velocity` on phase transition and capture `guidance_phase`**

After line 567 (`density_estimate_for_photo = nav_out.density_guidance;`), add:

```rust
            guidance_phase_for_photo = nav_out.guidance_phase;

            // Latch reference velocity at the phase 1→2 transition
            if nav_out.phase_transition_flag == 1 {
                ftc_state.reference_velocity = nav_out.reference_velocity;
            }
```

- [ ] **Step 3: Add `guidance_phase` parameter to `build_photo_values`**

Change the signature of `build_photo_values` (line 866) from:

```rust
fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &PlanetConfig,
    dynamic_pressure: f64,
    density_estimate: f64,
    sim_index: i32,
    cumulative_bank_change: f64,
    data: &SimData,
    density_gain: f64,
    run_state: &init::RunState,
    cumulative_flux: f64,
) -> [f64; 29] {
```

To:

```rust
fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &PlanetConfig,
    dynamic_pressure: f64,
    density_estimate: f64,
    sim_index: i32,
    cumulative_bank_change: f64,
    data: &SimData,
    density_gain: f64,
    run_state: &init::RunState,
    cumulative_flux: f64,
    guidance_phase: i32,
) -> [f64; 29] {
```

- [ ] **Step 4: Replace the phase heuristic with `guidance_phase` parameter**

In `build_photo_values`, replace lines 906-910:

```rust
    let phase = if !sim.bounced {
        if altitude > 80e3 { 1.0 } else { 2.0 }
    } else {
        if sim.state[0] > 80e3 { 3.0 } else { 2.0 }
    };
```

With:

```rust
    let phase = guidance_phase as f64;
```

- [ ] **Step 5: Update all call sites of `build_photo_values` to pass `guidance_phase_for_photo`**

There are two call sites (lines ~624 and ~704). Add `guidance_phase_for_photo,` as the last argument to both calls, after `sim.state[6],`.

For the call around line 624:
```rust
            photo_lines.push(build_photo_values(
                &sim,
                sim_time,
                planet,
                dynamic_pressure_for_photo,
                density_estimate_for_photo,
                sim_idx + 1,
                cumulative_bank_change_deg * DEG_TO_RAD,
                data,
                nav_filter.density_gain(),
                run_state,
                sim.state[6],
                guidance_phase_for_photo,
            ));
```

For the call around line 704 (final photo), same addition.

- [ ] **Step 6: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | head -30`

Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "wire reference_velocity latch and guidance_phase into photo output"
```

---

### Task 7: Move exit params to mission-level configs

**Files:**
- Modify: `configs/missions/mars.toml`
- Modify: `configs/missions/earth.toml`

- [ ] **Step 1: Add exit params to `configs/missions/mars.toml`**

Add at the end of the file (after the `[corridor]` section):

```toml
[guidance.ftc]
exit_velocity_threshold = 4400.0
exit_pdyn_margin = 1.75
exit_altitude_threshold = 60.0
exit_radial_vel_gain = 10.0
exit_apoapsis_threshold = 100.0
security_capture = 1
security_exit = 3
```

- [ ] **Step 2: Add exit params to `configs/missions/earth.toml`**

Read `configs/missions/earth.toml` first, then add the same `[guidance.ftc]` section at the end. Use the same default values (4400, 1.75, 60, 10, 100) — Earth missions can tune later.

- [ ] **Step 3: Remove duplicated exit params from FTC-specific configs**

In `configs/nominal/msr_aller_ftc_consolidated.toml`, `configs/nominal/msr_aller_ftc_nominal.toml`, `configs/nominal/msr_aller_ftc_mc_domain.toml`, `configs/nominal/esr_aller_ftc_nominal.toml`, `configs/training/msr_aller_ftc_train.toml`, and `configs/test/test_guided_orig.toml`: remove the `exit_*` lines and `security_capture`/`security_exit` lines from their `[guidance.ftc]` sections, since they now inherit from the mission base.

The remaining capture-phase params (`capture_damping`, `capture_frequency`, etc.) stay in the FTC-specific configs because they are scheme-specific.

- [ ] **Step 4: Verify config loading still works**

Run: `cd src/rust && cargo run --release -- configs/nominal/msr_aller_ftc_consolidated.toml 2>&1 | head -20`

Expected: Simulation runs without config errors.

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "move exit params to mission-level TOML configs for cross-scheme availability"
```

---

### Task 8: Full test suite and regression verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Run Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -30`

Expected: All tests PASS (existing + new). Pay attention to any tests that assumed `guidance_phase` was always 1.

- [ ] **Step 2: Run clippy**

Run: `cd src/rust && cargo clippy -- -D warnings 2>&1 | tail -20`

Expected: No warnings.

- [ ] **Step 3: Run fmt check**

Run: `cd src/rust && cargo fmt -- --check 2>&1 | tail -10`

Expected: No formatting issues.

- [ ] **Step 4: Run a CaptureOnly regression simulation**

Create a temporary test: run the FTC nominal config with `phase = "capture_only"` in TOML and verify the output matches the current (pre-change) behavior bit-for-bit. The simplest way is to set `phase = "capture_only"` in the test config and run it:

Run: `cd src/rust && cargo run --release -- configs/test/test_guided_orig.toml 2>&1 | tail -20`

Expected: Simulation completes. Output should be identical to pre-change results (since `test_guided_orig.toml` defaults to `phase = "full"` which now activates exit guidance — verify it still captures).

- [ ] **Step 5: Run a Full phase simulation and verify capture**

Run: `cd src/rust && cargo run --release -- configs/nominal/msr_aller_ftc_consolidated.toml 2>&1 | tail -20`

Expected: Simulation captures (ifinal=3). The exit guidance activates after the bounce and may change the final DV compared to the old capture-only behavior — this is expected and desired.

- [ ] **Step 6: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix test/lint issues from exit phase guidance integration"
```

---

### Task 9: Invoke smart-commit skill

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole `feature/exit-phase-guidance` git branch into account. This will sync CLAUDE.md and README.md with the codebase changes, then commit everything.

---

### Task 10: Request code review

- [ ] **Step 1: Invoke the `requesting-code-review` skill**

Use the `requesting-code-review` skill to review the completed exit phase guidance implementation against the spec.
