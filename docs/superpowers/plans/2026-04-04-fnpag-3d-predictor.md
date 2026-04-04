# FNPAG 3D Predictor Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FNPAG's planar 3-state predictor with a full 3D 6-state predictor using J2 gravity, planet rotation, RK4 integration, and correct inertial exit energy.

**Architecture:** Rewrite `predict_exit_energy` internals in `fnpag.rs`: expand `PredState` to 6 components, add a `pred_derivatives` helper matching `runner.rs` EOM (minus dispersions/winds), replace Euler with classic RK4, fix exit energy via existing `total_energy()`. The `fnpag_bank` secant method is structurally unchanged.

**Tech Stack:** Rust, nalgebra (existing), `gravity::gravity()`, `coordinates::total_energy()`, `coordinates::geodetic_from_spherical()`

**Spec:** `docs/superpowers/specs/2026-04-04-fnpag-3d-predictor-design.md`

---

### Task 1: Rewrite FNPAG predictor to 3D with RK4 and correct energy

This is one atomic change -- PredState expansion, new derivatives, new predictor, and updated initialization must all land together (code won't compile between them).

**Files:**
- Modify: `src/rust/src/gnc/guidance/fnpag.rs`

- [ ] **Step 1: Update imports**

Replace the imports at the top of `fnpag.rs` (lines 19-22):

```rust
use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;
```

With:

```rust
use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::physics::gravity;
```

- [ ] **Step 2: Expand PredState**

Replace the existing `PredState` (lines 45-51):

```rust
/// Simplified state for forward prediction.
#[derive(Clone, Copy)]
struct PredState {
    r: f64,     // radius (m)
    v: f64,     // velocity (m/s)
    gamma: f64, // flight path angle (rad)
}
```

With:

```rust
/// State for forward prediction (matches main sim's 6 translational DOFs).
#[derive(Clone, Copy)]
struct PredState {
    r: f64,     // radius (m)
    lon: f64,   // longitude (rad)
    lat: f64,   // latitude (rad)
    v: f64,     // relative velocity (m/s)
    gamma: f64, // flight path angle (rad)
    psi: f64,   // heading/azimuth (rad)
}
```

- [ ] **Step 3: Add pred_derivatives helper**

Insert the `pred_derivatives` function right after the `PredState` struct (before `predict_exit_energy`). This mirrors `runner.rs:1279-1357` but uses onboard atmosphere, no dispersions, no winds, no heat flux, and zero lateral lift:

```rust
/// Compute 3D trajectory derivatives for the onboard predictor.
///
/// Matches the main simulator EOM (runner.rs `compute_derivatives`) with:
/// - Onboard atmosphere model (no dispersions)
/// - J2/J3/J4 gravity via `gravity::gravity()`
/// - Planet rotation (Coriolis + centrifugal)
/// - Zero lateral lift (sin_bank = 0): predictor doesn't know roll sign
/// - Nominal aero coefficients at initial AoA
/// - No winds
fn pred_derivatives(
    s: &PredState,
    bank_angle: f64,
    planet: &PlanetConfig,
    data: &SimData,
) -> [f64; 6] {
    let (altitude, _) = geodetic_from_spherical(s.r, s.lon, s.lat, planet);
    let rho = data.atmosphere_onboard.density_at(altitude, &data.atmosphere);

    let cx = data.aero.interpolate_cx(data.entry.initial_aoa);
    let cz = data.aero.interpolate_cz(data.entry.initial_aoa).abs();

    let mass = data.capsule.mass;
    let sref = data.capsule.reference_area;
    let aero_factor = rho * sref / (2.0 * mass);
    let drag = aero_factor * cx * s.v * s.v;
    let lift = aero_factor * cz * s.v * s.v;

    let (gravtl, gravtr) = gravity::gravity(s.r, s.lat, planet);

    let cos_bank = bank_angle.cos();
    // sin_bank = 0: predictor assumes no lateral lift (roll sign unknown)
    let cos_gamma = s.gamma.cos();
    let sin_gamma = s.gamma.sin();
    let cos_psi = s.psi.cos();
    let sin_psi = s.psi.sin();
    let cos_lat = s.lat.cos();
    let sin_lat = s.lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;

    let omega = planet.omega;

    let dr = s.v * sin_gamma;
    let dlon = s.v * cos_gamma * sin_psi / (s.r * cos_lat);
    let dlat = s.v * cos_gamma * cos_psi / s.r;

    let dv = -drag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * s.r * cos_lat
            * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = if s.v.abs() > 1.0 {
        (lift * cos_bank / s.v) + (s.v * cos_gamma / s.r)
            - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / s.v)
            + (2.0 * omega * sin_psi * cos_lat)
            + (omega * omega * s.r * cos_lat
                * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma)
                / s.v)
    } else {
        0.0
    };

    // Lateral lift term is zero (sin_bank = 0), but gravity/Coriolis/centrifugal
    // still drive heading evolution.
    let dpsi = if s.v.abs() > 1.0 && cos_gamma.abs() > 1e-10 {
        (s.v * cos_gamma * sin_psi * tan_lat / s.r)
            + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
            + (gravtl * sin_psi / (s.v * cos_gamma))
            + (omega * omega * s.r * cos_lat * sin_lat * sin_psi / (s.v * cos_gamma))
    } else {
        0.0
    };

    [dr, dlon, dlat, dv, dgamma, dpsi]
}
```

- [ ] **Step 4: Replace predict_exit_energy**

Replace the entire `predict_exit_energy` function (lines 53-130) with the 3D RK4 version:

```rust
/// Predict exit energy by integrating 3D equations of motion forward.
///
/// Uses the same EOM as the main simulator (J2 gravity, planet rotation,
/// Coriolis/centrifugal) but with onboard atmosphere, no dispersions, no winds,
/// and zero lateral lift (sin_bank = 0). RK4 integration.
///
/// Integrates until atmosphere exit or crash.
fn predict_exit_energy(
    initial: PredState,
    bank_angle: f64,
    planet: &PlanetConfig,
    data: &SimData,
    exit_alt: f64,
    dt: f64,
) -> f64 {
    let req = planet.equatorial_radius;
    let max_steps = 2000;
    let mut s = initial;

    for _ in 0..max_steps {
        let alt = s.r - req;

        // Termination: crash
        if alt <= 0.0 {
            return 1e8;
        }
        // Termination: atmosphere exit (ascending)
        if alt >= exit_alt && s.gamma.sin() > 0.0 {
            return total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet);
        }

        // Classic RK4
        let k1 = pred_derivatives(&s, bank_angle, planet, data);

        let s2 = PredState {
            r: s.r + 0.5 * dt * k1[0],
            lon: s.lon + 0.5 * dt * k1[1],
            lat: s.lat + 0.5 * dt * k1[2],
            v: s.v + 0.5 * dt * k1[3],
            gamma: s.gamma + 0.5 * dt * k1[4],
            psi: s.psi + 0.5 * dt * k1[5],
        };
        let k2 = pred_derivatives(&s2, bank_angle, planet, data);

        let s3 = PredState {
            r: s.r + 0.5 * dt * k2[0],
            lon: s.lon + 0.5 * dt * k2[1],
            lat: s.lat + 0.5 * dt * k2[2],
            v: s.v + 0.5 * dt * k2[3],
            gamma: s.gamma + 0.5 * dt * k2[4],
            psi: s.psi + 0.5 * dt * k2[5],
        };
        let k3 = pred_derivatives(&s3, bank_angle, planet, data);

        let s4 = PredState {
            r: s.r + dt * k3[0],
            lon: s.lon + dt * k3[1],
            lat: s.lat + dt * k3[2],
            v: s.v + dt * k3[3],
            gamma: s.gamma + dt * k3[4],
            psi: s.psi + dt * k3[5],
        };
        let k4 = pred_derivatives(&s4, bank_angle, planet, data);

        s.r += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        s.lon += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
        s.lat += dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
        s.v += dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
        s.gamma += dt / 6.0 * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]);
        s.psi += dt / 6.0 * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]);

        // Safety: velocity can't go negative
        if s.v <= 0.0 {
            return 1e8;
        }
    }

    // Timeout -- didn't exit atmosphere
    total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet)
}
```

- [ ] **Step 5: Update fnpag_bank PredState initialization**

In `fnpag_bank`, replace the PredState construction (lines 153-158):

```rust
    // Current state for prediction
    let current = PredState {
        r: nav.position_estimated[0],
        v: nav.velocity_estimated[0],
        gamma: nav.velocity_estimated[1],
    };
```

With:

```rust
    // Current state for prediction (full 6-DOF from navigation)
    let current = PredState {
        r: nav.position_estimated[0],
        lon: nav.position_estimated[1],
        lat: nav.position_estimated[2],
        v: nav.velocity_estimated[0],
        gamma: nav.velocity_estimated[1],
        psi: nav.velocity_estimated[2],
    };
```

- [ ] **Step 6: Update module docstring**

Replace the module docstring (lines 1-17):

```rust
//! FNPAG -- Fully Numerical Predictor-corrector Aerocapture Guidance.
//!
//! Based on Ping Lu's algorithm (Journal of Guidance, Control, and Dynamics,
//! 2015). This is a modern predictor-corrector specifically designed for
//! aerocapture, using numerical forward prediction of the trajectory to
//! find the bank angle that achieves a target exit energy.
//!
//! Algorithm overview:
//! 1. Predict forward trajectory with current bank angle using 3D equations
//!    of motion (J2 gravity, planet rotation, onboard atmosphere model)
//! 2. Compute predicted exit orbital energy (inertial velocity)
//! 3. Use secant method to find the bank angle that achieves target energy
//! 4. Blend with equilibrium glide near atmosphere boundaries
//!
//! The predictor uses the same EOM as the main simulator but with onboard
//! atmosphere (no dispersions/winds) and zero lateral lift (roll sign unknown).
//! RK4 integration.
//!
//! The key insight vs FTC: FNPAG directly targets the exit orbital energy
//! rather than tracking a pre-computed reference trajectory. This makes it
//! inherently more robust to dispersions since it continuously re-plans.
```

- [ ] **Step 7: Verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -20`

Expected: Clean compilation (no errors).

- [ ] **Step 8: Run existing tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::fnpag 2>&1`

Expected: All 5 existing unit tests pass. The test fixtures set `position_estimated[1]` (lon) and `position_estimated[2]` (lat) to 0.0, and `velocity_estimated[2]` (psi) to 0.6. These are valid values for the 3D predictor. Output bank angles will differ numerically from before but remain finite and bounded.

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/gnc/guidance/fnpag.rs
git commit -m "feat: upgrade FNPAG predictor to 3D with J2 gravity, rotation, and RK4

Replace planar 3-state Euler predictor with full 6-DOF RK4 predictor.
Adds J2/J3/J4 gravity, Coriolis/centrifugal terms, and correct inertial
exit energy via total_energy(). Uses onboard atmosphere, no dispersions."
```

---

### Task 2: Add new unit tests for 3D-specific behavior

**Files:**
- Modify: `src/rust/src/gnc/guidance/fnpag.rs` (test module)

- [ ] **Step 1: Add exit energy inertial test**

Add this test after the existing `second_call_produces_finite_output` test:

```rust
    /// The predictor must use inertial (absolute) velocity for exit energy,
    /// not relative velocity. For a planet with nonzero omega, these differ.
    #[test]
    fn exit_energy_uses_inertial_velocity() {
        // Two predictions: one on a planet with rotation, one without.
        // With rotation, the inertial velocity is higher (prograde entry),
        // so the predicted exit energy should be higher (less negative).
        let nav = test_nav(5687.0);
        let data = test_sim_data();

        let planet_rotating = PlanetConfig::mars();
        let planet_static = PlanetConfig {
            omega: 0.0,
            ..PlanetConfig::mars()
        };

        let mut state_rot = FnpagState::new(64.77_f64.to_radians());
        let mut state_stat = FnpagState::new(64.77_f64.to_radians());

        // Both produce finite bank angles
        let bank_rot = fnpag_bank(&nav, &mut state_rot, &data, &planet_rotating);
        let bank_stat = fnpag_bank(&nav, &mut state_stat, &data, &planet_static);

        assert!(bank_rot.is_finite(), "rotating bank not finite: {bank_rot}");
        assert!(bank_stat.is_finite(), "static bank not finite: {bank_stat}");

        // The bank angles should differ because the energy model differs
        assert!(
            (bank_rot - bank_stat).abs() > 1e-6,
            "rotation should affect bank angle: rot={bank_rot:.6} stat={bank_stat:.6}"
        );
    }
```

- [ ] **Step 2: Add J2 latitude sensitivity test**

Add this test after the previous one:

```rust
    /// J2 gravity depends on latitude. The predictor should produce different
    /// bank angles for high-latitude vs equatorial entries (same speed/FPA).
    #[test]
    fn j2_sensitivity_with_latitude() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        // Equatorial entry (lat = 0)
        let nav_equator = test_nav(5687.0); // lat = 0.0 from test_nav

        // High-latitude entry (lat = 60 deg)
        let mut nav_high_lat = test_nav(5687.0);
        nav_high_lat.position_estimated[2] = 60.0_f64.to_radians();

        let mut state_eq = FnpagState::new(64.77_f64.to_radians());
        let mut state_hl = FnpagState::new(64.77_f64.to_radians());

        let bank_eq = fnpag_bank(&nav_equator, &mut state_eq, &data, &planet);
        let bank_hl = fnpag_bank(&nav_high_lat, &mut state_hl, &data, &planet);

        assert!(bank_eq.is_finite(), "equatorial bank not finite");
        assert!(bank_hl.is_finite(), "high-lat bank not finite");

        // J2 + 3D effects should produce measurably different bank commands
        assert!(
            (bank_eq - bank_hl).abs() > 1e-4,
            "J2 latitude effect too small: eq={bank_eq:.6} hl={bank_hl:.6}"
        );
    }
```

- [ ] **Step 3: Extend proptest to vary latitude and heading**

Replace the existing proptest block with a version that also varies lat and psi:

```rust
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For valid atmospheric entry conditions, FNPAG must always return a
            /// finite bank angle within [0, pi].
            #[test]
            fn output_always_finite_and_bounded(
                alt in 20_000.0..100_000.0_f64,
                vel in 3_000.0..6_000.0_f64,
                fpa in -0.15..0.0_f64,
                lat in -1.0..1.0_f64,
                psi in -3.0..3.0_f64,
            ) {
                let mut nav = test_nav(vel);
                let r = PlanetConfig::mars().equatorial_radius + alt;
                nav.position_estimated[0] = r;
                nav.position_estimated[2] = lat;
                nav.velocity_estimated[1] = fpa;
                nav.velocity_estimated[2] = psi;
                nav.density_guidance = 0.001;
                nav.dynamic_pressure_estimated = 0.5 * 0.001 * vel * vel;

                let mut state = FnpagState::new(64.77_f64.to_radians());
                let data = test_sim_data();
                let planet = PlanetConfig::mars();

                let bank = fnpag_bank(&nav, &mut state, &data, &planet);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= 0.0 - 1e-10, "bank negative: {}", bank);
                prop_assert!(bank <= std::f64::consts::PI + 1e-10, "bank > pi: {}", bank);
            }
        }
    }
```

- [ ] **Step 4: Run all FNPAG unit tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::fnpag -- --nocapture 2>&1`

Expected: All tests pass (existing + 2 new deterministic + updated proptest).

- [ ] **Step 5: Run integration tests that touch FNPAG**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test error_paths fnpag 2>&1`

Expected: All 5 FNPAG error path tests pass (zero velocity, zero density, extreme FPA up/down, very high altitude). These test finite/bounded output, not specific values.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/fnpag.rs
git commit -m "test: add 3D-specific FNPAG tests (inertial energy, J2 latitude, extended proptest)"
```

---

### Task 3: Regenerate FNPAG golden file

**Files:**
- Modify: `tests/reference_data/rust_golden/fnpag/final.golden_fnpag.csv`

- [ ] **Step 1: Build release binary**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release 2>&1 | tail -5`

Expected: Successful build.

- [ ] **Step 2: Run FNPAG golden config**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./src/rust/target/release/aerocapture configs/test/test_fnpag_golden.toml 2>&1`

Expected: Simulation completes. Produces `output/final.golden_fnpag.csv`.

- [ ] **Step 3: Replace golden file**

Run: `cp output/final.golden_fnpag.csv tests/reference_data/rust_golden/fnpag/final.golden_fnpag.csv`

- [ ] **Step 4: Run guidance regression test**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test guidance_regression fnpag 2>&1`

Expected: `guidance_regression::fnpag` passes (actual output matches newly regenerated golden).

- [ ] **Step 5: Verify other schemes' regressions are untouched**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test guidance_regression 2>&1`

Expected: All 6 regression tests pass. Only FNPAG output changed; other schemes are unaffected.

- [ ] **Step 6: Commit**

```bash
git add tests/reference_data/rust_golden/fnpag/final.golden_fnpag.csv
git commit -m "test: regenerate FNPAG golden file for 3D predictor"
```

---

### Task 4: Run full test suite and E2E validation

**Files:** None (validation only)

- [ ] **Step 1: Run full Rust test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -20`

Expected: All tests pass (unit + integration + E2E).

- [ ] **Step 2: Run clippy**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo clippy -- -D warnings 2>&1 | tail -10`

Expected: No warnings. Fix any clippy suggestions before proceeding.

- [ ] **Step 3: Run fmt check**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo fmt --check 2>&1`

Expected: No formatting differences.

- [ ] **Step 4: Run E2E FNPAG test**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test e2e guidance_fnpag 2>&1`

Expected: `guidance_fnpag_completes` passes (FNPAG sim runs to completion with the training config).

---

### Task 5: Smart commit (final)

- [ ] **Step 1: Invoke smart-commit skill**

Use the `smart-commit` skill to sync CLAUDE.md/README.md and create a final commit covering the whole branch.
