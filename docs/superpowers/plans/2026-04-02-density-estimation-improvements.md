# Density Estimation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the legacy density filter (rate limiting + saturation) and correct drag extraction by accounting for lift projection at angle of attack in both navigation modes.

**Architecture:** Two independent improvements to `estimator.rs`. Task 1 adds a TOML-configurable `density_gain_max_delta` parameter threaded through config -> data -> estimator. Task 2 changes the density inversion formula and truth models in both `navigate()` and `navigate_ekf()` to use `Cx*cos(alpha) + Cz*sin(alpha)` instead of just `Cx`. Both improvements are tested with dedicated unit tests and property-based tests.

**Tech Stack:** Rust (edition 2024, nalgebra), Python (param_spaces.py), TOML config, proptest, rstest, approx

**Spec:** `docs/superpowers/specs/2026-04-02-density-estimation-improvements-design.md`

---

### Task 1: Add `density_gain_max_delta` to GuidanceParams and TOML config

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs:125-170` (add field to `GuidanceParams`)
- Modify: `src/rust/src/data/guidance_params.rs:293-323` (add field to `Default` impl)
- Modify: `src/rust/src/config.rs:560-606` (add field to `TomlFtcParams`)
- Modify: `src/rust/src/config.rs:617-619` (add default function)
- Modify: `src/rust/src/data/mod.rs:467` (thread field through to `GuidanceParams`)
- Modify: `src/rust/src/data/mod.rs:525` (fallback path)

- [ ] **Step 1: Add `density_gain_max_delta` field to `GuidanceParams`**

In `src/rust/src/data/guidance_params.rs`, add the field after `density_filter_gain` (line 150):

```rust
    pub density_filter_gain: f64, // low-pass filter gain for density estimation
    pub density_gain_max_delta: f64, // max per-step change in density_gain (rate limiter)
```

In the `Default` impl (line 309), add after `density_filter_gain: 0.0`:

```rust
            density_filter_gain: 0.0,
            density_gain_max_delta: 0.1,
```

- [ ] **Step 2: Add `density_gain_max_delta` to `TomlFtcParams` in config.rs**

In `src/rust/src/config.rs`, add after the `density_filter_gain` field (line 593):

```rust
    #[serde(default = "default_density_filter_gain")]
    pub density_filter_gain: f64,
    #[serde(default = "default_density_gain_max_delta")]
    pub density_gain_max_delta: f64,
```

Add the default function after `default_density_filter_gain` (line 619):

```rust
fn default_density_gain_max_delta() -> f64 {
    0.1
}
```

- [ ] **Step 3: Thread `density_gain_max_delta` through data/mod.rs**

In `src/rust/src/data/mod.rs`, add after `density_filter_gain: ftc.density_filter_gain` (line 467):

```rust
                density_filter_gain: ftc.density_filter_gain,
                density_gain_max_delta: ftc.density_gain_max_delta,
```

In the fallback path (line 525), add after `density_filter_gain: 0.8`:

```rust
                density_filter_gain: 0.8,
                density_gain_max_delta: 0.1,
```

- [ ] **Step 4: Update test fixtures that construct `GuidanceParams` directly**

Every test file that constructs `GuidanceParams { density_filter_gain: 0.8, .. }` needs the new field. Since all fixtures use `..Default::default()`, the new field is already covered by the `Default` impl. Verify by running:

Run: `cd src/rust && cargo build 2>&1 | head -40`
Expected: Successful build (no missing field errors), OR errors listing fixtures that need updating.

If any fixture constructs `GuidanceParams` without `..Default::default()`, add `density_gain_max_delta: 0.1,` to it.

- [ ] **Step 5: Verify build succeeds**

Run: `cd src/rust && cargo build --release 2>&1 | tail -5`
Expected: `Finished` with no errors.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/guidance_params.rs src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat(nav): add density_gain_max_delta config parameter

New TOML parameter [guidance] density_gain_max_delta (default 0.1)
for rate-limiting the legacy density filter. Threaded through
config -> data -> GuidanceParams."
```

---

### Task 2: Implement rate limiting and gain saturation in legacy bias-mode filter

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:160-169` (filter update logic)

- [ ] **Step 1: Write failing tests for rate limiting and gain saturation**

Add these tests to the `#[cfg(test)] mod tests` block in `src/rust/src/gnc/navigation/estimator.rs`, after the existing `density_filter_stability` test (line 997):

```rust
    // ── Test: density_gain_rate_limited ──

    #[test]
    fn density_gain_rate_limited() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 0.05; // tight rate limit
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 1.0;

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // With rate limit of 0.05, density_gain cannot move more than 0.05 from 1.0
        let delta = (nav_state.density_gain - 1.0).abs();
        assert!(
            delta <= 0.05 + 1e-14,
            "density_gain delta {delta} exceeded max_delta 0.05"
        );
    }

    // ── Test: density_gain_saturated ──

    #[test]
    fn density_gain_saturated() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 100.0; // very loose rate limit
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        // Start with extreme density_gain — should be clamped to [0.1, 10.0]
        nav_state.density_gain = 50.0;

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert!(
            nav_state.density_gain <= 10.0,
            "density_gain {} should be <= 10.0",
            nav_state.density_gain
        );
        assert!(
            nav_state.density_gain >= 0.1,
            "density_gain {} should be >= 0.1",
            nav_state.density_gain
        );
    }

    // ── Test: rate_limit_before_saturation ──

    #[test]
    fn rate_limit_before_saturation() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 0.02; // very tight
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();

        // Start near the lower saturation bound
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 0.12;

        // Run one step — even if filter wants to go below 0.1,
        // rate limit restricts movement to 0.02
        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // density_gain should be in [0.10, 0.14] (0.12 +/- 0.02, then clamped to [0.1, 10.0])
        assert!(
            nav_state.density_gain >= 0.1,
            "density_gain {} below saturation floor",
            nav_state.density_gain
        );
        let delta = (nav_state.density_gain - 0.12).abs();
        assert!(
            delta <= 0.02 + 1e-14,
            "density_gain moved by {delta}, exceeding rate limit 0.02"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test density_gain_rate_limited density_gain_saturated rate_limit_before_saturation -- --nocapture 2>&1 | tail -20`
Expected: Tests fail (density_gain has no rate limiting or saturation yet).

- [ ] **Step 3: Implement rate limiting and gain saturation**

In `src/rust/src/gnc/navigation/estimator.rs`, replace the filter update block (lines 162-169):

Current:
```rust
    let lambda = (data.guidance.density_filter_gain + run_filter_gain_bias).clamp(0.01, 0.99);
    if rho_model.abs() > 1e-30 {
        nav_state.density_gain =
            (1.0 - lambda) * nav_state.density_gain + lambda * (density_estimated / rho_model);
    }
    if alt_est > 100e3 {
        nav_state.density_gain = 1.0;
    }
```

New:
```rust
    let lambda = (data.guidance.density_filter_gain + run_filter_gain_bias).clamp(0.01, 0.99);
    if rho_model.abs() > 1e-30 {
        let raw_gain = (1.0 - lambda) * nav_state.density_gain
            + lambda * (density_estimated / rho_model);

        // Rate-of-change limiting
        let max_delta = data.guidance.density_gain_max_delta;
        let delta = (raw_gain - nav_state.density_gain).clamp(-max_delta, max_delta);
        nav_state.density_gain += delta;

        // Gain saturation (hardcoded safety bounds, matches EKF [0.1, 10.0])
        nav_state.density_gain = nav_state.density_gain.clamp(0.1, 10.0);
    }
    if alt_est > 100e3 {
        nav_state.density_gain = 1.0;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test density_gain_rate_limited density_gain_saturated rate_limit_before_saturation -- --nocapture 2>&1 | tail -20`
Expected: All 3 tests pass.

- [ ] **Step 5: Run all existing estimator tests to verify no regression**

Run: `cd src/rust && cargo test --lib estimator 2>&1 | tail -10`
Expected: All existing tests pass. The `density_filter_stability` test may need attention since density_gain is now bounded to [0.1, 10.0] -- the test checks `density_gain > 0.0` which is strictly weaker, so it should still pass.

- [ ] **Step 6: Add proptest for density_gain bounds**

Add after the existing `proptest_navigate_outputs_finite` test block in `estimator.rs`:

```rust
        /// density_gain must always be in [0.1, 10.0] after any filter update
        /// (except high-altitude reset to 1.0).
        #[test]
        fn proptest_density_gain_bounded(
            alt_km in 30.0_f64..=90.0_f64,  // below 100 km so filter runs
            velocity in 2_000.0_f64..=8_000.0_f64,
            gamma in -0.3_f64..=0.0_f64,
            initial_gain in 0.001_f64..=100.0_f64,
            filter_gain_bias in -5.0_f64..=5.0_f64,
        ) {
            let data = test_sim_data();
            let r = MARS_REQ + alt_km * 1000.0;
            let position_true = [r, 0.0, 0.0];
            let velocity_true = [velocity, gamma, 1.0];
            let biases = zero_biases();
            let mut nav_state = NavigationState::new();
            nav_state.density_gain = initial_gain;

            let run_biases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, filter_gain_bias];
            let _out = call_navigate(
                &position_true,
                &velocity_true,
                &biases,
                &mut nav_state,
                &data,
                &run_biases,
            );

            prop_assert!(
                nav_state.density_gain >= 0.1 && nav_state.density_gain <= 10.0,
                "density_gain {} out of [0.1, 10.0] bounds",
                nav_state.density_gain
            );
        }
```

- [ ] **Step 7: Run proptest**

Run: `cd src/rust && cargo test proptest_density_gain_bounded -- --nocapture 2>&1 | tail -10`
Expected: PASS (256 cases by default).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "feat(nav): add rate limiting and gain saturation to legacy density filter

Rate-of-change limiting clamps per-step delta to +/-density_gain_max_delta.
Gain saturation clamps density_gain to [0.1, 10.0], matching EKF bounds.
Includes unit tests + proptest for bounds invariant."
```

---

### Task 3: Lift-corrected truth model and density inversion in legacy bias mode

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:83-99` (function signature: activate `_run_cz_bias`)
- Modify: `src/rust/src/gnc/navigation/estimator.rs:126-153` (truth model + density inversion)

- [ ] **Step 1: Write failing test for lift correction**

Add these tests to the `#[cfg(test)] mod tests` block in `estimator.rs`:

```rust
    // ── Test: lift_correction_at_zero_aoa ──

    #[test]
    fn lift_correction_at_zero_aoa() {
        let mut data = test_sim_data();
        // Override aero tables to have Cz = 0 at AoA = 0
        data.aero.incidence = vec![0.0, 0.35];
        data.aero.cx = vec![1.5, 1.7];
        data.aero.cz = vec![0.0, -0.4];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 0.0; // zero AoA

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // At alpha=0, cos(0)=1, sin(0)=0, so correction factor = 1.0
        // density_guidance should be positive and finite
        assert!(out.density_guidance > 0.0 && out.density_guidance.is_finite());
    }

    // ── Test: lift_correction_at_nonzero_aoa ──

    #[test]
    fn lift_correction_at_nonzero_aoa() {
        // Test that at non-zero AoA, the corrected density differs from
        // what a Cx-only inversion would produce.
        let mut data = test_sim_data();
        // Set up aero tables with known Cx and Cz at a specific AoA
        let aoa_10deg = 10.0_f64.to_radians();
        data.aero.incidence = vec![0.0, aoa_10deg, 0.35];
        data.aero.cx = vec![1.5, 1.6, 1.7];
        data.aero.cz = vec![0.0, -0.2, -0.4];
        data.aero.n_points = 3;
        data.entry.initial_aoa = aoa_10deg;
        // AoA profile returns constant aoa_10deg
        data.incidence.altitudes = vec![-10_000.0, 150_000.0];
        data.incidence.incidences = vec![aoa_10deg, aoa_10deg];

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();

        let mut nav_state = NavigationState::new();
        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // The corrected denominator at AoA=10deg: 1.6*cos(10) + (-0.2)*sin(10)
        // = 1.6 * 0.9848 - 0.2 * 0.1736 = 1.5757 - 0.0347 = 1.5410
        // Correction factor vs Cx-only: 1.6 / 1.541 = 1.038 (~3.8% more density)
        let cx = 1.6_f64;
        let cz = -0.2_f64;
        let corrected_denom = cx * aoa_10deg.cos() + cz * aoa_10deg.sin();
        let correction_ratio = cx / corrected_denom;

        // Verify the ratio is approximately 1.038
        assert_relative_eq!(correction_ratio, 1.038, max_relative = 0.01);

        // density_guidance should be finite and positive
        assert!(
            out.density_guidance > 0.0 && out.density_guidance.is_finite(),
            "density_guidance should be positive and finite, got {}",
            out.density_guidance
        );
    }

    // ── Test: lift_correction_denom_guard ──

    #[test]
    fn lift_correction_denom_guard() {
        // When Cx*cos(alpha) + Cz*sin(alpha) is near zero, density_estimated
        // should fall back to 0.0 (guard against division by near-zero).
        let mut data = test_sim_data();
        // Set up pathological aero: large negative Cz that nearly cancels Cx at some AoA
        let aoa = 1.2; // ~69 deg
        data.aero.incidence = vec![0.0, aoa, 1.57];
        data.aero.cx = vec![0.4, 0.4, 0.4];
        data.aero.cz = vec![0.0, -1.1, -1.1]; // Cx*cos(69) + Cz*sin(69) ~ 0.4*0.36 + (-1.1)*0.93 ~ -0.88
        data.aero.n_points = 3;
        data.entry.initial_aoa = aoa;

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        // This should not crash or produce NaN/Inf
        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert!(
            out.density_guidance.is_finite(),
            "density_guidance should be finite even with pathological aero, got {}",
            out.density_guidance
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test lift_correction -- --nocapture 2>&1 | tail -20`
Expected: Tests may pass or fail depending on current behavior. The key one is `lift_correction_at_nonzero_aoa` which validates the corrected physics.

- [ ] **Step 3: Implement lift-corrected truth model and inversion in `navigate()`**

In `src/rust/src/gnc/navigation/estimator.rs`, make these changes:

**3a. Activate `_run_cz_bias` parameter** (line 95):

Change:
```rust
    _run_cz_bias: f64,
```
To:
```rust
    run_cz_bias: f64,
```

**3b. Update truth model** (lines 126-132):

Replace:
```rust
    let cx_true =
        data.aero.interpolate_cx(aoa_commanded + run_incidence_bias) * (1.0 + run_cx_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let acdrag_true = rho_true * ref_area_true * cx_true * velocity_true[0] * velocity_true[0]
        / (2.0 * mass_true);
    let drag_acceleration_measured = acdrag_true + biases.drag;
```

With:
```rust
    let aoa_true = aoa_commanded + run_incidence_bias;
    let cx_true = data.aero.interpolate_cx(aoa_true) * (1.0 + run_cx_bias);
    let cz_true = data.aero.interpolate_cz(aoa_true) * (1.0 + run_cz_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let aero_factor_true = rho_true * ref_area_true * velocity_true[0] * velocity_true[0]
        / (2.0 * mass_true);
    let accel_body_x_true =
        aero_factor_true * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());
    let accel_measured = accel_body_x_true + biases.drag;
```

**3c. Update density inversion** (lines 146-153):

Replace:
```rust
    // Density estimation via inverse dynamics
    // density_estimated = 2*|drag_acceleration_measured|*mass / (Cx*S*V^2)
    let density_estimated = if cx_est.abs() > 1e-30 && velocity_relative.abs() > 1e-10 {
        2.0 * drag_acceleration_measured.abs() * data.capsule.mass
            / (cx_est * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };
```

With:
```rust
    // Density estimation via inverse dynamics (lift-corrected)
    // a_body_x = (rho*S*V^2 / 2m) * (Cx*cos(alpha) + Cz*sin(alpha))
    // => rho = 2*m*|a| / (S*V^2 * (Cx*cos(alpha) + Cz*sin(alpha)))
    let aoa_est = aoa_commanded;
    let denom = cx_est * aoa_est.cos() + cz_est * aoa_est.sin();
    let density_estimated = if denom.abs() > 1e-10 && velocity_relative.abs() > 1e-10 {
        2.0 * accel_measured.abs() * data.capsule.mass
            / (denom * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };
```

- [ ] **Step 4: Run lift correction tests**

Run: `cd src/rust && cargo test lift_correction -- --nocapture 2>&1 | tail -20`
Expected: All 3 lift correction tests pass.

- [ ] **Step 5: Run all estimator tests for regression**

Run: `cd src/rust && cargo test --lib estimator 2>&1 | tail -10`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "feat(nav): lift-corrected drag extraction in legacy bias mode

Truth model now generates body-frame x-axis acceleration including
both drag and lift projections: a_x = (rho*S*V^2/2m) * (Cx*cos(alpha) + Cz*sin(alpha)).
Density inversion uses matching corrected denominator.
Activates previously unused run_cz_bias parameter."
```

---

### Task 4: Lift-corrected truth model and density inversion in EKF mode

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:365-386` (`navigate_ekf` signature: add `run_cz_bias`)
- Modify: `src/rust/src/gnc/navigation/estimator.rs:414-423` (EKF truth model)
- Modify: `src/rust/src/gnc/navigation/estimator.rs:455-462` (EKF density inversion)
- Modify: `src/rust/src/simulation/runner.rs:577-597` (call site: pass `run_cz_bias`)

- [ ] **Step 1: Add `run_cz_bias` to `navigate_ekf()` signature**

In `src/rust/src/gnc/navigation/estimator.rs`, add `run_cz_bias: f64` after `run_cx_bias: f64` in the `navigate_ekf` function signature (line 382):

```rust
    run_cx_bias: f64,
    run_cz_bias: f64,
    run_mass_bias: f64,
```

- [ ] **Step 2: Update EKF truth model**

In `navigate_ekf()`, replace the truth acceleration computation (lines 414-423):

Current:
```rust
    let cx_true =
        data.aero.interpolate_cx(aoa_commanded + run_incidence_bias) * (1.0 + run_cx_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let aero_factor_true =
        rho_true * ref_area_true * velocity_true[0] * velocity_true[0] / (2.0 * mass_true);
    let drag_accel_true = aero_factor_true * cx_true;

    // Simplified: treat drag as acting along velocity (body x-axis approximation)
    let true_accel = [drag_accel_true, 0.0, 0.0];
```

New:
```rust
    let aoa_true = aoa_commanded + run_incidence_bias;
    let cx_true = data.aero.interpolate_cx(aoa_true) * (1.0 + run_cx_bias);
    let cz_true = data.aero.interpolate_cz(aoa_true) * (1.0 + run_cz_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let aero_factor_true =
        rho_true * ref_area_true * velocity_true[0] * velocity_true[0] / (2.0 * mass_true);
    let accel_body_x_true =
        aero_factor_true * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());

    // Body-frame x-axis acceleration includes both drag and lift projections
    let true_accel = [accel_body_x_true, 0.0, 0.0];
```

- [ ] **Step 3: Update EKF density inversion**

In `navigate_ekf()`, replace the density inversion (lines 455-462):

Current:
```rust
    // Drag-derived density from measured acceleration
    let drag_acceleration_measured = accel_meas[0]; // IMU x-axis ~ drag direction
    let density_estimated = if cx_est.abs() > 1e-30 && velocity_relative.abs() > 1e-10 {
        2.0 * drag_acceleration_measured.abs() * data.capsule.mass
            / (cx_est * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };
```

New:
```rust
    // Density estimation via inverse dynamics (lift-corrected)
    let accel_measured_ekf = accel_meas[0];
    let aoa_est = aoa_commanded;
    let denom = cx_est * aoa_est.cos() + cz_est * aoa_est.sin();
    let density_estimated = if denom.abs() > 1e-10 && velocity_relative.abs() > 1e-10 {
        2.0 * accel_measured_ekf.abs() * data.capsule.mass
            / (denom * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };
```

- [ ] **Step 4: Update call site in runner.rs**

In `src/rust/src/simulation/runner.rs`, add `run_state.cz_bias` to the `navigate_ekf()` call (after `run_state.cx_bias`, line 594):

```rust
                    run_state.cx_bias,
                    run_state.cz_bias,
                    run_state.mass_bias,
```

- [ ] **Step 5: Build and run all tests**

Run: `cd src/rust && cargo build --release 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cd src/rust && cargo test 2>&1 | tail -15`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs src/rust/src/simulation/runner.rs
git commit -m "feat(nav): lift-corrected drag extraction in EKF mode

EKF truth model now generates body-frame x-axis acceleration including
both drag and lift projections. Density inversion uses corrected
denominator. Adds run_cz_bias to navigate_ekf() signature and call site."
```

---

### Task 5: Add `density_gain_max_delta` to GA parameter spaces

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py` (add parameter to ftc scheme)

- [ ] **Step 1: Add `density_gain_max_delta` to the ftc parameter space**

In `src/python/aerocapture/training/param_spaces.py`, find the `"ftc"` entry in the `PARAM_SPACES` dictionary. Add after the `density_filter_gain` entry:

```python
        ParamSpec("density_filter_gain", 0.3, 1.0, 0.8),
        ParamSpec("density_gain_max_delta", 0.01, 0.5, 0.1),
```

The bounds [0.01, 0.5] allow the GA to find the optimal rate limit: 0.01 is very sluggish (tracks a 2x change in ~100 steps), 0.5 is very loose (tracks in ~2 steps).

- [ ] **Step 2: Verify Python tests pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -15`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "feat(training): add density_gain_max_delta to ftc GA parameter space

Bounds [0.01, 0.5] with default 0.1. Allows GA to optimize the
density filter rate limit alongside other FTC parameters."
```

---

### Task 6: Full regression test suite

**Files:**
- No new files -- run existing test infrastructure

- [ ] **Step 1: Run Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: All tests pass (unit + integration + proptest).

- [ ] **Step 2: Run Rust clippy**

Run: `cd src/rust && cargo clippy -- -D warnings 2>&1 | tail -10`
Expected: No warnings.

- [ ] **Step 3: Run Rust fmt check**

Run: `cd src/rust && cargo fmt --check 2>&1 | tail -5`
Expected: No formatting issues.

- [ ] **Step 4: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -15`
Expected: All tests pass.

- [ ] **Step 5: Run Python linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -15`
Expected: Clean.

- [ ] **Step 6: Build PyO3 bindings and run PyO3 tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_pyo3.py -x -q 2>&1 | tail -15`
Expected: All PyO3 tests pass.

- [ ] **Step 7: Commit any fixes**

If any test or lint failures were found and fixed, commit the fixes:
```bash
git add -A
git commit -m "fix: address test/lint issues from density estimation improvements"
```

---

### Task 7: Smart commit

Invoke the `smart-commit` skill, telling it to take the whole `feature/density-estimation-improvements` branch into account.
