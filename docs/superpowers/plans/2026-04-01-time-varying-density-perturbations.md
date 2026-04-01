# Time-Varying Density Perturbations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Ornstein-Uhlenbeck Gauss-Markov process that produces time-varying density perturbations during each simulation run, stacked multiplicatively on the existing static density bias.

**Architecture:** New `DensityPerturbationConfig` with Off/Low/Medium/High/Custom presets parsed from `[monte_carlo.density_perturbation]` TOML section. A pure `step_density_perturbation()` function computes exact OU transitions. The perturbation value lives on `RunState` (cloned per sim) and is updated each integration tick in `run_single()`. The existing `density_bias` and new `density_perturbation` are applied together at every density lookup site.

**Tech Stack:** Rust (rand, rand_distr for Normal distribution), TOML config, PyO3 bindings

**Spec:** `docs/superpowers/specs/2026-04-01-time-varying-density-perturbations-design.md`

---

### Task 1: Add `DensityPerturbationConfig` struct and presets

**Files:**
- Modify: `src/rust/src/data/dispersions.rs` (after `WindDispersionConfig` at line ~315)

- [ ] **Step 1: Write unit tests for the config struct and presets**

Add these tests inside the existing `#[cfg(test)] mod tests` block at the bottom of `dispersions.rs`:

```rust
#[test]
fn test_density_perturbation_config_off() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Off);
    assert_eq!(cfg.sigma, 0.0);
}

#[test]
fn test_density_perturbation_config_low() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Low);
    assert_eq!(cfg.tau, 120.0);
    assert_eq!(cfg.sigma, 0.05);
}

#[test]
fn test_density_perturbation_config_medium() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Medium);
    assert_eq!(cfg.tau, 60.0);
    assert_eq!(cfg.sigma, 0.10);
}

#[test]
fn test_density_perturbation_config_high() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::High);
    assert_eq!(cfg.tau, 30.0);
    assert_eq!(cfg.sigma, 0.20);
}

#[test]
fn test_density_perturbation_config_custom_defaults_to_medium() {
    let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Custom);
    assert_eq!(cfg.tau, 60.0);
    assert_eq!(cfg.sigma, 0.10);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test density_perturbation_config -- --nocapture`
Expected: compilation error -- `DensityPerturbationConfig` not defined.

- [ ] **Step 3: Implement the struct and presets**

Add after `WindDispersionConfig` (around line 315) in `dispersions.rs`:

```rust
/// Gauss-Markov (Ornstein-Uhlenbeck) density perturbation config.
/// Produces time-varying density multiplier that evolves during each run.
#[derive(Debug, Clone, Copy)]
pub struct DensityPerturbationConfig {
    pub tau: f64,   // correlation time (seconds)
    pub sigma: f64, // steady-state RMS amplitude (fractional)
}

impl DensityPerturbationConfig {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self { tau: 0.0, sigma: 0.0 },
            DispersionLevel::Low => Self { tau: 120.0, sigma: 0.05 },
            DispersionLevel::Medium => Self { tau: 60.0, sigma: 0.10 },
            DispersionLevel::High => Self { tau: 30.0, sigma: 0.20 },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }

    /// Returns true if the perturbation is effectively disabled.
    pub fn is_disabled(&self) -> bool {
        self.sigma <= 0.0 || self.tau <= 0.0
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test density_perturbation_config -- --nocapture`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat: add DensityPerturbationConfig struct with Off/Low/Medium/High presets"
```

---

### Task 2: Add `step_density_perturbation()` pure function

**Files:**
- Modify: `src/rust/src/data/dispersions.rs`

- [ ] **Step 1: Write unit tests for the step function**

Add to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_step_density_perturbation_disabled_sigma_zero() {
    assert_eq!(step_density_perturbation(0.5, 0.1, 60.0, 0.0, 1.0), 0.0);
}

#[test]
fn test_step_density_perturbation_disabled_tau_zero() {
    assert_eq!(step_density_perturbation(0.5, 0.1, 0.0, 0.10, 1.0), 0.0);
}

#[test]
fn test_step_density_perturbation_deterministic() {
    let a = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
    let b = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
    assert_eq!(a, b);
}

#[test]
fn test_step_density_perturbation_decay() {
    // With zero noise (normal_sample=0), the state should decay toward 0
    let x = step_density_perturbation(1.0, 0.1, 60.0, 0.10, 0.0);
    assert!(x < 1.0, "state should decay: got {}", x);
    assert!(x > 0.0, "state should remain positive with no noise: got {}", x);
}

#[test]
fn test_step_density_perturbation_statistical_properties() {
    // Run many steps from x=0 and check steady-state variance ~ sigma^2
    let tau = 60.0;
    let sigma = 0.10;
    let dt = 0.1;
    let n_steps = 100_000;

    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = 0.0;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let burn_in = 10_000; // let it reach steady state

    for i in 0..n_steps {
        let z = normal.sample(&mut rng);
        x = step_density_perturbation(x, dt, tau, sigma, z);
        if i >= burn_in {
            sum += x;
            sum_sq += x * x;
        }
    }

    let n = (n_steps - burn_in) as f64;
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;

    // Mean should be ~0
    assert!(mean.abs() < 0.01, "mean should be ~0, got {}", mean);
    // Variance should be ~sigma^2 = 0.01
    assert!(
        (variance - sigma * sigma).abs() < 0.002,
        "variance should be ~{}, got {}",
        sigma * sigma,
        variance
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test step_density_perturbation -- --nocapture`
Expected: compilation error -- `step_density_perturbation` not defined.

- [ ] **Step 3: Implement the step function**

Add after the `DensityPerturbationConfig` impl block in `dispersions.rs`:

```rust
/// Advance the Ornstein-Uhlenbeck density perturbation by one timestep.
///
/// Exact transition: x(t+dt) = x(t)*exp(-dt/tau) + sigma*sqrt(1 - exp(-2*dt/tau))*N(0,1)
///
/// Returns 0.0 when disabled (sigma <= 0 or tau <= 0).
pub fn step_density_perturbation(x: f64, dt: f64, tau: f64, sigma: f64, normal_sample: f64) -> f64 {
    if sigma <= 0.0 || tau <= 0.0 {
        return 0.0;
    }
    let decay = (-dt / tau).exp();
    x * decay + sigma * (1.0 - (-2.0 * dt / tau).exp()).sqrt() * normal_sample
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test step_density_perturbation -- --nocapture`
Expected: all 5 tests pass.

- [ ] **Step 5: Add proptest for finiteness**

Add to the test module (requires `proptest` dev-dependency, already in Cargo.toml):

```rust
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn step_always_finite(
            x in -10.0f64..10.0,
            dt in 0.001f64..10.0,
            tau in 0.01f64..1000.0,
            sigma in 0.0f64..1.0,
            z in -5.0f64..5.0,
        ) {
            let result = step_density_perturbation(x, dt, tau, sigma, z);
            prop_assert!(result.is_finite(), "got {}", result);
        }
    }
}
```

Note: if there is already a `mod proptests` block in the test module, add the new proptest inside that existing block instead of creating a second one.

- [ ] **Step 6: Run proptest**

Run: `cd src/rust && cargo test step_always_finite -- --nocapture`
Expected: PASS (proptest generates many random inputs, all finite).

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat: add step_density_perturbation() OU exact transition function with tests"
```

---

### Task 3: TOML parsing for `[monte_carlo.density_perturbation]`

**Files:**
- Modify: `src/rust/src/config.rs` (~line 802, `TomlMonteCarlo` struct)
- Modify: `src/rust/src/data/mod.rs` (~line 602, `build_dispersion_config` + `SimData`)
- Modify: `src/rust/src/data/dispersions.rs` (`DispersionConfig` struct ~line 319)

- [ ] **Step 1: Write config parsing integration test**

Add a new test in `src/rust/tests/config_loading.rs` (or inline in `config.rs` tests if that pattern is used):

Create a temporary TOML file in the test that includes:

```rust
#[test]
fn test_density_perturbation_toml_parsing() {
    let toml_str = r#"
        [monte_carlo]
        seed = 42
        [monte_carlo.density_perturbation]
        level = "high"
    "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.density_perturbation.is_some());
    let dp = mc.density_perturbation.unwrap();
    assert_eq!(dp.level, "high");
}

#[test]
fn test_density_perturbation_toml_custom() {
    let toml_str = r#"
        [monte_carlo]
        seed = 42
        [monte_carlo.density_perturbation]
        level = "custom"
        tau = 45.0
        sigma = 0.15
    "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    let dp = mc.density_perturbation.unwrap();
    assert_eq!(dp.level, "custom");
    assert_eq!(*dp.custom.get("tau").unwrap(), 45.0);
    assert_eq!(*dp.custom.get("sigma").unwrap(), 0.15);
}

#[test]
fn test_density_perturbation_toml_absent() {
    let toml_str = r#"
        [monte_carlo]
        seed = 42
    "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.density_perturbation.is_none());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test density_perturbation_toml -- --nocapture`
Expected: compilation error -- `density_perturbation` field not on `TomlMonteCarlo`.

- [ ] **Step 3: Add `density_perturbation` field to `TomlMonteCarlo`**

In `src/rust/src/config.rs`, add to the `TomlMonteCarlo` struct (after the `wind` field at ~line 813):

```rust
pub density_perturbation: Option<TomlMcDomain>,
```

This reuses the existing `TomlMcDomain` pattern (level string + flatten custom HashMap), same as atmosphere/aerodynamics/etc.

- [ ] **Step 4: Run TOML parsing tests to verify they pass**

Run: `cd src/rust && cargo test density_perturbation_toml -- --nocapture`
Expected: all 3 tests pass.

- [ ] **Step 5: Add `density_perturbation` field to `DispersionConfig`**

In `src/rust/src/data/dispersions.rs`, add to the `DispersionConfig` struct (~line 330, after `wind`):

```rust
pub density_perturbation: Option<DensityPerturbationConfig>,
```

- [ ] **Step 6: Wire up parsing in `build_dispersion_config()`**

In `src/rust/src/data/mod.rs`, add parsing logic in `build_dispersion_config()` (before the final `Ok(DispersionConfig { ... })` at ~line 814):

```rust
let density_perturbation = mc.density_perturbation.as_ref().and_then(|d| {
    let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
    if level == DispersionLevel::Off {
        return None;
    }
    let mut cfg = DensityPerturbationConfig::from_level(level);
    if level == DispersionLevel::Custom {
        if let Some(&v) = d.custom.get("tau") {
            cfg.tau = v;
        }
        if let Some(&v) = d.custom.get("sigma") {
            cfg.sigma = v;
        }
    }
    Some(cfg)
});
```

And add `density_perturbation,` to the `DispersionConfig { ... }` struct literal at ~line 814.

- [ ] **Step 7: Store the config on `SimData`**

In `src/rust/src/data/mod.rs`, add a field to the `SimData` struct (~line 182, after `sim_phase`):

```rust
/// Gauss-Markov density perturbation config (None = disabled)
pub density_perturbation: Option<dispersions::DensityPerturbationConfig>,
```

In the `SimData::from_toml()` method, after the `dispersion_config` is built (~line 607), extract and store it:

```rust
let density_perturbation = dispersion_config
    .as_ref()
    .and_then(|dc| dc.density_perturbation);
```

And add `density_perturbation,` to the `SimData { ... }` struct literal.

- [ ] **Step 8: Run full test suite to verify nothing broke**

Run: `cd src/rust && cargo test`
Expected: all tests pass. Any test that constructs `DispersionConfig` or `SimData` directly (in test helpers) will need the new field added with `density_perturbation: None`.

- [ ] **Step 9: Fix any test helpers that construct `DispersionConfig` or `SimData` directly**

Search for all direct `DispersionConfig {` and `SimData {` constructions in test code and add `density_perturbation: None` to each.

- [ ] **Step 10: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/dispersions.rs src/rust/src/data/mod.rs src/rust/tests/
git commit -m "feat: parse [monte_carlo.density_perturbation] TOML section with level presets"
```

---

### Task 4: Add `density_perturbation` to `RunState` and step it in the simulation loop

**Files:**
- Modify: `src/rust/src/simulation/init.rs` (`RunState` struct)
- Modify: `src/rust/src/simulation/runner.rs` (`run_single()` function)

- [ ] **Step 1: Add `density_perturbation` field to `RunState`**

In `src/rust/src/simulation/init.rs`, add to the `RunState` struct (after `density_bias` at ~line 21):

```rust
pub density_perturbation: f64, // time-varying GM perturbation (fractional), updated each tick
```

In `init_run_from_draw()`, initialize it to 0.0:

```rust
density_perturbation: 0.0,
```

- [ ] **Step 2: Clone `run_state` in `run_single()` and set up GM RNG**

In `src/rust/src/simulation/runner.rs`, at the top of `run_single()` (after line ~413 `let planet = &config.planet;`), add:

```rust
// Clone run_state so we can mutate density_perturbation each tick
let mut run_state = run_state.clone();

// Gauss-Markov density perturbation RNG (deterministic per sim)
let gm_config = data.density_perturbation;
let mut gm_rng = {
    use rand::SeedableRng;
    // Offset by 0xDE45 to avoid correlation with EKF RNG (which uses sim_idx * 10_000)
    rand::rngs::StdRng::seed_from_u64(config.random_seed as u64 + sim_idx as u64 * 10_000 + 0xDE45)
};
let gm_normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
```

- [ ] **Step 3: Update `run_single()` signature handling**

Since `run_state` is now cloned at the top of `run_single()`, every reference to `run_state` in the function body already works (it was `run_state.density_bias` etc., now referencing the local mutable clone). The function parameter stays as `run_state: &init::RunState` -- the clone happens inside.

- [ ] **Step 4: Step the GM process each integration tick**

In the main simulation loop, right after the `sequencer.update()` call at line ~511 and before the navigation block, add:

```rust
// Step Gauss-Markov density perturbation
if let Some(ref gm) = gm_config {
    if !gm.is_disabled() {
        use rand_distr::Distribution;
        let z: f64 = gm_normal.sample(&mut gm_rng);
        run_state.density_perturbation =
            crate::data::dispersions::step_density_perturbation(
                run_state.density_perturbation, dt, gm.tau, gm.sigma, z,
            );
    }
}
```

- [ ] **Step 5: Fix all `run_state` references from `&` to local clone**

Since `run_state` is now a local `let mut run_state = run_state.clone();`, all places that pass `run_state` as `&init::RunState` need to pass `&run_state` instead. The key call sites in `run_single()`:

- `estimator::navigate(... &nav_biases, nav_state, data, planet, run_state.density_bias, ...)` -- these pass individual fields, not the whole struct, so no change needed.
- `integrate_step(&mut sim, dt, planet, data, run_state)` -- change to `&run_state`
- `integrate_adaptive(&mut sim, dt, adaptive_config, planet, data, run_state)` -- change to `&run_state`
- `track_peak_values(&mut sim, altitude, sim_time, data, run_state)` -- change to `&run_state`
- `build_photo_values(... run_state, ...)` -- change to `&run_state`
- Any other place that passes `run_state` as a whole struct reference.

Search for all occurrences of `, run_state)` and `, run_state,` in `run_single()` and ensure they compile (the borrow checker will guide you).

- [ ] **Step 6: Run tests**

Run: `cd src/rust && cargo test`
Expected: all tests pass. The GM perturbation is 0.0 by default (no config), so behavior is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/simulation/init.rs src/rust/src/simulation/runner.rs
git commit -m "feat: step Gauss-Markov density perturbation each integration tick in run_single()"
```

---

### Task 5: Apply `density_perturbation` at all density lookup sites

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (lines ~582, ~654, ~979, ~1148, ~1241)
- Modify: `src/rust/src/gnc/navigation/estimator.rs` (lines ~119, ~401)
- Modify: `src/rust/src/physics/atmosphere.rs` (density function signature)

- [ ] **Step 1: Update `physics::atmosphere::density()` signature**

In `src/rust/src/physics/atmosphere.rs`, change the function at line ~7:

```rust
pub fn density(atm: &AtmosphereModel, altitude: f64, density_bias: f64, density_perturbation: f64) -> f64 {
    let rho = atm.density_at(altitude);
    rho * (1.0 + density_bias) * (1.0 + density_perturbation)
}
```

- [ ] **Step 2: Update inline density computations in `runner.rs`**

At each site that computes `data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias)`, add the GM factor. Change each to:

**Line ~582 (thermal computation):**
```rust
let rho_thermal =
    data.atmosphere.density_at(alt_for_thermal) * (1.0 + run_state.density_bias) * (1.0 + run_state.density_perturbation);
```

**Line ~654 (reference trajectory mode):**
```rust
let rho_truth = data.atmosphere.density_at(alt_truth) * (1.0 + run_state.density_bias) * (1.0 + run_state.density_perturbation);
```

**Line ~979 (photo value building -- `build_photo_values`):**
```rust
let rho_dispersed = rho_truth * (1.0 + run_state.density_bias) * (1.0 + run_state.density_perturbation);
```

**Line ~1148 (track_peak_values):**
```rust
let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias) * (1.0 + run_state.density_perturbation);
```

**Line ~1241 (compute_derivatives):**
```rust
let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias) * (1.0 + run_state.density_perturbation);
```

- [ ] **Step 3: Update navigation density in `estimator.rs`**

The navigation functions receive `run_density_bias: f64` as a parameter. Add `run_density_perturbation: f64` parameter to both functions.

**`navigate()` (at ~line 91):** Add parameter `run_density_perturbation: f64` after `run_density_bias: f64`.

At line ~119, change:
```rust
let rho_true = rho_true * (1.0 + run_density_bias) * (1.0 + run_density_perturbation);
```

**`navigate_ekf()` (at ~line 374):** Add parameter `run_density_perturbation: f64` after `run_density_bias: f64`.

At line ~401, change:
```rust
let rho_true = data.atmosphere.density_at(alt_true) * (1.0 + run_density_bias) * (1.0 + run_density_perturbation);
```

- [ ] **Step 4: Update call sites for navigation in `runner.rs`**

At line ~528, add `run_state.density_perturbation,` after `run_state.density_bias,`:
```rust
NavigationFilter::Bias(nav_state) => estimator::navigate(
    ...
    run_state.density_bias,
    run_state.density_perturbation,
    run_state.cx_bias,
    ...
),
```

At line ~559, add `run_state.density_perturbation,` after `run_state.density_bias,`:
```rust
} => estimator::navigate_ekf(
    ...
    run_state.density_bias,
    run_state.density_perturbation,
    run_state.cx_bias,
    ...
),
```

- [ ] **Step 5: Run tests**

Run: `cd src/rust && cargo test`
Expected: all tests pass. Any test that calls `navigate()` or `navigate_ekf()` directly needs the extra parameter (pass `0.0`).

- [ ] **Step 6: Fix any test call sites**

Search for `navigate(` and `navigate_ekf(` in test files and add the `0.0` density_perturbation parameter.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/physics/atmosphere.rs src/rust/src/simulation/runner.rs src/rust/src/gnc/navigation/estimator.rs
git commit -m "feat: apply density_perturbation multiplier at all density lookup sites"
```

---

### Task 6: Add trajectory output column for `density_perturbation`

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (photo array + trajectory mapping)
- Modify: `src/rust/src/lib.rs` (`RunOutput` trajectory type)
- Modify: `src/rust/aerocapture-py/src/results.rs` (column docs + array shape)

- [ ] **Step 1: Expand photo_lines array from 29 to 30 columns**

In `runner.rs`, change the `photo_lines` type at ~line 491:
```rust
let mut photo_lines: Vec<[f64; 30]> = Vec::new();
```

In `build_photo_values()`, add the density_perturbation value as the 30th element (index 29). Find the function's return array and append:
```rust
run_state.density_perturbation, // [29] density_perturbation (fractional)
```

Update the function return type from `[f64; 29]` to `[f64; 30]`.

- [ ] **Step 2: Expand trajectory from 16 to 17 columns in `RunOutput`**

In `src/rust/src/lib.rs`, change the trajectory type:
```rust
pub trajectory: Vec<[f64; 17]>,
```

Update the doc comment to include the new column:
```rust
/// Per-timestep state: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg,
/// heat_flux_kw_m2, time_s, energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg,
/// g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2, density_perturbation]
pub trajectory: Vec<[f64; 17]>,
```

- [ ] **Step 3: Update trajectory mapping in `runner.rs`**

In the trajectory mapping block (~line 240-279) where `photo_lines` are mapped to `RunOutput.trajectory`, add the 17th element:
```rust
p[29],       // [16] density_perturbation (fractional GM value)
```

- [ ] **Step 4: Update PyO3 results**

In `src/rust/aerocapture-py/src/results.rs`, update the trajectory docstring from `(N, 16)` to `(N, 17)` and add `density_perturbation` to the column list.

Search for `16` in the results.rs file and update any hardcoded trajectory width references to `17`.

- [ ] **Step 5: Run full test suite**

Run: `cd src/rust && cargo test`
Expected: all tests pass. Fix any array size mismatches the compiler catches.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/lib.rs src/rust/aerocapture-py/src/results.rs
git commit -m "feat: add density_perturbation as trajectory column 17"
```

---

### Task 7: Backward compatibility integration test

**Files:**
- Modify: `src/rust/tests/e2e.rs` (or create a new test)

- [ ] **Step 1: Write backward compatibility test**

Add an E2E test that runs a simulation WITHOUT the `[monte_carlo.density_perturbation]` section and verifies:
1. The simulation completes successfully
2. The `density_perturbation` trajectory column (index 16) is all zeros

```rust
#[test]
fn test_density_perturbation_absent_means_zero() {
    // Use an existing test config that has no density_perturbation section
    let (config, data) = load_test_config("configs/test/test_ref_orig.toml");
    let results = runner::run_for_api(&config, &data, false).unwrap();
    assert!(!results.is_empty());
    let result = &results[0];
    // If trajectory is populated, all GM values should be 0.0
    for row in &result.trajectory {
        assert_eq!(row[16], 0.0, "density_perturbation should be 0.0 when not configured");
    }
}
```

- [ ] **Step 2: Write enabled test**

Add a test that runs with GM enabled and verifies non-zero perturbation values:

```rust
#[test]
fn test_density_perturbation_enabled_produces_nonzero() {
    // Load a base config and add density_perturbation section via override
    let (mut config, mut data) = load_test_config("configs/test/test_ref_orig.toml");
    data.density_perturbation = Some(dispersions::DensityPerturbationConfig {
        tau: 60.0,
        sigma: 0.10,
    });
    config.n_sims = 1;
    let results = runner::run_for_api(&config, &data, true).unwrap();
    let result = &results[0];
    assert!(!result.trajectory.is_empty(), "trajectory should be populated");
    // At least some GM values should be non-zero
    let any_nonzero = result.trajectory.iter().any(|row| row[16] != 0.0);
    assert!(any_nonzero, "GM perturbation should produce non-zero values when enabled");
}
```

Note: Adapt the test setup to match the existing test infrastructure patterns in `e2e.rs`. The exact helper functions (`load_test_config` or similar) may differ -- use whatever pattern the existing E2E tests use to load configs and run simulations.

- [ ] **Step 3: Run tests**

Run: `cd src/rust && cargo test density_perturbation_absent density_perturbation_enabled -- --nocapture`
Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/tests/
git commit -m "test: add E2E tests for density perturbation backward compat and enabled behavior"
```

---

### Task 8: Update PyO3 bindings and Python tests

**Files:**
- Modify: `src/rust/aerocapture-py/src/results.rs` (if not already done in Task 6)
- Modify: Python test files that check trajectory column count

- [ ] **Step 1: Rebuild PyO3 bindings**

Run from repo root:
```bash
cd src/rust/aerocapture-py && maturin develop --release
```

- [ ] **Step 2: Update Python tests for trajectory column count**

Search for any Python test that asserts trajectory shape `(N, 16)` and update to `(N, 17)`. Likely in `tests/test_pyo3.py` or similar.

Also search for any Python code that references trajectory column indices or the trajectory column list in `charts.py` or `report.py`. The new column 16 (`density_perturbation`) should be documented but no chart changes are needed.

- [ ] **Step 3: Run Python tests**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/aerocapture-py/ tests/
git commit -m "feat: update PyO3 bindings for 17-column trajectory with density_perturbation"
```

---

### Task 9: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update relevant sections**

Update the following sections in `CLAUDE.md`:

1. **Architecture > Rust Simulator > `data/dispersions.rs`**: Add mention of `DensityPerturbationConfig` and `step_density_perturbation()`.

2. **Architecture > `simulation/init.rs`**: Note that `RunState` now includes `density_perturbation` (time-varying, updated each tick).

3. **Input Configuration**: Add documentation for the `[monte_carlo.density_perturbation]` section with level presets and custom tau/sigma.

4. **PyO3 Bindings**: Update trajectory column list from 16 to 17, adding `density_perturbation` at index 16.

5. **Conventions > Testing**: Update test count if significantly changed.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for time-varying density perturbation feature"
```

---

### Task 10: Run full CI checks

**Files:** None (verification only)

- [ ] **Step 1: Run Rust checks**

```bash
./check_all.sh
```

Expected: fmt, clippy, test, and release build all pass.

- [ ] **Step 2: Run Python checks**

```bash
./lint_code.sh
uv run pytest tests/ -v
```

Expected: ruff, mypy, and all pytest tests pass.

- [ ] **Step 3: Fix any issues found**

Address any clippy warnings, formatting issues, or test failures.

- [ ] **Step 4: Commit fixes if any**

```bash
git add -A
git commit -m "fix: address CI check findings for density perturbation feature"
```

---

### Task 11: Smart commit with IMPROVEMENTS.md cleanup

**Files:**
- Modify: `IMPROVEMENTS.md`

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill, taking the whole git branch into account. In addition to the normal smart-commit duties (CLAUDE.md, README.md sync), also update `IMPROVEMENTS.md`:

- Mark section 1.1 (Time-varying density perturbations) as completed or move it to a "Done" section
- Mark section 2.1 (Heat rate and heat load as guidance constraints) as done -- the thermal limiter was merged in PR #22
- Mark section 4.2 (Exit phase guidance) as done -- merged in PR #22
- Review other items against recent work on this branch and update status if any others have been addressed
