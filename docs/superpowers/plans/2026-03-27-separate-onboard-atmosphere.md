# Separate Truth vs Onboard Atmosphere Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a piecewise exponential onboard atmosphere model that navigation and guidance query instead of the truth table, so the density filter actually has to correct meaningful structural error.

**Architecture:** New `OnboardAtmosphereModel` enum in `data/atmosphere.rs` with two variants: `Identical` (delegates to truth table — backward compatible) and `PiecewiseExponential` (auto-fitted segments). `SimData` gets a new `atmosphere_onboard` field. Navigation (`estimator.rs`) and guidance (`fnpag.rs`, `equilibrium_glide.rs`) switch their `data.atmosphere.density_at()` calls to `data.atmosphere_onboard.density_at()`. TOML config gets an optional `[atmosphere.onboard]` section parsed in `config.rs`.

**Tech Stack:** Rust (nalgebra not needed for this — pure arithmetic), TOML/serde for config

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/rust/src/data/atmosphere.rs` | Modify | Add `ExponentialSegment`, `OnboardAtmosphereModel` enum, auto-fit logic, `density_at()` |
| `src/rust/src/config.rs` | Modify | Add `TomlOnboardAtmosphere` struct, parse `[atmosphere.onboard]` |
| `src/rust/src/data/mod.rs` | Modify | Add `atmosphere_onboard` to `SimData`, wire TOML → model construction |
| `src/rust/src/gnc/navigation/estimator.rs` | Modify | Replace `data.atmosphere.density_at()` with `data.atmosphere_onboard.density_at()` for onboard calls |
| `src/rust/src/gnc/guidance/fnpag.rs` | Modify | Replace `data.atmosphere.density_at()` with `data.atmosphere_onboard.density_at()` |
| `src/rust/src/gnc/guidance/equilibrium_glide.rs` | Modify | Replace `data.atmosphere.density_at()` with `data.atmosphere_onboard.density_at()` |
| `configs/test/*.toml` | Modify | Add `[atmosphere.onboard] mode = "identical"` for regression |
| 9× `test_sim_data()` helpers | Modify | Add `atmosphere_onboard` field |

---

### Task 1: Add `OnboardAtmosphereModel` data structures and `density_at()`

**Files:**
- Modify: `src/rust/src/data/atmosphere.rs`

- [ ] **Step 1: Write failing test — `PiecewiseExponential` density query**

Add to the existing `#[cfg(test)] mod tests` in `atmosphere.rs`:

```rust
#[test]
fn piecewise_exponential_single_segment() {
    let model = OnboardAtmosphereModel::PiecewiseExponential {
        segments: vec![ExponentialSegment {
            alt_low: 0.0,
            alt_high: 50_000.0,
            rho_ref: 0.02,
            scale_height: 10_000.0,
        }],
    };
    let truth = test_atm(); // existing helper
    // At alt_low the density should be rho_ref
    assert_abs_diff_eq!(model.density_at(0.0, &truth), 0.02, epsilon = 1e-10);
    // At one scale height above, density should be rho_ref * exp(-1)
    let expected = 0.02 * (-1.0_f64).exp();
    assert_abs_diff_eq!(model.density_at(10_000.0, &truth), expected, epsilon = 1e-10);
}

#[test]
fn piecewise_exponential_two_segments() {
    let model = OnboardAtmosphereModel::PiecewiseExponential {
        segments: vec![
            ExponentialSegment {
                alt_low: 0.0,
                alt_high: 20_000.0,
                rho_ref: 0.02,
                scale_height: 10_000.0,
            },
            ExponentialSegment {
                alt_low: 20_000.0,
                alt_high: 50_000.0,
                rho_ref: 0.002,
                scale_height: 8_000.0,
            },
        ],
    };
    let truth = test_atm();
    // In first segment
    let expected_low = 0.02 * (-15_000.0 / 10_000.0_f64).exp();
    assert_abs_diff_eq!(model.density_at(15_000.0, &truth), expected_low, epsilon = 1e-10);
    // In second segment
    let expected_high = 0.002 * (-5_000.0 / 8_000.0_f64).exp();
    assert_abs_diff_eq!(model.density_at(25_000.0, &truth), expected_high, epsilon = 1e-10);
}

#[test]
fn identical_mode_delegates_to_truth() {
    let truth = test_atm();
    let model = OnboardAtmosphereModel::Identical;
    assert_abs_diff_eq!(model.density_at(15_000.0, &truth), truth.density_at(15_000.0));
    assert_abs_diff_eq!(model.density_at(35_000.0, &truth), truth.density_at(35_000.0));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib data::atmosphere::tests -- --nocapture 2>&1 | tail -20`

Expected: compilation errors — `OnboardAtmosphereModel` and `ExponentialSegment` not defined.

- [ ] **Step 3: Implement `ExponentialSegment`, `OnboardAtmosphereModel` enum, and `density_at()`**

Add to `src/rust/src/data/atmosphere.rs`, after the `AtmosphereModel` impl block:

```rust
/// One altitude band of the onboard piecewise exponential model.
#[derive(Debug, Clone)]
pub struct ExponentialSegment {
    pub alt_low: f64,      // meters
    pub alt_high: f64,     // meters
    pub rho_ref: f64,      // kg/m^3 (density at alt_low)
    pub scale_height: f64, // meters
}

/// Onboard atmosphere model — degraded representation of truth.
#[derive(Debug, Clone)]
pub enum OnboardAtmosphereModel {
    /// Use the truth table directly (backward-compatible mode).
    Identical,
    /// Piecewise exponential segments auto-fitted or manually specified.
    PiecewiseExponential { segments: Vec<ExponentialSegment> },
}

impl OnboardAtmosphereModel {
    /// Query onboard density at a given altitude.
    ///
    /// For `Identical`, delegates to the truth table.
    /// For `PiecewiseExponential`, finds the containing segment and evaluates
    /// `rho_ref * exp(-(alt - alt_low) / H)`. Below the first segment uses
    /// the first segment's rho_ref. Above the last segment uses exponential
    /// extrapolation from the last segment.
    pub fn density_at(&self, altitude: f64, truth: &AtmosphereModel) -> f64 {
        match self {
            OnboardAtmosphereModel::Identical => truth.density_at(altitude),
            OnboardAtmosphereModel::PiecewiseExponential { segments } => {
                if segments.is_empty() {
                    return truth.density_at(altitude);
                }
                // Below first segment: clamp to first segment's rho_ref
                if altitude <= segments[0].alt_low {
                    return segments[0].rho_ref;
                }
                // Find containing segment
                for seg in segments {
                    if altitude <= seg.alt_high {
                        return seg.rho_ref
                            * (-(altitude - seg.alt_low) / seg.scale_height).exp();
                    }
                }
                // Above last segment: extrapolate from last segment
                let last = &segments[segments.len() - 1];
                last.rho_ref * (-(altitude - last.alt_low) / last.scale_height).exp()
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib data::atmosphere::tests -- --nocapture 2>&1 | tail -20`

Expected: all 3 new tests PASS, all existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/atmosphere.rs
git commit -m "feat(atmo): add OnboardAtmosphereModel with piecewise exponential density_at"
```

---

### Task 2: Add auto-fit from truth table

**Files:**
- Modify: `src/rust/src/data/atmosphere.rs`

- [ ] **Step 1: Write failing test — auto-fit produces segments that approximate truth**

Add to `atmosphere.rs` tests:

```rust
#[test]
fn auto_fit_produces_correct_segment_count() {
    let truth = AtmosphereModel {
        n_points: 5,
        altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
        densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
        ref_density: 5e-6,
        scale_factor: 1e-4,
        ref_altitude: 100_000.0,
        gas_constant: 1.3,
        density_profile: DensityProfile::default(),
    };
    let model = OnboardAtmosphereModel::fit_from_table(&truth, 3);
    match &model {
        OnboardAtmosphereModel::PiecewiseExponential { segments } => {
            assert_eq!(segments.len(), 3);
            // Segments should span the table range
            assert_abs_diff_eq!(segments[0].alt_low, 0.0);
            assert_abs_diff_eq!(segments[2].alt_high, 100_000.0);
            // Each segment should have positive scale height and density
            for seg in segments {
                assert!(seg.scale_height > 0.0, "scale_height must be positive");
                assert!(seg.rho_ref > 0.0, "rho_ref must be positive");
            }
        }
        _ => panic!("Expected PiecewiseExponential variant"),
    }
}

#[test]
fn auto_fit_approximates_truth_within_tolerance() {
    let truth = AtmosphereModel {
        n_points: 5,
        altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
        densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
        ref_density: 5e-6,
        scale_factor: 1e-4,
        ref_altitude: 100_000.0,
        gas_constant: 1.3,
        density_profile: DensityProfile::default(),
    };
    let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);
    // With 5 segments over 5 table points, fit should be reasonably close
    for &alt in &truth.altitudes {
        let rho_truth = truth.density_at(alt);
        let rho_onboard = model.density_at(alt, &truth);
        if rho_truth > 1e-10 {
            let rel_err = (rho_onboard - rho_truth).abs() / rho_truth;
            assert!(
                rel_err < 0.5,
                "relative error {:.2}% at alt={} m exceeds 50%",
                rel_err * 100.0,
                alt,
            );
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib data::atmosphere::tests::auto_fit -- --nocapture 2>&1 | tail -10`

Expected: compilation error — `fit_from_table` not defined.

- [ ] **Step 3: Implement `fit_from_table`**

Add this method to the `OnboardAtmosphereModel` impl block in `atmosphere.rs`:

```rust
impl OnboardAtmosphereModel {
    /// Auto-fit a piecewise exponential model from the truth atmosphere table.
    ///
    /// Divides the truth table altitude range into `n_segments` equal bands.
    /// For each band, samples the truth table at the band endpoints and any
    /// interior table points, then performs a linear regression on ln(rho) vs
    /// altitude to extract scale height H and reference density rho_ref.
    pub fn fit_from_table(truth: &AtmosphereModel, n_segments: usize) -> Self {
        if truth.n_points < 2 || n_segments == 0 {
            return OnboardAtmosphereModel::Identical;
        }

        let alt_min = truth.altitudes[0];
        let alt_max = truth.altitudes[truth.n_points - 1];
        let band_width = (alt_max - alt_min) / n_segments as f64;

        let mut segments = Vec::with_capacity(n_segments);
        for i in 0..n_segments {
            let alt_low = alt_min + i as f64 * band_width;
            let alt_high = alt_min + (i + 1) as f64 * band_width;

            // Sample truth densities within this band (at least 2 points: endpoints)
            let mut samples: Vec<(f64, f64)> = Vec::new();

            // Add band endpoints
            let rho_low = truth.density_at(alt_low);
            if rho_low > 0.0 {
                samples.push((alt_low, rho_low));
            }
            let rho_high = truth.density_at(alt_high);
            if rho_high > 0.0 {
                samples.push((alt_high, rho_high));
            }

            // Add interior table points
            for j in 0..truth.n_points {
                let alt_j = truth.altitudes[j];
                if alt_j > alt_low && alt_j < alt_high {
                    let rho_j = truth.densities[j];
                    if rho_j > 0.0 {
                        samples.push((alt_j, rho_j));
                    }
                }
            }

            // Linear regression on ln(rho) vs altitude: ln(rho) = a + b*alt
            // => rho_ref = exp(a + b*alt_low), scale_height = -1/b
            let (rho_ref, scale_height) = if samples.len() >= 2 {
                fit_exponential(&samples, alt_low)
            } else if let Some(&(_, rho)) = samples.first() {
                // Only one sample — use truth's exponential model scale factor
                (rho, 1.0 / truth.scale_factor)
            } else {
                // No valid samples — fallback
                (truth.ref_density, 1.0 / truth.scale_factor)
            };

            segments.push(ExponentialSegment {
                alt_low,
                alt_high,
                rho_ref,
                scale_height,
            });
        }

        OnboardAtmosphereModel::PiecewiseExponential { segments }
    }

    // ... existing density_at() stays here
}

/// Fit rho_ref and scale_height from samples using linear regression on ln(rho).
///
/// Model: ln(rho) = ln(rho_ref) - (alt - alt_low) / H
/// Which is: y = c + m*x where y=ln(rho), x=(alt-alt_low), c=ln(rho_ref), m=-1/H
fn fit_exponential(samples: &[(f64, f64)], alt_low: f64) -> (f64, f64) {
    let n = samples.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for &(alt, rho) in samples {
        let x = alt - alt_low;
        let y = rho.ln();
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        // Degenerate case — all samples at same altitude
        let rho_ref = (sum_y / n).exp();
        return (rho_ref, 10_000.0); // default scale height
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let rho_ref = intercept.exp();
    // slope = -1/H, so H = -1/slope
    let scale_height = if slope < -1e-15 {
        -1.0 / slope
    } else {
        // Slope is non-negative (density not decreasing) — use large scale height
        1e6
    };

    (rho_ref, scale_height)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib data::atmosphere::tests -- --nocapture 2>&1 | tail -20`

Expected: all atmosphere tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/atmosphere.rs
git commit -m "feat(atmo): add auto-fit piecewise exponential from truth table"
```

---

### Task 3: Add TOML config parsing for `[atmosphere.onboard]`

**Files:**
- Modify: `src/rust/src/config.rs`

- [ ] **Step 1: Write failing test — TOML with `[atmosphere.onboard]` section parses**

Add to the existing `#[cfg(test)] mod tests` in `config.rs` (or at the bottom of the file if no test module exists):

```rust
#[test]
fn parse_onboard_atmosphere_n_segments() {
    let toml_str = r#"
        [atmosphere.onboard]
        n_segments = 8
    "#;
    let parsed: TomlAtmosphereOnboard = toml::from_str(toml_str.trim_start_matches(
        |c: char| c.is_whitespace(),
    )).unwrap_or_else(|_| {
        // The section is nested — parse just the inner part
        let inner = r#"n_segments = 8"#;
        toml::from_str(inner).unwrap()
    });
    assert_eq!(parsed.n_segments, Some(8));
    assert_eq!(parsed.mode, None);
    assert!(parsed.segments.is_none());
}

#[test]
fn parse_onboard_atmosphere_identical_mode() {
    let parsed: TomlAtmosphereOnboard = toml::from_str(r#"mode = "identical""#).unwrap();
    assert_eq!(parsed.mode.as_deref(), Some("identical"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib config::tests::parse_onboard -- --nocapture 2>&1 | tail -10`

Expected: compilation error — `TomlAtmosphereOnboard` not defined.

- [ ] **Step 3: Implement TOML structs**

Add to `config.rs`, near the other TOML structs:

```rust
/// TOML config for explicit exponential segment override.
#[derive(Debug, Clone, Deserialize)]
pub struct TomlExponentialSegment {
    pub alt_low: f64,
    pub alt_high: f64,
    pub rho_ref: f64,
    pub scale_height: f64,
}

/// TOML config for the onboard atmosphere model.
///
/// Three modes:
/// - `mode = "identical"`: use truth table (backward compatible)
/// - `n_segments = N`: auto-fit N piecewise exponential segments from truth
/// - `segments = [...]`: explicit segment definitions
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TomlAtmosphereOnboard {
    /// Mode override: "identical" to use truth table directly
    pub mode: Option<String>,
    /// Number of segments for auto-fit (default: 5)
    pub n_segments: Option<usize>,
    /// Explicit segment definitions (overrides n_segments)
    pub segments: Option<Vec<TomlExponentialSegment>>,
}
```

Add the `onboard` field to `TomlData`:

```rust
#[derive(Debug, Deserialize)]
pub struct TomlData {
    // ... existing fields ...
    /// Onboard atmosphere model config (optional)
    #[serde(default)]
    pub onboard_atmosphere: Option<TomlAtmosphereOnboard>,
}
```

**Wait** — the spec says `[atmosphere.onboard]` in TOML, but TOML nesting under `[data]` would be `[data.onboard_atmosphere]`. The spec's `[atmosphere.onboard]` is a top-level section. Let's add it to `TomlConfig` instead, as a new optional top-level section called `atmosphere`:

Actually, looking at the TOML structure more carefully: the truth atmosphere path is in `[data] atmosphere = "..."`. We want `[atmosphere.onboard]` as a top-level section. But `atmosphere` under `[data]` is a string, not a table — so TOML would conflict. Let's use `[onboard_atmosphere]` as a top-level section to avoid ambiguity.

Add to `TomlConfig`:

```rust
pub struct TomlConfig {
    // ... existing fields ...
    /// Onboard atmosphere model config
    #[serde(default)]
    pub onboard_atmosphere: Option<TomlAtmosphereOnboard>,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib config::tests::parse_onboard -- --nocapture 2>&1 | tail -10`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat(config): add TomlAtmosphereOnboard for onboard atmosphere TOML config"
```

---

### Task 4: Wire `atmosphere_onboard` into `SimData`

**Files:**
- Modify: `src/rust/src/data/mod.rs`

- [ ] **Step 1: Add `atmosphere_onboard` field to `SimData`**

In `SimData` struct definition, add after the `atmosphere` field:

```rust
pub struct SimData {
    // ... existing fields ...
    pub atmosphere: atmosphere::AtmosphereModel,
    /// Onboard atmosphere model (degraded) for navigation and guidance
    pub atmosphere_onboard: atmosphere::OnboardAtmosphereModel,
    // ... rest unchanged ...
}
```

- [ ] **Step 2: Build onboard model in `SimData::from_toml()`**

In the `from_toml()` method, after the truth atmosphere is loaded (line ~501), add:

```rust
        // Atmosphere (always external)
        let atm_path = toml
            .data
            .atmosphere
            .as_ref()
            .ok_or_else(|| DataError("Missing data.atmosphere path".to_string()))?;
        let atm = atmosphere::AtmosphereModel::load(atm_path)?;

        // Onboard atmosphere model
        let atm_onboard = match &toml.onboard_atmosphere {
            Some(cfg) if cfg.mode.as_deref() == Some("identical") => {
                atmosphere::OnboardAtmosphereModel::Identical
            }
            Some(cfg) if cfg.segments.is_some() => {
                let segs = cfg.segments.as_ref().unwrap();
                atmosphere::OnboardAtmosphereModel::PiecewiseExponential {
                    segments: segs
                        .iter()
                        .map(|s| atmosphere::ExponentialSegment {
                            alt_low: s.alt_low,
                            alt_high: s.alt_high,
                            rho_ref: s.rho_ref,
                            scale_height: s.scale_height,
                        })
                        .collect(),
                }
            }
            Some(cfg) => {
                let n = cfg.n_segments.unwrap_or(5);
                atmosphere::OnboardAtmosphereModel::fit_from_table(&atm, n)
            }
            None => {
                // Default: auto-fit with 5 segments
                atmosphere::OnboardAtmosphereModel::fit_from_table(&atm, 5)
            }
        };
```

Update the `Ok(SimData { ... })` block to include:

```rust
        Ok(SimData {
            // ... existing fields ...
            atmosphere: atm,
            atmosphere_onboard: atm_onboard,
            // ... rest ...
        })
```

- [ ] **Step 3: Fix all `test_sim_data()` helpers across the codebase**

There are 9 `test_sim_data()` functions that construct `SimData` directly. Each needs the new field. Add to every `SimData { ... }` literal:

```rust
atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
```

Files to update (search for `fn test_sim_data`):
- `src/rust/src/simulation/init.rs`
- `src/rust/src/gnc/guidance/ftc.rs`
- `src/rust/src/gnc/navigation/estimator.rs`
- `src/rust/src/gnc/guidance/fnpag.rs`
- `src/rust/src/gnc/guidance/energy_controller.rs` (2 functions)
- `src/rust/src/gnc/guidance/predguid.rs` (2 functions)
- `src/rust/src/gnc/guidance/equilibrium_glide.rs`

- [ ] **Step 4: Build and verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build 2>&1 | tail -20`

Expected: successful compilation.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -30`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/mod.rs src/rust/src/data/atmosphere.rs \
        src/rust/src/simulation/init.rs src/rust/src/gnc/guidance/ftc.rs \
        src/rust/src/gnc/navigation/estimator.rs src/rust/src/gnc/guidance/fnpag.rs \
        src/rust/src/gnc/guidance/energy_controller.rs src/rust/src/gnc/guidance/predguid.rs \
        src/rust/src/gnc/guidance/equilibrium_glide.rs
git commit -m "feat(data): wire OnboardAtmosphereModel into SimData with TOML config"
```

---

### Task 5: Switch navigation to use onboard atmosphere

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs`

- [ ] **Step 1: Write failing test — navigation density_gain diverges from 1.0 with piecewise model**

Add to `estimator.rs` tests:

```rust
#[test]
fn density_gain_diverges_with_onboard_model() {
    use crate::data::atmosphere::{ExponentialSegment, OnboardAtmosphereModel};

    let mut data = test_sim_data();
    // Use a deliberately inaccurate onboard model
    data.atmosphere_onboard = OnboardAtmosphereModel::PiecewiseExponential {
        segments: vec![ExponentialSegment {
            alt_low: 0.0,
            alt_high: 150_000.0,
            rho_ref: 0.02, // different from truth table
            scale_height: 12_000.0,
        }],
    };

    let biases = NavigationBiases::default();
    let mut nav_state = NavigationState::new();
    let planet = Planet::Mars;
    let r = planet.equatorial_radius() + 50_000.0;

    let position = [r, 0.0, 0.0];
    let velocity = [5000.0, -0.15, 0.6];

    // Run several navigation steps
    for _ in 0..10 {
        navigate(
            &position, &velocity, -0.48, 10.0, &biases, &mut nav_state,
            &data, &planet, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        );
    }

    // Density gain should have moved away from 1.0 to compensate for model error
    assert!(
        (nav_state.density_gain - 1.0).abs() > 0.01,
        "density gain {} should diverge from 1.0 with inaccurate onboard model",
        nav_state.density_gain,
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::navigation::estimator::tests::density_gain_diverges -- --nocapture 2>&1 | tail -10`

Expected: FAIL — `density_gain` stays near 1.0 because navigation still queries truth table.

- [ ] **Step 3: Switch `rho_model` calls in `navigate()` and `navigate_ekf()` to use onboard model**

In `navigate()` (bias mode), change the `rho_model` computation (around line 145):

```rust
    // Model atmosphere density at estimated altitude — use ONBOARD model
    let rho_model = data.atmosphere_onboard.density_at(alt_est, &data.atmosphere);
```

Also change the exit density model call (around line 170):

```rust
    let rho_exit_model = data.atmosphere_onboard.density_at(alt_exit, &data.atmosphere);
```

In `navigate_ekf()`, make the same changes:

Line ~434 (rho_model for EKF):
```rust
    let rho_model = data.atmosphere_onboard.density_at(alt_est, &data.atmosphere);
```

Line ~489 (exit density for EKF):
```rust
    let rho_exit_model = data.atmosphere_onboard.density_at(alt_exit, &data.atmosphere);
```

**Important:** Do NOT change the `rho_true` lines (lines ~113-114, ~382). Those must stay on `data.atmosphere` — they model the true physical density that the IMU/drag acceleration actually measures.

- [ ] **Step 4: Run tests to verify the new test passes**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::navigation::estimator::tests -- --nocapture 2>&1 | tail -20`

Expected: `density_gain_diverges_with_onboard_model` PASSES, all existing tests still PASS (they use `Identical` mode).

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "feat(nav): switch density model queries to onboard atmosphere"
```

---

### Task 6: Switch guidance to use onboard atmosphere

**Files:**
- Modify: `src/rust/src/gnc/guidance/fnpag.rs`
- Modify: `src/rust/src/gnc/guidance/equilibrium_glide.rs`

- [ ] **Step 1: Switch FNPAG predictor density to onboard model**

In `fnpag.rs`, `predict_exit_energy()` function, change line ~93:

```rust
        // Atmospheric density (using onboard model for prediction)
        let rho = data.atmosphere_onboard.density_at(alt, &data.atmosphere);
```

And in the main `compute_bank_angle()` function (if there's a density call around line 167):

```rust
    let rho = data.atmosphere_onboard.density_at(altitude, &data.atmosphere);
```

- [ ] **Step 2: Switch equilibrium glide density to onboard model**

In `equilibrium_glide.rs`, change line 46:

```rust
    let rho = data.atmosphere_onboard.density_at(altitude, &data.atmosphere);
```

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -30`

Expected: all tests PASS (test helpers use `Identical` mode).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/fnpag.rs src/rust/src/gnc/guidance/equilibrium_glide.rs
git commit -m "feat(guidance): switch FNPAG and eq_glide density queries to onboard atmosphere"
```

---

### Task 7: Update golden test configs for backward compatibility

**Files:**
- Modify: `configs/test/test_ref_orig.toml`
- Modify: `configs/test/test_high_bank_orig.toml`
- Modify: `configs/test/test_neural_golden.toml`
- Modify: `configs/test/test_ftc_golden.toml`
- Modify: `configs/test/test_fnpag_golden.toml`
- Modify: `configs/test/test_pred_guid_golden.toml`
- Modify: `configs/test/test_energy_ctrl_golden.toml`
- Modify: `configs/test/test_eqglide_golden.toml`
- Modify: `configs/test/test_guided_orig.toml`
- Modify: `configs/test/test_wind_mars.toml`
- Modify: `configs/test/test_ekf_mars.toml`

- [ ] **Step 1: Add `[onboard_atmosphere]` section to all golden test configs**

Append to each test config file:

```toml
[onboard_atmosphere]
mode = "identical"
```

- [ ] **Step 2: Run golden regression tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test guidance_regression -- --nocapture 2>&1 | tail -30`

And run e2e tests:

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --test e2e -- --nocapture 2>&1 | tail -30`

Expected: all golden tests PASS with identical output.

- [ ] **Step 3: Run full Rust test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -30`

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/test/
git commit -m "test: add identical onboard atmosphere to golden test configs for regression"
```

---

### Task 8: Add integration test — piecewise model produces different densities than truth

**Files:**
- Modify: `src/rust/src/data/atmosphere.rs` (add test)

- [ ] **Step 1: Write integration test with Mars-like atmosphere**

Add to `atmosphere.rs` tests:

```rust
#[test]
fn auto_fit_mars_like_table_diverges_from_truth() {
    // Realistic Mars-like exponential decay table (simplified)
    let altitudes: Vec<f64> = (0..14).map(|i| i as f64 * 10_000.0).collect();
    let densities: Vec<f64> = altitudes
        .iter()
        .map(|&alt| 0.013 * (-alt / 9_000.0).exp())
        .collect();
    let n = altitudes.len();

    let truth = AtmosphereModel {
        n_points: n,
        altitudes: altitudes.clone(),
        densities: densities.clone(),
        ref_density: densities[n - 1],
        scale_factor: 1.0 / 9_000.0,
        ref_altitude: altitudes[n - 1],
        gas_constant: 1.3,
        density_profile: DensityProfile::default(),
    };

    let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);

    // The model should diverge from truth at midpoints between table entries
    // (piecewise fit with fewer segments than table points can't match exactly)
    let mut max_rel_err = 0.0_f64;
    for &alt in &[5_000.0, 15_000.0, 35_000.0, 55_000.0, 95_000.0] {
        let rho_truth = truth.density_at(alt);
        let rho_onboard = model.density_at(alt, &truth);
        if rho_truth > 1e-15 {
            let rel_err = (rho_onboard - rho_truth).abs() / rho_truth;
            max_rel_err = max_rel_err.max(rel_err);
        }
    }

    // Should have SOME error (not identical), but not wildly off
    assert!(
        max_rel_err > 1e-6,
        "onboard model should differ from truth; max_rel_err={}",
        max_rel_err,
    );
    assert!(
        max_rel_err < 1.0,
        "onboard model too far from truth; max_rel_err={}",
        max_rel_err,
    );
}

#[test]
fn piecewise_density_always_positive() {
    let truth = AtmosphereModel {
        n_points: 5,
        altitudes: vec![0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0],
        densities: vec![0.013, 0.003, 5e-4, 6e-5, 5e-6],
        ref_density: 5e-6,
        scale_factor: 1e-4,
        ref_altitude: 100_000.0,
        gas_constant: 1.3,
        density_profile: DensityProfile::default(),
    };
    let model = OnboardAtmosphereModel::fit_from_table(&truth, 5);

    // Check density is positive at many altitudes
    for alt_km in 0..=150 {
        let alt = alt_km as f64 * 1_000.0;
        let rho = model.density_at(alt, &truth);
        assert!(
            rho > 0.0,
            "density must be positive at alt={} m, got {}",
            alt,
            rho,
        );
        assert!(
            rho.is_finite(),
            "density must be finite at alt={} m, got {}",
            alt,
            rho,
        );
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib data::atmosphere::tests -- --nocapture 2>&1 | tail -20`

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/data/atmosphere.rs
git commit -m "test: add integration tests for onboard atmosphere model divergence and positivity"
```

---

### Task 9: Run full CI checks and verify

**Files:** none (verification only)

- [ ] **Step 1: Run Rust format check**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo fmt --check 2>&1`

Expected: no formatting issues. If there are, fix with `cargo fmt`.

- [ ] **Step 2: Run Clippy**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo clippy -- -D warnings 2>&1 | tail -20`

Expected: no warnings.

- [ ] **Step 3: Run full Rust test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -30`

Expected: all tests PASS.

- [ ] **Step 4: Build PyO3 bindings (smoke test)**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust/aerocapture-py && maturin develop --release 2>&1 | tail -10`

Expected: successful build. (PyO3 just passes `SimData` through — the new field is transparent.)

- [ ] **Step 5: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1 | tail -20`

Expected: all PASS.

- [ ] **Step 6: Fix any issues found above, then commit fixes if needed**

---

### Task 10: Smart commit — sync docs and final commit

Invoke the `smart-commit` skill, telling it to take the whole `feature/separate-onboard-atmosphere` branch into account.
