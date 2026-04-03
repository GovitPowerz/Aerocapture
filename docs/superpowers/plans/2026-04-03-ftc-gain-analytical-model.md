# FTC Gain Analytical Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 26-entry altitude-dependent pdyn lookup table in FTC guidance with an analytical exponential decay model and cosine fade, eliminating the ~1000x gain discontinuity at the table ceiling.

**Architecture:** Remove `DynamicPressureTableEntry`, `TomlPdynEntry`, and all table lookup code. Add 4 new scalar fields (`pressure_coeff_base`, `pressure_coeff_scale_height`, `gain_fade_start_km`, `gain_fade_end_km`) to the Rust data structs, TOML config, and GA parameter space. The new `compute_gains()` uses `base * exp(-h/H)` for the pressure coefficient and a cosine fade multiplier on both gains.

**Tech Stack:** Rust (nalgebra, proptest), Python (param_spaces.py), TOML configs

**Spec:** `docs/superpowers/specs/2026-04-03-ftc-gain-analytical-model-design.md`

---

### Task 1: Remove table structs and add new fields to data layer

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs:10-17` (remove `DynamicPressureTableEntry`)
- Modify: `src/rust/src/data/guidance_params.rs:125-171` (update `GuidanceParams`)
- Modify: `src/rust/src/data/guidance_params.rs:294-325` (update `Default` impl)

- [ ] **Step 1: Remove `DynamicPressureTableEntry` struct**

In `src/rust/src/data/guidance_params.rs`, delete lines 10-17:

```rust
/// Dynamic pressure reference table entry: altitude (km), linear coefficients a and b.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicPressureTableEntry {
    pub altitude: f64, // km (stored as-is from file)
    pub coeff_a: f64,
    pub coeff_b: f64,
}
```

- [ ] **Step 2: Replace `pdyn_table` field with 4 new fields in `GuidanceParams`**

In `src/rust/src/data/guidance_params.rs`, replace:

```rust
    // Pdyn = f(altitude) reference table
    pub pdyn_table: Vec<DynamicPressureTableEntry>,
```

with:

```rust
    // Analytical gain model (replaces pdyn altitude table)
    pub pressure_coeff_base: f64,         // base pressure coefficient for exponential decay
    pub pressure_coeff_scale_height: f64, // exponential decay scale height (km)
    pub gain_fade_start_km: f64,          // altitude where gain fade begins (km)
    pub gain_fade_end_km: f64,            // altitude where gains reach zero (km)
```

- [ ] **Step 3: Update `Default` impl for `GuidanceParams`**

In `src/rust/src/data/guidance_params.rs`, replace:

```rust
            pdyn_table: Vec::new(),
```

with:

```rust
            pressure_coeff_base: -0.001,
            pressure_coeff_scale_height: 10.0,
            gain_fade_start_km: 80.0,
            gain_fade_end_km: 100.0,
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -30`

Expected: Compilation errors in `config.rs`, `mod.rs`, and `ftc.rs` referencing the removed type and field. This is correct -- we'll fix those in subsequent tasks.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/guidance_params.rs
git commit -m "refactor: replace pdyn_table with analytical gain model fields in GuidanceParams"
```

---

### Task 2: Update TOML config structs

**Files:**
- Modify: `src/rust/src/config.rs:564-608` (update `TomlFtcParams`)
- Modify: `src/rust/src/config.rs:632-637` (remove `TomlPdynEntry`)

- [ ] **Step 1: Remove `TomlPdynEntry` struct**

In `src/rust/src/config.rs`, delete:

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlPdynEntry {
    pub altitude: f64,
    pub a: f64,
    pub b: f64,
}
```

- [ ] **Step 2: Replace `pdyn_table` field in `TomlFtcParams`**

In `src/rust/src/config.rs`, replace:

```rust
    #[serde(default)]
    pub pdyn_min: f64, // Pa
    #[serde(default)]
    pub pdyn_table: Vec<TomlPdynEntry>,
```

with:

```rust
    #[serde(default)]
    pub pdyn_min: f64, // Pa
    #[serde(default = "default_pressure_coeff_base")]
    pub pressure_coeff_base: f64,
    #[serde(default = "default_pressure_coeff_scale_height")]
    pub pressure_coeff_scale_height: f64, // km
    #[serde(default = "default_gain_fade_start_km")]
    pub gain_fade_start_km: f64,
    #[serde(default = "default_gain_fade_end_km")]
    pub gain_fade_end_km: f64,
```

- [ ] **Step 3: Add default functions**

In `src/rust/src/config.rs`, near the other default functions (after `default_longi_inh`), add:

```rust
fn default_pressure_coeff_base() -> f64 {
    -0.001
}
fn default_pressure_coeff_scale_height() -> f64 {
    10.0
}
fn default_gain_fade_start_km() -> f64 {
    80.0
}
fn default_gain_fade_end_km() -> f64 {
    100.0
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -30`

Expected: Errors in `data/mod.rs` referencing removed `pdyn_table` and `DynamicPressureTableEntry`. This is correct -- fixed in next task.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "refactor: replace pdyn_table TOML config with analytical gain model fields"
```

---

### Task 3: Wire new fields through data loading

**Files:**
- Modify: `src/rust/src/data/mod.rs:422-499` (FTC guidance params loading)

- [ ] **Step 1: Remove table mapping and wire new fields**

In `src/rust/src/data/mod.rs`, replace the block:

```rust
        // FTC guidance params
        let guidance = if let Some(ref ftc) = toml.guidance.ftc {
            let energy_scale = 1e6;
            let pdyn_table = ftc
                .pdyn_table
                .iter()
                .map(|e| guidance_params::DynamicPressureTableEntry {
                    altitude: e.altitude,
                    coeff_a: e.a,
                    coeff_b: e.b,
                })
                .collect();

            // Load reference trajectory from external file
```

with:

```rust
        // FTC guidance params
        let guidance = if let Some(ref ftc) = toml.guidance.ftc {
            let energy_scale = 1e6;

            // Load reference trajectory from external file
```

- [ ] **Step 2: Replace `pdyn_table` field assignment in `GuidanceParams` initializer**

In `src/rust/src/data/mod.rs`, replace:

```rust
                pdyn_min: ftc.pdyn_min,
                pdyn_table,
                ref_trajectory: ref_traj,
```

with:

```rust
                pdyn_min: ftc.pdyn_min,
                pressure_coeff_base: ftc.pressure_coeff_base,
                pressure_coeff_scale_height: ftc.pressure_coeff_scale_height,
                gain_fade_start_km: ftc.gain_fade_start_km,
                gain_fade_end_km: ftc.gain_fade_end_km,
                ref_trajectory: ref_traj,
```

- [ ] **Step 3: Update the `else` branch (no FTC params) default values**

In `src/rust/src/data/mod.rs`, replace:

```rust
                pdyn_table: vec![],
```

with:

```rust
                pressure_coeff_base: -0.001,
                pressure_coeff_scale_height: 10.0,
                gain_fade_start_km: 80.0,
                gain_fade_end_km: 100.0,
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -30`

Expected: Only errors in `ftc.rs` referencing the removed `pdyn_table` field. This is correct -- fixed in next task.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/mod.rs
git commit -m "refactor: wire analytical gain model fields through data loading"
```

---

### Task 4: Rewrite `compute_gains()` with analytical model

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs:87-137` (replace `compute_gains`)

- [ ] **Step 1: Replace `compute_gains()` function body**

In `src/rust/src/gnc/guidance/ftc.rs`, replace the entire `compute_gains` function (lines 87-137):

```rust
/// Compute guidance gains from altitude-based Pdyn model.
fn compute_gains(altitude: f64, aero_coefficients: &[f64; 2], data: &SimData) -> (f64, f64) {
    let pdyn_table = &data.guidance.pdyn_table;
    let alt_km = altitude / 1e3;

    // Find altitude bracket; use Option<usize> as "not found" sentinel.
    let mut found: Option<usize> = None;
    for i in 0..pdyn_table.len().saturating_sub(1) {
        if alt_km >= pdyn_table[i].altitude
            && alt_km < pdyn_table[i + 1].altitude
            && found.is_none()
        {
            found = Some(i);
        }
    }
    // If no bracket found, fall back to last entry
    let table_index = found.unwrap_or_else(|| {
        if pdyn_table.is_empty() {
            0
        } else {
            pdyn_table.len() - 1
        }
    });

    let pressure_coeff = if table_index < pdyn_table.len() {
        pdyn_table[table_index].coeff_a
    } else {
        1.0
    };

    // Gains
    let damping_capture = data.guidance.capture_damping;
    let frequency_capture = data.guidance.capture_frequency;
    let reference_area = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let cz = aero_coefficients[1]; // lift coefficient

    let gain_altitude_rate = if (reference_area * cz).abs() > 1e-30 {
        -2.0 * damping_capture * frequency_capture * mass / (reference_area * cz)
    } else {
        0.0
    };

    let gain_dynamic_pressure = if (pressure_coeff * reference_area * cz).abs() > 1e-30 {
        -frequency_capture * frequency_capture * mass / (pressure_coeff * reference_area * cz)
    } else {
        0.0
    };

    (gain_altitude_rate, gain_dynamic_pressure)
}
```

with:

```rust
/// Cosine fade: 1.0 below `start`, 0.0 above `end`, smooth cosine taper between.
/// Degenerate case: if `end <= start`, returns 1.0 (no fade).
fn cosine_fade(alt_km: f64, start: f64, end: f64) -> f64 {
    if end <= start {
        return 1.0;
    }
    let t = ((alt_km - start) / (end - start)).clamp(0.0, 1.0);
    0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Compute guidance gains using analytical exponential decay model.
fn compute_gains(altitude: f64, aero_coefficients: &[f64; 2], data: &SimData) -> (f64, f64) {
    let alt_km = altitude / 1e3;

    // Exponential decay pressure coefficient
    let pressure_coeff =
        data.guidance.pressure_coeff_base * (-alt_km / data.guidance.pressure_coeff_scale_height).exp();

    // Cosine fade: both gains taper to zero above the sensible atmosphere
    let fade = cosine_fade(alt_km, data.guidance.gain_fade_start_km, data.guidance.gain_fade_end_km);

    // Gains
    let damping_capture = data.guidance.capture_damping;
    let frequency_capture = data.guidance.capture_frequency;
    let reference_area = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let cz = aero_coefficients[1]; // lift coefficient

    let gain_altitude_rate = if (reference_area * cz).abs() > 1e-30 {
        fade * -2.0 * damping_capture * frequency_capture * mass / (reference_area * cz)
    } else {
        0.0
    };

    let gain_dynamic_pressure = if (pressure_coeff * reference_area * cz).abs() > 1e-30 {
        fade * -frequency_capture * frequency_capture * mass / (pressure_coeff * reference_area * cz)
    } else {
        0.0
    };

    (gain_altitude_rate, gain_dynamic_pressure)
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -5`

Expected: Clean compilation (no errors).

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "feat: replace pdyn table lookup with analytical exponential decay + cosine fade"
```

---

### Task 5: Add unit tests for cosine fade and analytical gains

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Add test module with cosine fade tests**

Append to `src/rust/src/gnc/guidance/ftc.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::data::SimData;
    use crate::data::guidance_params::GuidanceParams;

    #[test]
    fn cosine_fade_below_start_is_one() {
        assert_relative_eq!(cosine_fade(50.0, 80.0, 100.0), 1.0);
        assert_relative_eq!(cosine_fade(0.0, 80.0, 100.0), 1.0);
        assert_relative_eq!(cosine_fade(79.99, 80.0, 100.0), 1.0);
    }

    #[test]
    fn cosine_fade_above_end_is_zero() {
        assert_relative_eq!(cosine_fade(100.0, 80.0, 100.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(cosine_fade(150.0, 80.0, 100.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(cosine_fade(500.0, 80.0, 100.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn cosine_fade_midpoint_is_half() {
        assert_relative_eq!(cosine_fade(90.0, 80.0, 100.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn cosine_fade_monotonically_decreasing() {
        let start = 80.0;
        let end = 100.0;
        let n = 100;
        let mut prev = cosine_fade(start, start, end);
        for i in 1..=n {
            let alt = start + (end - start) * (i as f64) / (n as f64);
            let val = cosine_fade(alt, start, end);
            assert!(val <= prev, "fade increased at alt={alt}: {val} > {prev}");
            prev = val;
        }
    }

    #[test]
    fn cosine_fade_degenerate_end_le_start() {
        // end == start
        assert_relative_eq!(cosine_fade(90.0, 80.0, 80.0), 1.0);
        // end < start
        assert_relative_eq!(cosine_fade(90.0, 100.0, 80.0), 1.0);
    }

    #[test]
    fn pressure_coeff_decreases_with_altitude() {
        let base = -0.001_f64;
        let scale_height = 10.0;
        let coeff_at_0 = base * (-0.0 / scale_height).exp();
        let coeff_at_50 = base * (-50.0 / scale_height).exp();
        let coeff_at_100 = base * (-100.0 / scale_height).exp();
        // base is negative, so magnitude = -coeff
        assert!(coeff_at_0.abs() > coeff_at_50.abs());
        assert!(coeff_at_50.abs() > coeff_at_100.abs());
    }

    fn test_sim_data() -> SimData {
        use crate::data::{
            Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit,
            SphericalState, SuccessCriteria, TimePeriods,
        };
        use crate::data::aerodynamics::AeroTables;
        use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
        use crate::data::capsule::Capsule;
        use crate::data::incidence::IncidenceProfile;
        use crate::data::pilot::{PilotModel, PilotType};

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
                ..Default::default()
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
                pressure_coeff_base: -0.001,
                pressure_coeff_scale_height: 10.0,
                gain_fade_start_km: 80.0,
                gain_fade_end_km: 100.0,
                capture_damping: 0.7,
                capture_frequency: 0.072,
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
            density_perturbation: None,
        }
    }

    #[test]
    fn gains_zero_when_fade_is_zero() {
        let data = test_sim_data();
        let aero_coefficients = [0.5, 0.3]; // [Cx, Cz]
        let altitude_above_fade = 120_000.0; // 120 km in meters

        let (g_alt, g_pdyn) = compute_gains(altitude_above_fade, &aero_coefficients, &data);
        assert_relative_eq!(g_alt, 0.0, epsilon = 1e-30);
        assert_relative_eq!(g_pdyn, 0.0, epsilon = 1e-30);
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::ftc -- --nocapture 2>&1`

Expected: All 7 tests pass.

- [ ] **Step 3: Add proptest for gain finiteness**

Append inside the `mod tests` block (before the closing `}`):

```rust
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn gains_are_finite_for_any_altitude(
                alt_km in 0.0_f64..500.0,
                cz in -2.0_f64..2.0,
            ) {
                let data = test_sim_data();
                let aero_coefficients = [0.5, cz];
                let altitude_m = alt_km * 1e3;

                let (g_alt, g_pdyn) = compute_gains(altitude_m, &aero_coefficients, &data);
                prop_assert!(g_alt.is_finite(), "gain_altitude_rate is not finite at alt={alt_km} km");
                prop_assert!(g_pdyn.is_finite(), "gain_dynamic_pressure is not finite at alt={alt_km} km");
            }
        }
    }
```

- [ ] **Step 4: Run all FTC tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::ftc -- --nocapture 2>&1`

Expected: All 8 tests pass (7 unit + 1 proptest).

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "test: add unit tests and proptest for analytical gain model"
```

---

### Task 6: Update TOML configs

**Files:**
- Modify: `configs/training/msr_aller_ftc_train.toml`
- Modify: `configs/test/test_ftc_golden.toml`
- Modify: `configs/nominal/msr_aller_ftc_consolidated.toml`
- Modify: `configs/nominal/msr_aller_ftc_mc_domain.toml`
- Modify: `configs/nominal/msr_aller_ftc_nominal.toml`
- Modify: `configs/nominal/esr_aller_ftc_nominal.toml`
- Modify: `configs/test/test_guided_orig.toml`

- [ ] **Step 1: Update `msr_aller_ftc_train.toml`**

In `configs/training/msr_aller_ftc_train.toml`, replace the entire `pdyn_table` block (lines 28-55):

```toml
pdyn_table = [
    { altitude =  0.0000000000, a = -0.1645497562, b = 1.4897963360 },
    ...
    { altitude = 96.2469613900, a = -0.0000010000, b = 0.0022793682 },
]
```

with:

```toml
pressure_coeff_base = -0.001
pressure_coeff_scale_height = 10.0
gain_fade_start_km = 80.0
gain_fade_end_km = 100.0
```

- [ ] **Step 2: Add analytical gain fields to `test_ftc_golden.toml`**

In `configs/test/test_ftc_golden.toml`, add a `[guidance.ftc]` section after the `[guidance]` section:

```toml
[guidance.ftc]
pressure_coeff_base = -0.001
pressure_coeff_scale_height = 10.0
gain_fade_start_km = 80.0
gain_fade_end_km = 100.0
```

This ensures the golden test uses explicit values rather than relying on defaults (which previously gave an empty table and fallback `pressure_coeff = 1.0`).

- [ ] **Step 3: Update remaining nominal/test configs**

For each of these files, replace the `pdyn_table = [...]` block with the same 4 scalar fields:

- `configs/nominal/msr_aller_ftc_consolidated.toml`
- `configs/nominal/msr_aller_ftc_mc_domain.toml`
- `configs/nominal/msr_aller_ftc_nominal.toml`
- `configs/nominal/esr_aller_ftc_nominal.toml`
- `configs/test/test_guided_orig.toml`

Replace each `pdyn_table = [...]` block with:

```toml
pressure_coeff_base = -0.001
pressure_coeff_scale_height = 10.0
gain_fade_start_km = 80.0
gain_fade_end_km = 100.0
```

- [ ] **Step 4: Verify the Rust binary builds and runs**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release 2>&1 | tail -3`

Then: `cd /Users/govit/Git/Govit/Aerocapture && ./src/rust/target/release/aerocapture configs/test/test_ftc_golden.toml 2>&1 | tail -5`

Expected: Build succeeds. Sim runs to completion (no crash/panic).

- [ ] **Step 5: Commit**

```bash
git add configs/training/msr_aller_ftc_train.toml configs/test/test_ftc_golden.toml configs/nominal/ configs/test/test_guided_orig.toml
git commit -m "config: replace pdyn_table with analytical gain model fields in all FTC configs"
```

---

### Task 7: Regenerate FTC golden file

**Files:**
- Modify: `tests/reference_data/rust_golden/ftc/final.golden_ftc.csv`

- [ ] **Step 1: Build release binary**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release 2>&1 | tail -3`

Expected: Build succeeds.

- [ ] **Step 2: Regenerate golden file**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./src/rust/target/release/aerocapture configs/test/test_ftc_golden.toml`

Then copy the new output to the golden directory:

Run: `cp /Users/govit/Git/Govit/Aerocapture/final.golden_ftc.csv tests/reference_data/rust_golden/ftc/final.golden_ftc.csv`

Note: The exact output filename depends on the `results_suffix` in the config (`.golden_ftc`). Check what file was produced and copy it to the golden directory.

- [ ] **Step 3: Run the golden regression test**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test guidance_regression -- ftc --nocapture 2>&1`

Expected: FTC golden test passes.

- [ ] **Step 4: Run ALL golden regression tests to confirm no other scheme was affected**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test guidance_regression --nocapture 2>&1`

Expected: All 6 guidance regression tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/reference_data/rust_golden/ftc/
git commit -m "test: regenerate FTC golden file for analytical gain model"
```

---

### Task 8: Add GA parameter space entries

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py:104-114`

- [ ] **Step 1: Add 4 new ParamSpec entries to FTC param space**

In `src/python/aerocapture/training/param_spaces.py`, replace:

```python
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("altitude_damping", 0.3, 1.5, 0.7),
        ParamSpec("altitude_frequency", 0.01, 0.2, 0.08),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
    ],
```

with:

```python
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("altitude_damping", 0.3, 1.5, 0.7),
        ParamSpec("altitude_frequency", 0.01, 0.2, 0.08),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        ParamSpec("pressure_coeff_base", -0.01, -0.0001, -0.001),
        ParamSpec("pressure_coeff_scale_height", 5.0, 20.0, 10.0),
        ParamSpec("gain_fade_start_km", 60.0, 90.0, 80.0),
        ParamSpec("gain_fade_end_km", 85.0, 120.0, 100.0),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
    ],
```

- [ ] **Step 2: Run Python linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff check src/python/aerocapture/training/param_spaces.py`

Expected: No lint errors.

- [ ] **Step 3: Run Python tests related to param spaces**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -k "param" -v 2>&1 | tail -20`

Expected: All param-related tests pass. The chromosome length for FTC increases by 4 -- any test that hardcodes FTC chromosome length may need updating (check test output).

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "feat: add analytical gain model params to FTC GA parameter space"
```

---

### Task 9: Run full test suite and lint

**Files:** None (verification only)

- [ ] **Step 1: Run Rust test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1 | tail -20`

Expected: All tests pass.

- [ ] **Step 2: Run Rust clippy**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo clippy -- -D warnings 2>&1 | tail -10`

Expected: No warnings.

- [ ] **Step 3: Run Rust fmt check**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo fmt --check 2>&1`

Expected: No formatting issues (run `cargo fmt` if needed).

- [ ] **Step 4: Run Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v 2>&1 | tail -30`

Expected: All tests pass. If any test hardcodes FTC chromosome length, update it.

- [ ] **Step 5: Run full lint script**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -20`

Expected: Clean output.

---

### Task 10: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
