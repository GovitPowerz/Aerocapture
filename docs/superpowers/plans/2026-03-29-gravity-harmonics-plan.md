# Higher-Order Gravity Harmonics (J3, J4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded `Planet` enum with a TOML-configurable `PlanetConfig` struct and extend the gravity model with J3/J4 zonal harmonics.

**Architecture:** New `PlanetConfig` struct holds all planet constants (mu, radii, omega, J2, J3, J4). Parsed from a `[planet]` TOML section. Planet preset files in `configs/planets/` are base-inherited by mission configs. The gravity function gains two terms from Legendre polynomial derivatives P3 and P4.

**Tech Stack:** Rust (serde, nalgebra), TOML config, proptest for property-based gravity tests.

**Design spec:** `docs/superpowers/specs/2026-03-29-gravity-harmonics-design.md`

---

### Task 1: Create planet preset TOML files

**Files:**
- Create: `configs/planets/mars.toml`
- Create: `configs/planets/earth.toml`
- Create: `configs/planets/moon.toml`
- Create: `configs/planets/jupiter.toml`

- [ ] **Step 1: Create `configs/planets/mars.toml`**

```toml
# Mars planet constants (GMM-3 gravity model, Genova/Konopliv et al. 2016)

[planet]
name = "mars"
mu = 4.282829e13
equatorial_radius = 3393940.0
polar_radius = 3376780.0
omega = 7.088218e-5
j2 = 1.958616e-3
j3 = 3.145e-5
j4 = -1.538e-5
```

- [ ] **Step 2: Create `configs/planets/earth.toml`**

```toml
# Earth planet constants (WGS-84 / EGM96)

[planet]
name = "earth"
mu = 3.98600418e14
equatorial_radius = 6378137.0
polar_radius = 6356784.0
omega = 7.292115e-5
j2 = 1.08263e-3
j3 = -2.5327e-6
j4 = -1.6196e-6
```

- [ ] **Step 3: Create `configs/planets/moon.toml`**

```toml
# Moon planet constants (legacy Fortran values; J3/J4 not characterized)

[planet]
name = "moon"
mu = 3.249e14
equatorial_radius = 6051800.0
polar_radius = 6051800.0
omega = 2.9924e-7
j2 = 4.458e-6
j3 = 0.0
j4 = 0.0
```

- [ ] **Step 4: Create `configs/planets/jupiter.toml`**

```toml
# Jupiter planet constants (Juno / Iess et al. 2018)

[planet]
name = "jupiter"
mu = 1.26686e17
equatorial_radius = 71492000.0
polar_radius = 66854000.0
omega = 1.759e-4
j2 = 1.4736e-2
j3 = -4.2e-8
j4 = -5.866e-4
```

- [ ] **Step 5: Commit**

```bash
git add configs/planets/
git commit -m "feat: add planet preset TOML files (mars, earth, moon, jupiter)"
```

---

### Task 2: Add `PlanetConfig` struct and TOML deserialization

**Files:**
- Modify: `src/rust/src/config.rs:14-73` (Planet enum area)
- Modify: `src/rust/src/config.rs:133-153` (SimInput struct)
- Modify: `src/rust/src/config.rs:157-181` (TomlConfig struct)
- Modify: `src/rust/src/config.rs:294-305` (TomlMission struct)
- Modify: `src/rust/src/config.rs:936-974` (from_toml parsing)

- [ ] **Step 1: Add `PlanetConfig` struct below the existing `Planet` enum (don't remove enum yet)**

In `src/rust/src/config.rs`, after the `Planet` impl block (after line 73), add:

```rust
/// Planet physical constants, parsed from TOML [planet] section.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanetConfig {
    pub name: String,
    pub mu: f64,
    pub equatorial_radius: f64,
    pub polar_radius: f64,
    pub omega: f64,
    pub j2: f64,
    #[serde(default)]
    pub j3: f64,
    #[serde(default)]
    pub j4: f64,
}
```

- [ ] **Step 2: Add `#[cfg(test)]` factory methods to `PlanetConfig`**

Below the struct definition, add:

```rust
#[cfg(test)]
impl PlanetConfig {
    pub fn mars() -> Self {
        Self {
            name: "mars".into(),
            mu: 4.282829e13,
            equatorial_radius: 3393940.0,
            polar_radius: 3376780.0,
            omega: 7.088218e-5,
            j2: 1.958616e-3,
            j3: 3.145e-5,
            j4: -1.538e-5,
        }
    }

    pub fn earth() -> Self {
        Self {
            name: "earth".into(),
            mu: 3.98600418e14,
            equatorial_radius: 6378137.0,
            polar_radius: 6356784.0,
            omega: 7.292115e-5,
            j2: 1.08263e-3,
            j3: -2.5327e-6,
            j4: -1.6196e-6,
        }
    }

    pub fn moon() -> Self {
        Self {
            name: "moon".into(),
            mu: 3.249e14,
            equatorial_radius: 6051800.0,
            polar_radius: 6051800.0,
            omega: 2.9924e-7,
            j2: 4.458e-6,
            j3: 0.0,
            j4: 0.0,
        }
    }

    /// Mars-like planet with J3=J4=0 for backward-compat tests.
    pub fn mars_j2_only() -> Self {
        Self {
            j3: 0.0,
            j4: 0.0,
            ..Self::mars()
        }
    }
}
```

- [ ] **Step 3: Add `planet` field to `TomlConfig` and make `planet` optional in `TomlMission`**

In `src/rust/src/config.rs`, modify the `TomlConfig` struct (lines 157-181) to add a `planet` field:

```rust
#[derive(Debug, Deserialize)]
pub struct TomlConfig {
    pub mission: TomlMission,
    pub planet: PlanetConfig,  // ← NEW: required [planet] section
    pub guidance: TomlGuidance,
    #[serde(default)]
    pub simulation: TomlSimulation,
    pub data: TomlData,
    // ... rest unchanged ...
}
```

In `TomlMission` (lines 294-305), remove the `planet` field:

```rust
#[derive(Debug, Deserialize)]
pub struct TomlMission {
    #[serde(rename = "type")]
    pub mission_type: String,
    #[serde(default = "default_phase")]
    pub phase: String,
}
```

- [ ] **Step 4: Update `SimInput` to use `PlanetConfig` instead of `Planet`**

In `src/rust/src/config.rs`, change the `SimInput` struct (lines 133-153):

```rust
pub struct SimInput {
    pub mission_type: MissionType,
    pub planet: PlanetConfig,  // ← was: Planet
    pub n_sims: i32,
    // ... rest unchanged ...
}
```

- [ ] **Step 5: Update `from_toml()` to use the new `PlanetConfig` from TOML**

In `src/rust/src/config.rs`, in `SimInput::from_toml()` (around lines 936-974), remove the planet match block and use the TOML-parsed struct directly:

Replace lines 945-951:
```rust
let planet = match config.mission.planet.as_str() {
    "moon" => Planet::Moon,
    "earth" => Planet::Earth,
    "mars" => Planet::Mars,
    "jupiter" => Planet::Jupiter,
    other => return Err(ParseError(format!("Unknown planet: {}", other))),
};
```

With:
```rust
let planet = config.planet.clone();
```

And update the `SimInput` construction at line 972-974:
```rust
let sim_input = SimInput {
    mission_type,
    planet,  // now PlanetConfig, no change needed here
    // ...
};
```

- [ ] **Step 6: Verify it compiles (expect errors from Planet references — that's OK for now)**

Run: `cd src/rust && cargo check 2>&1 | head -40`

Expected: Compile errors about `Planet` being unused and various call sites still expecting `&Planet`. This confirms the struct is wired in. We'll fix call sites in Task 4.

- [ ] **Step 7: Commit (WIP — struct and parser wired, call sites not yet migrated)**

```bash
git add src/rust/src/config.rs
git commit -m "feat: add PlanetConfig struct with TOML deserialization"
```

---

### Task 3: Extend gravity function with J3/J4 (TDD)

**Files:**
- Modify: `src/rust/src/physics/gravity.rs:1-132`
- Test: inline `#[cfg(test)]` module in same file

- [ ] **Step 1: Write the failing test — J3/J4 zero matches J2 only**

In `src/rust/src/physics/gravity.rs`, add this test inside the existing `mod tests` block (after the last test, before the closing `}`):

```rust
    #[test]
    fn j3_j4_zero_matches_j2_only() {
        // A PlanetConfig with J3=J4=0 must produce bit-identical results to the
        // J2-only formula for any (r, lat) pair.
        let planet = PlanetConfig::mars_j2_only();
        let r = planet.equatorial_radius + 50_000.0;
        for lat_deg in [-60.0, -30.0, 0.0, 30.0, 60.0] {
            let lat = lat_deg.to_radians();
            let (gravtl, gravtr) = gravity(r, lat, &planet);
            // Manually compute J2-only values
            let mu = planet.mu;
            let req = planet.equatorial_radius;
            let j2 = planet.j2;
            let r2 = r * r;
            let r4 = r2 * r2;
            let sin_lat = lat.sin();
            let cos_lat = lat.cos();
            let sin2 = sin_lat * sin_lat;
            let req2 = req * req;
            let expected_tl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;
            let expected_tr = mu / r2 + 3.0 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / (2.0 * r4);
            assert_eq!(gravtl, expected_tl, "gravtl mismatch at lat={lat_deg}");
            assert_eq!(gravtr, expected_tr, "gravtr mismatch at lat={lat_deg}");
        }
    }
```

Also update the imports at the top of the test module — change `use crate::config::Planet;` to `use crate::config::PlanetConfig;`.

- [ ] **Step 2: Write the failing test — J3 breaks north-south symmetry**

```rust
    #[test]
    fn j3_breaks_north_south_symmetry() {
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let lat = 0.5_f64; // ~28.6 degrees
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        // J3 (odd harmonic) breaks exact antisymmetry
        assert!(
            (gravtl_pos + gravtl_neg).abs() > 1e-10,
            "J3 should break north-south symmetry: sum = {}",
            gravtl_pos + gravtl_neg
        );
    }
```

- [ ] **Step 3: Write the failing test — J3 lateral nonzero at equator**

```rust
    #[test]
    fn j3_lateral_nonzero_at_equator() {
        // J3 contributes cos(0)*(5*0-1) = -1 at equator, unlike J2 which is zero there
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius;
        let (gravtl, _) = gravity(r, 0.0, &planet);
        assert!(
            gravtl.abs() > 1e-12,
            "J3 lateral should be nonzero at equator: gravtl = {gravtl}"
        );
    }
```

- [ ] **Step 4: Write the failing test — J4 preserves symmetry**

```rust
    #[test]
    fn j4_preserves_lateral_antisymmetry() {
        // J4 (even harmonic) preserves antisymmetry in the lateral component
        // Test with a planet that has J3=0, only J4 active
        let planet = PlanetConfig {
            j3: 0.0,
            j4: -1.538e-5,
            ..PlanetConfig::mars_j2_only()
        };
        let r = planet.equatorial_radius + 50_000.0;
        let lat = 0.7;
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        assert_relative_eq!(gravtl_pos, -gravtl_neg, max_relative = 1e-14);
    }
```

- [ ] **Step 5: Write the failing test — J3/J4 are small corrections**

```rust
    #[test]
    fn j3_j4_small_correction_at_mars_surface() {
        let planet = PlanetConfig::mars();
        let planet_j2 = PlanetConfig::mars_j2_only();
        let r = planet.equatorial_radius;
        let lat = 0.5_f64;
        let (tl_full, tr_full) = gravity(r, lat, &planet);
        let (tl_j2, tr_j2) = gravity(r, lat, &planet_j2);
        let rel_tl = ((tl_full - tl_j2) / tl_j2).abs();
        let rel_tr = ((tr_full - tr_j2) / tr_j2).abs();
        assert!(rel_tl < 0.05, "J3+J4 lateral correction is {:.4}%, expected < 5%", rel_tl * 100.0);
        assert!(rel_tr < 0.05, "J3+J4 radial correction is {:.4}%, expected < 5%", rel_tr * 100.0);
    }
```

- [ ] **Step 6: Write the failing proptest — gravity magnitude is finite**

Add `use proptest::prelude::*;` to the test module imports.

```rust
    proptest! {
        #[test]
        fn gravity_magnitude_finite(
            alt_km in 0.0_f64..10000.0,
            lat_deg in -90.0_f64..90.0,
        ) {
            let planet = PlanetConfig::mars();
            let r = planet.equatorial_radius + alt_km * 1000.0;
            let lat = lat_deg.to_radians();
            let (gravtl, gravtr) = gravity(r, lat, &planet);
            prop_assert!(gravtl.is_finite(), "gravtl is not finite at alt={alt_km} lat={lat_deg}");
            prop_assert!(gravtr.is_finite(), "gravtr is not finite at alt={alt_km} lat={lat_deg}");
            prop_assert!(gravtr > 0.0, "gravtr should be positive (inward pull)");
        }
    }
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `cd src/rust && cargo test --lib physics::gravity -- --nocapture 2>&1 | tail -20`

Expected: FAIL — `gravity()` still takes `&Planet`, not `&PlanetConfig`.

- [ ] **Step 8: Implement the extended gravity function**

Replace the entire `gravity()` function in `src/rust/src/physics/gravity.rs` (lines 1-29) with:

```rust
//! Gravity model with J2/J3/J4 zonal harmonic corrections.

use crate::config::PlanetConfig;

/// Compute gravitational acceleration components in spherical coordinates.
///
/// Returns (gravtl, gravtr):
///   - gravtl: lateral (latitudinal) component (m/s^2), convention: -g_lat
///   - gravtr: radial component (m/s^2), convention: -g_r (positive inward)
///
/// Supports zonal harmonics J2, J3, J4. When J3=J4=0, reduces to J2-only model.
pub fn gravity(radius: f64, latitude: f64, planet: &PlanetConfig) -> (f64, f64) {
    let mu = planet.mu;
    let req = planet.equatorial_radius;
    let j2 = planet.j2;
    let j3 = planet.j3;
    let j4 = planet.j4;

    let r2 = radius * radius;
    let r4 = r2 * r2;
    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();
    let sin2 = sin_lat * sin_lat;
    let req2 = req * req;

    // ── Radial component (positive inward): gravtr = -g_r ──
    // Keplerian + J2
    let mut gravtr = mu / r2 + 1.5 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / r4;

    // J3: 2*mu*J3*R^3 * sin*(3 - 5*sin^2) / r^5
    if j3 != 0.0 {
        let r5 = r4 * radius;
        let req3 = req2 * req;
        gravtr += 2.0 * mu * j3 * req3 * sin_lat * (3.0 - 5.0 * sin2) / r5;
    }

    // J4: -(5/8)*mu*J4*R^4 * (3 - 30*sin^2 + 35*sin^4) / r^6
    if j4 != 0.0 {
        let r6 = r4 * r2;
        let req4 = req2 * req2;
        let sin4 = sin2 * sin2;
        gravtr -= 0.625 * mu * j4 * req4 * (3.0 - 30.0 * sin2 + 35.0 * sin4) / r6;
    }

    // ── Lateral component: gravtl = -g_lat ──
    // J2: 3*mu*J2*R^2 * sin*cos / r^4
    let mut gravtl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;

    // J3: (3/2)*mu*J3*R^3 * cos*(5*sin^2 - 1) / r^5
    if j3 != 0.0 {
        let r5 = r4 * radius;
        let req3 = req2 * req;
        gravtl += 1.5 * mu * j3 * req3 * cos_lat * (5.0 * sin2 - 1.0) / r5;
    }

    // J4: -(5/2)*mu*J4*R^4 * sin*cos*(3 - 7*sin^2) / r^6
    if j4 != 0.0 {
        let r6 = r4 * r2;
        let req4 = req2 * req2;
        gravtl -= 2.5 * mu * j4 * req4 * sin_lat * cos_lat * (3.0 - 7.0 * sin2) / r6;
    }

    (gravtl, gravtr)
}
```

- [ ] **Step 9: Update existing tests to use `PlanetConfig`**

In the test module of `gravity.rs`, replace all `Planet::Mars` with `PlanetConfig::mars()`, `Planet::Earth` with `PlanetConfig::earth()`, and `Planet::Moon` with `PlanetConfig::moon()`. Replace method calls with field access:
- `planet.equatorial_radius()` → `planet.equatorial_radius`
- `planet.polar_radius()` → `planet.polar_radius`
- `planet.mu()` → `planet.mu`

Update the `j2_lateral_symmetry` test — with real J3 values, exact antisymmetry no longer holds. Change to approximate:
```rust
    #[test]
    fn lateral_approximate_antisymmetry() {
        // With J3 active, exact antisymmetry is broken, but it's still nearly antisymmetric
        // because J3/J2 ~ 1.6% for Mars
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius;
        let lat = 0.7;
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        // The sum should be small relative to the individual values (dominated by J3)
        let asymmetry = (gravtl_pos + gravtl_neg).abs();
        let magnitude = gravtl_pos.abs().max(gravtl_neg.abs());
        assert!(asymmetry / magnitude < 0.05, "asymmetry ratio {:.4} exceeds 5%", asymmetry / magnitude);
    }
```

Update `j2_lateral_zero_at_equator` — rename and adjust for J3:
```rust
    #[test]
    fn lateral_at_equator_from_j3() {
        // J2 lateral is zero at equator (sin=0), but J3 contributes
        // cos(0)*(5*0-1) = -1, so lateral is nonzero
        let planet = PlanetConfig::mars();
        let (gravtl, _) = gravity(planet.equatorial_radius, 0.0, &planet);
        // Should be small but nonzero (J3 contribution only)
        assert!(gravtl.abs() > 1e-12);
        assert!(gravtl.abs() < 0.01); // Still a tiny correction
    }
```

- [ ] **Step 10: Run gravity tests**

Run: `cd src/rust && cargo test --lib physics::gravity -- --nocapture 2>&1`

Expected: All tests PASS (existing + new). If there are compile errors from other modules still using `&Planet`, use `--lib physics::gravity` to isolate.

Note: The full build may fail because other modules still reference `Planet`. That's expected — we'll fix those in Task 4.

- [ ] **Step 11: Commit**

```bash
git add src/rust/src/physics/gravity.rs
git commit -m "feat: extend gravity model with J3/J4 zonal harmonics (TDD)"
```

---

### Task 4: Update mission TOML configs to inherit from planet presets

**Files:**
- Modify: `configs/missions/mars.toml`
- Modify: `configs/missions/earth.toml`

- [ ] **Step 1: Update `configs/missions/mars.toml`**

Add planet base inheritance and remove the `planet` field from `[mission]`:

```toml
# Mars Sample Return — shared mission base config
# Inherited by all MSR training, test, and nominal configs via base = ["../missions/mars.toml"]
base = ["../planets/mars.toml"]

[mission]
type = "aerocapture"
```

Remove the line `planet = "mars"` from the `[mission]` section. Keep everything else unchanged.

- [ ] **Step 2: Update `configs/missions/earth.toml`**

Same pattern:

```toml
# Earth Sample Return — shared mission base config
# Inherited by all ESR configs via base = ["../missions/earth.toml"]
base = ["../planets/earth.toml"]

[mission]
type = "aerocapture"
```

Remove `planet = "earth"` from `[mission]`. Keep everything else unchanged.

- [ ] **Step 3: Commit**

```bash
git add configs/missions/
git commit -m "refactor: mission TOMLs inherit [planet] from planet presets"
```

---

### Task 5: Migrate all production call sites from `&Planet` to `&PlanetConfig`

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (~20 occurrences)
- Modify: `src/rust/src/orbit/elements.rs`
- Modify: `src/rust/src/orbit/maneuver.rs`
- Modify: `src/rust/src/gnc/navigation/coordinates.rs`
- Modify: `src/rust/src/gnc/navigation/estimator.rs`
- Modify: `src/rust/src/gnc/guidance/ftc.rs`
- Modify: `src/rust/src/gnc/guidance/equilibrium_glide.rs`
- Modify: `src/rust/src/gnc/guidance/fnpag.rs`
- Modify: `src/rust/src/gnc/guidance/neural.rs`
- Modify: `src/rust/src/gnc/guidance/lateral.rs`
- Modify: `src/rust/src/gnc/guidance/energy_controller.rs`
- Modify: `src/rust/src/gnc/guidance/predguid.rs`
- Modify: `src/rust/src/gnc/guidance/piecewise_constant.rs` (if it references Planet)
- Modify: `src/rust/src/lib.rs` (if it re-exports Planet)

This is a mechanical migration. For every file:

1. Replace `use crate::config::Planet;` with `use crate::config::PlanetConfig;`
2. Replace `planet: &Planet` with `planet: &PlanetConfig` in function signatures
3. Replace method calls with field access:
   - `planet.mu()` → `planet.mu`
   - `planet.equatorial_radius()` → `planet.equatorial_radius`
   - `planet.polar_radius()` → `planet.polar_radius`
   - `planet.omega()` → `planet.omega`
   - `planet.j2()` → `planet.j2`

- [ ] **Step 1: Migrate `simulation/runner.rs`**

Key changes:
- Line 1165: `planet: &Planet` → `planet: &PlanetConfig` (in `compute_derivatives`)
- Line 869: `planet: &Planet` → `planet: &PlanetConfig` (in `compute_trajectory_record`)
- Line 973: `planet: &Planet` → `planet: &PlanetConfig` (in `integrate_step`)
- Line 1013: `planet: &Planet` → `planet: &PlanetConfig` (in `integrate_adaptive`)
- Line 414: `let planet = &config.planet;` — no change needed (already field access)
- Line 415: `planet.equatorial_radius()` → `planet.equatorial_radius`
- Line 744: `planet.mu()` → `planet.mu`
- Line 892: `planet.mu()` → `planet.mu`
- Line 1209: `planet.omega()` → `planet.omega`

- [ ] **Step 2: Migrate `orbit/elements.rs`**

- Line 20: `planet.mu()` → `planet.mu`
- Line 21: `planet.equatorial_radius()` → `planet.equatorial_radius`
- Function signature: `planet: &Planet` → `planet: &PlanetConfig`

- [ ] **Step 3: Migrate `orbit/maneuver.rs`**

- Line 36: `planet.mu()` → `planet.mu`
- Line 37: `planet.equatorial_radius()` → `planet.equatorial_radius`
- Function signature: `planet: &Planet` → `planet: &PlanetConfig`

- [ ] **Step 4: Migrate `gnc/navigation/coordinates.rs`**

- Lines 15-16: `planet.equatorial_radius()` / `planet.polar_radius()` → field access
- Lines 82-83: same
- Line 198: `planet.omega()` → `planet.omega`
- Line 227: `planet.mu()` → `planet.mu`
- All function signatures: `&Planet` → `&PlanetConfig`

- [ ] **Step 5: Migrate all guidance modules**

For each of `ftc.rs`, `equilibrium_glide.rs`, `fnpag.rs`, `neural.rs`, `lateral.rs`, `energy_controller.rs`, `predguid.rs`, `piecewise_constant.rs`:

- Replace `use crate::config::Planet;` with `use crate::config::PlanetConfig;`
- Replace `planet: &Planet` with `planet: &PlanetConfig` in function signatures
- Replace all method calls with field access

- [ ] **Step 6: Migrate `gnc/navigation/estimator.rs`**

- Replace `Planet` import and any `&Planet` parameter types

- [ ] **Step 7: Update `lib.rs` re-exports (if any)**

Check if `src/rust/src/lib.rs` re-exports `Planet`. If so, change to `PlanetConfig`.

- [ ] **Step 8: Remove the `Planet` enum**

In `src/rust/src/config.rs`, delete:
- The `Planet` enum definition (lines 14-21)
- The entire `impl Planet` block (lines 23-73)

- [ ] **Step 9: Verify full crate compiles**

Run: `cd src/rust && cargo check 2>&1`

Expected: Warnings about unused imports in test modules (we'll fix those next), but no errors.

- [ ] **Step 10: Commit**

```bash
git add src/rust/
git commit -m "refactor: replace Planet enum with PlanetConfig across all call sites"
```

---

### Task 6: Update all Rust tests

**Files:**
- Modify: inline `#[cfg(test)]` modules in every file from Task 5
- Modify: `src/rust/tests/config_loading.rs`
- Modify: `src/rust/tests/error_paths.rs`
- Modify: `src/rust/tests/e2e.rs` (if it references Planet)
- Modify: `src/rust/tests/guidance_regression.rs` (if it references Planet)

- [ ] **Step 1: Update inline unit tests in all production modules**

For every file that has `#[cfg(test)]` with `Planet::Mars` or `Planet::Earth`:

- Replace `use crate::config::Planet;` with `use crate::config::PlanetConfig;`
- Replace `Planet::Mars` with `PlanetConfig::mars()`
- Replace `Planet::Earth` with `PlanetConfig::earth()`
- Replace `Planet::Moon` with `PlanetConfig::moon()`
- Replace method calls (`.mu()`, `.equatorial_radius()`, etc.) with field access

Files to update (from exploration — those with `Planet::` in tests):
- `src/rust/src/orbit/elements.rs`
- `src/rust/src/orbit/maneuver.rs`
- `src/rust/src/gnc/navigation/coordinates.rs`
- `src/rust/src/gnc/navigation/estimator.rs`
- `src/rust/src/gnc/guidance/ftc.rs`
- `src/rust/src/gnc/guidance/neural.rs`
- `src/rust/src/gnc/guidance/lateral.rs`
- `src/rust/src/gnc/guidance/equilibrium_glide.rs`
- `src/rust/src/gnc/guidance/energy_controller.rs`
- `src/rust/src/gnc/guidance/predguid.rs`
- `src/rust/src/gnc/guidance/fnpag.rs`

Note: In `error_paths.rs`, the line `const PLANET: Planet = Planet::Mars;` cannot use `const` with `PlanetConfig` (it contains a `String`). Replace with a helper function or `lazy_static`, or inline `PlanetConfig::mars()` at each usage point. Simplest: replace `const PLANET: Planet = Planet::Mars;` with a function:
```rust
fn planet() -> PlanetConfig {
    PlanetConfig::mars()
}
```
And replace all `&PLANET` with `&planet()` in that file.

- [ ] **Step 2: Update `src/rust/tests/config_loading.rs`**

Replace:
```rust
use aerocapture::config::{Planet, SimInput};
```
With:
```rust
use aerocapture::config::{PlanetConfig, SimInput};
```

Replace assertions like:
```rust
assert_eq!(config.planet, Planet::Mars);
```
With:
```rust
assert_eq!(config.planet.name, "mars");
```

- [ ] **Step 3: Update `src/rust/tests/error_paths.rs`**

Replace `const PLANET: Planet = Planet::Mars;` with `fn planet() -> PlanetConfig { PlanetConfig::mars() }` and adjust all usages.

- [ ] **Step 4: Run unit tests**

Run: `cd src/rust && cargo test --lib 2>&1 | tail -20`

Expected: All unit tests PASS.

- [ ] **Step 5: Run integration tests**

Run: `cd src/rust && cargo test --test config_loading --test error_paths 2>&1`

Expected: PASS (config_loading may need the TOML configs updated first — if it loads test configs that inherit from mission files, the `[planet]` section should flow through base inheritance).

- [ ] **Step 6: Commit**

```bash
git add src/rust/
git commit -m "refactor: update all Rust tests to use PlanetConfig"
```

---

### Task 7: Regenerate golden reference data and run full test suite

**Files:**
- Modify: `tests/reference_data/rust_golden/` (all golden CSV files)

- [ ] **Step 1: Build the release binary**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./build.sh`

Expected: Successful build of both the Rust binary and PyO3 bindings.

- [ ] **Step 2: Run the Rust check suite (fmt, clippy, unit tests)**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh 2>&1 | tail -40`

Expected: fmt OK, clippy OK, all unit tests PASS. Integration tests may fail due to stale golden data — that's expected.

- [ ] **Step 3: Regenerate golden reference data**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./tests/generate_golden.sh`

This re-runs all golden test configs and updates the reference CSV files in `tests/reference_data/rust_golden/`.

- [ ] **Step 4: Spot-check the golden data diff**

Run: `git diff --stat tests/reference_data/`

Expected: All golden CSV files show changes (small numerical differences from J3/J4). Verify the changes are small — the trajectory should shift slightly but not dramatically.

Run: `git diff tests/reference_data/rust_golden/ref/final.test_ref_orig.csv | head -20`

Look for small changes in trajectory values (velocity, position, orbital elements). Major changes would indicate a formula error.

- [ ] **Step 5: Run the full Rust integration test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -30`

Expected: All tests PASS (unit + integration + E2E).

- [ ] **Step 6: Run the Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v 2>&1 | tail -40`

Expected: All tests PASS. The PyO3 regression test (`test_pyo3_matches_subprocess`) compares PyO3 vs subprocess — both paths use the same updated binary, so they should still match.

- [ ] **Step 7: Run linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1`

Expected: ruff and mypy clean.

- [ ] **Step 8: Commit golden data**

```bash
git add tests/reference_data/
git commit -m "test: regenerate golden reference data with J3/J4 gravity"
```

---

### Task 8: Smart commit

- [ ] **Step 1: Invoke the `smart-commit` skill**

This takes the whole git branch into account, updates CLAUDE.md and README.md as needed, and creates a final commit.

---

### Task 9: Code review

- [ ] **Step 1: Invoke the `requesting-code-review` skill**

Review the completed work against the design spec.
