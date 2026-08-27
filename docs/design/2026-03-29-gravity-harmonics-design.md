# Higher-Order Gravity Harmonics (J3, J4) Design

**Date:** 2026-03-29
**Status:** Approved
**Scope:** Add J3/J4 zonal harmonics to gravity model, move all planet constants from hard-coded Rust enum to TOML configuration

## Problem

The current gravity model (`physics/gravity.rs`) uses J2 only. Planet constants (mu, radii, omega, J2) are hard-coded in a `Planet` enum in `config.rs`, requiring a Rust rebuild to change planets or tweak constants.

## Solution

1. Replace the `Planet` enum with a `PlanetConfig` struct parsed from a new TOML `[planet]` section.
2. Extend the gravity function with J3 and J4 zonal harmonic terms.
3. Create planet preset files (`configs/planets/*.toml`) that mission configs inherit via `base`.

## `PlanetConfig` Struct

```rust
/// Planet physical constants, parsed from TOML [planet] section.
pub struct PlanetConfig {
    pub name: String,           // display label ("mars", "earth", etc.)
    pub mu: f64,                // gravitational parameter GM (m^3/s^2)
    pub equatorial_radius: f64, // equatorial radius (m)
    pub polar_radius: f64,      // polar radius (m)
    pub omega: f64,             // sidereal rotation rate (rad/s)
    pub j2: f64,                // zonal harmonic J2
    pub j3: f64,                // zonal harmonic J3 (default 0.0)
    pub j4: f64,                // zonal harmonic J4 (default 0.0)
}
```

- `j3` and `j4` use `#[serde(default)]` so omitting them gives J2-only behavior.
- Public fields, no getters -- plain data struct.

## TOML Schema

New `[planet]` section in mission TOML files:

```toml
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

The old `[mission] planet = "mars"` field is removed. The `[mission]` section retains only `type`.

## Planet Preset Files

New directory `configs/planets/` with one file per planet. Mission configs inherit via `base`:

```
configs/planets/
  mars.toml       -- Mars constants (GMM-3 / Konopliv et al.)
  earth.toml      -- Earth constants (WGS-84 / EGM96)
  moon.toml       -- Moon constants (legacy Fortran values, J3=J4=0)
  jupiter.toml    -- Jupiter constants (Juno / Iess et al. 2018)
```

### Reference Values

**Mars** (GMM-3 gravity model, Genova et al. 2016 / Konopliv et al. 2016):

| Constant | Value | Unit |
|----------|-------|------|
| mu | 4.282829e13 | m^3/s^2 |
| equatorial_radius | 3393940.0 | m |
| polar_radius | 3376780.0 | m |
| omega | 7.088218e-5 | rad/s |
| J2 | 1.958616e-3 | -- |
| J3 | 3.145e-5 | -- |
| J4 | -1.538e-5 | -- |

**Earth** (WGS-84 / EGM96):

| Constant | Value | Unit |
|----------|-------|------|
| mu | 3.98600418e14 | m^3/s^2 |
| equatorial_radius | 6378137.0 | m |
| polar_radius | 6356784.0 | m |
| omega | 7.292115e-5 | rad/s |
| J2 | 1.08263e-3 | -- |
| J3 | -2.5327e-6 | -- |
| J4 | -1.6196e-6 | -- |

**Moon** (legacy Fortran values; J3/J4 set to 0.0):

| Constant | Value | Unit |
|----------|-------|------|
| mu | 3.249e14 | m^3/s^2 |
| equatorial_radius | 6051800.0 | m |
| polar_radius | 6051800.0 | m |
| omega | 2.9924e-7 | rad/s |
| J2 | 4.458e-6 | -- |
| J3 | 0.0 | -- |
| J4 | 0.0 | -- |

Note: The Moon values in the codebase do not match real lunar constants (the real Moon has R_eq ~ 1.738e6 m, J2 ~ 2.034e-4). These appear to be legacy Fortran placeholders. Preserving as-is to avoid unrelated scope creep.

**Jupiter** (Juno, Iess et al. 2018):

| Constant | Value | Unit |
|----------|-------|------|
| mu | 1.26686e17 | m^3/s^2 |
| equatorial_radius | 71492000.0 | m |
| polar_radius | 66854000.0 | m |
| omega | 1.759e-4 | rad/s |
| J2 | 1.4736e-2 | -- |
| J3 | -4.2e-8 | -- |
| J4 | -5.866e-4 | -- |

### Inheritance Chain

Before:
```
configs/training/msr_aller_eqglide_train.toml
  base = ["../missions/mars.toml", "common.toml"]
  # mars.toml has [mission] planet = "mars"
```

After:
```
configs/planets/mars.toml          -- [planet] section only
configs/missions/mars.toml
  base = ["../planets/mars.toml"]  -- inherits [planet]
  # [mission] section has type only, no planet field
configs/training/msr_aller_eqglide_train.toml
  base = ["../missions/mars.toml", "common.toml"]  -- unchanged
```

Training/test configs don't change -- they inherit `[planet]` transitively through the mission file.

## Gravity Function

The `gravity()` signature changes from `&Planet` to `&PlanetConfig`. J3 and J4 terms are added using standard zonal harmonic expressions derived from Legendre polynomials.

### Radial Component (gravtr)

Convention: `gravtr` is positive inward (magnitude of gravitational pull), i.e. `gravtr = -g_r` where `g_r` is the standard outward-positive radial acceleration. Derived from `gravtr_Jn = -(n+1) * mu * Jn * R_eq^n * Pn(sin(lat)) / r^{n+2}`, negated.

```
gravtr = mu/r^2
       + (3*mu*J2*R_eq^2) / (2*r^4) * (1 - 3*sin^2(lat))
       + (2*mu*J3*R_eq^3) / (r^5)   * sin(lat) * (3 - 5*sin^2(lat))
       - (5*mu*J4*R_eq^4) / (8*r^6) * (3 - 30*sin^2(lat) + 35*sin^4(lat))
```

### Lateral Component (gravtl)

Convention: `gravtl = -g_lat` where `g_lat = -(1/r) * dU/dlat`. Derived from `dPn/dlat` chain rule.

```
gravtl = (3*mu*J2*R_eq^2) / (r^4) * sin(lat) * cos(lat)
       + (3*mu*J3*R_eq^3) / (2*r^5) * cos(lat) * (5*sin^2(lat) - 1)     [NOTE: NOT antisymmetric]
       - (5*mu*J4*R_eq^4) / (2*r^6) * sin(lat) * cos(lat) * (3 - 7*sin^2(lat))
```

### Properties

- When J3=J4=0, returns exactly the current J2-only result.
- J3 breaks north-south symmetry (odd harmonic): gravtl(lat) != -gravtl(-lat).
- J4 preserves symmetry (even harmonic): its contribution to gravtl is antisymmetric.
- J3 and J4 corrections are small relative to J2 (Mars: J3/J2 ~ 1.6%, J4/J2 ~ 0.8%).

### FNPAG / EqGlide Predictors

No change. These continue using simplified g = mu/r^2 for computational speed. The impact note from the improvement list says gravity effects are "minor for aerocapture" during the atmospheric pass, and these predictors prioritize speed.

## Migration

### Removing the `Planet` Enum

The `Planet` enum is removed entirely. All ~15 production call sites switch from `&Planet` to `&PlanetConfig`:

| Old | New |
|-----|-----|
| `planet.mu()` | `planet.mu` |
| `planet.equatorial_radius()` | `planet.equatorial_radius` |
| `planet.polar_radius()` | `planet.polar_radius` |
| `planet.omega()` | `planet.omega` |
| `planet.j2()` | `planet.j2` |

### Production Call Sites

| File | Methods Used | Purpose |
|------|-------------|---------|
| `physics/gravity.rs` | mu, equatorial_radius, j2 (+j3, j4) | Gravity computation |
| `simulation/runner.rs` | equatorial_radius, mu, omega | EOM derivatives, orbital energy |
| `orbit/elements.rs` | mu, equatorial_radius, omega | Orbital element computation |
| `orbit/maneuver.rs` | mu, equatorial_radius | Delta-V calculation |
| `gnc/navigation/coordinates.rs` | equatorial_radius, polar_radius, omega, mu | Geodetic transforms, energy |
| `gnc/navigation/estimator.rs` | (via coordinates) | Navigation filter |
| `gnc/guidance/ftc.rs` | (passed as &PlanetConfig) | FTC guidance |
| `gnc/guidance/equilibrium_glide.rs` | mu | Orbital energy |
| `gnc/guidance/fnpag.rs` | mu, equatorial_radius | Simplified gravity predictor |
| `gnc/guidance/neural.rs` | mu | Feature normalization |
| `gnc/guidance/energy_controller.rs` | (via reference) | Energy tracking |
| `gnc/guidance/predguid.rs` | (via reference) | Drag tracking |

### TOML Parser Changes (`config.rs`)

- Remove `Planet` enum definition and all `impl Planet` blocks.
- Add `PlanetConfig` struct with serde `Deserialize`.
- Remove the `planet = "mars"` string-to-enum match block.
- Parse `[planet]` section directly into `PlanetConfig`.
- `SimInput` stores `PlanetConfig` instead of `Planet`.

### Test Fixtures

Tests that currently use `Planet::Mars` get factory functions:

```rust
impl PlanetConfig {
    #[cfg(test)]
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

    #[cfg(test)]
    pub fn earth() -> Self { /* ... */ }
}
```

Gated behind `#[cfg(test)]` so the TOML `[planet]` section is the only production path.

## Test Strategy

### Existing Tests (Migration)

- Replace `Planet::Mars` / `Planet::Earth` with `PlanetConfig::mars()` / `PlanetConfig::earth()`.
- All existing gravity tests must pass with the same assertions (J3/J4 corrections are tiny).

### New Gravity Tests

| Test | What It Verifies |
|------|-----------------|
| `j3_j4_zero_matches_j2_only` | With J3=J4=0, output is bit-identical to the old gravity function |
| `j3_breaks_north_south_symmetry` | gravtl(lat) != -gravtl(-lat) when J3 != 0 |
| `j4_preserves_symmetry` | J4-only lateral contribution is antisymmetric in latitude |
| `j3_lateral_nonzero_at_equator` | J3 lateral component is nonzero at equator (cos(0)*(5*0-1) = -1 != 0) |
| `j3_j4_small_correction` | |J3+J4 terms| / |J2 terms| < 5% at Mars surface |
| `mars_surface_gravity_ballpark` | Still ~3.72 m/s^2 with J3/J4 active |
| `gravity_decreases_with_altitude` | Monotonic decrease still holds with higher harmonics |
| proptest: `gravity_magnitude_finite` | No NaN/Inf for any valid (r, lat, planet) inputs |

### Existing Test Updates

- `j2_lateral_symmetry` -- update to verify approximate antisymmetry (J3 breaks exact symmetry, but J3/J2 ~ 1.6% so nearly antisymmetric).
- `j2_lateral_zero_at_equator` -- no longer exactly zero because J3 contributes a cos(lat)*(5*sin^2(lat) - 1) term that equals -1*cos(0) = -1 at equator. Rename/update.
- `j2_lateral_zero_at_pole` -- J3 adds cos(pi/2)*(5*1-1) which is still ~0. Stays valid.

### Golden Regression Tests

The existing golden reference data was validated against Fortran with J2-only. Since planet preset files now include real J3/J4 values, the golden test trajectories will shift slightly. The golden reference data in `tests/reference_data/` must be regenerated.

The changes will be small (J3/J4 are ~1-2% of J2) but nonzero, so exact regression matching will fail without re-baselining.

### PyO3 Bindings

No structural changes needed. PyO3 passes TOML paths to Rust, which parses them internally. The `[planet]` section flows through the existing config loading pipeline. PyO3 regression tests need re-baselining alongside Rust golden tests.

## Out of Scope

- Venus/Titan planet presets (now trivial to add but not this PR)
- Tesseral/sectorial harmonics (Cnm, Snm for m > 0)
- FNPAG/EqGlide predictor upgrades (stay with mu/r^2)
- Full spherical harmonic expansion (degree N > 4)
- Fixing the Moon's legacy constants (separate concern)

## Sources

- Earth: WGS-84 / EGM96 (J2=1.08263e-3, J3=-2.5327e-6, J4=-1.6196e-6)
- Mars: GMM-3 / Konopliv et al. 2016 (J3~3.145e-5, J4~-1.538e-5)
- Jupiter: Juno / Iess et al. 2018 Nature (J3~-4.2e-8, J4~-5.866e-4)
- Zonal harmonic formulas: standard Legendre polynomial derivatives of the gravitational potential
