# FNPAG 3D Predictor Upgrade

**Date**: 2026-04-04
**Status**: Proposed
**Scope**: `src/rust/src/gnc/guidance/fnpag.rs` (primary), golden file regeneration

## Problem

FNPAG produces significantly worse results than FTC across all metrics (DV cost, capture rate, constraint violations). The root cause is the predictor's simplified dynamics, which cause the secant method to converge to wrong bank angles:

1. **Planar model** (3-state: r, V, gamma) -- ignores latitude, heading, J2 lateral gravity
2. **No J2/J3/J4 gravity** -- uses simple mu/r^2 instead of zonal harmonic model
3. **No planet rotation** -- ignores Coriolis and centrifugal terms
4. **Euler integration** -- O(dt) global error with dt=2s over ~500s passes
5. **Exit energy bug** -- uses relative velocity (V_rel^2/2 - mu/r) instead of inertial velocity. On Mars at exit speeds, the rotation correction is O(2 MJ/kg) vs target energy of ~-6 MJ/kg -- a systematic bias of ~30%

FTC doesn't suffer from this because it tracks a pre-computed reference trajectory with feedback, never needing to predict forward. FNPAG's value proposition (continuous re-planning via forward prediction) becomes a liability when the predictions are wrong.

## Constraint

Guidance must not cheat: the predictor uses only information available to the GNC chain:
- Onboard atmosphere model (not truth table)
- Navigation-estimated state (not truth state)
- No wind knowledge
- No dispersion knowledge (no aero/mass biases)

The current code already respects this (line 93 uses `atmosphere_onboard.density_at()`). The upgrade preserves this constraint.

## Design

### 1. Expanded prediction state

Current:
```rust
struct PredState {
    r: f64,     // radius (m)
    v: f64,     // velocity (m/s)
    gamma: f64, // flight path angle (rad)
}
```

New:
```rust
struct PredState {
    r: f64,     // radius (m)
    lon: f64,   // longitude (rad)
    lat: f64,   // latitude (rad)
    v: f64,     // relative velocity (m/s)
    gamma: f64, // flight path angle (rad)
    psi: f64,   // heading/azimuth (rad)
}
```

Initialized from `NavigationOutput`, which already provides all 6 components:
```rust
let current = PredState {
    r: nav.position_estimated[0],
    lon: nav.position_estimated[1],
    lat: nav.position_estimated[2],
    v: nav.velocity_estimated[0],
    gamma: nav.velocity_estimated[1],
    psi: nav.velocity_estimated[2],
};
```

### 2. Equations of motion

The predictor derivatives match `runner.rs:1334-1351` with these differences:

- **Atmosphere**: `atmosphere_onboard.density_at(alt, &data.atmosphere)` (onboard model, no dispersions) -- same as current predictor
- **Gravity**: `gravity::gravity(r, lat, planet)` -- reuses existing J2/J3/J4 function
- **Planet rotation**: includes Coriolis and centrifugal terms via `planet.omega`
- **Lateral lift**: set to zero (`sin_bank = 0`). The predictor doesn't know the roll sign (lateral guidance decides that externally). Heading and latitude still evolve from J2 lateral gravity, Coriolis, and centrifugal forces. This is physically correct for energy prediction: roll reversals redistribute drag exposure over time but don't significantly change total energy dissipation.
- **No winds**: guidance has no wind knowledge
- **No dispersions**: uses nominal Cx/Cz at `data.entry.initial_aoa`, nominal mass and reference area
- **No heat flux**: not needed for energy prediction (6 derivatives instead of 8)

The derivative function is self-contained within `fnpag.rs` (private helper), not a call to `runner.rs::compute_derivatives` which requires `RunState` with dispersion data.

### 3. RK4 integration

Replace Euler with classic 4th-order Runge-Kutta:
- 4 derivative evaluations per step (vs 1 for Euler)
- O(dt^4) global error vs O(dt) -- orders of magnitude more accurate at the same dt
- Default `prediction_dt` stays at 2.0s (GA-tunable via existing `FnpagParams`)
- `max_steps` stays at 2000

Cost: ~4x more derivative evaluations per prediction, but each derivative is slightly more expensive (6 states + gravity call). Net predictor cost increase: ~5-6x. Since the predictor is a small fraction of overall simulation cost (the main integration loop dominates), this is acceptable. In the worst case (8 predictions per guidance tick * 2000 steps * 4 evals = 64,000 derivative evals), this is still fast for a single guidance call.

### 4. Exit energy correction

Replace:
```rust
let energy = s.v * s.v / 2.0 - mu / s.r;
```

With:
```rust
let energy = total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet);
```

This reuses the existing `coordinates::total_energy()` function which converts relative velocity to inertial velocity before computing E = V_abs^2/2 - mu/r. The same function the rest of the codebase uses. This fixes the ~30% systematic energy bias at Mars exit conditions.

The fix applies to both exit paths in the predictor:
- **Atmosphere exit** (successful prediction): `total_energy(s.r, s.lon, s.lat, s.v, s.gamma, s.psi, planet)`
- **Timeout** (max_steps reached): same `total_energy()` call

Crash and velocity-collapse cases still return the 1e8 penalty (unchanged -- these signal failure, not energy).

### 5. No changes to fnpag_bank or secant method

The `fnpag_bank` function structure is unchanged:
- Same initialization logic (two-point bracket at 40 deg and 90 deg)
- Same secant iteration (up to 5 iterations with convergence check)
- Same best-result tracking
- Same bank angle clamping

The only difference is that `predict_exit_energy` now returns more accurate predictions, so the secant method converges to better bank angles.

### 6. No new TOML parameters or config changes

All existing `FnpagParams` fields are unchanged:
- `energy_tol` -- convergence tolerance (J/kg)
- `prediction_dt` -- forward prediction timestep (s)
- `bank_min_deg`, `bank_max_high_deg`, `bank_max_low_deg` -- bank limits (deg)

No new config keys needed. The GA parameter space for FNPAG is unchanged.

## Files changed

| File | Change |
|------|--------|
| `src/rust/src/gnc/guidance/fnpag.rs` | Expand PredState, rewrite predict_exit_energy internals, add imports |

No other source files modified.

## Backward compatibility

- FNPAG bank angle commands will differ (better), so trajectories change
- **Golden file**: `tests/reference_data/rust_golden/` FNPAG golden CSV needs regeneration
- **Trained params**: `training_output/fnpag/best_params.json` needs retraining
- **Other schemes**: completely unaffected (FTC, EqGlide, EnergyCtrl, PredGuid, NN, PiecewiseConstant)

## Testing

### Existing tests (should still pass)

- `low_density_returns_previous_bank` -- early exit before predictor, unchanged
- `first_call_initializes_state` -- still picks one of two init brackets
- `output_finite_for_typical_state` -- finite + bounded invariant holds
- `second_call_produces_finite_output` -- secant path still finite
- Proptest `output_always_finite_and_bounded` -- same invariant

### New tests

1. **Exit energy uses inertial velocity**: predict from a known state, verify the returned energy matches `total_energy()` at the predicted exit point (not V_rel^2/2 - mu/r)
2. **J2 sensitivity**: compare predicted exit energy at high latitude (60 deg) vs equator (0 deg) for the same entry speed/FPA. J2 lateral gravity should produce measurably different predictions
3. **Proptest**: same invariants as current -- finite, bounded output for valid atmospheric states (already exists, will exercise the new code path)

### Validation

After implementation, retrain FNPAG via GA and compare MC performance against FTC:
- DV cost distribution
- Capture rate
- Constraint violation rates (thermal, g-load)

This is the real measure of whether the upgrade closes the gap with FTC.

## Implementation notes

- The derivative function is private to `fnpag.rs` -- it doesn't call `runner.rs::compute_derivatives` because that requires `RunState` (dispersion biases). Instead, write a focused `pred_derivatives` helper that takes `PredState`, `bank_angle`, `PlanetConfig`, `SimData` and returns `[f64; 6]`.
- For `geodetic_from_spherical` (needed to get altitude from radius for atmosphere lookup), import from `coordinates.rs`. Already used in the current `fnpag_bank` function.
- The RK4 implementation is a simple inline loop (not the Gill variant from `rk4.rs` which has a different API). ~20 lines of straightforward code.
