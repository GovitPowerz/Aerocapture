# Thermal Safety Limiter for Guidance

**Date**: 2026-04-01
**Scope**: IMPROVEMENTS.md section 2.1 -- heat rate and heat load as active guidance constraints

## Problem

Heat flux and cumulative heat load are tracked during simulation (instantaneous heat flux computed every integration step, cumulative load integrated as state[6]), but guidance is thermally blind. `NavigationOutput` has no thermal fields. The GA cost function penalizes violations after the simulation, but guidance cannot react during flight. The `data.constraints` struct is loaded from TOML but never referenced by any guidance code.

## Approach

A **safety limiter** that overrides bank angle toward full lift-up when approaching thermal limits. Not active thermal management -- guidance does its normal thing; the limiter only intervenes near constraint boundaries.

- Two independent limiters: heat flux (instantaneous spikes) and heat load (sustained exposure). Most restrictive wins.
- Smooth ramp from activation threshold to full override at 100% of limit. No discontinuities.
- 4 GA-tunable parameters per scheme: activation thresholds and ramp exponents for each limiter.
- Applies to the 5 unsigned-magnitude schemes (FTC, EqGlide, EnergyController, PredGuid, FNPAG).
- NN gets thermal margin fractions as 2 extra inputs (learns its own avoidance policy via GA).
- Piecewise Constant stays GA-only (cost function penalties sufficient).
- Backward compatible: default activation = 1.0 means the limiter is inert unless configured.

## Design

### 1. Thermal limiter module (`src/rust/src/gnc/guidance/thermal_limiter.rs`)

New file with a pure function and its parameter struct:

```rust
pub struct ThermalLimiterParams {
    pub heat_flux_activation: f64,       // fraction of max (0.6-0.95)
    pub heat_load_activation: f64,       // fraction of max (0.6-0.95)
    pub heat_flux_ramp_exponent: f64,    // ramp shape (1.0=linear, 2.0=quadratic)
    pub heat_load_ramp_exponent: f64,    // ramp shape
}
```

Default: all activations = 1.0 (never triggers), exponents = 1.0 (linear).

```rust
pub fn apply_thermal_limit(
    cos_bank_cmd: f64,
    heat_flux_fraction: f64,
    heat_load_fraction: f64,
    params: &ThermalLimiterParams,
) -> f64
```

Ramp logic (identical for both limiters):

```
fraction = q / q_max   (precomputed, passed as heat_flux_fraction or heat_load_fraction)

if fraction < activation:
    alpha = 0.0                                             -- no intervention
elif fraction >= 1.0:
    alpha = 1.0                                             -- full override
else:
    alpha = ((fraction - activation) / (1.0 - activation)) ^ exponent
```

Final blending:

```
alpha_max = max(alpha_flux, alpha_load)
cos_bank_limited = (1 - alpha_max) * cos_bank_cmd + alpha_max * 1.0
```

Where `1.0 = cos(0 deg) = full lift-up`. Output is always in [cos_bank_cmd, 1.0], which maps to a bank angle between the commanded value and 0 deg (maximum lift-up).

### 2. NavigationOutput extension (`estimator.rs`)

Two new fields:

```rust
pub heat_flux_fraction: f64,   // current_heat_flux / max_heat_flux
pub heat_load_fraction: f64,   // cumulative_heat_load / max_heat_load
```

Computed in `runner.rs` before calling `guidance_step()`. The runner already computes instantaneous heat flux (for peak tracking) and has `state[6]` (cumulative load) and `data.constraints`. If the constraint is 0.0 (not set), the fraction stays 0.0 -- limiter sees no threat.

### 3. Guidance dispatch integration (`ftc.rs`)

The limiter is called after the longitudinal bank angle is computed, before lateral guidance, for unsigned-magnitude schemes only:

```rust
// After scheme-specific bank angle computed (line ~173)
// Before skip_lateral check (line ~177)

if !skip_lateral && longitudinal_active == 1 {
    let cos_bank = bank_angle_longitudinal.cos();
    let cos_limited = thermal_limiter::apply_thermal_limit(
        cos_bank,
        nav.heat_flux_fraction,
        nav.heat_load_fraction,
        &data.guidance.thermal_limiter,
    );
    bank_angle_longitudinal = cos_limited.acos();
}
```

- `!skip_lateral` identifies unsigned-magnitude schemes (same guard used for lateral guidance)
- Only when `longitudinal_active == 1` (no intervention during securization or reference trajectory)
- Works in cos(bank) space, consistent with all feedback schemes
- `acos` result is always a valid bank angle magnitude in [0, pi]

### 4. NN thermal inputs (`neural.rs`)

Extend input vector from 6 to 8:

```rust
let input = [
    orbit.eccentricity - 1.0,
    (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0,
    2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0,
    -mu / (2.0 * orbit.semi_major_axis) / 6e6,
    (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0,
    accel_mag / 20.0 - 1.0,
    nav.heat_flux_fraction * 2.0 - 1.0,   // NEW: [-1, 1]
    nav.heat_load_fraction * 2.0 - 1.0,   // NEW: [-1, 1]
];
```

Normalization to [-1, 1] matches the existing convention. At fraction 0.0 (no heating) -> -1.0. At fraction 1.0 (at limit) -> +1.0.

**Breaking change**: NN input dimension changes from 6 to 8. Existing trained models need retraining. Old model JSON files will fail to load with a dimension mismatch (correct behavior -- no silent corruption).

The NN does NOT get the thermal limiter ramp. It learns its own thermal avoidance from the GA cost function. The fractions give it information; the GA teaches it policy.

### 5. Rust data structures and TOML config

**`guidance_params.rs`**: `ThermalLimiterParams` struct (as defined in section 1) + new field `thermal_limiter: ThermalLimiterParams` in `GuidanceParams`.

**`Constraints` struct** (`data/mod.rs`): Add `max_heat_load: f64` (J/m2, converted from kJ/m2 in TOML). Currently missing from the struct despite being defined in mission TOMLs.

**`config.rs`**: Parse optional `[guidance.thermal_limiter]` section. If absent, use defaults (limiter inactive). Parse `max_heat_load` from `[flight.constraints]` into the `Constraints` struct.

**TOML surface**: New optional section in training configs:

```toml
[guidance.thermal_limiter]
heat_flux_activation = 0.8
heat_load_activation = 0.85
heat_flux_ramp_exponent = 1.5
heat_load_ramp_exponent = 2.0
```

No mission TOML changes needed -- constraint limits are already defined.

### 6. Python GA integration

**`param_spaces.py`**: New shared parameter list:

```python
_THERMAL_LIMITER_PARAMS: list[ParamSpec] = [
    ParamSpec("thermal.heat_flux_activation", 0.6, 0.95, 1.0),
    ParamSpec("thermal.heat_load_activation", 0.6, 0.95, 1.0),
    ParamSpec("thermal.heat_flux_ramp_exponent", 0.5, 3.0, 1.0),
    ParamSpec("thermal.heat_load_ramp_exponent", 0.5, 3.0, 1.0),
]
```

Appended to the 5 unsigned-magnitude scheme param spaces alongside `_LATERAL_PARAMS` and `_EXIT_PARAMS`. Not added to `piecewise_constant` or `neural_network`.

Default activation of 1.0 means the GA starts with the limiter disabled and discovers whether thermal protection improves fitness.

**`evaluate.py`**: Route `thermal.*` prefixed params to `[guidance.thermal_limiter]` in the TOML override dict, following the same pattern as `exit.*` -> `[guidance.ftc]` and `lateral.*` -> `[guidance.lateral]`.

**NN training config**: Update default `layer_sizes` from `[6, 12, 2]` to `[8, 12, 2]`. No new ParamSpecs -- the GA already optimizes all weights/biases, which now include 2 extra input weights per hidden neuron.

### 7. Testing

**Rust unit tests** (`thermal_limiter.rs`, inline `#[cfg(test)]`):

- Below activation: returns `cos_bank_cmd` unchanged
- At 100% of limit: returns 1.0 (full lift-up)
- Mid-ramp: correct interpolation for linear (exp=1) and quadratic (exp=2)
- Both active, most restrictive wins: verify `alpha_max` picks the right one
- Zero constraint (fraction=0.0): limiter does nothing
- Proptest: random inputs -> output always in [cos_bank_cmd, 1.0], always finite

**Rust integration tests**: Run a sim with thermal limiter active, verify peak heat flux in final record is closer to (or below) the limit compared to without limiter.

**Python tests**:

- `_THERMAL_LIMITER_PARAMS` present in the 5 unsigned-magnitude scheme param spaces
- `thermal.*` overrides route to correct TOML section
- NN input dimension change (8 inputs) reflected in config/chromosome factories

**Backward compatibility**: Default activation=1.0 makes the limiter inert. Existing regression tests pass unchanged. No new golden reference data needed.

## Files modified

| File | Change |
|------|--------|
| `src/rust/src/gnc/guidance/thermal_limiter.rs` | **NEW** -- limiter module |
| `src/rust/src/gnc/guidance/mod.rs` | Add `pub mod thermal_limiter` |
| `src/rust/src/gnc/navigation/estimator.rs` | Add 2 fields to `NavigationOutput` |
| `src/rust/src/simulation/runner.rs` | Compute thermal fractions, populate `NavigationOutput` |
| `src/rust/src/gnc/guidance/ftc.rs` | Call limiter after longitudinal bank angle |
| `src/rust/src/gnc/guidance/neural.rs` | Extend input vector from 6 to 8 |
| `src/rust/src/data/guidance_params.rs` | Add `ThermalLimiterParams` struct + field |
| `src/rust/src/data/mod.rs` | Add `max_heat_load` to `Constraints` |
| `src/rust/src/config.rs` | Parse `[guidance.thermal_limiter]` + `max_heat_load` |
| `src/python/aerocapture/training/param_spaces.py` | Add `_THERMAL_LIMITER_PARAMS` |
| `src/python/aerocapture/training/evaluate.py` | Route `thermal.*` overrides |
| NN training TOML | Update `layer_sizes` default to `[8, 12, 2]` |
| `tests/` | New unit + integration + Python tests |
