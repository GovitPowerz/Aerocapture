# Density Estimation Improvements Design

## Summary

Two improvements to the navigation density estimation pipeline:

1. **3.1**: Harden the legacy bias-mode density filter with rate-of-change limiting and gain saturation bounds
2. **3.2**: Correct drag acceleration extraction by accounting for lift projection at angle of attack

## Motivation

The legacy bias-mode density filter has no bounds on `density_gain` and no protection against measurement spikes. A single bad sample at lambda=0.8 shifts the filter 80% toward the outlier. The EKF mode already has equivalent protections (density correction clamped to [0.1, 10.0], Kalman gain-weighted updates), but legacy mode lacks them.

The drag acceleration extraction in both navigation modes uses `rho = 2*|a|*m / (Cx*S*V^2)`, which assumes the measured acceleration is pure drag. In a body-frame accelerometer (or any realistic measurement), the x-axis reads both drag and lift projections: `a_x = (rho*S*V^2 / 2m) * (Cx*cos(alpha) + Cz*sin(alpha))`. At AoA=10 deg with L/D=0.12, the error is ~4%. At AoA=20 deg with L/D=0.24, it reaches ~16%.

## Scope

| Improvement | Legacy bias mode | EKF mode |
|-------------|-----------------|----------|
| 3.1 Rate limiting + gain saturation | Yes | No (already has equivalent) |
| 3.2 Lift-corrected drag extraction | Yes | Yes |

## 3.1 Legacy Bias-Mode Filter Hardening

### Current behavior

```
estimator.rs lines 162-169:

lambda = clamp(density_filter_gain + bias, 0.01, 0.99)
density_gain = (1 - lambda) * density_gain + lambda * (rho_est / rho_model)
if alt > 100 km: density_gain = 1.0
```

No bounds on `density_gain`, no outlier protection.

### New behavior

Two additions, applied in order after the exponential filter update:

**A. Rate-of-change limiting** -- clamp the per-step change in `density_gain`:

```
raw_gain = (1 - lambda) * density_gain + lambda * (rho_est / rho_model)
delta = clamp(raw_gain - density_gain, -max_delta, +max_delta)
density_gain = density_gain + delta
```

One new TOML parameter: `density_gain_max_delta` in `[guidance]` section, default 0.1. This limits density_gain movement to +/-0.1 per navigation step. At 10 Hz navigation rate, the filter can track a 2x density change in ~10 steps (1 second) -- fast enough for real atmospheric variability, slow enough to reject single-sample spikes.

**B. Gain saturation** -- hardcoded clamp on `density_gain` to [0.1, 10.0]:

```
density_gain = clamp(density_gain, 0.1, 10.0)
```

Applied after rate limiting, before the high-altitude reset. Matches the EKF density correction factor bounds. Hardcoded (not configurable) because this is a safety net, not a tuning knob.

### Combined filter update

```rust
// Exponential filter
let raw_gain = (1.0 - lambda) * nav_state.density_gain
    + lambda * (density_estimated / rho_model);

// Rate-of-change limiting
let max_delta = data.guidance.density_gain_max_delta;
let delta = (raw_gain - nav_state.density_gain).clamp(-max_delta, max_delta);
nav_state.density_gain += delta;

// Gain saturation (hardcoded safety bounds, matches EKF)
nav_state.density_gain = nav_state.density_gain.clamp(0.1, 10.0);

// High-altitude reset (unchanged)
if alt_est > 100e3 {
    nav_state.density_gain = 1.0;
}
```

### Configuration

New field in `GuidanceParams`:

```rust
pub density_gain_max_delta: f64,  // max per-step change in density_gain
```

Parsed from `[guidance] density_gain_max_delta` in TOML, default 0.1 if absent. GA-optimizable (add to `param_spaces.py` bounds for schemes that use bias-mode navigation).

### MC dispersions

No new dispersion parameter. The `density_gain_max_delta` is a filter design parameter, not a physical quantity with uncertainty. The existing `filter_gain_bias` dispersion on lambda already exercises the filter robustness.

## 3.2 Lift-Corrected Drag Extraction

### Physics

An accelerometer along the vehicle's body x-axis measures the total non-gravitational specific force projected onto that axis. With angle of attack alpha between the body x-axis and the velocity vector:

```
a_body_x = (D * cos(alpha) + L * sin(alpha)) / m
         = (rho * S * V^2 / 2m) * (Cx * cos(alpha) + Cz * sin(alpha))
```

Where D = q*S*Cx (drag force, along -V) and L = q*S*Cz (lift force, perpendicular to V).

**Bank angle does not affect this projection.** Bank angle rotates the lift vector around the velocity axis, but the body x-axis is always in the plane containing V and the body longitudinal axis. The projection onto body-x depends only on alpha.

### Current formula (both modes)

```
rho_est = 2 * |a_meas| * m / (Cx * S * V^2)
```

This assumes `a_meas = a_drag`, ignoring the lift term `Cz * sin(alpha)`.

### Corrected formula (both modes)

```
denom = Cx * cos(alpha) + Cz * sin(alpha)
rho_est = 2 * |a_meas| * m / (denom * S * V^2)
```

The denominator now accounts for both drag and lift projections onto the measurement axis. When alpha=0, this reduces to the current formula (cos(0)=1, sin(0)=0).

Guard: if `|denom| < 1e-10`, fall back to the current Cx-only formula. This threshold is more conservative than the existing Cx guard (1e-30) because the denominator combines two terms that could partially cancel. Physically unlikely for capsule-class vehicles (Cx >> |Cz|*sin(alpha)) but safe to guard.

### Truth model changes

Both navigation modes must also update their truth models to generate the corrected acceleration, so the measurement includes the lift component:

**Legacy bias mode** (`navigate()`, line 130-132):

Current:
```rust
let acdrag_true = rho_true * ref_area_true * cx_true * V^2 / (2.0 * mass_true);
let drag_acceleration_measured = acdrag_true + biases.drag;
```

New:
```rust
let cz_true = data.aero.interpolate_cz(aoa_commanded + run_incidence_bias) * (1.0 + run_cz_bias);
let aoa_true = aoa_commanded + run_incidence_bias;
let accel_body_x_true = rho_true * ref_area_true * V^2 / (2.0 * mass_true)
    * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());
let accel_measured = accel_body_x_true + biases.drag;
```

Note: `_run_cz_bias` (currently unused with underscore prefix) becomes `run_cz_bias` (active).

**EKF mode** (`navigate_ekf()`, line 420-423):

Current:
```rust
let drag_accel_true = aero_factor_true * cx_true;
let true_accel = [drag_accel_true, 0.0, 0.0];
```

New:
```rust
let cz_true = data.aero.interpolate_cz(aoa_commanded + run_incidence_bias) * (1.0 + run_cz_bias);
let aoa_true = aoa_commanded + run_incidence_bias;
let accel_body_x = aero_factor_true * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());
let true_accel = [accel_body_x, 0.0, 0.0];
```

Note: `run_cz_bias` must be added to `navigate_ekf()` function signature and its call site.

### Estimation changes (both modes)

Replace the density inversion in both `navigate()` (line 147-153) and `navigate_ekf()` (line 457-462):

```rust
let aoa = aoa_commanded;  // onboard estimate
let denom = cx_est * aoa.cos() + cz_est * aoa.sin();

let density_estimated = if denom.abs() > 1e-10 && velocity_relative.abs() > 1e-10 {
    2.0 * accel_measured.abs() * data.capsule.mass
        / (denom * data.capsule.reference_area * velocity_relative * velocity_relative)
} else {
    0.0
};
```

### Error magnitude

| AoA (deg) | Cx | Cz | Current denom | Corrected denom | Error (%) |
|-----------|----|----|---------------|-----------------|-----------|
| 0 | 1.5 | 0.0 | 1.50 | 1.50 | 0.0 |
| 10 | 1.6 | -0.2 | 1.60 | 1.541 | 3.8 |
| 20 | 1.7 | -0.4 | 1.70 | 1.461 | 16.4 |

At typical MSR AoA (~10 deg), the correction is ~4%. At higher AoA the error grows rapidly.

## Files Modified

| File | Change |
|------|--------|
| `src/rust/src/gnc/navigation/estimator.rs` | 3.1: rate limit + saturation in `navigate()`. 3.2: lift-corrected truth + inversion in both `navigate()` and `navigate_ekf()`. Remove underscore from `_run_cz_bias`. Add `run_cz_bias` to `navigate_ekf()` signature. |
| `src/rust/src/data/guidance_params.rs` | Add `density_gain_max_delta: f64` field |
| `src/rust/src/config.rs` | Parse `density_gain_max_delta` from TOML with default 0.1 |
| `src/rust/src/simulation/runner.rs` | Pass `run_cz_bias` to `navigate_ekf()` call site |
| `src/python/aerocapture/training/param_spaces.py` | Add `density_gain_max_delta` bounds for relevant schemes |

## Testing

### Unit tests (estimator.rs)

- **Rate limiting**: verify density_gain delta is clamped to +/-max_delta per step
- **Gain saturation**: verify density_gain stays within [0.1, 10.0] for extreme inputs
- **Rate limit + saturation interaction**: verify rate limiting applies first, then saturation
- **Lift correction at alpha=0**: verify corrected formula equals current formula (cos(0)=1, sin(0)=0)
- **Lift correction at alpha=10 deg**: verify density_estimated differs from uncorrected by expected ~4%
- **Lift correction at alpha=20 deg**: verify ~16% correction
- **Denominator guard**: verify fallback to Cx-only when denom near zero

### Property-based tests (proptest)

- density_gain always in [0.1, 10.0] after any sequence of filter updates
- density_estimated >= 0 for any valid inputs
- Lift correction factor always >= 0.5 (prevents sign flip) for realistic alpha range

### Regression

- Existing density filter tests must pass (behavior changes only for non-zero AoA cases)
- Run nominal MC batch with default AoA profile and compare DV distribution -- expect small shift from lift correction, no degradation

## Backward Compatibility

- `density_gain_max_delta` defaults to 0.1 -- existing configs work unchanged
- Gain saturation at [0.1, 10.0] may clip extreme cases that previously diverged -- this is intentional (safety net)
- Lift correction changes density estimation at non-zero AoA -- this is a physics improvement, not a regression. At alpha=0 the formulas are identical. Validation against the legacy reference (which used alpha=0 for the FTC guided case) should be unaffected.
