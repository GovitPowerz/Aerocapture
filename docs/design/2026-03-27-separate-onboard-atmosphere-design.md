# Separate Truth vs Onboard Atmosphere Models

**Date:** 2026-03-27
**Ref:** IMPROVEMENTS.md section 1.2, TODO.md "Simulation -- Medium Impact"

## Problem

The simulator uses a single `AtmosphereModel` for both truth physics and onboard navigation/guidance. The only difference is a Monte Carlo density bias multiplier applied to truth. This means the onboard model is unrealistically close to truth -- the navigation density filter and EKF density correction state have almost nothing to correct, and guidance schemes see perfect atmospheric knowledge.

Real onboard systems carry a pre-flight analytical atmosphere model (typically exponential fits to limited prior data), which diverges structurally from the actual atmosphere. The navigation filter's job is to estimate and correct this divergence in real time.

## Solution

Introduce a **piecewise exponential onboard atmosphere model** that coexists with the existing tabulated truth model. The onboard model is auto-fitted from the truth table at initialization, creating a realistic structural error that varies with altitude. Guidance and navigation use the degraded onboard model; physics propagation continues to use the full truth table.

## Onboard Model: Piecewise Exponential

### Structure

N altitude segments (default: 5), each with its own reference density and scale height:

```
rho_onboard(z) = rho_ref_i * exp(-(z - z_low_i) / H_i)    for z in [z_low_i, z_high_i]
```

Segment boundaries are placed at equal altitude intervals spanning the truth table range.

### Auto-Fitting

At simulation init, for each segment:
1. Sample the truth table densities at altitudes within `[z_low, z_high]`
2. Perform linear regression on `ln(rho)` vs `z` (least-squares fit in log-space)
3. Extract `H_i = -1/slope` and `rho_ref_i = exp(intercept)`

This produces a model that approximates truth on average but diverges locally -- exactly the kind of structured error a real onboard model would have.

### Continuity

Segment fits are independent (no continuity enforcement at boundaries). This is intentional: real analytical models have discontinuities at altitude band boundaries. The navigation filter should handle this gracefully. If this causes numerical issues, C0 continuity can be added later by constraining endpoint densities to match.

## TOML Configuration

### Auto-fit (default)

No configuration needed. If `[atmosphere.onboard]` is absent, the onboard model is auto-fitted with 5 segments.

```toml
# Optional: control number of segments
[atmosphere.onboard]
n_segments = 6
```

### Explicit override

Full manual control over segment parameters:

```toml
[atmosphere.onboard]
segments = [
  { alt_low = 0.0, alt_high = 20000.0, rho_ref = 0.015, scale_height = 8000.0 },
  { alt_low = 20000.0, alt_high = 40000.0, rho_ref = 0.004, scale_height = 7500.0 },
  # ...
]
```

When `segments` is present, `n_segments` is ignored.

### Identical mode (regression testing)

To reproduce current behavior exactly (onboard = truth table):

```toml
[atmosphere.onboard]
mode = "identical"
```

This uses the truth tabulated model for onboard queries. Useful for regression testing and A/B comparison.

## Usage Split

| Component | Model | Field | Rationale |
|-----------|-------|-------|-----------|
| `compute_derivatives()` (runner.rs) | truth + MC dispersions | `data.atmosphere` | Ground truth dynamics |
| `track_peak_values()` (runner.rs) | truth + MC dispersions | `data.atmosphere` | Actual loads on vehicle |
| `estimate()` bias mode (estimator.rs) | onboard for `rho_model` | `data.atmosphere_onboard` | Filter corrects onboard |
| `estimate()` EKF mode (estimator.rs) | onboard for `rho_model` | `data.atmosphere_onboard` | EKF corrects onboard |
| `predict_exit_energy()` (fnpag.rs) | onboard | `data.atmosphere_onboard` | Guidance uses onboard knowledge |
| `compute_bank_angle()` (equilibrium_glide.rs) | onboard | `data.atmosphere_onboard` | Guidance uses onboard knowledge |
| `compute_bank_angle()` (energy_controller.rs) | none | N/A | Operates on pdyn/hdot feedback, no density query |
| `compute_bank_angle()` (predguid.rs) | none | N/A | Operates on drag acceleration feedback, no density query |
| Photo/trajectory output | both | both fields | Truth density column unchanged; add onboard model density column |

Note: FTC, neural network, and piecewise constant guidance do not query the atmosphere model directly -- they operate on state feedback (pdyn, hdot, energy). No changes needed for these schemes.

## Monte Carlo Dispersions

- **Truth:** `density_bias` multiplier applied as today (via `DensityProfile` altitude-dependent envelope)
- **Onboard:** deterministic -- same piecewise exponential every run for a given truth table
- **Effect:** The nav filter must correct both:
  1. Structural error (piecewise exponential vs tabulated truth) -- systematic, altitude-dependent
  2. MC perturbation (density_bias) -- random, varies per run

This is realistic: the real atmosphere varies run-to-run, but the onboard model is frozen at launch.

## Data Structures

### New: `OnboardAtmosphereModel`

```rust
pub struct ExponentialSegment {
    pub alt_low: f64,       // meters
    pub alt_high: f64,      // meters
    pub rho_ref: f64,       // kg/m^3 (density at alt_low)
    pub scale_height: f64,  // meters
}

pub enum OnboardAtmosphereModel {
    /// Use truth table directly (backward-compatible mode)
    Identical,
    /// Piecewise exponential segments
    PiecewiseExponential {
        segments: Vec<ExponentialSegment>,
    },
}

impl OnboardAtmosphereModel {
    /// Auto-fit from truth table
    pub fn fit_from_table(truth: &AtmosphereModel, n_segments: usize) -> Self;

    /// Query onboard density (delegates to truth table if Identical)
    pub fn density_at(&self, altitude: f64, truth: &AtmosphereModel) -> f64;
}
```

### Modified: `SimData`

```rust
pub struct SimData {
    pub atmosphere: AtmosphereModel,              // truth (unchanged)
    pub atmosphere_onboard: OnboardAtmosphereModel, // new
    // ... rest unchanged
}
```

## Backward Compatibility

- Default behavior (no TOML changes): auto-fit with 5 segments. This changes simulation results vs the current "identical" behavior.
- Existing configs that need exact regression: add `[atmosphere.onboard] mode = "identical"`.
- Golden test configs (`configs/test/`) should use `mode = "identical"` to preserve bit-exact validation.
- All other configs get the more realistic onboard model by default.

## Files Affected

### Rust (core changes)
- `src/rust/src/data/atmosphere.rs` -- add `OnboardAtmosphereModel`, `ExponentialSegment`, auto-fit logic
- `src/rust/src/data/mod.rs` -- add `atmosphere_onboard` to `SimData`, parse TOML config
- `src/rust/src/config.rs` -- parse `[atmosphere.onboard]` section
- `src/rust/src/simulation/runner.rs` -- pass onboard model to nav/guidance calls
- `src/rust/src/gnc/navigation/estimator.rs` -- use onboard model for `rho_model`
- `src/rust/src/gnc/guidance/fnpag.rs` -- use onboard model for predictor density
- `src/rust/src/gnc/guidance/equilibrium_glide.rs` -- use onboard model

### Rust (tests)
- Unit tests for `OnboardAtmosphereModel`: fitting accuracy, segment boundary behavior, identical mode
- Integration tests: verify truth != onboard density at sample altitudes
- Update golden test configs to use `mode = "identical"`

### PyO3
- `src/rust/aerocapture-py/src/results.rs` -- expose onboard density in trajectory data if adding a column

### Python
- No changes expected (atmosphere model is Rust-internal; Python only sees output columns)

### Configs
- `configs/test/*.toml` -- add `[atmosphere.onboard] mode = "identical"` for regression
- `configs/missions/mars.toml` -- optionally document the new section
- Training configs -- no changes needed (auto-fit default is appropriate)

## Testing Strategy

1. **Unit tests:** Fitting accuracy (fitted model within X% of truth at segment midpoints), segment boundary queries, exponential extrapolation above/below table
2. **Property tests (proptest):** For any altitude in table range, `|onboard - truth| / truth < max_relative_error` where max error depends on n_segments
3. **Identical mode regression:** Run golden test with `mode = "identical"`, verify bit-exact match with current output
4. **Integration test:** Run same MC scenario with identical vs piecewise_exponential, verify that density filter gain diverges from 1.0 (proving the filter is actually correcting something)
5. **Sanity check:** Onboard density should be positive everywhere, monotonically decreasing with altitude within each segment

## Success Criteria

1. Navigation density filter produces meaningful corrections (gain != 1.0) under default config
2. Golden tests pass unchanged with `mode = "identical"`
3. All existing Rust and Python tests pass
4. GA training still converges (guidance schemes handle onboard model error gracefully)
