# Predictive Roll Reversal (First-Order Inclination Projection)

**Date:** 2026-04-05
**Supersedes:** 2026-03-28-roll-reversal-design.md (corridor-based lateral guidance)
**Schemes affected:** `equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag`, `ftc`
**Schemes NOT affected:** `neural_network`, `piecewise_constant` (signed output, bypass lateral guidance)

## Problem

The current roll sign management uses a reactive, velocity-dependent inclination error corridor:

```
corridor_width = (V / corridor_slope)^4 + corridor_intercept
reverse when |i_err| > corridor_width
```

This has no awareness of the rate of change of inclination error. Consequences:

1. **Unnecessary reversals** -- triggers when error is large but already converging (the current roll direction is fixing it).
2. **Late reversals** -- waits until error exceeds corridor even when it's clearly diverging fast.
3. **Low-velocity chatter** -- the 4th-power velocity dependence narrows the corridor aggressively at low speed, causing rapid reversals near trajectory end.

## Approach

Replace the corridor-based logic with a first-order inclination projection. Instead of asking "is the error too large right now?", ask "will the error still be a problem in tau seconds?"

This is the analytical/derivative-based approach: compute the inclination error rate via finite difference, extrapolate forward by a tunable lookahead horizon tau, and reverse only when the projected error exceeds a threshold. No forward integration needed.

## Design

### 1. Core Algorithm

**Rate computation:** Finite difference between consecutive guidance ticks.

```
di_err_dt = (i_err_current - i_err_previous) / (t_current - t_previous)
```

Previous inclination error and guidance time are stored in `LateralState`. On the first tick (`prev_inclination_error = None`), no reversal can fire.

**Projection:**

```
i_err_projected = i_err + di_err_dt * tau
```

When error is large but shrinking, `di_err_dt * tau` opposes `i_err`, reducing the projected error. No reversal needed. When error is small but growing, the projection amplifies it, triggering early correction.

**Reversal decision:**

```
if |i_err_projected| > threshold:
    desired_sign = if i_err_projected > 0 { -1.0 } else { 1.0 }
    if desired_sign != roll_sign AND n_reversals < max_reversals:
        reverse, increment counter
```

Same sign convention as current code (positive error -> negative roll sign).

**Anti-chatter:** `min_reversal_interval` (seconds) prevents re-triggering within a short window after a reversal. Guards against finite-difference lag: right after a reversal, the rate still reflects the old trend for 1-2 ticks until orbital dynamics respond to the new roll direction.

**Guards (kept from current):**

- Energy window: `lateral_inhibition <= energy <= lateral_activation`
- Max reversals budget
- Degenerate bank angle (near 0 or pi): roll sign is physically meaningless at these extremes
- `tau <= 0.0`: early return (inactive lateral guidance, matching default behavior)

### 2. Data Structures (Rust)

**`LateralParams`** -- complete replacement of current struct:

```rust
pub struct LateralParams {
    pub tau: f64,                    // Lookahead horizon (seconds)
    pub threshold: f64,              // Projected inclination error threshold (radians)
    pub min_reversal_interval: f64,  // Minimum time between reversals (seconds)
    pub lateral_activation: f64,     // Energy upper bound (J/kg)
    pub lateral_inhibition: f64,     // Energy lower bound (J/kg)
    pub max_reversals: i32,          // Reversal budget
}
```

Default produces inactive guidance (`tau = 0.0` triggers early return).

**`LateralState`** -- adds history tracking:

```rust
pub struct LateralState {
    pub roll_sign: f64,                      // Current roll direction (+-1.0)
    pub n_reversals: i32,                    // Reversals executed
    pub prev_inclination_error: Option<f64>, // Previous tick's i_err (None on first tick)
    pub prev_time: f64,                      // Previous tick's guidance_time
    pub last_reversal_time: f64,             // Time of most recent reversal
}
```

Initialization via `LateralState::new(initial_bank)`: `roll_sign` from sign of `initial_bank`, `n_reversals = 0`, `prev_inclination_error = None`, `prev_time = 0.0`, `last_reversal_time = f64::NEG_INFINITY` (so the first reversal is never blocked by the interval check).

**Function signature** -- unchanged:

```rust
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    planet: &PlanetConfig,
) -> bool
```

Caller in `dispatch.rs` requires no changes.

### 3. TOML Configuration

New `[guidance.lateral]` section (replaces current fields):

```toml
[guidance.lateral]
tau = 15.0                    # Lookahead horizon (seconds)
threshold = 0.5               # Projected inclination error threshold (degrees -> rad)
min_reversal_interval = 5.0   # Minimum time between reversals (seconds)
lateral_activation = -2.5     # Energy upper bound (MJ/kg -> J/kg)
lateral_inhibition = -8.0     # Energy lower bound (MJ/kg -> J/kg)
max_reversals = 5             # Reversal budget (integer)
```

Unit conversions in config parsing:
- `threshold`: degrees -> radians
- `lateral_activation` / `lateral_inhibition`: MJ/kg -> J/kg
- All others: no conversion

**Backward compatibility:** Old configs with `corridor_slope` / `corridor_intercept` will fail to parse. This is intentional (clean replacement). Configs without `[guidance.lateral]` are unaffected (default = inactive).

**Affected training TOMLs:**
- `configs/training/msr_aller_eqglide_train.toml`
- `configs/training/msr_aller_energy_controller_train.toml`
- `configs/training/msr_aller_pred_guid_train.toml`
- `configs/training/msr_aller_fnpag_train.toml`
- `configs/training/msr_aller_ftc_train.toml`

### 4. GA Parameter Space (Python)

Replace `_LATERAL_PARAMS` in `param_spaces.py`:

| Gene | Bounds | Default | Unit |
|---|---|---|---|
| `lateral.tau` | [2.0, 60.0] | 15.0 | seconds |
| `lateral.threshold` | [0.01, 2.0] | 0.5 | degrees (TOML units) |
| `lateral.min_reversal_interval` | [1.0, 30.0] | 5.0 | seconds |
| `lateral.lateral_activation` | [-5.0, -0.5] | -2.5 | MJ/kg |
| `lateral.lateral_inhibition` | [-10.0, -2.0] | -8.0 | MJ/kg |
| `lateral.max_reversals` | [1.0, 10.0] | 5.0 | integer |

`evaluate.py` and `compare_guidance.py` require no changes -- the `lateral.` prefix routing and `max_reversals` integer rounding already handle the new parameter names.

### 5. Testing

**Unit tests** (in `lateral.rs` `#[cfg(test)]`):

- `no_reversal_on_first_tick` -- `prev_inclination_error` is None, verify no reversal fires
- `no_reversal_when_error_converging` -- i_err large but di_err_dt opposing, projected error below threshold
- `reversal_when_error_diverging` -- i_err moderate and growing, projected error exceeds threshold
- `respects_min_reversal_interval` -- trigger reversal, then immediately present diverging error; second reversal suppressed until interval elapses
- `respects_max_reversals` -- budget exhaustion
- `no_reversal_outside_energy_window` -- energy gating
- `no_reversal_when_bank_degenerate` -- near 0 or pi
- `tau_zero_produces_inactive` -- default params never fire

**Property-based tests** (proptest):

- `roll_sign_always_pm_one` -- invariant across random nav states
- `n_reversals_monotonic` -- non-decreasing across sequences
- `n_reversals_bounded_by_max` -- never exceeds budget
- `projected_error_finite` -- no NaN/Inf for valid inputs

**Golden file regeneration:** All 6 golden CSV files need regeneration (any scheme with lateral guidance active produces different trajectories).

**Integration:** Existing PyO3 tests and CI pipeline cover end-to-end.
