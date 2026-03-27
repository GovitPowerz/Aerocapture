# Roll Reversal for Unsigned-Magnitude Guidance Schemes

**Date:** 2026-03-28
**Schemes affected:** `equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag`
**Schemes NOT affected:** `neural_network`, `piecewise_constant` (signed output, bypass lateral guidance)

## Problem

The lateral guidance / roll reversal logic exists in `ftc.rs:397–466` but is effectively dead:

1. **Zero-width energy window** — `lateral_activation` and `lateral_inhibition` both default to 0.0 MJ/kg, so the energy gate never opens.
2. **`reversal_active` immediate reset** — Armed and disarmed on the same line (ftc.rs:453–455), so the smooth roll-through sweep never executes.
3. **`guidance_active[1]` permanently killed after first reversal** — Only one reversal decision ever fires per trajectory.

The four unsigned-magnitude schemes (EqGlide, EnergyController, PredGuid, FNPAG) produce bank angle magnitudes but have no working mechanism to control the bank angle sign for inclination management.

## Approach

Extract lateral guidance into a clean shared module with per-scheme TOML-configurable parameters. Fix the bugs. Add lateral parameters to the GA search space so each scheme's corridor shape and activation windows can evolve independently.

NN and PiecewiseConstant are completely unaffected — the existing `skip_lateral` gate in `ftc.rs` prevents lateral guidance from ever running for those two schemes.

## Design

### 1. New `lateral.rs` Module (`gnc/guidance/lateral.rs`)

**`LateralParams` struct** (deserialized from TOML, per-scheme):

| Field | Type | Unit | Description |
|---|---|---|---|
| `corridor_slope` | f64 | m/s | Velocity scaling for corridor width |
| `corridor_intercept` | f64 | rad | Baseline corridor width at low velocity |
| `lateral_activation` | f64 | MJ/kg | Energy at which lateral guidance arms |
| `lateral_inhibition` | f64 | MJ/kg | Energy below which lateral guidance disarms |
| `max_reversals` | u32 | — | Cap on total reversals per trajectory |

**`LateralState` struct** (mutable, per-run):

| Field | Type | Initial | Description |
|---|---|---|---|
| `roll_sign` | f64 | 1.0 | Current roll direction (±1) |
| `n_reversals` | u32 | 0 | Number of reversals executed |
| `lateral_active` | bool | false | Whether lateral channel is currently active |

**`lateral_guidance()` function:**
- Signature: `fn lateral_guidance(params: &LateralParams, state: &mut LateralState, nav: &NavigationState, target_inclination: f64, energy: f64, bank_magnitude: f64) -> f64`
- Returns the roll sign (±1.0)
- Logic:
  1. Check energy window: `lateral_inhibition <= energy <= lateral_activation` → set `lateral_active` (energy decreases during entry; activation is the upper threshold, inhibition the lower)
  2. If not active, return current `roll_sign` unchanged
  3. Compute current inclination from nav state via `elements::from_spherical`
  4. Compute `inclination_error = target_inclination - current_inclination`
  5. Compute `corridor_width = (velocity / corridor_slope)^4 + corridor_intercept`
  6. If `|inclination_error| >= corridor_width` AND `bank_magnitude > epsilon` AND `n_reversals < max_reversals`:
     - If `inclination_error > corridor_width`: `roll_sign = -1.0`
     - If `inclination_error < -corridor_width`: `roll_sign = +1.0`
     - If sign changed: increment `n_reversals`
  7. Return `roll_sign`

### 2. TOML Configuration

New optional `[guidance.lateral]` sub-section:

```toml
[guidance.lateral]
corridor_slope = 13080.458
corridor_intercept = 0.0
lateral_activation = -2.5
lateral_inhibition = -8.0
max_reversals = 5
```

**Backward compatibility:** If `[guidance.lateral]` is absent, check for old flat keys (`guidance.corridor_slope`, etc.) as fallback. If neither exists, defaults keep lateral guidance inactive (zero-width energy window). Existing configs produce identical behavior.

**Base inheritance:** Training configs override `[guidance.lateral]` independently via deep merge.

### 3. Integration into `guidance_step`

Flow in `ftc.rs::guidance_step`:

```
1. Dispatch to scheme → unsigned bank magnitude
2. if skip_lateral (NN, PiecewiseConstant):
      bank_commanded = signed output directly (unchanged)
   else:
      roll_sign = lateral_guidance(&lateral_params, &mut lateral_state, nav, target_incl, energy, bank_mag)
      bank_commanded = magnitude * roll_sign
3. Rate saturation (unchanged)
4. → pilot → physics
```

No changes to any individual scheme's longitudinal logic. `LateralState` becomes a field on `FtcState`, replacing the old individual `roll_sign`, `n_reversals`, and related fields. Initialized in `init.rs` with `roll_sign = 1.0, n_reversals = 0, lateral_active = false`.

### 4. GA Parameter Space

Five new genes per scheme (EqGlide, EnergyController, PredGuid, FNPAG) in `param_spaces.py`:

| Gene | Type | Bounds | Scale |
|---|---|---|---|
| `lateral.corridor_slope` | float | [5000, 20000] | linear |
| `lateral.corridor_intercept` | float | [0.0, 0.1] | linear |
| `lateral.lateral_activation` | float | [-5.0, -0.5] | linear |
| `lateral.lateral_inhibition` | float | [-10.0, -2.0] | linear |
| `lateral.max_reversals` | int | [1, 10] | linear |

Decode in `evaluate.py` uses dot-path keys: `"guidance.lateral.corridor_slope"`, etc. `max_reversals` rounded to integer during decode.

NN and PiecewiseConstant parameter spaces are untouched.

### 5. Bug Fixes

1. **Remove `reversal_active` immediate reset** (ftc.rs:453–455) — The roll_sign flip is sufficient; rate saturation limits the physical bank rate. Delete the dead `reversal_active` arm/disarm and the associated sweep machinery.

2. **Remove `guidance_active[1]` permanent kill** (ftc.rs:218) — Let the energy window and `max_reversals` be the only gates. Multiple reversals become possible.

### 6. Testing

**Unit tests (`lateral.rs`):**
- Inclination error triggers reversal at corridor boundary
- Respects `max_reversals` cap
- Energy window gating (active only between activation/inhibition thresholds)
- No reversal when bank magnitude is near zero
- `roll_sign` output is always ±1

**Property-based tests (proptest):**
- `corridor_width` is always positive for any velocity
- `roll_sign` output is always ±1
- `n_reversals` is monotonically non-decreasing

**Integration test:**
- Run one of the four schemes with lateral params configured, verify bank angle sign flips during trajectory

**Regression:**
- Existing tests with lateral off (default config) must produce bit-identical results — zero-width energy window means lateral never activates
