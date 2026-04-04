# Bank Angle Rate and Acceleration Command Shaping

**Date:** 2026-04-03
**Status:** Design approved
**Approach:** Dispatch-layer S-curve shaper (Approach 1)

## Problem

Guidance schemes compute bank angle commands with zero awareness of the vehicle's rotational rate and acceleration limits. The pilot model (`gnc/control/pilot.rs`) enforces rate limits, but guidance doesn't anticipate this. The result:

1. **Feedback gap:** The dispatch layer computes rate from the *last commanded* angle (`bank_angle_previous`), not the *pilot-realized* angle. If the pilot hasn't finished slewing, guidance underestimates the true angular distance remaining.
2. **Hard saturation:** The dispatch hard-clamps commands at `max_bank_rate` (bang-bang). This produces discontinuous rate profiles that the pilot (especially second-order) struggles to track, creating persistent lag.

Worst offenders: Piecewise Constant (hard segment boundaries), FNPAG (secant method jumps), FTC (predictor-corrector steps).

## Design

Two complementary changes in the dispatch layer, transparent to all 7 guidance schemes.

### 1. Feedback Fix: Realized Angle Baseline

**Current:** `state.bank_angle_previous` is set to the last *commanded* angle (dispatch.rs line 243).

**Change:** The runner passes the pilot-realized bank angle (`sim.bank_angle`) into `guidance_step()`. The dispatch uses this as the baseline for rate/acceleration calculations.

- `bank_angle_previous` renamed to `bank_angle_realized`, updated from pilot state
- Rate computation: `angle_diff = shortest_angle_diff(realized, commanded)`
- If the pilot fell behind, guidance sees the real gap and corrects accordingly

**Always active** -- this is a correctness fix, not a feature toggle. Existing tuned parameters may need minor adjustment since guidance will see larger deltas when the pilot lags (correct behavior that was previously masked).

### 2. S-Curve Command Shaper

A new `CommandShaper` struct in the dispatch layer, persisted in `GuidanceState`.

**State:**
```
CommandShaper {
    shaped_rate: f64,    // current shaped bank rate (rad/s), carried between ticks
}
```

**Algorithm per guidance tick:**

1. Compute raw angle error: `error = shortest_angle_diff(realized_angle, raw_commanded_angle)`
2. Compute raw rate: `raw_rate = error / guidance_period`
3. Apply acceleration limit to shaped_rate:
   - `rate_delta = raw_rate - shaped_rate`
   - `max_rate_delta = max_bank_acceleration * guidance_period`
   - `shaped_rate += clamp(rate_delta, -max_rate_delta, max_rate_delta)`
4. Clamp shaped_rate to `[-max_bank_rate, +max_bank_rate]`
5. Compute shaped command: `shaped_angle = realized_angle + shaped_rate * guidance_period`

**Behavior:**
- Large step commands get smoothed into trapezoidal rate profiles (ramp up, hold, ramp down)
- Small corrections pass through nearly unchanged
- Direction reversals decelerate before re-accelerating (physically correct)
- `shaped_rate` carries momentum between ticks

**Edge cases:**
- First tick / reset: `shaped_rate = 0.0`
- Wrap-aware throughout via `shortest_angle_diff`

### 3. TOML Configuration

New optional section `[guidance.command_shaping]`:

```toml
[guidance.command_shaping]
enabled = true                    # default: true when section present, false when absent
max_bank_acceleration = 5.0       # deg/s^2
```

**Backward compatibility:** Section absent = shaping disabled, dispatch falls back to current hard-clamp. The feedback fix (realized angle baseline) applies regardless.

**Rust side:** New `CommandShapingConfig` struct in `config.rs`, loaded as `Option<CommandShapingConfig>`. Dispatch checks `is_some() && enabled`. `max_bank_acceleration` is specified in deg/s^2 in TOML and converted to rad/s^2 at config load time (same pattern as `max_bank_rate` in `Capsule`).

`max_bank_acceleration` has no default -- must be specified when section present. Reasonable range: ~2-15 deg/s^2 (with `max_bank_rate` typically 15 deg/s, gives ramp-up times of 1-7.5 s).

### 4. GA Integration

**`param_spaces.py`:** New `shaping.` prefix:

```python
_SHAPING_PARAMS = [
    ("shaping.max_bank_acceleration", 2.0, 15.0, False),  # deg/s^2, linear scale
]
```

Added to all scheme param spaces (shaper is universal).

**`evaluate.py`:** Add `shaping.` to prefix routing -> `[guidance.command_shaping]` overrides.

**`compare_guidance.py`:** Load `shaping.*` keys from `best_params.json`.

**`train.py`:** No changes -- PyO3 override dict handles dot-path keys generically.

### 5. GuidanceOutput Changes

- `bank_rate`: reflects shaped rate (post-shaper, pre-pilot). When shaping disabled, same as today.
- `rate_saturated`: set to 1 when either acceleration limit or rate limit is hit.
- No new fields. Transparent to downstream (photo output, CSV, analysis).

### 6. Testing Strategy

**Rust unit tests** (dispatch.rs `#[cfg(test)]`):

1. **Shaper disabled:** Behavior identical to current hard-clamp (regression guard).
2. **Realized baseline:** Rate uses realized angle, not previous command. Pilot-lag scenario confirms real gap detected.
3. **Acceleration limiting:** Step 0 -> 90 deg, verify rate ramps at `max_bank_acceleration` not instant jump.
4. **Rate clamping preserved:** `max_bank_rate` still caps shaped rate after ramp-up.
5. **Direction reversal:** +90 then -90 in successive ticks, verify deceleration before reversal (S-curve).
6. **Wrap-around:** +170 to -170 deg, verify shortest path through 180.
7. **Small corrections pass through:** 2 deg change passes nearly unmodified.

**Proptest properties:**
- Shaped rate always in `[-max_bank_rate, max_bank_rate]`
- Rate change between ticks always in `[-max_bank_acceleration * dt, +max_bank_acceleration * dt]`
- Shaped angle always finite and in `[-pi, pi]`

**Integration/E2E:** Existing golden configs with `[guidance.command_shaping]` added -- verify no NaN/divergence. Golden files unchanged (shaping disabled by default).

## Files Modified

### Rust
- `src/rust/src/config.rs` -- `CommandShapingConfig` struct, TOML parsing
- `src/rust/src/data/guidance_params.rs` -- store parsed shaping config
- `src/rust/src/gnc/guidance/dispatch.rs` -- `CommandShaper` struct, realized baseline, S-curve algorithm, `GuidanceState` changes
- `src/rust/src/simulation/runner.rs` -- pass `sim.bank_angle` into `guidance_step()`

### Python
- `src/python/aerocapture/training/param_spaces.py` -- `_SHAPING_PARAMS`, add to all schemes
- `src/python/aerocapture/training/evaluate.py` -- `shaping.` prefix routing
- `src/python/aerocapture/training/compare_guidance.py` -- `shaping.*` key loading

### Config
- `configs/missions/mars.toml` (or similar) -- example `[guidance.command_shaping]` section (commented out)
- `configs/training/common.toml` -- shaping defaults for training

## Not In Scope

- Per-scheme rate awareness (Approach 2) -- schemes don't receive rate budgets
- Pilot-model-aware predictive shaping (Approach 3) -- no coupling to pilot parameters
- Changes to photo/CSV output columns
- New report charts for rate shaping metrics
