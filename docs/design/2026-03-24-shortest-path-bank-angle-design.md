# Shortest-Path Bank Angle Control

**Date:** 2026-03-24
**Scope:** `src/rust/src/gnc/control/`, `src/rust/src/gnc/guidance/ftc.rs`, `src/rust/src/simulation/runner.rs`

## Problem

The bank angle control chain does not account for angular wrap-around at ±π. When the commanded bank angle crosses the ±180° boundary (e.g. a reversal from +170° to -170°), three issues arise:

1. **Rate saturation** (`ftc.rs:257-272`) computes `bank_rate = (commanded - previous) / dt` as a raw difference. A 20° reversal through ±180° produces a -340° difference, causing the rate limiter to clamp in the wrong direction — the spacecraft takes the 340° long way around instead of the 20° short path.

2. **Pilot dynamics** (`pilot.rs:41,52`) compute the error between commanded and realized bank angle as a raw subtraction. First-order and second-order models will see a 340° error instead of 20° and overshoot or oscillate.

3. **Bank consumption tracking** (`runner.rs:518`) accumulates `|new_bank - old_bank|`, which registers 340° instead of 20° for wrap-around transitions, inflating the performance metric.

## Decision

Add a single `shortest_angle_diff(from, to) -> f64` utility function and apply it at all three sites. No changes to guidance scheme outputs, lateral guidance logic, or roll reversal decision logic.

## Design

### New file: `src/rust/src/gnc/control/angle_utils.rs`

```rust
use std::f64::consts::{PI, TAU};

/// Shortest signed angular difference from `from` to `to`, in [-PI, PI].
///
/// Returns the smallest rotation needed to get from `from` to `to`,
/// with positive meaning counterclockwise and negative meaning clockwise.
pub fn shortest_angle_diff(from: f64, to: f64) -> f64 {
    let mut d = (to - from) % TAU;
    if d > PI { d -= TAU; }
    if d < -PI { d += TAU; }
    d
}
```

Register the module in `src/rust/src/gnc/control/mod.rs`.

### Call site 1: Rate saturation in `ftc.rs` (lines 257-278)

Replace raw subtraction with `shortest_angle_diff`:

```rust
let angle_diff = shortest_angle_diff(state.bank_angle_previous, state.bank_angle_commanded);
let bank_rate = angle_diff / guidance_period;

if bank_rate.abs() - max_bank_rate > 1e-10 {
    rate_saturated = 1;
    state.bank_angle_commanded =
        state.bank_angle_previous + max_bank_rate.copysign(angle_diff) * guidance_period;
}

// Cumulative tracking uses shortest path too
if bank_rate.abs() > 1e-10 {
    state.cumulative_bank_change += angle_diff.abs();
}
```

`copysign` replaces the `if commanded > previous` direction check, which is correct even across ±π wrap.

### Call site 2: Pilot dynamics in `pilot.rs` (lines 41, 52)

Replace raw subtraction with `shortest_angle_diff` for the error computation:

```rust
// FirstOrder (line 41): error = commanded - state.bank_angle
let error = shortest_angle_diff(state.bank_angle, commanded);

// SecondOrder (line 52): error = state.bank_angle - commanded
let error = shortest_angle_diff(commanded, state.bank_angle);
```

The argument order preserves the original sign convention in each model.

### Call site 3: Bank consumption in `runner.rs` (line 518)

Replace raw absolute difference:

```rust
let bank_change = shortest_angle_diff(sim.bank_angle, pilot_state.bank_angle).abs();
```

### What stays the same

- All 7 guidance scheme outputs — unchanged
- Lateral guidance and roll reversal decision logic in `ftc.rs` — unchanged
- The angle wrapping in the reversal incremental block (lines 240-252) — still needed for the step-by-step path, but rate saturation now handles wrap correctly downstream
- The `roll_path` field (hardcoded to 1) — dead code, separate cleanup

## Testing

### Unit tests for `shortest_angle_diff` (in `angle_utils.rs`)

- **Property tests (proptest):** result always in [-PI, PI]; approximate antisymmetry (`(diff(a,b) + diff(b,a)).abs() < 1e-15` — float `%` is not perfectly symmetric); `|diff| <= PI`
- **Specific cases:**
  - `(170°, -170°)` → `+20°` (short path through +180°)
  - `(-170°, 170°)` → `-20°` (short path through -180°)
  - `(0°, 180°)` → exactly `+PI` (since `PI % TAU == PI` and `d > PI` is false)
  - `(0°, 0°)` → `0°`
  - `(PI, -PI)` → `0°` (same angle)
  - `(-PI, PI)` → `0°` (same angle)
- **NaN/infinity:** inputs must be finite; `shortest_angle_diff` propagates NaN for non-finite inputs (standard IEEE 754 behavior, no panic). Add `debug_assert!(from.is_finite() && to.is_finite())` to catch upstream corruption in debug builds.

### Pilot dynamics tests (in `pilot.rs`)

- Add test: first-order pilot with commanded at -170° and current at +170° should move toward -170° through +180° (rate positive), not through 0° (rate negative)

### Integration regression

- Run existing FTC guided test config and verify results are near-identical (small differences expected only where wrap-around was previously wrong — i.e. during roll reversals near ±180°)
