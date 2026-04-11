# Event Detection for DOPRI45 Adaptive Integrator

**Date:** 2026-04-07
**Branch:** `feature/advanced-sampling-sensitivity` (will move to dedicated branch for implementation)

## Problem

The simulator detects events (bounce, atmosphere exit, crash, phase transition) via post-integration threshold checks evaluated once per outer tick (typically 1 s). With 1 s ticks at Mars entry velocities (~5.7 km/s), events can be missed by up to 1 second / 5.7 km of position error. The DOPRI45 adaptive integrator sub-steps within each tick but has no event awareness -- it can step over zero-crossings without noticing.

## Approach

Dense output interpolation + Brent's method root-finding, integrated into the DOPRI45 adaptive sub-stepping loop. DOPRI45 only -- the fixed Gill RK4 path is unchanged.

The DOPRI45 method has a standard 4th-order continuous extension (Dormand & Prince 1986, Hairer/Norsett/Wanner "Solving ODEs I" Table 6.2). Given an accepted step with stage derivatives k1..k7, it produces a polynomial interpolant valid at any point within the step -- no extra derivative evaluations needed.

After each accepted substep, event functions are evaluated at the new state. If a sign change is detected, Brent's method locates the zero-crossing on the dense output interpolant to ~1 ms precision (typically 5-7 iterations).

## Event Function Framework

An event is a scalar function `g(state) -> f64` where a sign change indicates the event occurred.

```rust
struct EventDef {
    /// Scalar function evaluated on the state vector. Zero-crossing = event.
    eval: fn(&[f64; 8], &EventContext) -> f64,
    /// Direction: +1 (rising only), -1 (falling only), 0 (both).
    direction: i8,
    /// What to do when triggered.
    action: EventAction,
}

enum EventAction {
    Terminate(TermReason),
    PhaseTransition,
    Record,
}
```

### Four Event Functions

| Event | g(state) | Direction | Action |
|-------|----------|-----------|--------|
| Bounce | `sin(gamma)` | +1 (rising through zero) | `Record` (sets bounced flag, records bounce state) |
| Atmosphere exit | `altitude - exit_altitude` | +1 (rising through threshold) | `Terminate(AtmosphereExit)` |
| Ground crash | `altitude` | -1 (falling through zero) | `Terminate(Crash)` |
| Phase transition | `exit_velocity_threshold - V_relative` | +1 (V dropping below threshold) | `PhaseTransition` |

`EventContext` carries read-only data needed by event functions (planet radius, exit altitude, velocity threshold) to avoid coupling event evaluation to the full `SimData`.

Bounce and phase transition have guard conditions: bounce only fires if `!sim.bounced`, phase transition only fires if `sim.bounced && !exit_phase_locked`. Guards are checked after root-finding locates the crossing, not during evaluation.

## Dense Output

New function in `dopri45.rs`:

```rust
pub fn dopri45_dense(
    y0: &[f64; N],
    h: f64,
    theta: f64,
    k: &[[f64; N]; 7],
) -> [f64; N]
```

Uses the standard quartic interpolation coefficients:
```
b_i(theta) = theta * (b_i + theta * (bp_i + theta * (bpp_i + theta * bppp_i)))
y(t_n + theta*h) = y_n + h * sum(b_i(theta) * k_i)
```

28 constants total (4 coefficients x 7 stages), fixed from the literature.

Stage derivatives are made available via a new function `dopri45_step_with_stages` that returns `k1..k7` alongside `StepResult`. The existing `dopri45_step` remains unchanged for non-event callers.

## Root-Finding Within Substeps

1. Before each substep: cache `g_i(y_n)` for all active event functions.
2. After an accepted substep: evaluate `g_i(y_{n+1})`.
3. For each event with a sign change matching its direction filter:
   - Brent's method on `g_i(dopri45_dense(y0, h, theta, k))` with theta in [0, 1].
   - Converge until bracket width < `tol_event / h` in theta-space (tol_event = 1e-3 s).
4. If multiple events trigger in the same substep, take the earliest (smallest theta).
5. Interpolate full 8-component state at the winning theta.

After locating:
- Overwrite `sim.state` with interpolated state.
- Advance `sim_time` to event time.
- Invalidate FSAL (`dopri.fsal_valid = false`).
- Execute event action.
- `Record` events (bounce): resume integration from event point for the remainder of the outer tick. `sim_time` is not adjusted mid-tick -- it still advances by full `dt` at the tick boundary. The precise event time is stored in `EventRecord.time` and copied to `sim.bounce_time`.
- `Terminate` events (crash, atmosphere exit): break out of both the substep loop and the main sim loop. `sim_time` is adjusted to the precise event time (entry time of tick + time_offset). Final record uses this adjusted time.
- `PhaseTransition`: record transition time in `EventRecord`, resume integration. Guidance picks up the new phase on the next GNC tick. `sim_time` not adjusted mid-tick.

## Sim Loop Integration

### New flow

```
main loop tick -> GNC -> integrate_adaptive_with_events -> post-tick composite checks only
```

New function signature:

```rust
fn integrate_adaptive_with_events(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    events: &[EventDef],
    event_ctx: &EventContext,
) -> AdaptiveStepResult

struct AdaptiveStepResult {
    stats: AdaptiveStepStats,
    triggered: Option<TriggeredEvent>,
}

struct TriggeredEvent {
    event_index: usize,
    time_offset: f64,
    state: [f64; 8],
}
```

The main loop delegates the 4 root-found events to the integrator. Composite checks remain as post-tick:
- Atmospheric apoapsis crash (bounced + descending + below exit altitude)
- Trapped orbit detection (bounced + 2*a < r_exit)
- NaN safety net
- Wall-clock timeout

### Phase transition detail

The precise velocity threshold crossing is located by the integrator, but the actual phase state change (`guidance_phase = 2`, `exit_phase_locked = true`, latching `reference_velocity`) still happens in the navigation layer on its next tick. The event detection gives us the exact moment; on the next navigation tick, the estimator sees the post-crossing state and transitions. No restructuring of the GNC chain.

### Fixed RK4

Completely untouched. Same post-tick event checks as today.

## Trajectory Output

### New fields on SimState

```rust
event_records: Vec<EventRecord>,

struct EventRecord {
    time: f64,
    state: [f64; 8],
    event_type: EventType,  // Bounce, AtmosphereExit, Crash, PhaseTransition
}
```

### How event records reach output

- **Trajectory data** (PyO3 `include_trajectories=True`, (N, 17) array): event records interleaved at correct time position. Trajectory output is no longer strictly uniform in time. Consumers already handle variable-length arrays.
- **final_record** (52-element array): `bounce_alt`, `bounce_time`, `sim_time` now reflect precise event times. No layout changes.
- **Photo CSV** (legacy format): unchanged. Photo rows at cadence ticks only.

### PyO3 API

No changes. `BatchResults.trajectories` already returns variable-length arrays. Column layout (17 columns) unchanged.

## Testing

### Unit tests (dopri45.rs)

- Dense output accuracy: harmonic oscillator, verify at theta = 0.0/0.25/0.5/0.75/1.0 against analytical solution to 4th-order.
- Dense output boundaries: theta=0 returns y0 exactly, theta=1 matches accepted y5.
- Brent convergence: standalone test on sin(theta), verify convergence within budget and tolerance.

### Integration tests (tests/)

- Bounce precision: compare event-detected bounce time against fine-dt bisection reference, agreement within 1 ms.
- Atmosphere exit precision: same approach for exit altitude crossing.
- Crash detection: verify event fires before altitude goes negative.
- Phase transition timing: verify transition at correct velocity within 1 ms.
- Multiple events in one tick: bounce + exit close together, earliest wins.
- Guard conditions: second FPA zero-crossing after bounce does not re-trigger.
- Fixed RK4 non-regression: bit-identical output to current baseline.

### Proptest

- For random Mars entry corridor initial conditions, event-detected bounce time is between the two consecutive tick boundaries where current threshold check fires.
- Event state interpolation produces finite values for all 8 components.

### Golden files

DOPRI45 test configs: regenerate (bounce_time, bounce_alt, sim_time shift by up to 1 s). Fixed-RK4 golden files: untouched.

## File Changes

| File | Change |
|------|--------|
| `src/rust/src/integration/dopri45.rs` | Dense output coefficients, `dopri45_dense()`, `dopri45_step_with_stages()`, Brent's root-finder |
| `src/rust/src/integration/mod.rs` | New submodule `events.rs` |
| `src/rust/src/integration/events.rs` | `EventDef`, `EventAction`, `EventRecord`, `TriggeredEvent`, `EventContext`, `check_events_and_locate()` |
| `src/rust/src/simulation/runner.rs` | `SimState` gets `event_records`; `integrate_adaptive_with_events()`; main loop delegates 4 events to integrator, keeps composite checks; trajectory output interleaves event rows |
| Tests: `src/rust/src/integration/dopri45.rs` | Dense output unit tests, Brent convergence test |
| Tests: `src/rust/tests/` | Event detection integration tests |
| Golden files: `tests/reference_data/rust_golden/` | Regenerate DOPRI45 configs only |

### Not touched

- Python analysis code
- TOML configs (no new configuration -- event detection is always on for DOPRI45, 1 ms tolerance is a hardcoded constant)
- PyO3 bindings API
- Fixed RK4 path
- GNC chain (navigation, guidance, control)
- Report/chart generation
- `sequencer.rs`, `rk4.rs`, `estimator.rs`
