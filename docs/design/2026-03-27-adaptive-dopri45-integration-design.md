# Adaptive Integration: Dormand-Prince 4(5)

**Date:** 2026-03-27
**Status:** Approved
**TODO ref:** §10.1 — Adaptive RK4 step sizing

## Motivation

The current fixed-step Gill-variant RK4 integrator uses a constant `dt = 1.0 s` for all phases of flight. This is:

1. **Too coarse during the deep atmospheric pass** — density varies exponentially, aero forces spike during peak heating/g-load, and 1 s steps may not capture the dynamics accurately.
2. **Not robust** — some guidance schemes or MC dispersions create rapid dynamics (bank reversals, density spikes) where fixed 1 s steps can lose accuracy with no feedback on error magnitude.

## Approach

Replace the physics integration (not the GNC cadence) with a Dormand-Prince 4(5) embedded Runge-Kutta method (DOPRI45) that provides local error estimation and adaptive step sizing. The existing Gill RK4 is retained as the default and fallback.

### Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integrator | Dormand-Prince 4(5) | Industry standard for non-stiff ODEs; 6 evals/step with FSAL vs 12 for step-doubling on Gill |
| Scope of adaptivity | Integration sub-stepping only | GNC subsystems keep their fixed cadences; outer loop ticks unchanged |
| Tolerance config | Single `rtol` knob + hardcoded per-component `atol` | Minimal config surface; `atol` chosen from physical scales |
| Coexistence | Transitional — Gill stays as default, DOPRI45 opt-in via TOML | Preserves validated Fortran-matching behavior; adaptive path validated independently |

## Design

### 1. DOPRI45 Integrator

**New file:** `src/rust/src/integration/dopri45.rs`

Standard 7-stage Dormand-Prince with FSAL (First Same As Last) optimization — accepted steps cost 6 derivative evaluations. The 4th-order and 5th-order solutions are computed from the same stages; their difference provides the local error estimate.

**Key types:**

```rust
pub struct Dopri45State {
    k7: [f64; 8],       // FSAL: last stage from previous step
    fsal_valid: bool,    // false on first step
}

pub struct StepResult {
    pub accepted: bool,
    pub error_norm: f64,  // scaled error norm (<= 1.0 means accepted)
    pub dt_next: f64,     // suggested next step size
}
```

**Function signature:**

```rust
pub fn dopri45_step(
    state: &mut [f64; 8],
    dt: f64,
    dopri: &mut Dopri45State,
    atol: &[f64; 8],
    rtol: f64,
    deriv_fn: &mut impl FnMut(&[f64; 8]) -> [f64; 8],
) -> StepResult
```

On rejection, `state` is restored from an internal copy. The caller retries with `dt_next`.

**Error norm** (mixed tolerance, standard formulation):

```
scale_i = atol[i] + rtol * |y_i|
err_i   = |y4_i - y5_i| / scale_i
error_norm = sqrt(mean(err_i²))
```

Accept when `error_norm <= 1.0`.

**Step-size controller** — PI controller (Gustafsson) to prevent oscillating step sizes:

```
dt_next = dt * clamp(fac * (1/err)^beta1 * (err_prev/err)^beta2, facmin, facmax)
```

Constants: `fac = 0.9`, `facmin = 0.2`, `facmax = 5.0`, `beta1 = 0.7/5`, `beta2 = 0.4/5`. On first step or after rejection, falls back to elementary controller (`beta2 = 0`).

**Default absolute tolerances:**

| Component | `atol` | Rationale |
|-----------|--------|-----------|
| r (m) | 1.0 | 1 m on ~3.4e6 m radius |
| lon (rad) | 1e-8 | ~0.03 m at Mars equator |
| lat (rad) | 1e-8 | ~0.03 m at Mars equator |
| V (m/s) | 1e-3 | 1 mm/s on ~5700 m/s |
| gamma (rad) | 1e-8 | ~0.03 m position equiv |
| psi (rad) | 1e-8 | ~0.03 m position equiv |
| flux (kJ/m²) | 1e-2 | 0.01 kJ/m² on O(1000) total |
| time (s) | 1e-6 | Machine-level for identity derivative |

Default `rtol = 1e-6`.

### 2. Runner Integration

**Modified file:** `src/rust/src/simulation/runner.rs`

The outer simulation loop is unchanged — it ticks at GNC cadence via the sequencer. The change is at the integration call site (currently line 615). A new function replaces the single `integrate_step()` call with adaptive sub-stepping that covers exactly the same outer tick duration.

```rust
fn integrate_adaptive(
    sim: &mut SimState,
    dt_outer: f64,
    dopri: &mut Dopri45State,
    config: &AdaptiveConfig,
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) -> AdaptiveStepStats
```

**Sub-stepping logic within one outer tick:**

1. Start with `t_remaining = dt_outer`, initial sub-step `h = min(h_suggested, t_remaining)`
2. Attempt DOPRI45 step of size `h`
3. If accepted: advance, `t_remaining -= h`, use `dt_next` for next attempt
4. If rejected: shrink to `dt_next`, retry (state already restored by `dopri45_step`)
5. When `t_remaining < h`: clamp `h = t_remaining` (exact landing on tick boundary)
6. Safety: cap at 1000 sub-steps per outer tick — if exceeded, log warning and proceed

**`AdaptiveStepStats`:** `{ n_substeps: u32, n_rejections: u32 }` — diagnostics only, not stored.

**Mode dispatch:** The runner checks `IntegrationMode` once before the main loop and dispatches to `integrate_step()` (Gill) or `integrate_adaptive()` (DOPRI45). No branching inside the hot path.

```rust
pub enum IntegrationMode {
    FixedGill,
    AdaptiveDopri45(AdaptiveConfig),
}

pub struct AdaptiveConfig {
    pub rtol: f64,       // relative tolerance (default 1e-6)
    pub initial_dt: f64, // initial sub-step guess (default 0.1 s)
    pub min_dt: f64,     // floor (default 1e-6 s)
    pub max_dt: f64,     // ceiling (default = periods.integration)
}
```

**`SimState` extension:** Add `dopri: Dopri45State` field alongside existing `accumulator` and `gill_toggle` (retained for fixed-step path).

### 3. TOML Configuration

**New optional section** in config files. When absent, behavior is identical to today.

```toml
[integration]
mode = "adaptive"   # "fixed" (default) or "adaptive"
rtol = 1e-6         # relative tolerance, default 1e-6
initial_dt = 0.1    # initial sub-step guess (s), default 0.1
min_dt = 1e-6       # floor to prevent sub-step collapse (s), default 1e-6
max_dt = 2.0        # ceiling (clamped to outer tick), default = periods.integration
```

Parsed in `config.rs` into an `IntegrationConfig` struct. Participates in `deep_merge` / `resolve_toml_bases` like any other section. The `atol` array is not exposed — hardcoded defaults only.

### 4. Testing Strategy

**Unit tests (`integration/dopri45.rs`):**

- Butcher tableau row-sum consistency
- Exact integration of polynomials up to degree 5
- Harmonic oscillator (one full period)
- FSAL continuity: `k7` from step N equals `k1` at step N+1
- Rejection + recovery on stiff-ish problem (`dy/dt = -1000*y` with large initial dt)
- PI controller respects `facmin`/`facmax` bounds
- Error norm scaling: `atol` dominates near zero, `rtol` dominates for large values

**Integration tests:**

- `integrate_adaptive` covers exactly `dt_outer` (no overshoot/undershoot beyond float epsilon)
- 1000-step safety cap triggers on pathologically tight tolerance
- Matches Gill on smooth, low-dynamics trajectories within tolerance

**Regression tests:**

- All existing golden tests run unchanged with `FixedGill` (default) — zero diff
- New golden reference: `test_ref_orig.toml` with `mode = "adaptive"` produces valid capture trajectory
- Proptest: random entry conditions + bank profiles — adaptive mode never crashes, always covers full outer tick, produces finite state values

**PyO3 tests:**

- Override dict sets `integration.mode = "adaptive"` and `integration.rtol` via dot-path notation
- `run()` with adaptive returns valid `SimResult` with all expected fields

### 5. Scope Boundaries — Explicit Exclusions

- **No GNC cadence changes.** Nav/guidance/pilot fire at configured periods regardless of integration mode.
- **No dense output / interpolation.** Photo snapshots and trajectory data are recorded at outer tick boundaries only.
- **No stiffness detection or implicit methods.** DOPRI45 is explicit; truly stiff scenarios just take tiny steps.
- **No adaptive outer tick.** The outer loop advances at `periods.integration`; adaptivity is purely within each tick.
- **No performance optimization beyond FSAL.** No SIMD, no intra-integrator parallelism. MC-level rayon parallelism is where it matters.
- **`atol` not exposed in TOML.** Hardcoded defaults; can be exposed later if needed.
