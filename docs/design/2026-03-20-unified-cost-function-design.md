# Unified Cost Function, Pending Crash Detection & Sentinel Corridor

**Date:** 2026-03-20
**Status:** Proposed

## Problem Statement

The current GA training cost function has three issues:

1. **Extreme DV penalties:** Rust returns `1e30` for non-exited trajectories, Python applies `1e6` for non-captures. These extreme values dominate the RMS aggregation and wash out gradient signal for the GA optimizer.

2. **Missing crash detection:** A trajectory that exits the atmosphere on a captured orbit with apoapsis below the atmosphere ceiling (~130 km for Mars) is guaranteed to re-enter and crash, but the sim currently scores it by its orbital correction DV rather than flagging it as a crash.

3. **Poor corridor boundary resolution:** The `capture_min_pdyn` (hyperbolic boundary) and `crash_max_pdyn` envelopes are built opportunistically from GA population history, which may not explore the full corridor — especially the extremes (full lift-up / full lift-down).

## Design

### 1. Pending Crash Detection (Rust)

**File:** `src/rust/src/simulation/runner.rs`

After the simulation exits the atmosphere (`TermReason::AtmosphereExit`) and orbital elements are computed, add a reclassification check:

```
let captured = ecc < 1.0 && energy < 0.0;
if term == AtmosphereExit && captured && orbit.apoapsis_alt < exit_altitude {
    term = PendingCrash
}
```

**Critical guard:** The `captured` check is required because hyperbolic orbits have negative SMA, making `apoapsis_alt` negative — which would otherwise pass the `< exit_altitude` check and misclassify every hyperbolic exit as a pending crash.

**Order of operations in runner.rs post-termination block:**
1. Compute orbital elements (existing, line 606)
2. Compute `captured = ecc < 1.0 && energy < 0.0`
3. Reclassify pending crash (requires `captured` from step 2)
4. Compute DV based on final classification (step 2 determines which path)

- `exit_altitude` = `data.final_conditions.altitude` (Mars: 130,988 m)
- `orbit.apoapsis_alt` is already computed (meters)
- New `TermReason::PendingCrash` variant maps to `ifinal = 4`
- Python `classify_trajectories` treats `ifinal == 4` identically to `ifinal == 1` (crash): `crash = (ifinal == 1.0) | (ifinal == 4.0)`
- In `RunOutput`, pending crashes set `captured = false` — they are gravitationally captured but operationally doomed, and downstream filters (final_report stats, compare_guidance) should exclude them from capture statistics

### 2. Meaningful DV for All Outcomes (Rust)

**File:** `src/rust/src/simulation/runner.rs` (post-termination DV computation)

Instead of routing all outcomes through `compute_deltav` (which returns `1e30` for non-exits), compute DV contextually:

```
match (term, captured) {
    (AtmosphereExit, true)  -> compute_deltav()           // real orbital correction DV
    (AtmosphereExit, false) -> HYPERBOLIC_BASE + v_excess  // velocity surplus over escape
    (Crash | PendingCrash)  -> CRASH_BASE * (1.0 - 0.5 * sim_time / max_time)
    (Timeout)               -> CRASH_BASE * (1.0 - 0.5 * sim_time / max_time)
}
```

**Constants (module-level `const` in runner.rs, not TOML-configurable):**
- `HYPERBOLIC_BASE: f64 = 10_000.0` — clearly worse than any realistic captured DV
- `CRASH_BASE: f64 = 20_000.0` — clearly worse than any hyperbolic exit

**Crash time decay** uses proportional scaling to stay safe regardless of `max_time`:
```
virtual_dv = CRASH_BASE * (1.0 - 0.5 * sim_time / max_time)
```
Range: `[CRASH_BASE * 0.5, CRASH_BASE]` = `[10,000, 20,000]` — always positive, always worse than hyperbolic, and the GA still gets a gradient favoring longer survival.

**Hyperbolic excess velocity:**
- `v_excess = speed_abs - sqrt(2 * mu / r)` — minimum impulse needed to capture into any bound orbit
- `speed_abs` and `mu/r` are already computed in the post-termination block

**`maneuver.rs` change:** Remove the `ifinal != 3 -> 1e30` early return. `compute_deltav` is now only called for confirmed captured trajectories.

**Cost landscape after log compression (T=1000 m/s):**

| Outcome | Raw virtual DV | After f(dv) |
|---------|---------------|-------------|
| Great capture | 100 m/s | 100 |
| OK capture | 800 m/s | 800 |
| Threshold | 1,000 m/s | 1,000 |
| Bad capture | 5,000 m/s | 2,609 |
| Barely hyperbolic | ~10,000 m/s | 3,302 |
| Very hyperbolic (+1 km/s excess) | 11,000 m/s | 3,397 |
| Late crash (t=2000s, max=3000s) | 13,333 m/s | 3,590 |
| Immediate crash (t=100s, max=3000s) | 19,667 m/s | 3,979 |

### 3. Unified Smooth Cost Function (Python)

**File:** `src/python/aerocapture/training/evaluate.py` — `compute_cost()`

Replace the two-tier branching system with a single smooth function:

```python
def log_cap(dv: np.ndarray, threshold: float = 1000.0) -> np.ndarray:
    """C1-continuous log-capped cost: linear below threshold, log above."""
    dv = np.maximum(dv, 1e-6)  # safety floor: avoids log(0) if DV is ever 0
    below = dv <= threshold
    result = np.empty_like(dv)
    result[below] = dv[below]
    result[~below] = threshold * (1.0 + np.log(dv[~below] / threshold))
    return result
```

**Properties:**
- C0 continuous at threshold: both sides evaluate to `T`
- C1 continuous at threshold: both sides have derivative `1`
- Monotonically increasing for all `dv > 0`
- Compresses outliers: `f(20000) ≈ 3996` vs raw `20000`
- Safety floor at `1e-6` prevents `log(0)` in edge cases

**Simulation failure path:** When `evaluate_chromosome()` cannot run the simulation at all (subprocess timeout, binary crash), it returns `(1e30, None)` as the cost, bypassing `compute_cost()` entirely. This path is unchanged — `log_cap` is never called with that value.

**New `compute_cost()` — keeps existing kwargs signature pattern:**

```python
def compute_cost(
    final_conditions: np.ndarray, *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,
    heat_flux_limit: float = 200.0,
    g_load_weight: float = 1000.0,
    heat_flux_weight: float = 1000.0,
) -> float:
    dv = final_conditions[:, 41]       # always meaningful now
    g_max = final_conditions[:, 17]
    q_max = final_conditions[:, 16]

    cost = log_cap(dv, threshold=dv_threshold)

    # Soft constraint penalties (unchanged)
    g_penalty = g_load_weight * np.maximum((g_max - g_load_limit) / g_load_limit, 0) ** 2
    q_penalty = heat_flux_weight * np.maximum((q_max - heat_flux_limit) / heat_flux_limit, 0) ** 2

    cost += g_penalty + q_penalty
    return float(np.sqrt(np.mean(cost ** 2)))  # RMS aggregation
```

Callers continue to use `compute_cost(final, **cost_kwargs)` — no call-site changes needed.

**Removed:**
- Branching on `hyperbolic | (dv_total > 1e10)`
- The `1e6 + 1e3 * |energy|` non-capture penalty tier
- The `np.clip(dv, 0, 1e4)` cap

**TOML addition** in `configs/training/common.toml`:
```toml
[cost_function]
dv_threshold = 1000.0   # NEW: log-cap threshold (m/s)
```

**TOML parsing** in `train.py`: add `dv_threshold` extraction from `cfg.cost_function` into `cost_kwargs` alongside the existing `g_load_limit`, `heat_flux_limit`, etc.

### 4. Sentinel Chromosomes in Piecewise-Constant Training (Python)

**File:** `src/python/aerocapture/training/train.py` — piecewise_constant generation loop

After evaluating the GA population each generation, evaluate 11 additional constant-bank-angle chromosomes:

```python
SENTINEL_BANK_ANGLES = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]  # degrees
```

Each sentinel is a 10-segment chromosome with all segments set to the same bank angle: `[angle] * 10`.

**Mechanics:**
- Evaluated with the **same MC seed set** as the GA population that generation
- Use the existing `run_batch()` / `evaluate_chromosome()` pipeline
- Trajectories classified via `classify_trajectories()` and fed to `CorridorAccumulator.update()` alongside population trajectories
- Sentinels do NOT participate in selection, crossover, or mutation
- Results excluded from fitness stats / JSONL logging

**Computational overhead:** 11 x `n_sims` extra sims per generation (~27% overhead for `n_pop=40`). Fully parallelizable via Rayon.

**Physical coverage:**
- 0° (full lift-up): traces the hyperbolic/ricochet lower boundary
- 90° (zero lift): near-ballistic midline
- 180° (full lift-down): traces the crash upper boundary
- Intermediate angles: fill transition zones

**Why only 0°–180° (not negative angles):** Piecewise-constant uses signed bank angles where the sign controls lateral roll direction. But the corridor is plotted in energy-vs-pdyn space, which depends only on the bank angle magnitude (the lift component in the vertical plane). A trajectory at -45° has the same pdyn/energy profile as +45° — only the ground track differs. So 0°–180° covers the full corridor.

**No changes needed to:** `evaluate.py`, `corridor.py`.

## Testing Strategy

### Rust Tests

- **Pending crash detection:** Unit test — sim that exits atmosphere with captured orbit but apoapsis < 130 km -> verify `ifinal == 4`
- **Hyperbolic NOT reclassified:** Test that a hyperbolic exit (ecc > 1, negative apoapsis_alt) stays `ifinal == 3`, NOT reclassified as pending crash
- **Virtual DV for hyperbolic:** Test that a hyperbolic exit gets `dv = HYPERBOLIC_BASE + v_excess` (finite, > 10,000)
- **Virtual DV for crash:** Test proportional decay: `dv = CRASH_BASE * (1 - 0.5 * t / max_t)` stays in `[10000, 20000]`
- **maneuver.rs:** Update `non_exit_returns_penalty` test — `compute_deltav` is no longer called for non-exits; verify the new virtual DV path instead
- **Regression:** Golden test configs produce identical results for captured trajectories (real DV path unchanged)

### Python Tests

- **`log_cap`:** Verify C0/C1 continuity at threshold (value and numerical derivative match), correct linear/log behavior on each side, safety floor at dv=0
- **Unified `compute_cost`:** Mock final_records with mix of captures, hyperbolic, crashes — verify no NaN/inf, monotonic ordering (crash > hyperbolic > bad capture > good capture)
- **`classify_trajectories` with ifinal=4:** Verify pending crash classified same as crash
- **`metrics.py` capture_rate:** Verify updated threshold works with new cost scale
- **`compare_guidance.py`:** Verify no `dv > 1e10` filtering remains
- **Sentinel integration:** Test that sentinel chromosomes are correctly constructed and their trajectories feed the accumulator
- **TOML parsing:** Verify `dv_threshold` is correctly parsed from TOML and propagated to `cost_kwargs`

### Property-Based Tests

- `log_cap(dv)` monotonically increasing for all `dv > 0` (hypothesis)
- `log_cap` numerically stable at exact threshold boundary from both sides (hypothesis)
- Virtual DV always finite and positive for any termination scenario (proptest)
- Crash virtual DV always in `[CRASH_BASE/2, CRASH_BASE]` for any `sim_time` and `max_time` (proptest)
- Cost ordering invariant: `cost(crash) > cost(hyperbolic) > cost(bad_capture)` (proptest)

## Files Modified

| File | Change |
|------|--------|
| `src/rust/src/simulation/runner.rs` | Pending crash detection, virtual DV computation, module-level constants |
| `src/rust/src/orbit/maneuver.rs` | Remove `1e30` early return (only called for captures now) |
| `src/python/aerocapture/training/evaluate.py` | Unified `compute_cost()` with `log_cap()` |
| `src/python/aerocapture/training/corridor.py` | Treat `ifinal == 4` as crash in `classify_trajectories` |
| `src/python/aerocapture/training/train.py` | Sentinel chromosome evaluation, `dv_threshold` TOML parsing into `cost_kwargs` |
| `src/python/aerocapture/training/metrics.py` | Update `capture_rate` default threshold from `1e6` to `3000.0` (above worst captured cost ~2600, below all non-capture costs ~3300+, giving ~700 gap) |
| `src/python/aerocapture/training/compare_guidance.py` | Remove `dv > 1e10` filtering; fix 5 pre-existing column index bugs (energy: 8->7, ecc: 10->9, apo_err: 31->30, peri_err: 30->29, DV: 42->41); add `ifinal != 4` exclusion from capture stats; parse `dv_threshold` from TOML into cost_kwargs |
| `src/python/aerocapture/training/final_report.py` | Update `captured` derivation to exclude pending crashes: `(ecc < 1.0) & (energy < 0) & (ifinal != 4)` |
| `configs/training/common.toml` | Add `dv_threshold = 1000.0` to `[cost_function]` |
| `tests/test_cost.py` | Rewrite tests for new unified cost function (remove two-tier tests) |
| `tests/test_corridor.py` | Add `ifinal == 4` classification tests |
