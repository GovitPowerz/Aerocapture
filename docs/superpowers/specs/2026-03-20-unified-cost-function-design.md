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
if term == AtmosphereExit && orbit.apoapsis_alt < exit_altitude {
    term = PendingCrash
}
```

- `exit_altitude` = `data.final_conditions.altitude` (Mars: 130,988 m)
- `orbit.apoapsis_alt` is already computed (meters)
- New `TermReason::PendingCrash` variant maps to `ifinal = 4`
- Python `classify_trajectories` treats `ifinal == 4` identically to `ifinal == 1` (crash)

### 2. Meaningful DV for All Outcomes (Rust)

**File:** `src/rust/src/simulation/runner.rs` (post-termination DV computation)

Instead of routing all outcomes through `compute_deltav` (which returns `1e30` for non-exits), compute DV contextually:

```
match (term, captured) {
    (AtmosphereExit, true)  -> compute_deltav()           // real orbital correction DV
    (AtmosphereExit, false) -> HYPERBOLIC_BASE + v_excess  // velocity surplus over escape
    (Crash | PendingCrash)  -> CRASH_BASE - CRASH_TIME_DECAY * sim_time
    (Timeout)               -> CRASH_BASE - CRASH_TIME_DECAY * sim_time
}
```

**Constants (hardcoded, not TOML-configurable):**
- `HYPERBOLIC_BASE = 10,000 m/s` — clearly worse than any realistic captured DV
- `CRASH_BASE = 20,000 m/s` — clearly worse than any hyperbolic exit
- `CRASH_TIME_DECAY = 1.0 m/s/s` — small gradient so GA prefers longer survival

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
| Late crash (t=2000s) | 18,000 m/s | 3,890 |
| Immediate crash (t=100s) | 19,900 m/s | 3,990 |

### 3. Unified Smooth Cost Function (Python)

**File:** `src/python/aerocapture/training/evaluate.py` — `compute_cost()`

Replace the two-tier branching system with a single smooth function:

```python
def log_cap(dv: np.ndarray, threshold: float = 1000.0) -> np.ndarray:
    """C1-continuous log-capped cost: linear below threshold, log above."""
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

**New `compute_cost()`:**

```python
def compute_cost(final_records, cost_kwargs):
    dv = final_records[:, 41]          # always meaningful now
    g_max = final_records[:, 17]
    q_max = final_records[:, 16]

    cost = log_cap(dv, threshold=cost_kwargs.get("dv_threshold", 1000.0))

    # Soft constraint penalties (unchanged)
    g_limit = cost_kwargs["g_load_limit"]
    q_limit = cost_kwargs["heat_flux_limit"]
    g_penalty = cost_kwargs["g_load_weight"] * np.maximum((g_max - g_limit) / g_limit, 0) ** 2
    q_penalty = cost_kwargs["heat_flux_weight"] * np.maximum((q_max - q_limit) / q_limit, 0) ** 2

    cost += g_penalty + q_penalty
    return np.sqrt(np.mean(cost ** 2))  # RMS aggregation
```

**Removed:**
- Branching on `hyperbolic | (dv_total > 1e10)`
- The `1e6 + 1e3 * |energy|` non-capture penalty tier
- The `np.clip(dv, 0, 1e4)` cap

**TOML addition** in `configs/training/common.toml`:
```toml
[cost_function]
dv_threshold = 1000.0   # NEW: log-cap threshold (m/s)
```

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

**No changes needed to:** `evaluate.py`, `corridor.py`, `final_report.py`.

## Testing Strategy

### Rust Tests

- **Pending crash detection:** Unit test — sim with extreme bank-down that exits atmosphere with apoapsis < 130 km -> verify `ifinal == 4`
- **Virtual DV for hyperbolic:** Test that a hyperbolic exit gets `dv.total = HYPERBOLIC_BASE + v_excess` (finite, > 10,000)
- **Virtual DV for crash:** Test that crash gets `dv.total = CRASH_BASE - decay * time` (finite, > 18,000 for short sim)
- **maneuver.rs:** Update `non_exit_returns_penalty` test — `compute_deltav` is no longer called for non-exits; verify the new virtual DV path instead
- **Regression:** Golden test configs produce identical results for captured trajectories (real DV path unchanged)

### Python Tests

- **`log_cap`:** Verify C0/C1 continuity at threshold (value and numerical derivative match), correct linear/log behavior on each side
- **Unified `compute_cost`:** Mock final_records with mix of captures, hyperbolic, crashes — verify no NaN/inf, monotonic ordering (crash > hyperbolic > bad capture > good capture)
- **`classify_trajectories` with ifinal=4:** Verify pending crash classified same as crash
- **Sentinel integration:** Test that sentinel chromosomes are correctly constructed and their trajectories feed the accumulator

### Property-Based Tests

- `log_cap(dv)` monotonically increasing for all `dv > 0` (hypothesis)
- Virtual DV always finite and positive for any termination scenario (proptest)
- Cost ordering invariant: `cost(crash) > cost(hyperbolic) > cost(bad_capture)` (proptest)

## Files Modified

| File | Change |
|------|--------|
| `src/rust/src/simulation/runner.rs` | Pending crash detection, virtual DV computation |
| `src/rust/src/orbit/maneuver.rs` | Remove `1e30` early return (only called for captures now) |
| `src/python/aerocapture/training/evaluate.py` | Unified `compute_cost()` with `log_cap()` |
| `src/python/aerocapture/training/corridor.py` | Treat `ifinal == 4` as crash in `classify_trajectories` |
| `src/python/aerocapture/training/train.py` | Sentinel chromosome evaluation in piecewise_constant loop |
| `configs/training/common.toml` | Add `dv_threshold = 1000.0` to `[cost_function]` |
