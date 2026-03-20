# Unified Cost Function Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-tier cost function (1e30 penalties, 1e6 non-capture branch) with a unified C1-continuous log-capped function, add pending-crash detection in Rust, and integrate sentinel chromosomes for corridor boundary resolution.

**Architecture:** Rust-side changes (pending crash detection + meaningful virtual DVs for all outcomes) feed into a simplified Python cost function (`log_cap`). Sentinel constant-bank-angle chromosomes are evaluated alongside the GA population during piecewise_constant training to improve corridor envelopes. The cost landscape compresses from [0, 1e30] to [0, ~4000] while preserving monotonic ordering across all outcome tiers.

**Tech Stack:** Rust (nalgebra, serde, toml), Python (numpy, deap), PyO3 (maturin), pytest, hypothesis, proptest

**Spec:** `docs/superpowers/specs/2026-03-20-unified-cost-function-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/rust/src/simulation/runner.rs` | Modify | Add `PendingCrash` variant, virtual DV computation, constants |
| `src/rust/src/orbit/maneuver.rs` | Modify | Remove `1e30` early return, guard with assertion |
| `src/python/aerocapture/training/evaluate.py` | Modify | Replace `compute_cost` with `log_cap`-based unified version |
| `src/python/aerocapture/training/corridor.py` | Modify | Handle `ifinal == 4` in `classify_trajectories` |
| `src/python/aerocapture/training/train.py` | Modify | Add sentinel evaluation, parse `dv_threshold` |
| `src/python/aerocapture/training/metrics.py` | Modify | Update `capture_rate` threshold |
| `src/python/aerocapture/training/compare_guidance.py` | Modify | Fix 5 column index bugs, add `dv_threshold`, exclude `ifinal=4` |
| `src/python/aerocapture/training/final_report.py` | Modify | Exclude `ifinal=4` from captured stats |
| `configs/training/common.toml` | Modify | Add `dv_threshold` |
| `tests/test_cost.py` | Rewrite | New tests for unified cost function |
| `tests/test_corridor.py` | Modify | Add `ifinal=4` classification tests |

---

## Task 1: Rust — Add `PendingCrash` Variant and Reclassification

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:64-71` (TermReason enum)
- Modify: `src/rust/src/simulation/runner.rs:606-634` (post-termination block)

- [ ] **Step 1: Add `PendingCrash` to `TermReason` enum**

In `src/rust/src/simulation/runner.rs`, change the enum at line 64:

```rust
/// Termination reason
#[derive(Debug, Clone, Copy, PartialEq)]
enum TermReason {
    None,
    Crash,
    Timeout,
    AtmosphereExit,
    PendingCrash,
}
```

- [ ] **Step 2: Add reclassification logic after orbital elements**

Replace the `ifinal` assignment block (lines 630–634) with:

```rust
// Pending crash: captured orbit with apoapsis below atmosphere ceiling
let captured = orbit.eccentricity < 1.0 && energy < 0.0;
let exit_altitude = data.final_conditions.altitude;
if term == TermReason::AtmosphereExit && captured && orbit.apoapsis_alt < exit_altitude {
    term = TermReason::PendingCrash;
}

let ifinal = match term {
    TermReason::AtmosphereExit => 3,
    TermReason::Crash => 1,
    TermReason::PendingCrash => 4,
    _ => 2,
};
```

Note: `data.final_conditions.altitude` is in meters (Mars: 130,988 m). `orbit.apoapsis_alt` is also in meters.

- [ ] **Step 3: Build to verify compilation**

Run: `cd src/rust && cargo build --release 2>&1 | tail -5`
Expected: successful build (warnings about unused `captured` are OK for now — used in Task 2)

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "feat(rust): add PendingCrash termination variant (ifinal=4)

Reclassify atmosphere exits where the captured orbit has apoapsis
below the atmosphere ceiling as pending crashes. These trajectories
are gravitationally captured but operationally doomed to re-enter."
```

---

## Task 2: Rust — Virtual DV for All Outcomes

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:616-641` (DV computation)
- Modify: `src/rust/src/orbit/maneuver.rs:37-43` (remove 1e30 early return)

- [ ] **Step 1: Add module-level constants and DeltaV import in `runner.rs`**

Add the import near the top of the file (with other `use` statements):

```rust
use crate::orbit::maneuver::DeltaV;
```

Add constants after the imports, before the structs (around line 20):

```rust
/// Virtual DV base for hyperbolic exits (m/s).
/// Set above any realistic captured orbit correction DV.
const HYPERBOLIC_BASE: f64 = 10_000.0;

/// Virtual DV base for crash/timeout (m/s).
/// Set above any hyperbolic virtual DV to maintain cost ordering.
const CRASH_BASE: f64 = 20_000.0;
```

- [ ] **Step 2: Replace DV computation in `run_sim`**

Replace the current `compute_deltav` call (lines 635–641) with context-aware DV. Note: `max_time` is already a local variable (line 416: `let max_time = config.max_time;`) — reuse it directly:

```rust
let deltav = if term == TermReason::AtmosphereExit && captured {
    // Real orbital correction DV
    maneuver::compute_deltav(
        &orbit,
        &data.target_orbit,
        &data.parking_orbit,
        planet,
    )
} else if term == TermReason::AtmosphereExit {
    // Hyperbolic exit: excess velocity over escape speed
    let v_escape = (2.0 * mu / sim.state[0]).sqrt();
    let v_excess = (speed_abs - v_escape).max(0.0);
    DeltaV {
        dv1: 0.0,
        dv2: 0.0,
        dv3: 0.0,
        total: HYPERBOLIC_BASE + v_excess,
    }
} else {
    // Crash, PendingCrash, or Timeout: proportional time decay
    let virtual_dv = CRASH_BASE * (1.0 - 0.5 * sim_time / max_time);
    DeltaV {
        dv1: 0.0,
        dv2: 0.0,
        dv3: 0.0,
        total: virtual_dv,
    }
};
```

Note: For virtual DV cases (hyperbolic/crash), `dv1/dv2/dv3` are 0.0 — they have no physical meaning. This means `final_record[40]` (`dv1+dv2`) will be 0.0 while `final_record[41]` (`total`) holds the virtual value. This is correct — individual burn components are undefined for non-captures.

- [ ] **Step 3: Update `run_for_api` captured flag**

In `run_for_api()` (line 232–237), `energy` and `ecc` are already extracted from `r.final_line[7]` and `r.final_line[9]` at lines 207–208. Add `ifinal_val` and update the `captured` field:

```rust
let ifinal_val = r.final_line[31] as i32;
crate::RunOutput {
    trajectory,
    final_record: r.final_line,
    captured: ecc < 1.0 && energy < 0.0 && ifinal_val != 4,
    dispersions: r.dispersions,
}
```

Also update the `RunOutput` docstring in `lib.rs` (line ~17) to reflect the new semantics:

```rust
/// True if orbit is bound (ecc < 1 && energy < 0) and not a pending crash (ifinal != 4).
pub captured: bool,
```

- [ ] **Step 4: Update `maneuver.rs` — remove `ifinal` parameter**

The function is now only called for confirmed captures. Remove the `ifinal` parameter and the `1e30` early return:

```rust
/// Compute delta-V cost for orbit correction.
///
/// Only called for confirmed captured trajectories (ecc < 1, energy < 0).
/// Maneuver 1 at apoapsis: correct periapsis to target
/// Maneuver 2 at new periapsis: correct apoapsis (circularize)
/// Maneuver 3: inclination plane change at ascending/descending node
pub fn compute_deltav(
    orbit: &OrbitalElements,
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &Planet,
) -> DeltaV {
    let mu = planet.mu();
    let req = planet.equatorial_radius();

    let rapoge = req + orbit.apoapsis_alt;
    let rperig = req + orbit.periapsis_alt;
    // ... rest unchanged from line 48 onwards ...
```

- [ ] **Step 5: Build to verify compilation**

Run: `cd src/rust && cargo build --release 2>&1 | tail -5`
Expected: successful build

- [ ] **Step 6: Update maneuver.rs tests**

**Delete** `non_exit_returns_penalty` — no longer applicable (function is only called for captures).

**Update all remaining tests** to remove the `ifinal` parameter (was the 2nd arg). Each test changes from 5-param to 4-param calls:

- `exit_returns_finite_cost` (line 167): `compute_deltav(&orbit, 3, &target, ...)` → `compute_deltav(&orbit, &target, ...)`
- `total_is_sum_of_abs` (line 179): same signature change
- `optimal_has_zero_dv3` (line 192): uses `compute_deltav_optimal` — no change needed
- `zero_inclination_error_small_dv3` (lines 207, 216): two calls, same signature change

- [ ] **Step 7: Run Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass. Some integration/E2E tests may need updating if they check DV values for non-capture cases — those now get virtual DVs instead of 1e30.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/orbit/maneuver.rs
git commit -m "feat(rust): unified virtual DV for all termination outcomes

Replace 1e30 sentinel values with physics-based virtual DVs:
- Captured: real orbital correction DV (unchanged)
- Hyperbolic: HYPERBOLIC_BASE (10000) + excess velocity
- Crash/PendingCrash/Timeout: CRASH_BASE (20000) * proportional decay

Remove ifinal parameter from compute_deltav — now only called
for confirmed captures."
```

---

## Task 3: Rust — Integration & Property-Based Tests for New Termination Logic

**Files:**
- Modify: `src/rust/tests/` (integration tests that check DV or ifinal values)
- Modify: `src/rust/src/simulation/runner.rs` (add proptest module)

- [ ] **Step 1: Check which integration tests reference 1e30 or ifinal**

Run: `grep -rn "1e30\|ifinal\|non_exit" src/rust/tests/`

Update any tests that expect `1e30` DV for non-capture trajectories to expect the new virtual DV ranges:
- Crash: `[10_000, 20_000]`
- Hyperbolic: `>= 10_000`
- Captured: finite, positive, realistic

Note: integration tests in `src/rust/tests/` may have NO matches — the `1e30` was only in `maneuver.rs` unit tests (already updated in Task 2). If no matches, skip to Step 2.

- [ ] **Step 2: Add proptest for virtual DV in `runner.rs`**

Add to the existing `#[cfg(test)]` module in `runner.rs` (or create one if not present):

```rust
#[cfg(test)]
mod virtual_dv_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn crash_virtual_dv_in_range(
            sim_time in 0.0f64..10000.0,
            max_time in 100.0f64..10000.0,
        ) {
            let virtual_dv = CRASH_BASE * (1.0 - 0.5 * sim_time / max_time);
            // Always positive (even if sim_time > max_time)
            // Range: [CRASH_BASE * 0.5, CRASH_BASE] when sim_time in [0, max_time]
            prop_assert!(virtual_dv.is_finite());
            if sim_time <= max_time {
                prop_assert!(virtual_dv >= CRASH_BASE * 0.5);
                prop_assert!(virtual_dv <= CRASH_BASE);
            }
        }

        #[test]
        fn hyperbolic_virtual_dv_above_base(
            v_excess in 0.0f64..5000.0,
        ) {
            let virtual_dv = HYPERBOLIC_BASE + v_excess;
            prop_assert!(virtual_dv >= HYPERBOLIC_BASE);
            prop_assert!(virtual_dv.is_finite());
        }
    }

    #[test]
    fn cost_ordering_crash_gt_hyperbolic_gt_capture() {
        let capture_dv = 500.0;  // realistic captured DV
        let hyperbolic_dv = HYPERBOLIC_BASE + 100.0;  // barely hyperbolic
        let crash_dv = CRASH_BASE * 0.9;  // typical crash
        assert!(crash_dv > hyperbolic_dv);
        assert!(hyperbolic_dv > capture_dv);
    }
}
```

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Run golden regression tests**

Run: `cd src/rust && cargo test --test e2e 2>&1 | tail -20`
Expected: captured trajectory golden tests still pass (DV unchanged for captures)

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/tests/
git commit -m "test(rust): add proptest for virtual DV ranges and cost ordering"
```

---

## Task 4: Rebuild PyO3 Bindings

**Files:**
- Build: `src/rust/aerocapture-py/`

- [ ] **Step 1: Rebuild PyO3 module**

Run: `cd src/rust/aerocapture-py && uv run maturin develop --release 2>&1 | tail -5`
Expected: successful build of `aerocapture_rs` module

- [ ] **Step 2: Quick smoke test**

Run: `uv run python -c "import aerocapture_rs; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit (no source changes, just verify build)**

No commit needed — this is a build step.

---

## Task 5: Python — Unified `compute_cost` with `log_cap`

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:229-279`
- Rewrite: `tests/test_cost.py`

- [ ] **Step 1: Write failing tests for `log_cap`**

Create/rewrite `tests/test_cost.py`:

```python
"""Tests for the unified cost function with log-cap compression."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aerocapture.training.evaluate import compute_cost, log_cap


class TestLogCap:
    """Tests for the C1-continuous log-cap function."""

    def test_linear_below_threshold(self) -> None:
        dv = np.array([100.0, 500.0, 999.0])
        result = log_cap(dv, threshold=1000.0)
        np.testing.assert_array_almost_equal(result, dv)

    def test_log_above_threshold(self) -> None:
        dv = np.array([2000.0, 5000.0, 10000.0])
        result = log_cap(dv, threshold=1000.0)
        expected = 1000.0 * (1.0 + np.log(dv / 1000.0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_c0_continuity_at_threshold(self) -> None:
        """Both sides evaluate to T at the threshold."""
        t = 1000.0
        below = log_cap(np.array([t - 1e-10]), threshold=t)[0]
        above = log_cap(np.array([t + 1e-10]), threshold=t)[0]
        assert abs(below - above) < 1e-6

    def test_c1_continuity_at_threshold(self) -> None:
        """Numerical derivative matches from both sides (slope = 1)."""
        t = 1000.0
        eps = 1e-6
        left_deriv = (log_cap(np.array([t]), t)[0] - log_cap(np.array([t - eps]), t)[0]) / eps
        right_deriv = (log_cap(np.array([t + eps]), t)[0] - log_cap(np.array([t]), t)[0]) / eps
        assert abs(left_deriv - 1.0) < 1e-3
        assert abs(right_deriv - 1.0) < 1e-3

    def test_safety_floor(self) -> None:
        """DV at 0 or negative doesn't produce NaN/inf."""
        result = log_cap(np.array([0.0, -1.0]), threshold=1000.0)
        assert np.all(np.isfinite(result))

    @given(st.floats(min_value=0.01, max_value=1e6))
    @settings(max_examples=200)
    def test_monotonically_increasing(self, dv: float) -> None:
        eps = 1.0
        v1 = log_cap(np.array([dv]), threshold=1000.0)[0]
        v2 = log_cap(np.array([dv + eps]), threshold=1000.0)[0]
        assert v2 >= v1


class TestUnifiedComputeCost:
    """Tests for the unified compute_cost function."""

    @staticmethod
    def _make_final(n: int, dv: float = 200.0, g: float = 5.0, q: float = 100.0) -> np.ndarray:
        """Build a mock final_records array with n rows."""
        arr = np.zeros((n, 52))
        arr[:, 41] = dv   # dv_total
        arr[:, 17] = g    # g_max
        arr[:, 16] = q    # q_max
        return arr

    def test_good_capture_cost_equals_dv(self) -> None:
        final = self._make_final(5, dv=200.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert abs(cost - 200.0) < 1.0  # RMS of identical values = that value

    def test_bad_capture_log_compressed(self) -> None:
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        # log_cap(5000, 1000) = 1000*(1+ln(5)) ≈ 2609
        assert 2500 < cost < 2700

    def test_crash_dv_produces_high_cost(self) -> None:
        """Virtual DV of 20000 (fresh crash) -> log_cap ≈ 3996."""
        final = self._make_final(5, dv=20000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert 3900 < cost < 4100

    def test_cost_ordering(self) -> None:
        """crash > hyperbolic > bad_capture > good_capture."""
        good = compute_cost(self._make_final(5, dv=200.0))
        bad = compute_cost(self._make_final(5, dv=5000.0))
        hyper = compute_cost(self._make_final(5, dv=10500.0))
        crash = compute_cost(self._make_final(5, dv=19000.0))
        assert good < bad < hyper < crash

    def test_g_load_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, g=10.0))
        with_penalty = compute_cost(self._make_final(5, dv=200.0, g=20.0))
        assert with_penalty > no_penalty

    def test_heat_flux_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, q=100.0))
        with_penalty = compute_cost(self._make_final(5, dv=200.0, q=300.0))
        assert with_penalty > no_penalty

    def test_custom_dv_threshold(self) -> None:
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost_low_t = compute_cost(final, dv_threshold=500.0)
        cost_high_t = compute_cost(final, dv_threshold=2000.0)
        # Lower threshold = more compression = lower cost
        assert cost_low_t < cost_high_t

    @given(st.floats(min_value=1.0, max_value=50000.0))
    @settings(max_examples=100)
    def test_cost_always_finite(self, dv: float) -> None:
        final = self._make_final(3, dv=dv, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert np.isfinite(cost)
        assert cost >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py -v 2>&1 | tail -20`
Expected: FAIL (either `log_cap` not found, or `compute_cost` behavior differs)

- [ ] **Step 3: Implement `log_cap` and update `compute_cost`**

In `src/python/aerocapture/training/evaluate.py`:

Add the `log_cap` function before `compute_cost` (around line 229):

```python
def log_cap(dv: npt.NDArray[np.float64], threshold: float = 1000.0) -> npt.NDArray[np.float64]:
    """C1-continuous log-capped cost: linear below threshold, log above.

    Properties:
        - C0 continuous at threshold: both sides evaluate to T
        - C1 continuous at threshold: both sides have derivative 1
        - Monotonically increasing for all dv > 0
    """
    dv = np.maximum(dv, 1e-6)  # safety floor
    below = dv <= threshold
    result = np.empty_like(dv)
    result[below] = dv[below]
    result[~below] = threshold * (1.0 + np.log(dv[~below] / threshold))
    return result
```

Replace `compute_cost` (lines 229–279) with:

```python
def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,
    heat_flux_limit: float = 200.0,
    g_load_weight: float = 1000.0,
    heat_flux_weight: float = 1000.0,
) -> float:
    """Compute RMS cost from simulation final conditions.

    Uses log-capped delta-V as the primary objective with normalized
    soft constraint penalties for g-load and heat flux exceedances.

    All termination outcomes now produce meaningful DV values from Rust:
    - Captured: real orbital correction DV
    - Hyperbolic: 10000 + excess velocity
    - Crash/PendingCrash/Timeout: 20000 * proportional time decay

    Returns:
        RMS cost value. Lower is better.
    """
    dv_total = final_conditions[:, 41]
    g_max = final_conditions[:, 17]
    q_max = final_conditions[:, 16]

    costs = log_cap(dv_total, threshold=dv_threshold)

    g_penalty = g_load_weight * np.maximum((g_max - g_load_limit) / g_load_limit, 0) ** 2
    q_penalty = heat_flux_weight * np.maximum((q_max - heat_flux_limit) / heat_flux_limit, 0) ** 2
    costs = costs + g_penalty + q_penalty

    return float(np.sqrt(np.mean(costs**2)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py -v 2>&1 | tail -30`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py tests/test_cost.py
git commit -m "feat(python): unified log-capped cost function

Replace two-tier cost (1e6 non-capture penalty + clipped DV) with
single C1-continuous log_cap function. All DV values are now
meaningful from Rust, so no branching on capture status needed."
```

---

## Task 6: Python — Update `classify_trajectories` for `ifinal=4`

**Files:**
- Modify: `src/python/aerocapture/training/corridor.py:66`
- Modify: `tests/test_corridor.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_corridor.py`:

```python
def test_pending_crash_classified_as_crash(self) -> None:
    """ifinal=4 (pending crash) should be classified same as ifinal=1."""
    fr = np.zeros((2, 52))
    # First row: real crash (ifinal=1)
    fr[0, 31] = 1.0
    # Second row: pending crash (ifinal=4)
    fr[1, 31] = 4.0
    labels = classify_trajectories(fr)
    assert labels[0] == "crash"
    assert labels[1] == "crash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corridor.py::TestClassifyTrajectories::test_pending_crash_classified_as_crash -v`
Expected: FAIL (ifinal=4 currently falls through unclassified)

- [ ] **Step 3: Update `classify_trajectories`**

In `src/python/aerocapture/training/corridor.py`, line 66, change:

```python
crash = ifinal == 1.0
```

to:

```python
crash = (ifinal == 1.0) | (ifinal == 4.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_corridor.py -v 2>&1 | tail -20`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/corridor.py tests/test_corridor.py
git commit -m "feat(python): classify ifinal=4 (pending crash) as crash"
```

---

## Task 7: Python — Update `metrics.py`, `final_report.py`, `compare_guidance.py`

**Files:**
- Modify: `src/python/aerocapture/training/metrics.py:49`
- Modify: `src/python/aerocapture/training/final_report.py:199`
- Modify: `src/python/aerocapture/training/compare_guidance.py:135-160`

- [ ] **Step 1: Update `capture_rate` threshold in `metrics.py`**

Change line 49:

```python
def capture_rate(costs: npt.NDArray[np.float64], capture_threshold: float = 3000.0) -> float:
    """Fraction of individuals with cost below capture threshold.

    Default threshold 3000 separates captured trajectories (max ~2600
    after log compression) from non-captures (min ~3300).
    """
    return float(int(np.sum(costs < capture_threshold)) / len(costs))
```

- [ ] **Step 2: Update `final_report.py` captured derivation**

In `src/python/aerocapture/training/final_report.py`, add `_COL_IFINAL = 31` to the existing column constants block (lines 27–41, alongside `_COL_ECC`, `_COL_ENERGY`, etc.), then at line 199 change:

```python
captured = (ecc < 1.0) & (energy < 0)
```

to:

```python
ifinal = final_array[:, _COL_IFINAL]
captured = (ecc < 1.0) & (energy < 0) & (ifinal != 4.0)
```

- [ ] **Step 3: Fix `compare_guidance.py` — 5 column index bugs + updates**

In `src/python/aerocapture/training/compare_guidance.py`, fix lines 135–152:

```python
energy = final[:, 7]       # was 8 (SMA)
ecc = final[:, 9]          # was 10 (inclination)
ifinal = final[:, 31]
captured = (ecc < 1.0) & (energy < 0) & (ifinal != 4.0)

metrics: dict = {
    "n_sims": len(final),
    "captured": int(captured.sum()),
    "capture_rate": float(captured.sum()) / len(final) * 100,
    "cost": compute_cost(final, **(cost_kwargs or {})),
}

if captured.any():
    metrics["apo_err_mean"] = float(np.abs(final[captured, 30]).mean())   # was 31
    metrics["apo_err_std"] = float(np.abs(final[captured, 30]).std())     # was 31
    metrics["peri_err_mean"] = float(np.abs(final[captured, 29]).mean())  # was 30
    metrics["peri_err_std"] = float(np.abs(final[captured, 29]).std())    # was 30
    dv = final[captured, 41]  # was 42
    metrics["dv_mean"] = float(np.mean(dv))
    metrics["dv_std"] = float(np.std(dv))
```

Also add `dv_threshold` parsing in the `__main__` block (around line 212):

```python
cost_kwargs = {
    "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
    "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
    "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
    "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
    "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
}
```

Remove the `dv > 1e10` → `np.nan` filtering (line ~152) — no longer needed since all DVs are meaningful.

- [ ] **Step 4: Run full Python test suite**

Run: `uv run pytest tests/ -v 2>&1 | tail -30`
Expected: all tests PASS (some may need updating if they test the old `compute_cost` behavior)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/metrics.py \
       src/python/aerocapture/training/final_report.py \
       src/python/aerocapture/training/compare_guidance.py
git commit -m "fix(python): update downstream consumers for unified cost

- metrics.py: capture_rate threshold 1e6 -> 3000
- final_report.py: exclude ifinal=4 from captured stats
- compare_guidance.py: fix 5 column index bugs, add dv_threshold,
  exclude ifinal=4, remove stale dv>1e10 filtering"
```

---

## Task 8: Python — Parse `dv_threshold` from TOML + Update `common.toml`

**Files:**
- Modify: `src/python/aerocapture/training/train.py:270-275`
- Modify: `configs/training/common.toml`

- [ ] **Step 1: Add `dv_threshold` to `common.toml`**

In `configs/training/common.toml`, add to the `[cost_function]` section:

```toml
[cost_function]
dv_threshold = 1000.0
g_load_limit = 15.0
heat_flux_limit = 200.0
g_load_weight = 1000.0
heat_flux_weight = 1000.0
```

- [ ] **Step 2: Parse `dv_threshold` in `train.py`**

In `src/python/aerocapture/training/train.py`, update lines 270–275:

```python
cost_kwargs = {
    "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
    "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
    "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
    "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
    "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
}
```

- [ ] **Step 3: Run linter and type checker**

Run: `./lint_code.sh 2>&1 | tail -20`
Expected: no new errors

- [ ] **Step 4: Commit**

```bash
git add configs/training/common.toml src/python/aerocapture/training/train.py
git commit -m "feat: add dv_threshold to TOML config and train.py parsing"
```

---

## Task 9: Python — Sentinel Chromosomes in Piecewise-Constant Training

**Files:**
- Modify: `src/python/aerocapture/training/train.py:565-587` (corridor accumulation block)

- [ ] **Step 1: Add sentinel constant at module level**

Near the top of `train.py` (with other constants):

```python
# Constant bank angles for corridor boundary sentinels (degrees).
# 0° = full lift-up (hyperbolic boundary), 180° = full lift-down (crash boundary).
# Only magnitude affects energy-vs-pdyn corridor; sign only affects lateral track.
_SENTINEL_BANK_ANGLES = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
```

- [ ] **Step 2: Add sentinel evaluation after population corridor update**

After the existing corridor accumulation block (line 587), add sentinel evaluation:

```python
# Sentinel chromosomes: constant bank angles for corridor boundary resolution
sentinel_overrides: list[dict[str, object]] = []
for bank in _SENTINEL_BANK_ANGLES:
    ovr_s: dict[str, object] = {
        f"guidance.{section}.bank_angle_{i}": float(bank)
        for i in range(10)
    }
    ovr_s["guidance.type"] = config.guidance_type
    ovr_s["simulation.n_sims"] = 1
    sentinel_overrides.append(ovr_s)

sentinel_results = _aero_rs.run_batch(  # type: ignore[union-attr]
    toml_path=corr_toml_path,
    overrides_list=sentinel_overrides,
    include_trajectories=True,
)
sentinel_labels = classify_traj(
    sentinel_results.final_records,
    delta_za_low=corridor_acc.delta_za_low,
    delta_za_high=corridor_acc.delta_za_high,
)
corridor_acc.update(sentinel_results.trajectories, sentinel_labels)
```

Note: This reuses `section`, `corr_toml_path`, `corridor_acc`, and `classify_traj` that are already in scope from the existing corridor block.

- [ ] **Step 3: Run linter**

Run: `./lint_code.sh 2>&1 | tail -20`
Expected: no new errors

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(python): add sentinel chromosomes for corridor boundaries

Evaluate 11 constant-bank-angle chromosomes (0° to 180° in 18° steps)
alongside the GA population during piecewise_constant training.
Uses same MC seeds as population. Trajectories feed into
CorridorAccumulator for better corridor envelope resolution."
```

---

## Task 10: Full Integration Test Pass

**Files:**
- All modified files

- [ ] **Step 1: Run Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all pass

- [ ] **Step 2: Run Rust clippy**

Run: `cd src/rust && cargo clippy -- -D warnings 2>&1 | tail -20`
Expected: no warnings

- [ ] **Step 3: Rebuild PyO3**

Run: `cd src/rust/aerocapture-py && uv run maturin develop --release 2>&1 | tail -5`
Expected: success

- [ ] **Step 4: Run Python tests**

Run: `uv run pytest tests/ -v 2>&1 | tail -30`
Expected: all pass

- [ ] **Step 5: Run linter + type checker**

Run: `./lint_code.sh 2>&1 | tail -20`
Expected: clean

- [ ] **Step 6: Run full check**

Run: `./check_all.sh 2>&1 | tail -20`
Expected: all pass

- [ ] **Step 7: Commit any remaining fixes**

```bash
git add -A
git commit -m "fix: address integration test issues from unified cost function"
```

---

**Checkpoint compatibility note:** Existing training checkpoints store `cost_history` values computed with the old cost function (~1e6 scale for non-captures). Resuming from old checkpoints will produce a dramatic cost discontinuity in convergence reports. Old checkpoints should be discarded and training restarted fresh.

---

## Task 11: Smart Commit (Final)

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole `feature/unified-cost-function` git branch into account. This will sync CLAUDE.md and README.md with the codebase changes, then commit everything.
