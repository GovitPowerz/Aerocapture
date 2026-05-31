# NN Input Vector v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pdyn_error` + 3 live correction-DV inputs to the NN candidate vector (32 -> 35), renormalize all inputs data-driven so they fill `[-1, 1]`, and move all three decoder masks to a shared 17-input set.

**Architecture:** Rust `build_nn_input` owns all normalization (Python calls it via PyO3 and never re-derives scales). New inputs are append-only at indices 32-34. A one-time Python calibration script collects raw input distributions and emits new `const S_*` scale literals to paste into `neural.rs`. Decoder retraining is the user's step after this plan lands.

**Tech Stack:** Rust (nalgebra, neural.rs), PyO3 (`aerocapture_rs`), Python (numpy), pytest, cargo test.

**Spec:** `docs/superpowers/specs/2026-06-01-nn-input-vector-v2-design.md`

---

## File Structure

- `src/rust/src/data/neural.rs` — `NN_FULL_INPUT_SIZE` const (32 -> 35); `validate_mask` already uses the const.
- `src/rust/src/gnc/guidance/neural.rs` — `build_nn_input`: 3 new DV inputs + hyperbolic guard, renormalized scale constants, updated 35-entry doc comment, inline `#[cfg(test)]` tests.
- `src/python/aerocapture/training/ablation.py` — `NN_INPUT_NAMES` (32 -> 35).
- `src/python/aerocapture/training/nn_input_report.py` — any literal 32 in name handling.
- `src/python/aerocapture/training/calibrate_inputs.py` — **new** calibration utility.
- `tests/test_calibrate_inputs.py` — **new** unit tests for calibration helpers.
- `configs/training/msr_aller_nn_{train_consolidated,scaledpi_train,delta_train}.toml` — 17-input mask + `input_size`.
- `tests/reference_data/rust_golden/neural*.csv` — regenerated golden.

---

## Task 1: Bump candidate vector to 35 + add 3 live correction-DV inputs (provisional scale)

**Files:**
- Modify: `src/rust/src/data/neural.rs` (the `NN_FULL_INPUT_SIZE` const)
- Modify: `src/rust/src/gnc/guidance/neural.rs` (add `S_DV` const, DV inputs in `build_nn_input`, doc comment)
- Test: inline `#[cfg(test)]` module in `src/rust/src/gnc/guidance/neural.rs`

- [ ] **Step 1: Bump the size constant**

In `src/rust/src/data/neural.rs`, find `pub const NN_FULL_INPUT_SIZE: usize = 32;` and change to:

```rust
pub const NN_FULL_INPUT_SIZE: usize = 35;
```

- [ ] **Step 2: Add the provisional DV scale + import**

At the top of `src/rust/src/gnc/guidance/neural.rs` with the other `const S_*` definitions (lines ~55-60), add:

```rust
// Live correction-DV scale (provisional; finalized by calibrate_inputs.py in Task 5).
// ~150 m/s typical component => 150/sinh(1) ~= 128.
const S_DV: f64 = 1.28e+02;
// Hyperbolic / open-orbit sentinel: maps to asinh = 1.5 (out-of-band, bounded).
const DV_SENTINEL: f64 = S_DV * 2.129279; // sinh(1.5) = 2.129279...
```

Ensure `compute_deltav` is reachable. Near the top imports add (if not present):

```rust
use crate::orbit::maneuver;
```

- [ ] **Step 3: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `src/rust/src/gnc/guidance/neural.rs`:

```rust
#[test]
fn full_vector_is_35_wide() {
    let nav = test_nav();
    let data = test_sim_data_with_ref_traj();
    let planet = PlanetConfig::mars();
    let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
    let inp = build_nn_input(
        &nav, Some(&full_mask), None, 0.0, &data, &planet,
        50.0_f64.to_radians(), 0.0, None, 0.0, 0.0, 0.0, 0.0,
    );
    assert_eq!(inp.len(), 35, "candidate vector must be 35 wide");
}

#[test]
fn dv_inputs_sentinel_when_hyperbolic() {
    // test_nav() is at hyperbolic entry (e >= 1) => DV inputs saturate to the sentinel.
    let nav = test_nav();
    let data = test_sim_data_with_ref_traj();
    let planet = PlanetConfig::mars();
    let orbit = elements::from_spherical(
        nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2],
        nav.velocity_estimated[0], nav.velocity_estimated[1], nav.velocity_estimated[2],
        &planet,
    );
    assert!(orbit.eccentricity >= 1.0, "fixture must be hyperbolic for this test");
    let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
    let inp = build_nn_input(
        &nav, Some(&full_mask), None, 0.0, &data, &planet,
        50.0_f64.to_radians(), 0.0, None, 0.0, 0.0, 0.0, 0.0,
    );
    let expected = (DV_SENTINEL / S_DV).asinh(); // == 1.5
    for idx in 32..35 {
        assert!((inp[idx] - expected).abs() < 1e-9, "input[{idx}] = {} != sentinel {expected}", inp[idx]);
    }
}

#[test]
fn dv_inputs_live_when_elliptical() {
    // Lower the radial speed so the osculating orbit closes (e < 1).
    let mut nav = test_nav();
    nav.velocity_estimated[0] *= 0.45; // bleed energy -> elliptical
    let data = test_sim_data_with_ref_traj();
    let planet = PlanetConfig::mars();
    let orbit = elements::from_spherical(
        nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2],
        nav.velocity_estimated[0], nav.velocity_estimated[1], nav.velocity_estimated[2],
        &planet,
    );
    assert!(orbit.eccentricity < 1.0, "fixture must be elliptical (got e={})", orbit.eccentricity);
    let dv = maneuver::compute_deltav(&orbit, &data.target_orbit, &data.parking_orbit, &planet);
    let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
    let inp = build_nn_input(
        &nav, Some(&full_mask), None, 0.0, &data, &planet,
        50.0_f64.to_radians(), 0.0, None, 0.0, 0.0, 0.0, 0.0,
    );
    assert!((inp[32] - (dv.dv1 / S_DV).asinh()).abs() < 1e-9);
    assert!((inp[33] - (dv.dv2 / S_DV).asinh()).abs() < 1e-9);
    assert!((inp[34] - (dv.dv3 / S_DV).asinh()).abs() < 1e-9);
    assert!(inp[32].is_finite() && inp[33].is_finite() && inp[34].is_finite());
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd src/rust && cargo test --lib full_vector_is_35 dv_inputs_ -- --nocapture`
Expected: FAIL — `inp.len()` is 32, DV indices panic on out-of-bounds.

- [ ] **Step 5: Implement the DV inputs in build_nn_input**

In `src/rust/src/gnc/guidance/neural.rs`, after the existing reference-trajectory block (`full_input[19] = pdyn_error / 2e3;`) and before the exit-bank-teacher block (index 20), the `full_input` array literal must already be sized 35 via Step 1. Add the DV computation near the end of `build_nn_input`, just before the masking/return logic:

```rust
    // -- Live correction-DV inputs (indices 32-34) --
    // compute_deltav is only valid for a closed (elliptical) osculating orbit;
    // pre-capture the orbit is hyperbolic (apoapsis undefined) -> saturate to sentinel.
    let (dv1, dv2, dv3) = if orbit.eccentricity < 1.0 {
        let dv = maneuver::compute_deltav(&orbit, &data.target_orbit, &data.parking_orbit, planet);
        if dv.dv1.is_finite() && dv.dv2.is_finite() && dv.dv3.is_finite() {
            (dv.dv1, dv.dv2, dv.dv3)
        } else {
            (DV_SENTINEL, DV_SENTINEL, DV_SENTINEL)
        }
    } else {
        (DV_SENTINEL, DV_SENTINEL, DV_SENTINEL)
    };
    full_input[32] = (dv1 / S_DV).asinh();
    full_input[33] = (dv2 / S_DV).asinh();
    full_input[34] = (dv3 / S_DV).asinh();
```

Update the input-name doc comment at the top of the file (lines ~7-10) to add the three new entries:

```rust
//!  32 predicted_dv1   33 predicted_dv2   34 predicted_dv3
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd src/rust && cargo test --lib full_vector_is_35 dv_inputs_ -- --nocapture`
Expected: PASS (3 tests).

- [ ] **Step 7: Run the full neural test module + clippy**

Run: `cd src/rust && cargo test --lib neural && cargo clippy --lib -- -D warnings`
Expected: PASS, no clippy warnings.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): add live correction-DV inputs (32-34), NN_FULL_INPUT_SIZE 32->35"
```

---

## Task 2: Extend Python NN_INPUT_NAMES to 35

**Files:**
- Modify: `src/python/aerocapture/training/ablation.py` (`NN_INPUT_NAMES`)
- Modify: `src/python/aerocapture/training/nn_input_report.py` (only if it has its own 32-length name list)
- Test: `tests/test_ablation.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_ablation.py` add:

```python
def test_nn_input_names_has_35_with_dv() -> None:
    from aerocapture.training.ablation import NN_INPUT_NAMES
    assert len(NN_INPUT_NAMES) == 35
    assert NN_INPUT_NAMES[32] == "predicted_dv1"
    assert NN_INPUT_NAMES[33] == "predicted_dv2"
    assert NN_INPUT_NAMES[34] == "predicted_dv3"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ablation.py::test_nn_input_names_has_35_with_dv -v`
Expected: FAIL — length 32.

- [ ] **Step 3: Append the three names**

In `src/python/aerocapture/training/ablation.py`, find the end of the `NN_INPUT_NAMES` list (ends at `"periapsis_alt"` at index 31) and append:

```python
    "predicted_dv1",   # 32
    "predicted_dv2",   # 33
    "predicted_dv3",   # 34
```

If `nn_input_report.py` defines or asserts its own 32-length list, point it at `NN_INPUT_NAMES` instead (grep first: `rg -n "32|INPUT_NAMES" src/python/aerocapture/training/nn_input_report.py`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ablation.py::test_nn_input_names_has_35_with_dv -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/ablation.py src/python/aerocapture/training/nn_input_report.py tests/test_ablation.py
git commit -m "feat(nn): NN_INPUT_NAMES 32->35 (predicted_dv1/2/3)"
```

---

## Task 3: Rebuild PyO3 + write the calibration script

**Files:**
- Create: `src/python/aerocapture/training/calibrate_inputs.py`
- Test: `tests/test_calibrate_inputs.py`

- [ ] **Step 1: Rebuild PyO3 so collect_nn_inputs sees the 35-wide vector**

Run: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
Expected: builds `aerocapture_rs` successfully.

- [ ] **Step 2: Write the failing helper tests**

Create `tests/test_calibrate_inputs.py`:

```python
import math
import numpy as np
from aerocapture.training.calibrate_inputs import (
    invert_transform, derive_asinh_scale, derive_affine,
)


def test_invert_asinh_roundtrip() -> None:
    raw = np.array([-500.0, 0.0, 880.0, 3000.0])
    s = 880.0
    norm = np.arcsinh(raw / s)
    back = invert_transform(norm, ("asinh", s))
    assert np.allclose(back, raw, atol=1e-9)


def test_invert_affine_roundtrip() -> None:
    raw = np.array([0.0, 25.0, 50.0])
    # current transform: norm = raw/50 - 1  => ("affine", a=1/50, b=-1)
    norm = raw / 50.0 - 1.0
    back = invert_transform(norm, ("affine", 1.0 / 50.0, -1.0))
    assert np.allclose(back, raw, atol=1e-9)


def test_derive_asinh_scale_puts_p99_at_one() -> None:
    s = derive_asinh_scale(p1=-200.0, p99=180.0)
    # max(|p1|,|p99|)=200 ; asinh(200/s) should be 1.0
    assert math.isclose(math.asinh(200.0 / s), 1.0, rel_tol=1e-9)


def test_derive_affine_maps_p1_p99_to_pm1() -> None:
    center, half = derive_affine(p1=10.0, p99=50.0)
    assert math.isclose((10.0 - center) / half, -1.0, rel_tol=1e-9)
    assert math.isclose((50.0 - center) / half, 1.0, rel_tol=1e-9)


def test_derive_affine_floors_degenerate_halfwidth() -> None:
    # near-constant input must not blow up
    center, half = derive_affine(p1=5.0, p99=5.0)
    assert half >= 1e-6
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_calibrate_inputs.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 4: Write the calibration script**

Create `src/python/aerocapture/training/calibrate_inputs.py`:

```python
"""Calibrate NN input normalization scales from observed raw distributions.

Runs the deployed NN over a reserved seed pool, collects the normalized 35-wide
candidate trace, inverts each input's KNOWN current transform to recover the raw
distribution, then emits new scale constants so each input's [p1, p99] fills
~[-1, 1]. Heavy-tailed / acceleration / DV inputs -> asinh; bounded -> affine.

One-time tool: paste the emitted Rust const block into neural.rs (Task 5), then
re-run nn_input_report to verify ~1% saturation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Reserved seed pool, disjoint from train/val/final-eval/report streams.
CALIBRATION_SEED_OFFSET = 6_000_000

# Inputs that always use asinh (heavy-tailed / spiky), regardless of tail ratio.
_FORCE_ASINH = {
    2, 3, 5, 11, 12, 13, 14, 18, 19, 31, 32, 33, 34,
}
# Bounded inputs to skip entirely (binary / tanh / sin-cos already in [-1,1]).
_SKIP = {15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

# Current transform per index (must mirror build_nn_input at calibration time).
# Forms: ("asinh", s) | ("affine", a, b) meaning norm = a*raw + b | ("raw",).
CURRENT_TRANSFORMS: dict[int, tuple] = {
    0: ("affine", 1.0, -1.0),            # ecc - 1
    1: ("affine", 3.0 / 5.0, 0.0),       # deg * 3/5
    2: ("asinh", 8.802043e02),
    3: ("asinh", 5.554906e06),
    4: ("affine", 2.0 / 3e3, -3.0),      # (raw/3e3 - 1.5)*2
    5: ("affine", 1.0 / 20.0, -1.0),     # raw/20 - 1
    6: ("affine", 2.0, -1.0),            # frac*2 - 1
    7: ("affine", 2.0, -1.0),
    8: ("affine", 1.0 / 65.0, -1.0),     # (raw-65)/65
    9: ("affine", 1.0 / 0.3, 0.0),
    10: ("affine", 2.0 / math.pi, 0.0),
    11: ("affine", 1.0 / 50.0, -1.0),    # raw/50 - 1
    12: ("affine", 1.0 / 10.0, 0.0),     # raw/10
    13: ("asinh", 3.259362e07),
    14: ("asinh", 6.626041e07),
    16: ("raw",),
    17: ("affine", 1.0 / 2e3, -1.0),     # raw/2e3 - 1
    18: ("asinh", 7.333648e02),
    19: ("affine", 1.0 / 2e3, 0.0),      # raw/2e3
    31: ("asinh", 9.158960e05),
    32: ("asinh", 1.28e02),              # provisional S_DV
    33: ("asinh", 1.28e02),
    34: ("asinh", 1.28e02),
}

# Rust const name per asinh index (for the emitted block).
_ASINH_CONST_NAME = {
    2: "S_RADIAL_VELOCITY", 3: "S_ORBITAL_ENERGY", 5: "S_ACCEL_MAGNITUDE",
    11: "S_DRAG_ACCEL", 12: "S_LIFT_ACCEL", 13: "S_SMA_ERROR",
    14: "S_APOAPSIS_ALT", 18: "S_HDOT_NOMINAL", 19: "S_PDYN_ERROR",
    31: "S_PERIAPSIS_ALT", 32: "S_DV", 33: "S_DV", 34: "S_DV",
}


def invert_transform(norm: np.ndarray, transform: tuple) -> np.ndarray:
    kind = transform[0]
    if kind == "asinh":
        (_, s) = transform
        return s * np.sinh(norm)
    if kind == "affine":
        (_, a, b) = transform
        return (norm - b) / a
    if kind == "raw":
        return norm
    raise ValueError(f"unknown transform {transform!r}")


def derive_asinh_scale(p1: float, p99: float) -> float:
    span = max(abs(p1), abs(p99))
    span = max(span, 1e-12)
    return span / math.sinh(1.0)


def derive_affine(p1: float, p99: float) -> tuple[float, float]:
    center = (p1 + p99) / 2.0
    half = max((p99 - p1) / 2.0, 1e-6)
    return center, half


def _collect_raw(toml_path: str, n_sims: int) -> dict[int, np.ndarray]:
    import aerocapture_rs

    from aerocapture.training.evaluate import make_reserved_seeds

    seeds = make_reserved_seeds(0, CALIBRATION_SEED_OFFSET, n_sims)
    recs = aerocapture_rs.collect_nn_inputs(toml_path, seeds, overrides=None)
    cols: dict[int, list[np.ndarray]] = {}
    for r in recs:
        x = np.asarray(r["X"])  # (T, 35) normalized
        for idx in range(x.shape[1]):
            cols.setdefault(idx, []).append(x[:, idx])
    raw: dict[int, np.ndarray] = {}
    for idx, parts in cols.items():
        norm = np.concatenate(parts)
        norm = norm[np.isfinite(norm)]
        if idx in CURRENT_TRANSFORMS:
            raw[idx] = invert_transform(norm, CURRENT_TRANSFORMS[idx])
    return raw


def calibrate(toml_path: str, n_sims: int) -> str:
    from aerocapture.training.ablation import NN_INPUT_NAMES

    raw = _collect_raw(toml_path, n_sims)
    lines: list[str] = []
    lines.append("// === calibrated input scales (calibrate_inputs.py) ===")
    seen_const: set[str] = set()
    table: list[str] = []
    for idx in sorted(raw):
        if idx in _SKIP:
            continue
        vals = raw[idx]
        p1, p50, p99 = np.percentile(vals, [1, 50, 99])
        name = NN_INPUT_NAMES[idx] if idx < len(NN_INPUT_NAMES) else f"idx{idx}"
        if idx in _FORCE_ASINH:
            s = derive_asinh_scale(p1, p99)
            const = _ASINH_CONST_NAME.get(idx, f"S_IDX{idx}")
            if const not in seen_const:
                lines.append(f"const {const}: f64 = {s:.6e}; // {name}: p1={p1:.3g} p99={p99:.3g}")
                seen_const.add(const)
            table.append(f"  [{idx:2d}] {name:22s} asinh  p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> s={s:.4e}")
        else:
            center, half = derive_affine(p1, p99)
            table.append(f"  [{idx:2d}] {name:22s} affine p1={p1:11.4g} p50={p50:11.4g} p99={p99:11.4g} -> center={center:.6e} half={half:.6e}")
    report = "\n".join(["RAW DISTRIBUTION TABLE:", *table, "", *lines])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate NN input normalization scales")
    ap.add_argument("--toml", required=True)
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--output", default=None, help="optional path to write the report")
    args = ap.parse_args()
    report = calibrate(args.toml, args.n_sims)
    print(report)
    if args.output:
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run to verify helper tests pass**

Run: `uv run pytest tests/test_calibrate_inputs.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Lint**

Run: `./lint_code.sh`
Expected: ruff + mypy clean (calibrate_inputs.py + test).

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/calibrate_inputs.py tests/test_calibrate_inputs.py
git commit -m "feat(nn): add calibrate_inputs.py (data-driven input scale derivation)"
```

---

## Task 4: Run calibration to derive the real scale constants

**Files:** none modified (produces the const block consumed by Task 5).

- [ ] **Step 1: Run the calibration against the existing atan2 config**

The existing 13-input config + model load fine (mask indices < 35); `collect_nn_inputs` returns the full 35-wide candidate trace regardless of mask, including the live DV inputs and `pdyn_error`.

Run:
```bash
uv run python -m aerocapture.training.calibrate_inputs \
  --toml configs/training/msr_aller_nn_train_consolidated.toml \
  --n-sims 300 --output /tmp/nn_calib.txt
```
Expected: prints a RAW DISTRIBUTION TABLE + a `const S_* = ...;` block. `S_DV` should land near 1e2; `S_DRAG_ACCEL` / `S_LIFT_ACCEL` are the headline fixes.

- [ ] **Step 2: Sanity-check the output**

Confirm every emitted scale is finite and positive, and the affine half-widths are not floored at `1e-6` for masked inputs (a floored half-width means a near-constant input — investigate before using). Masked inputs to verify in the table: indices 0, 5, 6, 7, 11, 12, 19, 32, 33, 34.

- [ ] **Step 3: No commit** (this task only produces the const values pasted in Task 5).

---

## Task 5: Apply calibrated scales + renormalize build_nn_input, verify coverage

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (scale consts + normalization lines)
- Test: inline `#[cfg(test)]` (finiteness regression) + `nn_input_report` coverage check

- [ ] **Step 1: Replace the scale constants**

In `src/rust/src/gnc/guidance/neural.rs`, replace the `const S_*` block with the calibration output from Task 4 (keep `S_DV` and recompute `DV_SENTINEL = S_DV * 2.129279`). Add the new asinh consts (`S_ACCEL_MAGNITUDE`, `S_DRAG_ACCEL`, `S_LIFT_ACCEL`, `S_PDYN_ERROR`) and affine consts as `const C_*`/`const H_*` literals from the table, e.g.:

```rust
const C_ECC_EXCESS: f64 = /* center from table */;
const H_ECC_EXCESS: f64 = /* half from table */;
```

- [ ] **Step 2: Rewrite the affected normalization lines**

Apply the new transforms in `build_nn_input`. Asinh for accelerations + pdyn_error; affine for the bounded ones. Concretely change these lines (values via the consts from Step 1):

```rust
    full_input[0] = (orbit.eccentricity - 1.0 - C_ECC_EXCESS) / H_ECC_EXCESS; // ecc excess (affine, calibrated)
    full_input[5] = (accel_mag / S_ACCEL_MAGNITUDE).asinh();                  // accel magnitude (asinh)
    full_input[6] = (nav.heat_flux_fraction - C_HEAT_FLUX) / H_HEAT_FLUX;     // heat flux fraction (affine)
    full_input[7] = (nav.heat_load_fraction - C_HEAT_LOAD) / H_HEAT_LOAD;     // heat load fraction (affine)
    full_input[11] = (nav.acceleration_estimated[0] / S_DRAG_ACCEL).asinh();  // drag accel (asinh)
    full_input[12] = (nav.acceleration_estimated[1] / S_LIFT_ACCEL).asinh();  // lift accel (asinh)
    full_input[19] = (pdyn_error / S_PDYN_ERROR).asinh();                     // pdyn error (asinh)
```

Also apply the calibrated affine/asinh to the remaining non-skipped, non-masked inputs the calibration emitted (1, 4, 8, 9, 10, 16, 17) so the full candidate vector is consistently normalized per the spec's "all inputs" decision. Use the same `(raw - C_x) / H_x` affine pattern.

- [ ] **Step 3: Run finiteness + golden-adjacent regression**

Run: `cd src/rust && cargo test --lib neural -- --nocapture`
Expected: PASS. The existing `build_nn_input` finiteness tests must still pass (they assert finite, not exact values). If a test asserted an exact normalized value for a renormalized index (e.g. an `input_5`/`input_11`/`input_12` value test), update its expected value to the new transform — show the new expected inline.

- [ ] **Step 4: Rebuild PyO3 + verify coverage with the report**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
uv run python -m aerocapture.training.nn_input_report \
  training_output/neural_network_islands \
  --toml configs/training/msr_aller_nn_train_consolidated.toml --n-sims 200
uv run python -c "
import json
s=json.load(open('training_output/neural_network_islands/nn_input_report/summary.json'))
by={r['name']:r for r in s['inputs']}
for nm in ['drag_accel','lift_accel','accel_magnitude','pdyn_error']:
    if nm in by: print(nm, round(100*by[nm]['frac_out_of_range'],1), '%', 'p1', round(by[nm]['p1'],2), 'p99', round(by[nm]['p99'],2))
"
```
Expected: `drag_accel` / `lift_accel` / `accel_magnitude` / `pdyn_error` now report `<~2%` out-of-range with p1/p99 near `[-1, 1]` (the headline fix). If any still saturate badly, re-run Task 4 calibration with more sims and re-paste.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): data-driven renormalization of all inputs (asinh accels + pdyn_error, calibrated affine)"
```

---

## Task 6: Move all three decoder masks to the shared 17-input set

**Files:**
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`
- Modify: `configs/training/msr_aller_nn_scaledpi_train.toml`
- Modify: `configs/training/msr_aller_nn_delta_train.toml`

- [ ] **Step 1: Edit each config's mask + input_size**

In all three configs, replace the `input_mask` line with:

```toml
input_mask = [0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30, 32, 33, 34]
```

and update the comment block above it to:

```toml
# 17 inputs: 13-mask + pdyn_error(19) + live correction-DV predicted_dv1/2/3 (32,33,34).
# All inputs renormalized data-driven (asinh accels + pdyn_error, calibrated affine) to
# fill [-1,1]. predicted_dv* = per-tick compute_deltav on current osculating orbit,
# asinh, hyperbolic-guarded to a +1.5 sentinel pre-capture.
```

Change the first `[[network.architecture]]` `input_size = 13` to `input_size = 17` in each.

- [ ] **Step 2: Verify all three load with the 17-input mask matching input_size**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
expect=[0,2,3,5,6,7,11,12,18,19,27,28,29,30,32,33,34]
for f in ['nn_train_consolidated','nn_scaledpi_train','nn_delta_train']:
    c=load_toml_with_bases(f'configs/training/msr_aller_{f}.toml')
    n=c['network']; m=n['input_mask']; i0=n['architecture'][0]['input_size']
    assert m==expect and i0==len(m)==17, (f, m, i0)
    print(f, 'OK')
"
```
Expected: three `OK` lines.

- [ ] **Step 3: Commit**

```bash
git add configs/training/msr_aller_nn_train_consolidated.toml configs/training/msr_aller_nn_scaledpi_train.toml configs/training/msr_aller_nn_delta_train.toml
git commit -m "feat(nn): shared 17-input mask (13 + pdyn_error + predicted_dv1/2/3)"
```

---

## Task 7: Regenerate the neural golden + fix size-asserting tests

**Files:**
- Modify: `tests/reference_data/rust_golden/neural*.csv` (regenerated)
- Modify: any Rust/Python test asserting the 32-element candidate size

- [ ] **Step 1: Find tests/asserts that hardcode 32**

Run: `rg -rn "== 32|len.*32|NN_FULL_INPUT_SIZE" src/rust/src tests src/python | rg -v "// |#"`
Expected: a small list. For each that asserts the candidate vector is 32, change to 35 (show the change inline when editing). The cross-language equivalence tests use explicit input vectors sized to `arch[0].input_size`, not `NN_FULL_INPUT_SIZE` — leave them unless they literally assert 32.

- [ ] **Step 2: Identify the neural golden config + regenerate**

Run: `rg -n "neural" src/rust/tests/*.rs | rg -i golden` to find the golden config path (e.g. `configs/test/*neural*.toml`). Regenerate:
```bash
cargo build --release --manifest-path src/rust/Cargo.toml
./src/rust/target/release/aerocapture <neural_golden_config.toml>
# copy the produced photo CSV over tests/reference_data/rust_golden/neural*.csv
```
(Use the exact paths the golden test reads — inspect the test to confirm source/target file names.)

- [ ] **Step 3: Run the Rust golden + full suite**

Run: `cd src/rust && cargo test`
Expected: PASS. The 5 non-neural goldens stay bit-identical (their guidance does not use `build_nn_input`); the neural golden now matches the regenerated file.

- [ ] **Step 4: Commit**

```bash
git add tests/reference_data/rust_golden src/rust/src src/rust/tests
git commit -m "test(nn): regenerate neural golden + bump candidate-size asserts to 35"
```

---

## Task 8: Full verification + sync docs/commit the branch

**Files:** none (verification + smart-commit).

- [ ] **Step 1: Full Rust check**

Run: `./check_all.sh`
Expected: cargo test + fmt --check + clippy + release build all pass.

- [ ] **Step 2: Full Python check**

Run: `./lint_code.sh && uv run pytest tests -q`
Expected: ruff + mypy clean, all tests pass.

- [ ] **Step 3: Confirm the three configs are ready to train**

The chromosome width changed (13 -> 17 inputs); auto-resume will refuse on stale checkpoints. Training is the user's step — note in the handoff that they must clear the target dirs or use `--from-scratch`.

- [ ] **Step 4: smart-commit the whole branch**

Invoke the `smart-commit` skill, telling it to take the whole `feature/nn-input-rescale` branch into account (sync CLAUDE.md — the 32->35 candidate vector, the `predicted_dv` inputs, the `asinh` renormalization, the calibrate_inputs.py tool, and the 17-input masks all need documenting in the architecture section).

---

## Self-Review Notes

- **Spec coverage:** candidate 32->35 (T1), live DV + guard (T1), pdyn_error in mask (T6), renormalize all data-driven (T3/T4/T5), 17-input masks (T6), touch list Rust+Python+configs+tests (T1/T2/T5/T6/T7), calibration workflow (T3/T4/T5). All covered.
- **Retrain is the user's step** — plan stops at a tested, committed pipeline ready to train.
- **Calibration-derived literals** are produced by Task 4's script run and pasted in Task 5; the code structure (which inputs are asinh vs affine, const names, formulas) is fully specified — only numeric literals are runtime-derived, which is the nature of a calibration step.
