# Piecewise-Constant Bank Guidance & Corridor Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GA-optimizable `piecewise_constant` guidance scheme (10 bank angle segments over orbital energy) that produces optimized reference trajectories and builds corridor envelopes from GA population history.

**Architecture:** New Rust guidance module dispatched from `ftc::guidance_step`, using individual TOML fields (`bank_angle_0..9`) that work automatically with the existing GA pipeline (`PARAM_SPACES` + `GUIDANCE_TOML_SECTIONS` → `decode_params_from_chromosome` → dot-path overrides). No special `evaluate.py` branch needed. Python-side `CorridorAccumulator` class updates running max/min envelopes per generation via a single `run_batch` call that serves both fitness evaluation and corridor accumulation.

**Spec deviation:** The spec describes `guidance.bank_angles = [list]` and a special `evaluate.py` branch. This plan uses individual flat fields (`bank_angle_0..9`) instead, which plugs directly into the existing GA pipeline with zero special cases — the same approach used by all other non-NN schemes. The spec should be updated to reflect this pragmatic choice.

**Tech Stack:** Rust (nalgebra, serde), Python (numpy, deap), PyO3/maturin, TOML configs

**Spec:** `docs/superpowers/specs/2026-03-19-piecewise-constant-corridor-design.md`

---

### File Map

**Create:**
- `src/rust/src/gnc/guidance/piecewise_constant.rs` — Guidance function: energy → segment → signed bank angle
- `configs/training/msr_aller_piecewise_constant_train.toml` — Training config for the new scheme

**Modify:**
- `src/rust/src/config.rs` — Add `PiecewiseConstant` enum variant + TOML parsing
- `src/rust/src/data/guidance_params.rs` — Add `PiecewiseConstantParams` runtime struct
- `src/rust/src/data/mod.rs` — Wire params into `SimData`
- `src/rust/src/gnc/guidance/ftc.rs` — Add dispatch arm + skip lateral guidance for this scheme
- `src/rust/src/gnc/guidance/mod.rs` — Export new module
- `src/python/aerocapture/training/param_spaces.py` — Add PARAM_SPACES + GUIDANCE_TOML_SECTIONS entries + REQUIRES_REF_TRAJECTORY
- `src/python/aerocapture/training/corridor.py` — Add `CorridorAccumulator` class
- `src/python/aerocapture/training/train.py` — Corridor accumulation per generation + ref trajectory check + save artifacts
- `src/python/aerocapture/training/final_report.py` — 4-layer fill visualization
- `configs/missions/mars.toml` — Add `delta_za_restricted` to `[corridor]`

**Test:**
- `src/rust/tests/` — Integration tests for piecewise_constant guidance
- `tests/test_corridor.py` — CorridorAccumulator unit tests
- `tests/test_param_spaces.py` or inline — REQUIRES_REF_TRAJECTORY tests

---

### Task 1: Rust — PiecewiseConstant params and config parsing

**Files:**
- Modify: `src/rust/src/config.rs`
- Modify: `src/rust/src/data/guidance_params.rs`
- Modify: `src/rust/src/data/mod.rs`

- [ ] **Step 1: Add `PiecewiseConstantParams` to `guidance_params.rs`**

After the existing `FnpagParams` struct (~line 140), add:

```rust
/// Piecewise-constant bank angle guidance parameters.
/// 10 segments uniformly distributed over the energy range.
/// Bank angles are signed (negative = implicit roll reversal).
#[derive(Debug, Clone)]
pub struct PiecewiseConstantParams {
    pub bank_angles: [f64; 10],  // radians, signed
    pub energy_min: f64,          // J/kg (NOT MJ/kg)
    pub energy_max: f64,          // J/kg (NOT MJ/kg)
}

impl Default for PiecewiseConstantParams {
    fn default() -> Self {
        Self {
            bank_angles: [65.0_f64.to_radians(); 10],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        }
    }
}
```

Add the field to `GuidanceParams` struct:
```rust
pub piecewise_constant: PiecewiseConstantParams,
```

- [ ] **Step 2: Add `GuidanceType::PiecewiseConstant` to `config.rs`**

Add variant to enum (~line 91):
```rust
PiecewiseConstant,
```

Add TOML parsing in the match (~line 727):
```rust
"piecewise_constant" => GuidanceType::PiecewiseConstant,
```

Add TOML struct for serde parsing (near the other `TomlXxxParams` structs):
```rust
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TomlPiecewiseConstantParams {
    #[serde(default = "default_bank_65")]
    pub bank_angle_0: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_1: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_2: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_3: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_4: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_5: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_6: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_7: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_8: f64,
    #[serde(default = "default_bank_65")]
    pub bank_angle_9: f64,
    #[serde(default = "default_energy_min")]
    pub energy_min: f64,  // MJ/kg in TOML, converted to J/kg
    #[serde(default = "default_energy_max")]
    pub energy_max: f64,  // MJ/kg in TOML, converted to J/kg
}

fn default_bank_65() -> f64 { 65.0 }
fn default_energy_min() -> f64 { -6.0 }
fn default_energy_max() -> f64 { 5.0 }
```

Add field to `TomlGuidance` struct:
```rust
#[serde(default)]
pub piecewise_constant: TomlPiecewiseConstantParams,
```

Wire the TOML struct into `PiecewiseConstantParams` in the config-to-data conversion (in `data/mod.rs`, following the pattern of other schemes):
```rust
piecewise_constant: PiecewiseConstantParams {
    bank_angles: [
        toml_guidance.piecewise_constant.bank_angle_0.to_radians(),
        toml_guidance.piecewise_constant.bank_angle_1.to_radians(),
        // ... all 10
        toml_guidance.piecewise_constant.bank_angle_9.to_radians(),
    ],
    energy_min: toml_guidance.piecewise_constant.energy_min * 1e6,
    energy_max: toml_guidance.piecewise_constant.energy_max * 1e6,
},
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd src/rust && cargo build --release`
Expected: Compiles with no errors (warning about unused `PiecewiseConstant` variant is OK for now).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/guidance_params.rs src/rust/src/data/mod.rs
git commit -m "feat: add PiecewiseConstant guidance type and config parsing"
```

---

### Task 2: Rust — Implement piecewise_constant guidance function

**Files:**
- Create: `src/rust/src/gnc/guidance/piecewise_constant.rs`
- Modify: `src/rust/src/gnc/guidance/mod.rs`

- [ ] **Step 1: Create `piecewise_constant.rs` with the guidance function**

```rust
//! Piecewise-constant bank angle guidance.
//!
//! Divides the orbital energy range into 10 uniform segments, each with
//! a constant bank angle in [-180°, +180°]. The bank angle sign is part
//! of the profile (negative = implicit roll reversal). No navigation
//! feedback, no lateral guidance — pure open-loop bank profile.
//!
//! GA-optimized to produce reference trajectories and corridor envelopes.

use crate::config::Planet;
use crate::data::guidance_params::PiecewiseConstantParams;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Number of segments in the piecewise-constant bank profile.
const N_SEGMENTS: usize = 10;

/// Compute piecewise-constant bank angle from current orbital energy.
///
/// Returns the **signed** bank angle in radians. Unlike other schemes
/// that return magnitude only (with roll sign applied by lateral guidance),
/// piecewise_constant encodes the sign directly in the bank profile.
pub fn piecewise_constant_bank(
    nav: &NavigationOutput,
    params: &PiecewiseConstantParams,
    planet: &Planet,
) -> f64 {
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    segment_bank_angle(energy, params)
}

/// Pure lookup: energy → segment → bank angle.
/// Exposed for unit testing without needing a full NavigationOutput.
pub fn segment_bank_angle(energy: f64, params: &PiecewiseConstantParams) -> f64 {
    let e_min = params.energy_min;
    let e_max = params.energy_max;

    if e_max <= e_min {
        return params.bank_angles[0];
    }

    // Segment 0 = highest energy (entry), segment 9 = lowest energy (deep capture)
    // Energy DECREASES during flight, so segment index increases as energy drops
    let frac = (e_max - energy) / (e_max - e_min);
    let seg = (frac * N_SEGMENTS as f64).floor() as i64;
    let seg = seg.clamp(0, (N_SEGMENTS - 1) as i64) as usize;

    params.bank_angles[seg]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn test_params() -> PiecewiseConstantParams {
        PiecewiseConstantParams {
            bank_angles: [
                60.0_f64.to_radians(),
                50.0_f64.to_radians(),
                40.0_f64.to_radians(),
                30.0_f64.to_radians(),
                20.0_f64.to_radians(),
                -20.0_f64.to_radians(),
                -30.0_f64.to_radians(),
                -40.0_f64.to_radians(),
                -50.0_f64.to_radians(),
                -60.0_f64.to_radians(),
            ],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        }
    }

    #[test]
    fn entry_energy_gives_segment_0() {
        let params = test_params();
        // Energy at entry (highest, near energy_max)
        let bank = segment_bank_angle(4.5e6, &params);
        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn deep_capture_gives_last_segment() {
        let params = test_params();
        // Energy at deep capture (lowest, near energy_min)
        let bank = segment_bank_angle(-5.5e6, &params);
        assert_relative_eq!(bank, -60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn mid_energy_gives_middle_segment() {
        let params = test_params();
        // Energy at midpoint: (5.0e6 + -6.0e6) / 2 = -0.5e6
        // frac = (5.0e6 - (-0.5e6)) / (5.0e6 - (-6.0e6)) = 5.5e6 / 11.0e6 = 0.5
        // seg = floor(0.5 * 10) = 5
        let bank = segment_bank_angle(-0.5e6, &params);
        assert_relative_eq!(bank, -20.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn above_range_clamps_to_segment_0() {
        let params = test_params();
        let bank = segment_bank_angle(10.0e6, &params);
        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn below_range_clamps_to_last_segment() {
        let params = test_params();
        let bank = segment_bank_angle(-20.0e6, &params);
        assert_relative_eq!(bank, -60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn negative_bank_angle_preserved() {
        let params = test_params();
        // Segment 6 has -30°
        let bank = segment_bank_angle(-3.0e6, &params);
        assert!(bank < 0.0, "expected negative bank, got {}", bank);
    }

    #[test]
    fn returns_signed_value() {
        let mut params = test_params();
        params.bank_angles[0] = -PI / 3.0; // -60° for segment 0
        let bank = segment_bank_angle(4.9e6, &params);
        assert!(bank < 0.0, "bank should be negative: {}", bank);
        assert_relative_eq!(bank.abs(), PI / 3.0, epsilon = 1e-10);
    }
}
```

- [ ] **Step 2: Export the module in `mod.rs`**

Add to `src/rust/src/gnc/guidance/mod.rs`:
```rust
pub mod piecewise_constant;
```

- [ ] **Step 3: Build and run unit tests**

Run: `cd src/rust && cargo test piecewise_constant`
Expected: All 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/piecewise_constant.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "feat: implement piecewise_constant guidance function with unit tests"
```

---

### Task 3: Rust — Guidance dispatch and lateral guidance bypass

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs`

- [ ] **Step 1: Add piecewise_constant dispatch arm in `guidance_step`**

In the match statement at line 151, add after the `Fnpag` arm:
```rust
GuidanceType::PiecewiseConstant => {
    piecewise_constant::piecewise_constant_bank(nav, &data.guidance.piecewise_constant, planet)
}
```

Add import at top of `ftc.rs`:
```rust
use super::piecewise_constant;
```

- [ ] **Step 2: Bypass lateral guidance for PiecewiseConstant**

The piecewise_constant function returns a **signed** bank angle, so lateral guidance (roll reversals) must be skipped. Instead of an early return (which would skip rate saturation and cumulative bank tracking), disable lateral guidance and set `bank_angle_commanded` directly:

Before the lateral guidance section (~line 169, `let mut lateral_active: i32;`), add:
```rust
// PiecewiseConstant provides signed bank angle — skip lateral guidance entirely
if guidance_type == GuidanceType::PiecewiseConstant {
    state.bank_angle_commanded = bank_angle_longitudinal;
    state.roll_sign = if bank_angle_longitudinal >= 0.0 { 1.0 } else { -1.0 };
}
```

Then force `lateral_active = 0` for PiecewiseConstant by adding after the lateral energy check (~line 177):
```rust
if guidance_type == GuidanceType::PiecewiseConstant {
    lateral_active = 0;
}
```

This reuses the existing roll-rate-saturation logic (lines 231-246) and cumulative bank tracking (lines 249-252) without duplication. The combine logic (line 201) won't override `bank_angle_commanded` because `lateral_active == 0` skips the lateral guidance path.

- [ ] **Step 3: Build and verify**

Run: `cd src/rust && cargo build --release && cargo test`
Expected: All existing tests pass. No regressions.

- [ ] **Step 4: Rebuild PyO3 bindings**

Run: `./build.sh`
Expected: Binary + PyO3 module built successfully.

- [ ] **Step 5: Create training config (needed for smoke test)**

Create `configs/training/msr_aller_piecewise_constant_train.toml`:
```toml
base = ["../training/common.toml", "../missions/mars.toml"]

[guidance]
type = "piecewise_constant"

[guidance.piecewise_constant]
energy_min = -6.0
energy_max = 5.0

[simulation]
n_sims = 50
results_suffix = "piecewise_constant"
```

- [ ] **Step 6: Verify piecewise_constant works end-to-end**

```bash
uv run python -c "
import aerocapture_rs as aero
import numpy as np

result = aero.run('configs/training/msr_aller_piecewise_constant_train.toml', overrides={
    'guidance.piecewise_constant.bank_angle_0': 60.0,
    'guidance.piecewise_constant.bank_angle_1': 65.0,
    'guidance.piecewise_constant.bank_angle_2': 70.0,
    'guidance.piecewise_constant.bank_angle_3': -65.0,
    'guidance.piecewise_constant.bank_angle_4': -60.0,
    'guidance.piecewise_constant.bank_angle_5': 55.0,
    'guidance.piecewise_constant.bank_angle_6': 50.0,
    'guidance.piecewise_constant.bank_angle_7': 45.0,
    'guidance.piecewise_constant.bank_angle_8': -40.0,
    'guidance.piecewise_constant.bank_angle_9': 35.0,
})
fr = result.final_record
print(f'Captured: {result.captured}')
print(f'Energy: {fr[7]:.2f} MJ/kg, Ecc: {fr[9]:.4f}')
print(f'Apo err: {fr[30]:.1f} km, DV total: {fr[41]:.1f} m/s')
"
```
Expected: Simulation runs and produces a captured or near-captured trajectory.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs configs/training/msr_aller_piecewise_constant_train.toml
git commit -m "feat: dispatch piecewise_constant in guidance_step with lateral bypass"
```

---

### Task 4: Python — Add piecewise_constant to GA pipeline

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py`
- Modify: `src/python/aerocapture/training/train.py` (CLI choices)

- [ ] **Step 1: Add PARAM_SPACES and GUIDANCE_TOML_SECTIONS entries**

In `param_spaces.py`, add to `PARAM_SPACES` dict:
```python
"piecewise_constant": [
    ParamSpec("bank_angle_0", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_1", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_2", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_3", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_4", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_5", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_6", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_7", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_8", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_9", -180.0, 180.0, 65.0),
],
```

Add to `GUIDANCE_TOML_SECTIONS`:
```python
"piecewise_constant": "piecewise_constant",
```

Add new constant:
```python
# Schemes that require a pre-computed reference trajectory
REQUIRES_REF_TRAJECTORY: set[str] = {"energy_controller", "pred_guid", "fnpag", "ftc"}
```

- [ ] **Step 2: Add `"piecewise_constant"` to CLI choices in `train.py`**

Find the `--guidance` argument `choices` list and add `"piecewise_constant"`.

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py src/python/aerocapture/training/train.py
git commit -m "feat: add piecewise_constant to GA pipeline (param_spaces + CLI)"
```

---

### Task 5: Python — CorridorAccumulator class

**Files:**
- Modify: `src/python/aerocapture/training/corridor.py`
- Modify: `tests/test_corridor.py`

- [ ] **Step 1: Write failing tests for CorridorAccumulator**

Add to `tests/test_corridor.py`:
```python
from aerocapture.training.corridor import CorridorAccumulator, classify_trajectories

class TestCorridorAccumulator:
    def test_init_creates_nan_envelopes(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        assert acc.energy_bins.shape == (200,)
        assert np.all(np.isnan(acc.crash_max_pdyn))
        assert np.all(np.isnan(acc.restricted_max_pdyn))
        assert np.all(np.isnan(acc.restricted_min_pdyn))
        assert np.all(np.isnan(acc.capture_min_pdyn))

    def test_update_populates_envelopes(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        # Crash boundary should be populated (we have crash + non-crash trajectories)
        assert not np.all(np.isnan(acc.crash_max_pdyn))
        # Capture boundary should be populated
        assert not np.all(np.isnan(acc.capture_min_pdyn))

    def test_update_is_incremental(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        crash_after_first = acc.crash_max_pdyn.copy()
        # Second update with same data should not change envelopes
        acc.update(trajs, labels)
        np.testing.assert_array_equal(acc.crash_max_pdyn, crash_after_first)

    def test_checkpoint_roundtrip(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        state = acc.to_checkpoint()
        acc2 = CorridorAccumulator.from_checkpoint(state)
        np.testing.assert_array_equal(acc.crash_max_pdyn, acc2.crash_max_pdyn)
        np.testing.assert_array_equal(acc.restricted_max_pdyn, acc2.restricted_max_pdyn)

    def test_to_corridor_data(self) -> None:
        acc = CorridorAccumulator(energy_min=-6e6, energy_max=5e6, delta_za_restricted=200.0)
        trajs, labels = _make_trajectories_with_labels()
        acc.update(trajs, labels)
        data = acc.to_corridor_data(nominal=trajs[0])
        assert "schema_version" in data
        assert int(data["schema_version"][0]) == 4
        assert "envelope_crash_pdyn" in data
        assert "envelope_restricted_max_pdyn" in data
        assert "envelope_restricted_min_pdyn" in data
        assert "envelope_capture_pdyn" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_corridor.py::TestCorridorAccumulator -v`
Expected: FAIL (CorridorAccumulator not defined yet).

- [ ] **Step 3: Implement CorridorAccumulator**

Add to `corridor.py`:
```python
class CorridorAccumulator:
    """Incremental corridor envelope accumulator.

    Maintains 4 running envelopes updated per generation:
    - crash_max_pdyn: max pdyn of non-crash (above = crash zone)
    - restricted_max_pdyn: max pdyn of corridor-classified (|apo_err| < delta_za)
    - restricted_min_pdyn: min pdyn of corridor-classified
    - capture_min_pdyn: min pdyn of all captured (below = hyperbolic zone)
    """

    def __init__(self, energy_min: float, energy_max: float, delta_za_restricted: float = 200.0, n_bins: int = 200) -> None:
        self.n_bins = n_bins
        self.delta_za_restricted = delta_za_restricted
        bins = np.linspace(energy_min, energy_max, n_bins + 1)
        self.energy_bins = (bins[:-1] + bins[1:]) / 2
        self._bin_edges = bins
        self.crash_max_pdyn = np.full(n_bins, np.nan)
        self.restricted_max_pdyn = np.full(n_bins, np.nan)
        self.restricted_min_pdyn = np.full(n_bins, np.nan)
        self.capture_min_pdyn = np.full(n_bins, np.nan)

    def update(
        self,
        trajectories: list[npt.NDArray[np.float64]],
        labels: npt.NDArray[np.str_],
    ) -> None:
        """Update envelopes with a batch of classified trajectories."""
        non_crash = (labels != "crash") & (labels != "timeout")
        corridor = labels == "corridor"
        captured = (labels == "corridor") | (labels == "undershoot") | (labels == "overshoot")

        self._update_envelope(trajectories, non_crash, self.crash_max_pdyn, use_max=True)
        self._update_envelope(trajectories, corridor, self.restricted_max_pdyn, use_max=True)
        self._update_envelope(trajectories, corridor, self.restricted_min_pdyn, use_max=False)
        self._update_envelope(trajectories, captured, self.capture_min_pdyn, use_max=False)

    def _update_envelope(
        self,
        trajectories: list[npt.NDArray[np.float64]],
        mask: npt.NDArray[np.bool_],
        envelope: npt.NDArray[np.float64],
        use_max: bool,
    ) -> None:
        if not mask.any():
            return
        for i in np.where(mask)[0]:
            t = np.asarray(trajectories[i])
            if t.ndim != 2 or t.shape[0] == 0:
                continue
            e_arr = t[:, _TRAJ_COL_ENERGY]
            p_arr = t[:, _TRAJ_COL_PDYN]
            bin_idx = np.clip(np.digitize(e_arr, self._bin_edges) - 1, 0, self.n_bins - 1)
            for b in range(self.n_bins):
                m = bin_idx == b
                if not m.any():
                    continue
                val = float(np.max(p_arr[m]) if use_max else np.min(p_arr[m]))
                if np.isnan(envelope[b]):
                    envelope[b] = val
                elif use_max:
                    envelope[b] = max(envelope[b], val)
                else:
                    envelope[b] = min(envelope[b], val)

    def to_checkpoint(self) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "corridor_energy_bins": self.energy_bins,
            "corridor_crash_max_pdyn": self.crash_max_pdyn,
            "corridor_restricted_max_pdyn": self.restricted_max_pdyn,
            "corridor_restricted_min_pdyn": self.restricted_min_pdyn,
            "corridor_capture_min_pdyn": self.capture_min_pdyn,
            "corridor_delta_za": np.array([self.delta_za_restricted]),
        }

    @classmethod
    def from_checkpoint(cls, state: dict[str, npt.NDArray[np.float64]]) -> "CorridorAccumulator":
        energy_bins = state["corridor_energy_bins"]
        n_bins = len(energy_bins)
        delta_za = float(state["corridor_delta_za"][0])
        acc = cls(energy_min=0.0, energy_max=1.0, delta_za_restricted=delta_za, n_bins=n_bins)
        acc.energy_bins = energy_bins
        # Reconstruct bin edges from centers
        half = (energy_bins[1] - energy_bins[0]) / 2 if n_bins > 1 else 0.5
        acc._bin_edges = np.concatenate([[energy_bins[0] - half], energy_bins + half])
        acc.crash_max_pdyn = state["corridor_crash_max_pdyn"].copy()
        acc.restricted_max_pdyn = state["corridor_restricted_max_pdyn"].copy()
        acc.restricted_min_pdyn = state["corridor_restricted_min_pdyn"].copy()
        acc.capture_min_pdyn = state["corridor_capture_min_pdyn"].copy()
        return acc

    def to_corridor_data(self, nominal: npt.NDArray[np.float64] | None = None) -> dict[str, npt.NDArray[np.float64]]:
        """Export as corridor cache dict (schema v4)."""
        # Smooth envelopes for visualization
        smoothed = {}
        for name, arr in [
            ("envelope_crash_pdyn", self.crash_max_pdyn),
            ("envelope_restricted_max_pdyn", self.restricted_max_pdyn),
            ("envelope_restricted_min_pdyn", self.restricted_min_pdyn),
            ("envelope_capture_pdyn", self.capture_min_pdyn),
        ]:
            s = arr.copy()
            valid = ~np.isnan(s)
            if valid.any() and (~valid).any():
                s[~valid] = np.interp(self.energy_bins[~valid], self.energy_bins[valid], s[valid])
            if valid.sum() > 5:
                s = uniform_filter1d(s, size=5)
            smoothed[name] = s

        return {
            "schema_version": np.array([4]),
            "energy_bins": self.energy_bins,
            **smoothed,
            "nominal": nominal if nominal is not None else np.empty((0, 12)),
            "delta_za_km": np.array([self.delta_za_restricted]),
        }
```

Update `_SCHEMA_VERSION = 4` at the top of the file. Update `load_corridor` to accept schema 4.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_corridor.py -v`
Expected: All tests pass (old + new).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/corridor.py tests/test_corridor.py
git commit -m "feat: add CorridorAccumulator for incremental envelope building"
```

---

### Task 6: Python — train.py integration (corridor accumulation + ref trajectory)

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

- [ ] **Step 1: Add ref trajectory check at startup**

Early in `main()`, after config loading, add:
```python
from aerocapture.training.param_spaces import REQUIRES_REF_TRAJECTORY

if cfg.guidance_type in REQUIRES_REF_TRAJECTORY:
    # Check for ref trajectory
    ref_traj_path = corr_dir / "ref_trajectory.dat"
    if not ref_traj_path.exists():
        print(f"\nERROR: No reference trajectory found for mission '{mission_name}'.")
        print(f"Run piecewise_constant training first:")
        print(f"  uv run python -m aerocapture.training.train --guidance piecewise_constant --toml <config>")
        sys.exit(1)
    # Override ref trajectory path in config
    # ... (pass via TOML override to Rust sim)
```

- [ ] **Step 2: Initialize CorridorAccumulator during piecewise_constant training**

At GA loop start (before generation loop), if `guidance_type == "piecewise_constant"`:
```python
if cfg.guidance_type == "piecewise_constant":
    from aerocapture.training.corridor import CorridorAccumulator
    energy_range = [
        float(toml_data.get("guidance", {}).get("piecewise_constant", {}).get("energy_min", -6.0)) * 1e6,
        float(toml_data.get("guidance", {}).get("piecewise_constant", {}).get("energy_max", 5.0)) * 1e6,
    ]
    delta_za_r = float(toml_data.get("corridor", {}).get("delta_za_restricted", 200.0))
    corridor_acc = CorridorAccumulator(energy_range[0], energy_range[1], delta_za_restricted=delta_za_r)
```

- [ ] **Step 3: Replace per-chromosome evaluation with single batch call for piecewise_constant**

For piecewise_constant, replace the per-chromosome `evaluate_chromosome` loop with a single `run_batch` call that serves BOTH fitness evaluation AND corridor accumulation (avoids doubling compute cost). In the offspring evaluation section:

```python
if cfg.guidance_type == "piecewise_constant" and corridor_acc is not None:
    # Single batch call: fitness + corridor in one pass
    offspring_overrides = []
    for ind in offspring:
        params = decode_params_from_chromosome(ind, cfg)
        section = GUIDANCE_TOML_SECTIONS[cfg.guidance_type]
        ovr = {f"guidance.{section}.{k}": v for k, v in params.items()}
        ovr["guidance.type"] = cfg.guidance_type
        ovr["simulation.n_sims"] = 1
        if mc_seed is not None:
            ovr["monte_carlo.seed"] = mc_seed
        offspring_overrides.append(ovr)

    batch_results = aero.run_batch(cfg.sim.toml_config, offspring_overrides, include_trajectories=True)

    # Extract fitness from final_records
    for i in range(len(offspring)):
        fr_i = batch_results.final_records[i]
        offspring_costs[i] = compute_cost(fr_i, cost_kwargs)

    # Update corridor envelopes
    labels = classify_trajectories(batch_results.final_records, delta_za=corridor_acc.delta_za_restricted)
    corridor_acc.update(batch_results.trajectories, labels)
else:
    # Standard per-chromosome evaluation for other schemes
    for i in range(len(offspring)):
        cost, _ = evaluate_chromosome(offspring[i], base_network, config, ...)
        offspring_costs[i] = cost
```

- [ ] **Step 4: Save corridor + ref trajectory at end of piecewise_constant training**

After the GA loop completes, if piecewise_constant:
```python
if cfg.guidance_type == "piecewise_constant" and corridor_acc is not None:
    # Re-run best individual with trajectories to get nominal
    best_params = decode_params_from_chromosome(best_overall_chrom, cfg)
    section = GUIDANCE_TOML_SECTIONS[cfg.guidance_type]
    best_ovr = {f"guidance.{section}.{k}": v for k, v in best_params.items()}
    best_ovr["guidance.type"] = cfg.guidance_type
    best_result = aero.run(cfg.sim.toml_config, overrides=best_ovr, include_trajectories=True)
    nom_traj = np.asarray(best_result.trajectory)

    # Save corridor_boundaries.npz
    corr_data = corridor_acc.to_corridor_data(nominal=nom_traj)
    save_corridor(corr_data, corr_dir / "corridor_boundaries.npz")

    # Generate ref_trajectory.dat (7-column format)
    # Derive radial_vel from trajectory cols: V * sin(FPA)
    # Col 3 = vel_m_s, col 4 = fpa_deg
    vel = nom_traj[:, 3]
    fpa_rad = np.radians(nom_traj[:, 4])
    radial_vel = vel * np.sin(fpa_rad)
    energy_j = nom_traj[:, 8] * 1e6  # MJ/kg → J/kg
    pdyn_pa = nom_traj[:, 9] * 1e3   # kPa → Pa
    incl_rad = np.radians(nom_traj[:, 11])
    time_s = nom_traj[:, 7]
    bank_rad = np.radians(nom_traj[:, 10])
    cos_bank = np.cos(bank_rad)

    ref_data = np.column_stack([energy_j, pdyn_pa, radial_vel, radial_vel, incl_rad, time_s, cos_bank])
    ref_path = corr_dir / "ref_trajectory.dat"
    np.savetxt(str(ref_path), ref_data, fmt="  %.16E")
    print(f"  Reference trajectory saved to {ref_path}")
```

- [ ] **Step 5: Add corridor accumulator to checkpoint save/restore**

Extend `save_checkpoint` and `load_checkpoint` to include corridor accumulator state (the 4 envelope arrays + energy bins).

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: integrate corridor accumulator and ref trajectory check in train.py"
```

---

### Task 7: Python — 4-layer corridor visualization in final_report.py

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py`

- [ ] **Step 1: Update `_draw_pdyn_zones` for 4-layer fill with schema v4**

Replace the current 2-layer fill logic with 4-layer:
```python
def _draw_pdyn_zones(ax, corridor_data):
    if corridor_data is None:
        return

    energy = corridor_data.get("energy_bins")
    if energy is None or len(energy) == 0:
        return

    y_data_max = ax.get_ylim()[1]
    y_axis_max = y_data_max * 1.3
    ax.set_ylim(bottom=0, top=y_axis_max)
    x_lo, x_hi = ax.get_xlim()

    # Layer 1: Grey above restricted_max_pdyn (transition zone)
    env_r_max = corridor_data.get("envelope_restricted_max_pdyn")
    if env_r_max is not None and not np.all(np.isnan(env_r_max)):
        valid = ~np.isnan(env_r_max)
        ax.fill_between(energy[valid], env_r_max[valid], y_axis_max, color="#BDBDBD", alpha=0.5, zorder=4)

    # Layer 2: Red above crash_max_pdyn (crash zone — overpaints grey)
    env_crash = corridor_data.get("envelope_crash_pdyn")
    if env_crash is not None and not np.all(np.isnan(env_crash)):
        valid = ~np.isnan(env_crash)
        ax.fill_between(energy[valid], env_crash[valid], y_axis_max, color="#E57373", alpha=0.5, zorder=4.1)

    # Layer 3: Grey below restricted_min_pdyn (transition zone)
    env_r_min = corridor_data.get("envelope_restricted_min_pdyn")
    if env_r_min is not None and not np.all(np.isnan(env_r_min)):
        valid = ~np.isnan(env_r_min)
        ax.fill_between(energy[valid], 0, env_r_min[valid], color="#BDBDBD", alpha=0.5, zorder=4.2)

    # Layer 4: Red below capture_min_pdyn (hyperbolic zone — overpaints grey)
    env_capture = corridor_data.get("envelope_capture_pdyn")
    if env_capture is not None and not np.all(np.isnan(env_capture)):
        valid = ~np.isnan(env_capture)
        ax.fill_between(energy[valid], 0, env_capture[valid], color="#E57373", alpha=0.5, zorder=4.3)

    # Annotations
    mid_e = (x_lo + x_hi) / 2
    ax.text(mid_e, y_axis_max * 0.92, "Crash", ...)
    ax.text(mid_e, y_axis_max * 0.02, "Hyperbolic exit", ...)
    ax.text(x_hi * 0.9, y_axis_max * 0.02, "Entry", ...)
    ax.text(x_lo * 0.9, y_axis_max * 0.02, "Atm. exit", ...)
```

- [ ] **Step 2: Update legend**

```python
legend_elements = [
    Patch(facecolor="#2196F3", alpha=0.4, label="MC captured"),
    Patch(facecolor="#E57373", alpha=0.5, label="Crash / Hyperbolic exit"),
    Patch(facecolor="#BDBDBD", alpha=0.5, label="Transition zone"),
]
```

- [ ] **Step 3: Verify `load_corridor` handles schema v4**

`_SCHEMA_VERSION` was already set to 4 in Task 5. Verify `load_corridor` correctly rejects old schemas and loads v4 files with the new restricted envelope keys.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/final_report.py src/python/aerocapture/training/corridor.py
git commit -m "feat: restore 4-layer corridor visualization with restricted envelopes"
```

---

### Task 8: TOML config update + Rust integration test + final verification

**Files:**
- Modify: `configs/missions/mars.toml`
- Create: `src/rust/tests/piecewise_constant_regression.rs` (or add to existing `guidance_regression.rs`)

- [ ] **Step 1: Add `delta_za_restricted` to `mars.toml`**

Keep existing `delta_za` and `n_sims` (other code may still depend on them). Add the new field:
```toml
[corridor]
delta_za = 500.0            # km, existing (used by corridor.py for other schemes)
delta_za_restricted = 200.0  # km, for restricted corridor envelopes (GA)
n_sims = 10000              # existing
```

- [ ] **Step 2: Add Rust integration test for piecewise_constant**

Add a test to `src/rust/tests/guidance_regression.rs` (or a new file) that runs a full simulation with piecewise_constant guidance and verifies it doesn't panic and produces a reasonable trajectory:

```rust
#[test]
fn piecewise_constant_basic() {
    let config = from_toml_file("../../configs/training/msr_aller_piecewise_constant_train.toml");
    let data = SimData::from_config(&config).unwrap();
    let results = run_for_api(&config, &data, false, false).unwrap();
    assert!(!results.is_empty());
    let r = &results[0];
    assert!(r.final_record[0].is_finite()); // altitude is finite
}
```

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass.

- [ ] **Step 4: Run full Python test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 5: Run linting**

Run: `./lint_code.sh && ./check_all.sh`
Expected: Clean.

- [ ] **Step 6: Smart commit**

Use the `smart-commit` skill to commit everything, update CLAUDE.md and README.md.
