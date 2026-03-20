# Corridor Visualization Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite corridor computation and rendering to produce a physically meaningful 5-zone corridor plot (crash/undershoot/corridor/overshoot/hyperbolic) with proper envelope extraction from classified MC trajectories.

**Architecture:** Two-phase MC in `corridor.py` (Phase 1: dispersed MC for envelope classification, Phase 2: bank-only MC for nominal). Four-layer fill rendering in `final_report.py` replaces the broken midpoint-split hack. TOML `[corridor]` section provides `delta_za` and `n_sims`.

**Tech Stack:** Python (numpy, scipy, matplotlib), PyO3 bindings (`aerocapture_rs`), TOML config, Rust (docstring-only fix)

**Spec:** `docs/superpowers/specs/2026-03-17-corridor-visualization-design.md`

---

### Task 1: Add `[corridor]` section to mission TOMLs

**Files:**
- Modify: `configs/missions/mars.toml:89` (append after `[data]` section)
- Modify: `configs/missions/earth.toml:89` (append after `[data]` section)

- [ ] **Step 1: Add `[corridor]` section to `configs/missions/mars.toml`**

Append after line 89:

```toml
[corridor]
delta_za = 200.0  # km, apoapsis error tolerance for restricted corridor boundaries
n_sims = 10000    # number of MC sims for corridor boundary computation
```

- [ ] **Step 2: Add `[corridor]` section to `configs/missions/earth.toml`**

Append after line 89 (earth has much larger orbits — apoapsis target ~600k km):

```toml
[corridor]
delta_za = 100000.0  # km, apoapsis error tolerance (scaled for Earth high orbit)
n_sims = 10000
```

- [ ] **Step 3: Commit**

```bash
git add configs/missions/mars.toml configs/missions/earth.toml
git commit -m "feat: add [corridor] config section to mission TOMLs"
```

---

### Task 2: Fix PyO3 trajectory docstring

**Files:**
- Modify: `src/rust/aerocapture-py/src/results.rs:18-22`

- [ ] **Step 1: Fix the docstring**

Replace the incorrect docstring at lines 18-22:

```rust
    /// Per-timestep trajectory as an (N, 12) NumPy array.
    ///
    /// Columns: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, heat_flux, time_s,
    ///           energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg].
    /// Empty if trajectories were not requested.
```

- [ ] **Step 2: Verify it compiles**

Run: `cd src/rust && cargo check --manifest-path aerocapture-py/Cargo.toml`
Expected: compiles with no errors (docstring-only change)

- [ ] **Step 3: Commit**

```bash
git add src/rust/aerocapture-py/src/results.rs
git commit -m "fix: correct trajectory column docstring in PyO3 results.rs (cols 8-11)"
```

---

### Task 3: Rewrite `corridor.py` — classification and envelope extraction

**Files:**
- Modify: `src/python/aerocapture/training/corridor.py` (full rewrite)
- Test: `tests/test_corridor.py` (new file)

- [ ] **Step 1: Write tests for trajectory classification**

Create `tests/test_corridor.py` with unit tests for the classification function:

```python
"""Tests for corridor boundary computation."""

from __future__ import annotations

import numpy as np
import pytest

from aerocapture.training.corridor import classify_trajectories


def _make_final_records(
    n_crash: int = 5,
    n_undershoot: int = 10,
    n_corridor: int = 50,
    n_overshoot: int = 10,
    n_hyperbolic: int = 5,
    n_timeout: int = 2,
    delta_za: float = 200.0,
) -> np.ndarray:
    """Create synthetic final_records with known classification counts."""
    n = n_crash + n_undershoot + n_corridor + n_overshoot + n_hyperbolic + n_timeout
    fr = np.zeros((n, 52))
    idx = 0

    # Crash: ifinal=1
    fr[idx : idx + n_crash, 31] = 1.0
    idx += n_crash

    # Undershoot: captured, apo_err < -delta_za
    fr[idx : idx + n_undershoot, 31] = 3.0  # atmosphere exit
    fr[idx : idx + n_undershoot, 7] = -1.0  # energy < 0
    fr[idx : idx + n_undershoot, 9] = 0.5  # ecc < 1
    fr[idx : idx + n_undershoot, 30] = -(delta_za + 50)  # apo_err < -delta_za
    idx += n_undershoot

    # Corridor: captured, -delta_za <= apo_err <= +delta_za
    fr[idx : idx + n_corridor, 31] = 3.0
    fr[idx : idx + n_corridor, 7] = -1.0
    fr[idx : idx + n_corridor, 9] = 0.5
    fr[idx : idx + n_corridor, 30] = np.linspace(-delta_za + 10, delta_za - 10, n_corridor)
    idx += n_corridor

    # Overshoot: captured, apo_err > +delta_za
    fr[idx : idx + n_overshoot, 31] = 3.0
    fr[idx : idx + n_overshoot, 7] = -1.0
    fr[idx : idx + n_overshoot, 9] = 0.5
    fr[idx : idx + n_overshoot, 30] = delta_za + 50
    idx += n_overshoot

    # Hyperbolic: not captured, atmosphere exit
    fr[idx : idx + n_hyperbolic, 31] = 3.0
    fr[idx : idx + n_hyperbolic, 7] = 1.0  # energy > 0
    fr[idx : idx + n_hyperbolic, 9] = 1.5  # ecc > 1
    idx += n_hyperbolic

    # Timeout: ifinal=2
    fr[idx : idx + n_timeout, 31] = 2.0
    idx += n_timeout

    return fr


class TestClassifyTrajectories:
    def test_correct_counts(self) -> None:
        fr = _make_final_records()
        labels = classify_trajectories(fr, delta_za=200.0)
        assert (labels == "crash").sum() == 5
        assert (labels == "undershoot").sum() == 10
        assert (labels == "corridor").sum() == 50
        assert (labels == "overshoot").sum() == 10
        assert (labels == "hyperbolic").sum() == 5
        assert (labels == "timeout").sum() == 2

    def test_crash_priority_over_captured(self) -> None:
        """Crash classification takes priority even if orbital elements say captured."""
        fr = np.zeros((1, 52))
        fr[0, 31] = 1.0  # crash
        fr[0, 7] = -1.0  # energy < 0 (would be "captured")
        fr[0, 9] = 0.5  # ecc < 1
        fr[0, 30] = 0.0  # in corridor
        labels = classify_trajectories(fr, delta_za=200.0)
        assert labels[0] == "crash"

    def test_empty_input(self) -> None:
        fr = np.zeros((0, 52))
        labels = classify_trajectories(fr, delta_za=200.0)
        assert len(labels) == 0

    def test_all_crash(self) -> None:
        fr = np.zeros((10, 52))
        fr[:, 31] = 1.0
        labels = classify_trajectories(fr, delta_za=200.0)
        assert (labels == "crash").sum() == 10

    def test_boundary_values(self) -> None:
        """Trajectories exactly at +-delta_za are in-corridor."""
        fr = np.zeros((2, 52))
        fr[:, 31] = 3.0
        fr[:, 7] = -1.0
        fr[:, 9] = 0.5
        fr[0, 30] = -200.0  # exactly -delta_za
        fr[1, 30] = 200.0  # exactly +delta_za
        labels = classify_trajectories(fr, delta_za=200.0)
        assert labels[0] == "corridor"
        assert labels[1] == "corridor"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_corridor.py -v`
Expected: FAIL — `classify_trajectories` does not exist yet

- [ ] **Step 3: Write tests for envelope extraction**

Append to `tests/test_corridor.py`:

```python
from aerocapture.training.corridor import compute_envelopes


def _make_trajectories_with_labels(
    n_per_class: int = 20,
    n_steps: int = 50,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create synthetic trajectories with known classification labels.

    Returns (trajectories, labels) where trajectories have distinct pdyn ranges
    per class to make envelope testing deterministic.
    """
    rng = np.random.default_rng(seed)
    trajs: list[np.ndarray] = []
    labels_list: list[str] = []

    energy_range = np.linspace(4.0, -6.0, n_steps)

    for cls, pdyn_base in [("crash", 2.5), ("undershoot", 1.8), ("corridor", 1.0), ("overshoot", 0.5), ("hyperbolic", 0.2)]:
        for _ in range(n_per_class):
            t = np.zeros((n_steps, 12))
            t[:, 8] = energy_range  # energy col
            t[:, 9] = pdyn_base + rng.normal(0, 0.05, n_steps)  # pdyn col
            trajs.append(t)
            labels_list.append(cls)

    return trajs, np.array(labels_list)


class TestComputeEnvelopes:
    def test_returns_four_envelopes(self) -> None:
        trajs, labels = _make_trajectories_with_labels()
        result = compute_envelopes(trajs, labels, delta_za=200.0, n_bins=50)
        assert "energy_bins" in result
        assert "envelope_undershoot_pdyn" in result
        assert "envelope_crash_pdyn" in result
        assert "envelope_overshoot_pdyn" in result
        assert "envelope_hyperbolic_pdyn" in result
        assert len(result["energy_bins"]) == 50

    def test_crash_envelope_above_undershoot(self) -> None:
        """Crash boundary (p99 of non-crash) should be above undershoot boundary."""
        trajs, labels = _make_trajectories_with_labels()
        result = compute_envelopes(trajs, labels, delta_za=200.0, n_bins=50)
        crash = result["envelope_crash_pdyn"]
        under = result["envelope_undershoot_pdyn"]
        valid = ~np.isnan(crash) & ~np.isnan(under)
        if valid.any():
            assert np.all(crash[valid] >= under[valid] - 0.1)  # allow small tolerance from smoothing

    def test_hyperbolic_envelope_below_overshoot(self) -> None:
        """Hyperbolic boundary (p1 of captured) should be below overshoot boundary."""
        trajs, labels = _make_trajectories_with_labels()
        result = compute_envelopes(trajs, labels, delta_za=200.0, n_bins=50)
        hyper = result["envelope_hyperbolic_pdyn"]
        over = result["envelope_overshoot_pdyn"]
        valid = ~np.isnan(hyper) & ~np.isnan(over)
        if valid.any():
            assert np.all(hyper[valid] <= over[valid] + 0.1)

    def test_empty_class_produces_nan_envelope(self) -> None:
        """If no crashes exist, crash envelope should be all NaN."""
        trajs, labels = _make_trajectories_with_labels()
        # Remove all crash trajectories
        non_crash = labels != "crash"
        trajs_filtered = [trajs[i] for i in range(len(trajs)) if non_crash[i]]
        labels_filtered = labels[non_crash]
        result = compute_envelopes(trajs_filtered, labels_filtered, delta_za=200.0, n_bins=50)
        assert np.all(np.isnan(result["envelope_crash_pdyn"]))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_corridor.py::TestComputeEnvelopes -v`
Expected: FAIL — `compute_envelopes` does not exist yet

- [ ] **Step 5: Write tests for cache format**

Append to `tests/test_corridor.py`:

```python
from aerocapture.training.corridor import save_corridor, load_corridor


class TestCorridorCache:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "corridor.npz"
        data = {
            "schema_version": np.array([2]),
            "energy_bins": np.linspace(-6, 4, 50),
            "envelope_undershoot_pdyn": np.random.default_rng(0).random(50),
            "envelope_crash_pdyn": np.random.default_rng(1).random(50),
            "envelope_overshoot_pdyn": np.random.default_rng(2).random(50),
            "envelope_hyperbolic_pdyn": np.random.default_rng(3).random(50),
            "nominal": np.random.default_rng(4).random((100, 12)),
            "nominal_bank_deg": np.array([65.0]),
            "nominal_dv": np.array([150.0]),
            "nominal_dv_total": np.array([180.0]),
            "target_apoapsis_km": np.array([500.13]),
            "delta_za_km": np.array([200.0]),
            "n_sims": np.array([10000]),
            "classification_counts": np.array([500, 1000, 6000, 1000, 1500]),
        }
        save_corridor(data, path)
        loaded = load_corridor(path)
        assert loaded is not None
        assert loaded["schema_version"][0] == 2
        np.testing.assert_array_equal(loaded["energy_bins"], data["energy_bins"])

    def test_load_old_cache_returns_none(self, tmp_path: Path) -> None:
        """Old cache format (no schema_version) returns None with warning."""
        path = tmp_path / "old_corridor.npz"
        np.savez_compressed(str(path), nominal=np.zeros((10, 12)), traj_lengths=np.array([10]))
        loaded = load_corridor(path)
        assert loaded is None  # schema mismatch

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.npz"
        loaded = load_corridor(path)
        assert loaded is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_corridor.py::TestCorridorCache -v`
Expected: FAIL — functions have wrong signatures / behavior

- [ ] **Step 7: Implement `classify_trajectories` function**

Rewrite `src/python/aerocapture/training/corridor.py`. Replace the entire file content with:

```python
"""Aerocapture corridor boundary computation via Monte Carlo.

Computes corridor boundaries (4 envelopes) and optimal nominal trajectory
using two MC phases:

1. **Phase 1** (bank angle dispersed [0deg,180deg] + all mission dispersions):
   Classify each trajectory (crash/undershoot/corridor/overshoot/hyperbolic),
   extract 4 pdyn envelopes at p99/p1 percentiles.

2. **Phase 2** (bank angle dispersed [0deg,180deg], no other dispersions):
   Find the trajectory with lowest |dv1|+|dv2| as the optimal constant-bank nominal.

Usage (standalone):
    uv run python -m aerocapture.training.corridor \\
        --toml configs/missions/mars.toml \\
        --output corridor_boundaries.npz
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

# Trajectory column indices (12-column format)
_TRAJ_COL_ENERGY = 8
_TRAJ_COL_PDYN = 9

# Final record column indices (52-column format)
_COL_ENERGY = 7
_COL_ECC = 9
_COL_PERI_ALT = 14
_COL_APO_ERR = 30
_COL_IFINAL = 31
_COL_DV_APO_PERI = 40  # |dv1| + |dv2|
_COL_DV_TOTAL = 41
_DV_CRASH_SENTINEL = 1e10

# Cache schema version
_SCHEMA_VERSION = 2

# Default corridor parameters
_DEFAULT_DELTA_ZA = 200.0
_DEFAULT_N_SIMS = 10000


def _viable_capture_mask(final_records: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return boolean mask for viable captures in a batch of final records (N, 52).

    Viable = bound orbit (ecc < 1, energy < 0) + periapsis above surface + DV not sentinel.
    """
    ecc = final_records[:, _COL_ECC]
    energy = final_records[:, _COL_ENERGY]
    peri_alt = final_records[:, _COL_PERI_ALT]
    dv = final_records[:, _COL_DV_APO_PERI]
    return (ecc < 1.0) & (energy < 0.0) & (peri_alt > 0.0) & (dv < _DV_CRASH_SENTINEL)


def classify_trajectories(
    final_records: npt.NDArray[np.float64],
    delta_za: float = _DEFAULT_DELTA_ZA,
) -> npt.NDArray[np.str_]:
    """Classify each trajectory by outcome.

    Priority order: crash > timeout > hyperbolic > captured sub-categories.

    Returns array of strings: "crash", "undershoot", "corridor", "overshoot",
    "hyperbolic", or "timeout".
    """
    n = len(final_records)
    labels = np.empty(n, dtype="U12")

    if n == 0:
        return labels

    ifinal = final_records[:, _COL_IFINAL]
    energy = final_records[:, _COL_ENERGY]
    ecc = final_records[:, _COL_ECC]
    apo_err = final_records[:, _COL_APO_ERR]

    # Step 1: crash (ifinal == 1) — highest priority
    crash = ifinal == 1.0
    labels[crash] = "crash"

    # Step 2: timeout (ifinal == 2) — discard
    timeout = ifinal == 2.0
    labels[timeout] = "timeout"

    # Step 3: atmosphere exit (ifinal == 3)
    atm_exit = ifinal == 3.0
    captured = atm_exit & (ecc < 1.0) & (energy < 0.0)

    # Hyperbolic: atmosphere exit but not captured
    hyperbolic = atm_exit & ~captured & ~crash & ~timeout
    labels[hyperbolic] = "hyperbolic"

    # Captured sub-categories by apoapsis error
    undershoot = captured & (apo_err < -delta_za)
    overshoot = captured & (apo_err > delta_za)
    corridor = captured & ~undershoot & ~overshoot

    labels[undershoot] = "undershoot"
    labels[overshoot] = "overshoot"
    labels[corridor] = "corridor"

    return labels


def compute_envelopes(
    trajectories: list[npt.NDArray[np.float64]],
    labels: npt.NDArray[np.str_],
    delta_za: float = _DEFAULT_DELTA_ZA,
    n_bins: int = 200,
) -> dict[str, npt.NDArray[np.float64]]:
    """Extract 4 pdyn envelope curves from classified trajectories.

    Returns dict with keys: energy_bins, envelope_{undershoot,crash,overshoot,hyperbolic}_pdyn.
    Each envelope array has NaN where insufficient data exists.
    """
    # Shared energy axis from ALL trajectories
    all_energies: list[float] = []
    for t in trajectories:
        t_arr = np.asarray(t)
        if t_arr.ndim == 2 and t_arr.shape[0] > 0:
            all_energies.extend(t_arr[:, _TRAJ_COL_ENERGY].tolist())

    if not all_energies:
        empty = np.full(n_bins, np.nan)
        return {
            "energy_bins": np.linspace(-6, 4, n_bins),
            "envelope_undershoot_pdyn": empty.copy(),
            "envelope_crash_pdyn": empty.copy(),
            "envelope_overshoot_pdyn": empty.copy(),
            "envelope_hyperbolic_pdyn": empty.copy(),
        }

    e_all = np.array(all_energies)
    bins = np.linspace(e_all.min(), e_all.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def _envelope(mask: npt.NDArray[np.bool_], percentile: float) -> npt.NDArray[np.float64]:
        """Compute percentile of pdyn per energy bin for trajectories matching mask."""
        result = np.full(n_bins, np.nan)
        if not mask.any():
            return result

        # Collect all (energy, pdyn) points from matching trajectories
        e_pts: list[float] = []
        p_pts: list[float] = []
        for i in np.where(mask)[0]:
            t_arr = np.asarray(trajectories[i])
            if t_arr.ndim != 2 or t_arr.shape[0] == 0:
                continue
            e_pts.extend(t_arr[:, _TRAJ_COL_ENERGY].tolist())
            p_pts.extend(t_arr[:, _TRAJ_COL_PDYN].tolist())

        if not e_pts:
            return result

        e_arr = np.array(e_pts)
        p_arr = np.array(p_pts)
        bin_idx = np.clip(np.digitize(e_arr, bins) - 1, 0, n_bins - 1)

        for b in range(n_bins):
            m = bin_idx == b
            count = int(m.sum())
            if count >= 3:  # minimum bin occupancy
                result[b] = np.percentile(p_arr[m], percentile)

        # Interpolate NaN gaps from neighbors
        valid = ~np.isnan(result)
        if valid.any() and (~valid).any():
            result[~valid] = np.interp(
                bin_centers[~valid], bin_centers[valid], result[valid]
            )

        # Smooth to reduce jaggedness
        valid_after = ~np.isnan(result)
        if valid_after.sum() > 5:
            result[valid_after] = uniform_filter1d(result[valid_after], size=5)

        return result

    # Envelope A (undershoot boundary): p99 of captured trajectories with apo_err >= -delta_za
    # "corridor" and "overshoot" qualify; "undershoot" has apo_err < -delta_za so excluded
    mask_a = (labels == "corridor") | (labels == "overshoot")
    if not mask_a.any():
        warnings.warn("No captured trajectories with apo_err >= -delta_za — undershoot envelope empty", stacklevel=2)
    envelope_undershoot = _envelope(mask_a, 99)

    # Envelope B (crash boundary): p99 of ALL non-crashing trajectories
    mask_b = (labels != "crash") & (labels != "timeout")
    if not mask_b.any():
        warnings.warn("No non-crashing trajectories — crash envelope empty", stacklevel=2)
    envelope_crash = _envelope(mask_b, 99)

    # Envelope C (overshoot boundary): p1 of captured trajectories with apo_err <= +delta_za
    # These are: corridor + undershoot (captured with apo_err <= +delta_za)
    mask_c = (labels == "corridor") | (labels == "undershoot")
    if not mask_c.any():
        warnings.warn("No captured trajectories with apo_err <= +delta_za — overshoot envelope empty", stacklevel=2)
    envelope_overshoot = _envelope(mask_c, 1)

    # Envelope D (hyperbolic boundary): p1 of ALL captured trajectories
    mask_d = (labels == "corridor") | (labels == "undershoot") | (labels == "overshoot")
    if not mask_d.any():
        warnings.warn("No captured trajectories — hyperbolic envelope empty", stacklevel=2)
    envelope_hyperbolic = _envelope(mask_d, 1)

    return {
        "energy_bins": bin_centers,
        "envelope_undershoot_pdyn": envelope_undershoot,
        "envelope_crash_pdyn": envelope_crash,
        "envelope_overshoot_pdyn": envelope_overshoot,
        "envelope_hyperbolic_pdyn": envelope_hyperbolic,
    }


def _run_mc_constant_bank(
    toml_path: str,
    n_sims: int,
    bank_lo: float,
    bank_hi: float,
    seed: int,
    with_dispersions: bool,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Run MC with constant bank angle uniformly dispersed in [bank_lo, bank_hi].

    When with_dispersions=False, sets dispersion level to "none" to disable all
    mission dispersions (only bank angle varies).

    Returns (final_records (N, 52), trajectories list of (T_i, 12)).
    """
    try:
        import aerocapture_rs as aero  # type: ignore[import-not-found, import-untyped]
    except ImportError as e:
        msg = "PyO3 aerocapture_rs module required for corridor computation"
        raise ImportError(msg) from e

    rng = np.random.default_rng(seed)
    bank_angles = rng.uniform(bank_lo, bank_hi, n_sims)

    overrides_list = []
    for bank in bank_angles:
        ovr: dict[str, object] = {
            "guidance.type": "ftc",  # dummy — overridden by reference_trajectory
            "guidance.reference_trajectory": True,
            "guidance.reference_bank_angle": float(bank),
            "simulation.n_sims": 1,
        }
        if not with_dispersions:
            ovr["monte_carlo.dispersion_level"] = "none"
            ovr["monte_carlo.seed"] = seed
        overrides_list.append(ovr)

    results = aero.run_batch(  # type: ignore[attr-defined]
        toml_path,
        overrides_list,
        include_trajectories=True,
    )
    final_records: npt.NDArray[np.float64] = results.final_records
    trajectories: list[npt.NDArray[np.float64]] = results.trajectories
    return final_records, trajectories


def _read_corridor_config(toml_path: str) -> tuple[float, int]:
    """Read delta_za and n_sims from TOML [corridor] section.

    Falls back to defaults if section is absent.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(Path(toml_path))
    corridor = data.get("corridor", {})
    delta_za = float(corridor.get("delta_za", _DEFAULT_DELTA_ZA))
    n_sims = int(corridor.get("n_sims", _DEFAULT_N_SIMS))
    return delta_za, n_sims


def compute_corridor(
    toml_path: str,
    n_sims: int | None = None,
    seed: int = 42,
    delta_za: float | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute corridor boundaries and optimal nominal via Monte Carlo.

    1. Phase 1 (bank [0deg,180deg] + all dispersions): classify trajectories,
       extract 4 pdyn envelopes.
    2. Phase 2 (bank [0deg,180deg], no dispersions): find min-DV nominal.

    Args:
        toml_path: Path to mission TOML config.
        n_sims: Number of MC sims per phase (overrides TOML [corridor].n_sims).
        seed: Random seed.
        delta_za: Apoapsis error tolerance in km (overrides TOML [corridor].delta_za).

    Returns dict with corridor cache data (schema version 2).
    """
    toml_str = str(Path(toml_path).resolve())

    # Read config defaults, allow CLI overrides
    cfg_delta_za, cfg_n_sims = _read_corridor_config(toml_str)
    delta_za = delta_za if delta_za is not None else cfg_delta_za
    n_sims = n_sims if n_sims is not None else cfg_n_sims

    # Read target apoapsis
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(Path(toml_str))
    target_apo = float(toml_data.get("flight", {}).get("target_orbit", {}).get("apoapsis", 0.0))

    # Phase 1: Full MC (bank + all dispersions) -> envelopes
    print(f"  Phase 1: Running {n_sims}-sim full MC (bank [0deg,180deg] + dispersions)...")
    fr_full, traj_full = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed, with_dispersions=True)
    labels = classify_trajectories(fr_full, delta_za=delta_za)

    counts = np.array([
        int((labels == "crash").sum()),
        int((labels == "undershoot").sum()),
        int((labels == "corridor").sum()),
        int((labels == "overshoot").sum()),
        int((labels == "hyperbolic").sum()),
    ])
    print(f"    Classification: crash={counts[0]}, under={counts[1]}, corridor={counts[2]}, over={counts[3]}, hyper={counts[4]}")

    envelopes = compute_envelopes(traj_full, labels, delta_za=delta_za)

    # Phase 2: Bank-only MC (no dispersions) -> nominal
    print(f"  Phase 2: Running {n_sims}-sim bank-only MC (no dispersions)...")
    fr_nom, traj_nom = _run_mc_constant_bank(toml_str, n_sims, 0.0, 180.0, seed + 1, with_dispersions=False)
    viable_nom = _viable_capture_mask(fr_nom)
    n_viable_nom = int(viable_nom.sum())
    print(f"    Viable captures: {n_viable_nom}/{n_sims}")

    if n_viable_nom > 0:
        dv_values = fr_nom[viable_nom, _COL_DV_APO_PERI]
        best_idx_in_viable = int(np.argmin(dv_values))
        best_idx = np.where(viable_nom)[0][best_idx_in_viable]
        nom_traj = traj_nom[best_idx]
        nom_dv = float(dv_values[best_idx_in_viable])
        nom_bank = float(nom_traj[0, 10]) if nom_traj.size > 0 else 0.0
        nom_dv_total = float(fr_nom[best_idx, _COL_DV_TOTAL])
        print(f"    Nominal: bank={nom_bank:.1f}deg, DV(apo+peri)={nom_dv:.1f} m/s, DV(total)={nom_dv_total:.1f} m/s")
    else:
        print("  WARNING: No viable captures in bank-only MC")
        nom_traj = np.empty((0, 12))
        nom_bank = 0.0
        nom_dv = 0.0
        nom_dv_total = 0.0

    return {
        "schema_version": np.array([_SCHEMA_VERSION]),
        **envelopes,
        "nominal": nom_traj,
        "nominal_bank_deg": np.array([nom_bank]),
        "nominal_dv": np.array([nom_dv]),
        "nominal_dv_total": np.array([nom_dv_total]),
        "target_apoapsis_km": np.array([target_apo]),
        "delta_za_km": np.array([delta_za]),
        "n_sims": np.array([n_sims]),
        "classification_counts": counts,
    }


def save_corridor(data: dict[str, npt.NDArray[np.float64]], output_path: Path) -> None:
    """Save corridor data to a compressed .npz file."""
    np.savez_compressed(str(output_path), **data)  # type: ignore[arg-type]
    print(f"  Corridor data saved to {output_path}")


def load_corridor(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load corridor data from a .npz file.

    Returns None if file not found or schema version mismatch.
    """
    if not path.exists():
        return None
    npz = np.load(str(path))
    data = {k: npz[k] for k in npz.files}

    # Check schema version
    version = data.get("schema_version", np.array([0]))
    if int(version[0]) != _SCHEMA_VERSION:
        warnings.warn(
            f"Corridor cache {path} has schema version {int(version[0])}, "
            f"expected {_SCHEMA_VERSION}. Recomputation required.",
            stacklevel=2,
        )
        return None

    return data


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute aerocapture corridor boundaries via Monte Carlo")
    parser.add_argument("--toml", type=str, required=True, help="Mission TOML config (e.g., configs/missions/mars.toml)")
    parser.add_argument("--output", type=str, default="corridor_boundaries.npz", help="Output .npz file path")
    parser.add_argument("--n-sims", type=int, default=None, help="Number of MC sims (default: from TOML or 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--delta-za", type=float, default=None, help="Apoapsis error tolerance km (default: from TOML or 200)")
    args = parser.parse_args()

    print(f"Computing corridor boundaries for {args.toml}...")
    corridor = compute_corridor(args.toml, n_sims=args.n_sims, seed=args.seed, delta_za=args.delta_za)
    save_corridor(corridor, Path(args.output))

    counts = corridor["classification_counts"]
    print(f"\n  Summary: crash={counts[0]}, undershoot={counts[1]}, corridor={counts[2]}, overshoot={counts[3]}, hyperbolic={counts[4]}")
    nom = corridor["nominal"]
    if nom.size > 0:
        print(f"  Nominal: {nom.shape[0]} timesteps, energy [{nom[:, _TRAJ_COL_ENERGY].min():.2f}, {nom[:, _TRAJ_COL_ENERGY].max():.2f}] MJ/kg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run all corridor tests**

Run: `uv run pytest tests/test_corridor.py -v`
Expected: ALL PASS

- [ ] **Step 9: Run linter**

Run: `./lint_code.sh`
Expected: No errors from ruff/mypy on corridor.py

- [ ] **Step 10: Commit**

```bash
git add src/python/aerocapture/training/corridor.py tests/test_corridor.py
git commit -m "feat: rewrite corridor.py with trajectory classification and 4-envelope extraction"
```

---

### Task 4: Rewrite corridor rendering in `final_report.py`

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:490-716`
- Modify: `tests/test_final_report.py` (update existing tests, add new ones)

- [ ] **Step 1: Write tests for the new corridor rendering**

Add to `tests/test_final_report.py`:

```python
class TestCorridorRendering:
    def test_corridor_png_with_new_format_corridor_data(self, tmp_path: Path) -> None:
        """Corridor PNG renders correctly with schema-v2 corridor data."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(50, with_trajectories=True, n_captured=40, n_hyper=10)
        corridor_data = {
            "schema_version": np.array([2]),
            "energy_bins": np.linspace(-6, 4, 50),
            "envelope_undershoot_pdyn": np.linspace(1.5, 0.0, 50),
            "envelope_crash_pdyn": np.linspace(2.0, 0.0, 50),
            "envelope_overshoot_pdyn": np.linspace(0.8, 0.0, 50),
            "envelope_hyperbolic_pdyn": np.linspace(0.3, 0.0, 50),
            "nominal": np.column_stack([
                np.zeros((30, 8)),
                np.linspace(4, -5, 30),  # energy
                np.linspace(0, 1.2, 30),  # pdyn
                np.full(30, 65.0),  # bank
                np.full(30, 50.0),  # incl
            ]),
            "nominal_bank_deg": np.array([65.0]),
            "nominal_dv": np.array([150.0]),
            "nominal_dv_total": np.array([180.0]),
        }
        # Save corridor data to npz
        corr_path = tmp_path / "corridor_boundaries.npz"
        np.savez_compressed(str(corr_path), **corridor_data)

        output = tmp_path / "report.html"
        generate_final_report(eval_data, "eqglide", 50.0, output, corridor_path=corr_path)
        corridor_png = tmp_path / "report_corridors.png"
        assert corridor_png.exists()
        assert corridor_png.stat().st_size > 1000

    def test_guided_nominal_is_min_dv(self, tmp_path: Path) -> None:
        """Guided nominal should be the min-DV captured trajectory, not first-by-index."""
        from aerocapture.training.final_report import _select_guided_nominal

        n = 50
        final_array = _make_captured_array(n)
        captured = np.ones(n, dtype=bool)
        # Set DV values: make index 30 the minimum
        final_array[:, 41] = 200.0
        final_array[30, 41] = 50.0

        trajectories = _make_trajectories(n)
        idx, dv = _select_guided_nominal(final_array, captured, trajectories)
        assert idx == 30
        assert dv == pytest.approx(50.0)

    def test_guided_nominal_none_when_no_captures(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import _select_guided_nominal

        final_array = _make_all_hyperbolic(20)
        captured = np.zeros(20, dtype=bool)
        trajectories = _make_trajectories(20)
        idx, dv = _select_guided_nominal(final_array, captured, trajectories)
        assert idx is None
        assert dv is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_final_report.py::TestCorridorRendering -v`
Expected: FAIL — `_select_guided_nominal` doesn't exist, corridor rendering uses old format

- [ ] **Step 3: Implement `_select_guided_nominal` and rewrite `_draw_pdyn_zones`**

In `src/python/aerocapture/training/final_report.py`, make these changes:

**3a. Add `_select_guided_nominal` function** (after `_compute_envelope`, around line 525):

```python
def _select_guided_nominal(
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
    trajectories: list[npt.NDArray[np.float64]],
) -> tuple[int | None, float | None]:
    """Select guided nominal as the min total-DV captured trajectory.

    Returns (index, dv_total) or (None, None) if no captures.
    """
    if not captured.any():
        return None, None

    cap_indices = np.where(captured)[0]
    dv_values = final_array[cap_indices, _COL_DV_TOTAL]
    best_in_cap = int(np.argmin(dv_values))
    best_idx = int(cap_indices[best_in_cap])

    t = np.asarray(trajectories[best_idx])
    if t.ndim != 2 or t.shape[0] == 0:
        return None, None

    return best_idx, float(dv_values[best_in_cap])
```

**3b. Delete the OLD `_draw_pdyn_zones` function entirely** (lines 531-583), including the `_COLOR_CRASH` and `_COLOR_HYPERBOLIC` constants at lines 527-528. The old function body imports `_unpack_trajectories` from `corridor.py` which no longer exists. **Replace** with the new 4-layer fill approach:

```python
_COLOR_CRASH = "#E57373"       # light red for crash zone
_COLOR_HYPERBOLIC_ZONE = "#90A4AE"  # blue-grey for hyperbolic exit zone
_COLOR_UNDERSHOOT = "#BDBDBD"  # grey for undershoot zone
_COLOR_OVERSHOOT = "#BDBDBD"   # grey for overshoot zone


def _draw_pdyn_zones(
    ax: Any,  # matplotlib Axes
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
) -> None:
    """Draw 4-layer corridor zones on the pdyn panel.

    Layers (back to front):
    1. Grey fill above undershoot envelope (Envelope A)
    2. Red fill above crash envelope (Envelope B) — overpaints grey
    3. Grey fill below overshoot envelope (Envelope C)
    4. Red fill below hyperbolic envelope (Envelope D) — overpaints grey

    The white gap between Envelopes A and C is the viable corridor.
    """
    if corridor_data is None:
        return

    energy = corridor_data.get("energy_bins")
    if energy is None or len(energy) == 0:
        return

    # Add 30% headroom above the data
    y_data_max = ax.get_ylim()[1]
    y_axis_max = y_data_max * 1.3
    ax.set_ylim(bottom=0, top=y_axis_max)

    x_lo, x_hi = ax.get_xlim()

    # Layer 1: Grey above undershoot boundary (Envelope A)
    env_under = corridor_data.get("envelope_undershoot_pdyn")
    if env_under is not None and not np.all(np.isnan(env_under)):
        valid = ~np.isnan(env_under)
        ax.fill_between(energy[valid], env_under[valid], y_axis_max, color=_COLOR_UNDERSHOOT, alpha=0.5, zorder=4, label="Undershoot")

    # Layer 2: Red above crash boundary (Envelope B) — overpaints grey
    env_crash = corridor_data.get("envelope_crash_pdyn")
    if env_crash is not None and not np.all(np.isnan(env_crash)):
        valid = ~np.isnan(env_crash)
        ax.fill_between(energy[valid], env_crash[valid], y_axis_max, color=_COLOR_CRASH, alpha=0.5, zorder=4.1, label="Crash")

    # Layer 3: Grey below overshoot boundary (Envelope C)
    env_over = corridor_data.get("envelope_overshoot_pdyn")
    if env_over is not None and not np.all(np.isnan(env_over)):
        valid = ~np.isnan(env_over)
        ax.fill_between(energy[valid], 0, env_over[valid], color=_COLOR_OVERSHOOT, alpha=0.5, zorder=4.2, label="Overshoot")

    # Layer 4: Red below hyperbolic boundary (Envelope D) — overpaints grey
    env_hyper = corridor_data.get("envelope_hyperbolic_pdyn")
    if env_hyper is not None and not np.all(np.isnan(env_hyper)):
        valid = ~np.isnan(env_hyper)
        ax.fill_between(energy[valid], 0, env_hyper[valid], color=_COLOR_CRASH, alpha=0.5, zorder=4.3, label="Hyperbolic exit")

    # Annotations
    mid_e = (x_lo + x_hi) / 2
    ax.text(mid_e, y_axis_max * 0.92, "Crash", ha="center", fontsize=10, fontstyle="italic", color="#B71C1C", zorder=6)
    ax.text(mid_e, y_axis_max * 0.02, "Hyperbolic exit", ha="center", fontsize=10, fontstyle="italic", color="#37474F", zorder=6)
    ax.text(x_hi * 0.9, y_axis_max * 0.02, "Entry", fontsize=8, color="#37474F", ha="right", zorder=6)
    ax.text(x_lo * 0.9, y_axis_max * 0.02, "Atm. exit", fontsize=8, color="#37474F", ha="left", zorder=6)
```

**3c. Update `_generate_corridor_png`** to use `_select_guided_nominal` and the new `_draw_pdyn_zones` signature:

- Replace the guided nominal selection block (lines 628-636) with a call to `_select_guided_nominal`
- Update the `_draw_pdyn_zones` call (line 656) to pass only `corridor_data` (remove `trajectories` and `captured` args)
- Update the legend elements to match new zone names/colors

Specifically, replace lines 627-636:

```python
    # Guidance nominal: min-DV captured trajectory from final-evaluation MC
    guid_nom: npt.NDArray[np.float64] | None = None
    guid_nom_dv: float | None = None
    if captured.any():
        guid_idx, guid_nom_dv = _select_guided_nominal(final_array, captured, trajectories)
        if guid_idx is not None:
            t = np.asarray(trajectories[guid_idx])
            if t.ndim == 2 and t.shape[0] > 0:
                guid_nom = t
```

**Note:** Keep `_compute_envelope` (lines 490-524) — it's still used by the inclination and bank angle panels (lines 659-662). Only `_draw_pdyn_zones` and its old color constants are replaced.

And replace the `_draw_pdyn_zones` call at line 655-656:

```python
        if y_col == _TRAJ_COL_PDYN:
            _draw_pdyn_zones(ax, corridor_data)
```

And update the legend (lines 678-686):

```python
    legend_elements: list[Any] = [
        Patch(facecolor="#2196F3", alpha=0.4, label="MC captured"),
        Patch(facecolor=_COLOR_CRASH, alpha=0.5, label="Crash / Hyperbolic exit"),
        Patch(facecolor=_COLOR_UNDERSHOOT, alpha=0.5, label="Undershoot / Overshoot"),
    ]
```

- [ ] **Step 4: Run all final_report tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run linter**

Run: `./lint_code.sh`
Expected: No errors from ruff/mypy

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/final_report.py tests/test_final_report.py
git commit -m "feat: 4-layer corridor rendering with min-DV guided nominal selection"
```

---

### Task 5: Update `train.py` caller

**Files:**
- Modify: `src/python/aerocapture/training/train.py:850`

- [ ] **Step 1: Update `compute_corridor` call to use TOML defaults**

At line 850, change:
```python
                    corr_data = compute_corridor(str(base_toml_path), n_sims=1000, seed=final_seed)
```
to:
```python
                    corr_data = compute_corridor(str(base_toml_path), seed=final_seed)
```

This lets `n_sims` and `delta_za` come from the TOML `[corridor]` section (default 10000 and 200.0).

- [ ] **Step 2: Run linter**

Run: `./lint_code.sh`
Expected: Clean

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "fix: use TOML-configured n_sims for corridor computation in train.py"
```

---

### Task 6: Integration test and cleanup

**Files:**
- Verify no dangling imports reference removed functions

- [ ] **Step 1: Verify no other files import the removed functions**

Run: `grep -r "load_corridor_trajectories\|_pack_trajectories\|_unpack_trajectories" src/ tests/`
Expected: Only in `corridor.py` itself (already removed in Task 3). If found elsewhere, update those callers.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (no regressions)

- [ ] **Step 3: Run linter on full codebase**

Run: `./lint_code.sh`
Expected: Clean

- [ ] **Step 4: Commit cleanup**

```bash
git add -A
git commit -m "chore: remove dead corridor code (pack/unpack trajectories, old load helper)"
```

---

### Task 7: Smart commit (final)

- [ ] **Step 1: Invoke `smart-commit` skill**

Use the `smart-commit` skill to update CLAUDE.md and README.md with the corridor visualization changes, then commit everything on the branch.

Take the whole git branch into account when updating documentation.
