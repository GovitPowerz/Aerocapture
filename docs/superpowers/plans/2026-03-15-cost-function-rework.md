# Cost Function Rework Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace orbit altitude errors with delta-V as the primary cost function objective, add normalized soft constraint penalties for g-load and heat flux, and populate the existing but unused peak-value columns in the Rust simulator.

**Architecture:** Two-layer change — (1) Rust simulator tracks peak heat flux, g-load, and dynamic pressure during integration, populating existing placeholder columns 16-24; (2) Python cost function swaps orbit errors for delta-V + normalized constraint penalties with TOML-configurable thresholds and weights.

**Tech Stack:** Rust (nalgebra), Python (numpy), TOML config, pytest, cargo test

**Spec:** `docs/superpowers/specs/2026-03-15-cost-function-rework-design.md`

---

## Chunk 1: Rust Peak Value Tracking

### Task 1: Add peak value tracking to the integration loop

The `SimState` struct already has `max_heat_flux`, `max_load_factor`, `max_dyn_pressure` fields and their altitude/time companions (lines 51-60 of `runner.rs`), initialized to 0.0 (lines 356-364). The final record output already reads from them (lines 609-617). The only missing piece is updating them during integration.

**Files:**
- Modify: `src/rust/src/simulation/runner.rs:500-524` (after `integrate_step`, before termination checks)

- [ ] **Step 1: Add `track_peak_values` helper function**

Add this function above `compute_derivatives` (before line 737):

```rust
/// Update peak tracking values (heat flux, load factor, dynamic pressure)
/// after each integration step.
fn track_peak_values(
    sim: &mut SimState,
    altitude: f64,
    data: &SimData,
    run_state: &init::RunState,
) {
    let v = sim.state[3];
    let sim_time = sim.state[7];
    let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias);

    // Heat flux (W/m²) — same formula as dflux in compute_derivatives
    let heat_flux = data.capsule.cq * rho.sqrt() * v.powf(3.05);

    // Dynamic pressure (Pa)
    let pdyn = 0.5 * rho * v * v;

    // Load factor (m/s²) — aerodynamic acceleration magnitude
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho * ref_area * v * v / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    if heat_flux > sim.max_heat_flux {
        sim.max_heat_flux = heat_flux;
        sim.alt_max_flux = altitude;
        sim.time_max_flux = sim_time;
    }
    if load_factor > sim.max_load_factor {
        sim.max_load_factor = load_factor;
        sim.alt_max_load = altitude;
        sim.time_max_load = sim_time;
    }
    if pdyn > sim.max_dyn_pressure {
        sim.max_dyn_pressure = pdyn;
        sim.alt_max_pdyn = altitude;
        sim.time_max_pdyn = sim_time;
    }
}
```

Note: `load_factor` is stored in m/s² here; it gets divided by `G0` (9.81) when written to `final_record[17]` at line 610. This matches the existing convention in the struct comment (`data/mod.rs:123`).

- [ ] **Step 2: Call `track_peak_values` in the integration loop**

In the main loop, after altitude computation (line 503-504) and before termination checks (line 506), insert:

```rust
        track_peak_values(&mut sim, altitude, data, run_state);
```

This goes right after:
```rust
        let (altitude, _lat_geo) =
            geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);
```

And before:
```rust
        // === Termination checks ===
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd src/rust && cargo build --release`
Expected: Compiles with no errors or warnings.

- [ ] **Step 4: Run existing Rust tests**

Run: `cd src/rust && cargo test`
Expected: All ~176 tests pass. No regressions — we're only populating previously-zero fields.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "feat(rust): track peak heat flux, g-load, and dynamic pressure during integration"
```

### Task 2: Add Rust tests for peak value columns

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (add inline test in the existing `run_output_tests` module at line 807)

- [ ] **Step 1: Write integration test using existing `load_test_config` helper**

First, refactor `load_test_config()` (line 812) to accept an optional config filename. Then add integration tests. All go in the existing `mod run_output_tests` block in `runner.rs`:

```rust
    fn load_config(config_name: &str) -> (SimInput, SimData) {
        // Data file paths in TOML configs are relative to repo root
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let repo_root = std::path::PathBuf::from(&manifest)
            .join("../..")
            .canonicalize()
            .unwrap();
        std::env::set_current_dir(&repo_root).unwrap();

        let content = std::fs::read_to_string(config_name).expect("test config");
        let (sim_config, toml_config) = SimInput::from_toml(&content).expect("parse");
        let sim_data = SimData::from_toml(&toml_config, &sim_config).expect("data");
        (sim_config, sim_data)
    }

    // Update load_test_config to delegate:
    fn load_test_config() -> (SimInput, SimData) {
        load_config("configs/test/test_ref_orig.toml")
    }

    #[test]
    fn peak_values_populated_for_atmospheric_trajectory() {
        // Use high-bank config which enters atmosphere deeply
        let (config, data) = load_config("configs/test/test_high_bank_orig.toml");
        let results = run_for_api(&config, &data).expect("run");
        let rec = &results[0].final_record;

        // Columns 16-18: peak heat flux (kW/m²), load factor (g), dynamic pressure (kPa)
        assert!(rec[16] > 0.0, "max_heat_flux should be > 0, got {}", rec[16]);
        assert!(rec[17] > 0.0, "max_load_factor should be > 0, got {}", rec[17]);
        assert!(rec[18] > 0.0, "max_dyn_pressure should be > 0, got {}", rec[18]);

        // Columns 19-24: altitudes and times at peak values
        assert!(rec[19] > 0.0, "alt_max_flux should be > 0, got {}", rec[19]);
        assert!(rec[20] > 0.0, "alt_max_load should be > 0, got {}", rec[20]);
        assert!(rec[21] > 0.0, "alt_max_pdyn should be > 0, got {}", rec[21]);
        assert!(rec[22] > 0.0, "time_max_flux should be > 0, got {}", rec[22]);
        assert!(rec[23] > 0.0, "time_max_load should be > 0, got {}", rec[23]);
        assert!(rec[24] > 0.0, "time_max_pdyn should be > 0, got {}", rec[24]);

        // Physical plausibility for Mars entry:
        // Heat flux: 10-500 kW/m² typical, load factor: 1-30 g typical
        assert!(rec[16] > 10.0 && rec[16] < 500.0,
            "peak heat flux {:.1} kW/m² outside reasonable Mars entry range", rec[16]);
        assert!(rec[17] > 1.0 && rec[17] < 30.0,
            "peak load factor {:.1} g outside reasonable Mars entry range", rec[17]);
    }
```

- [ ] **Step 2: Write unit test for `track_peak_values` with known inputs**

Add a separate unit test that directly calls `track_peak_values` with hand-constructed inputs and asserts exact computed values. This requires constructing a minimal `SimState`, `SimData` with known `cq`/aero tables, and a zeroed `RunState`:

```rust
    #[test]
    fn track_peak_values_heat_flux_formula() {
        // Known inputs: cq=1e-4, rho=0.01 kg/m³, v=5000 m/s
        // Expected heat flux = cq * sqrt(rho) * v^3.05
        //                    = 1e-4 * 0.1 * 5000_f64.powf(3.05)
        // Store in max_heat_flux (W/m²), final_record divides by 1e3 → kW/m²
        //
        // This test verifies the formula is correct by checking the SimState
        // fields after calling track_peak_values with synthetic inputs.
        // The exact construction of SimData/SimState/RunState with mock values
        // may require test helpers — adapt to whatever constructors are available.
        // The key assertion: sim.max_heat_flux == cq * rho.sqrt() * v.powf(3.05)
        //
        // If constructing full SimData is too complex for a unit test, an
        // acceptable alternative is to add a standalone pure function
        // `compute_heat_flux(cq, rho, v) -> f64` and unit-test that.
        todo!("Implement with project-specific test helpers for SimData construction")
    }
```

Note: the implementer should adapt the test to use whatever test infrastructure is available for constructing `SimData`. If creating a full `SimData` from scratch is impractical, extracting the heat flux and load factor formulas into small pure functions and unit-testing those is an acceptable alternative.

- [ ] **Step 2: Run the new tests**

Run: `cd src/rust && cargo test peak_values -- --nocapture`
Expected: PASS — all assertions hold for an atmospheric entry trajectory.

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass including the new ones.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "test(rust): verify peak heat flux, g-load, dynamic pressure columns are populated"
```

---

## Chunk 2: Python Cost Function Rewrite

### Task 3: Fix pre-existing Python 2 except syntax bug

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:212`

- [ ] **Step 1: Fix the syntax**

Change line 212 from:
```python
    except subprocess.TimeoutExpired, FileNotFoundError:
```
To:
```python
    except (subprocess.TimeoutExpired, FileNotFoundError):
```

- [ ] **Step 2: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "fix: Python 2 except syntax on subprocess fallback path"
```

### Task 4: Rewrite `compute_cost()` with delta-V primary + constraint penalties

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:229-284`

- [ ] **Step 1: Write failing tests for the new cost function**

Replace the contents of `tests/test_cost.py` with tests for the new formula. The `_make_row` helper needs `g_max` (column 17) and `q_max` (column 16) parameters:

```python
"""Tests for compute_cost: delta-V primary with normalized constraint penalties.

Column layout of final_conditions (0-indexed, 52-column):
    7  = energy (MJ/kg), >0 → hyperbolic
    9  = eccentricity, >1 → hyperbolic
    16 = max heat flux (kW/m²)
    17 = max g-load (g)
    27 = sim_time (s)
    41 = dv_total (m/s)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.evaluate import compute_cost
from hypothesis import given, settings
from hypothesis import strategies as st

N_COLS = 52


def _make_row(
    *,
    energy: float = -1.0,
    ecc: float = 0.5,
    sim_time: float = 300.0,
    dv_total: float = 0.0,
    g_max: float = 0.0,
    q_max: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Build a single-row final_conditions array with the given values."""
    row = np.zeros((1, N_COLS))
    row[0, 7] = energy
    row[0, 9] = ecc
    row[0, 16] = q_max
    row[0, 17] = g_max
    row[0, 27] = sim_time
    row[0, 41] = dv_total
    return row


class TestCostDeltaVPrimary:
    def test_zero_dv_zero_cost(self) -> None:
        """Captured with zero delta-V and no constraint violations → cost = 0."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_dv_is_primary_cost(self) -> None:
        """For captured trajectory, cost ≈ delta-V when no constraints violated."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=150.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(150.0, abs=1e-6)

    def test_dv_clipped_at_10000(self) -> None:
        """Delta-V above 10000 m/s is clipped."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=50000.0)
        cost = compute_cost(row)
        assert cost <= 10001.0  # 10000 + possible small constraint penalty

    def test_bogus_dv_treated_as_noncapture(self) -> None:
        """dv_total > 1e10 (bogus Fortran value) → non-capture penalty path."""
        row = _make_row(energy=-1.0, ecc=0.5, dv_total=1e30)
        cost = compute_cost(row)
        assert cost > 1e6, f"Bogus dv should trigger non-capture penalty, got {cost}"


class TestCostConstraintPenalties:
    def test_gload_below_limit_no_penalty(self) -> None:
        """G-load at or below limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, g_max=14.0)
        cost = compute_cost(row, g_load_limit=15.0)
        assert cost == pytest.approx(100.0, abs=1e-6)

    def test_gload_above_limit_adds_penalty(self) -> None:
        """G-load above limit adds quadratic normalized penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, g_max=16.5)
        cost_with = compute_cost(row, g_load_limit=15.0, g_load_weight=1000.0)
        cost_without = compute_cost(row, g_load_limit=15.0, g_load_weight=0.0)
        assert cost_with > cost_without

    def test_heat_flux_below_limit_no_penalty(self) -> None:
        """Heat flux at or below limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, q_max=180.0)
        cost = compute_cost(row, heat_flux_limit=200.0)
        assert cost == pytest.approx(100.0, abs=1e-6)

    def test_heat_flux_above_limit_adds_penalty(self) -> None:
        """Heat flux above limit adds quadratic normalized penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, q_max=250.0)
        cost_with = compute_cost(row, heat_flux_limit=200.0, heat_flux_weight=1000.0)
        cost_without = compute_cost(row, heat_flux_limit=200.0, heat_flux_weight=0.0)
        assert cost_with > cost_without

    def test_normalized_exceedance_symmetry(self) -> None:
        """10% g-load exceedance = 10% heat flux exceedance at equal weights."""
        # 10% over g_limit=10 → g_max=11
        row_g = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0, g_max=11.0)
        cost_g = compute_cost(row_g, g_load_limit=10.0, g_load_weight=1000.0, heat_flux_weight=0.0)
        # 10% over q_limit=100 → q_max=110
        row_q = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0, q_max=110.0)
        cost_q = compute_cost(row_q, heat_flux_limit=100.0, heat_flux_weight=1000.0, g_load_weight=0.0)
        assert cost_g == pytest.approx(cost_q, rel=1e-10)

    def test_weight_zero_disables_penalty(self) -> None:
        """Setting weight to 0 disables that constraint penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=50.0, g_max=100.0, q_max=1000.0)
        cost = compute_cost(row, g_load_weight=0.0, heat_flux_weight=0.0)
        assert cost == pytest.approx(50.0, abs=1e-6)


class TestCostHyperbolic:
    def test_hyperbolic_penalized(self) -> None:
        """energy > 0 AND ecc > 1 → Level 0 penalty above 1e6."""
        row = _make_row(energy=5.0, ecc=2.0)
        cost = compute_cost(row)
        assert cost > 1e6

    def test_hyperbolic_higher_than_captured(self) -> None:
        """Hyperbolic always costs more than a well-captured orbit."""
        hyperbolic = _make_row(energy=1.0, ecc=1.5)
        captured = _make_row(energy=-1.0, ecc=0.5, dv_total=500.0)
        assert compute_cost(hyperbolic) > compute_cost(captured)

    def test_parabolic_boundary_classified_as_captured(self) -> None:
        """Energy=0, ecc=1 (strict >) → classified as captured."""
        row = _make_row(energy=0.0, ecc=1.0, dv_total=0.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12)


class TestCostAggregation:
    def test_multi_sim_rms(self) -> None:
        """Stacking identical rows produces the same cost as a single row."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0)
        stacked = np.tile(row, (5, 1))
        assert compute_cost(row) == pytest.approx(compute_cost(stacked), abs=1e-9)


class TestCostProperties:
    @given(
        energy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        ecc=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
        dv_total=st.floats(0.0, 1e4, allow_nan=False, allow_infinity=False),
        g_max=st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False),
        q_max=st.floats(0.0, 1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_cost_always_finite_nonneg(
        self, energy: float, ecc: float, dv_total: float, g_max: float, q_max: float,
    ) -> None:
        """For any finite inputs, compute_cost returns a finite, non-negative value."""
        row = _make_row(energy=energy, ecc=ecc, dv_total=dv_total, g_max=g_max, q_max=q_max)
        cost = compute_cost(row)
        assert np.isfinite(cost), f"cost is not finite: {cost}"
        assert cost >= 0.0, f"cost is negative: {cost}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cost.py -v`
Expected: Multiple failures — current `compute_cost` doesn't accept keyword args and uses orbit errors, not delta-V.

- [ ] **Step 3: Rewrite `compute_cost` in evaluate.py**

Replace lines 229-284 of `src/python/aerocapture/training/evaluate.py` with:

```python
def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    *,
    g_load_limit: float = 15.0,
    heat_flux_limit: float = 200.0,
    g_load_weight: float = 1000.0,
    heat_flux_weight: float = 1000.0,
) -> float:
    """Compute RMS cost from simulation final conditions.

    Uses delta-V as the primary objective with normalized soft constraint
    penalties for g-load and heat flux exceedances.

    Final file columns (0-indexed, 52-column layout):
        7  = orbital energy (MJ/kg), >0 hyperbolic, <0 bound
        9  = eccentricity, >1 hyperbolic
        16 = peak heat flux (kW/m²)
        17 = peak g-load (g)
        27 = total simulation time (s)
        41 = total delta-V to reach target orbit (m/s)

    Cost hierarchy:
        Non-capture (hyperbolic or bogus ΔV): 1e6 + 1e3 * |energy| - 0.1 * sim_time
        Captured: ΔV + w_g * max((g-g_lim)/g_lim, 0)² + w_q * max((q-q_lim)/q_lim, 0)²

    Returns:
        RMS cost value. Lower is better.
    """
    energy = final_conditions[:, 7]  # MJ/kg
    ecc = final_conditions[:, 9]  # dimensionless
    sim_time = final_conditions[:, 27]  # s
    dv_total = final_conditions[:, 41]  # m/s
    g_max = final_conditions[:, 17]  # g
    q_max = final_conditions[:, 16]  # kW/m²

    hyperbolic = (ecc > 1.0) | (energy > 0)

    costs = np.zeros(len(final_conditions))

    # Non-capture OR bogus delta-V: energy-based penalty
    bad = hyperbolic | (dv_total > 1e10)
    costs[bad] = 1e6 + 1e3 * np.abs(energy[bad]) - 0.1 * sim_time[bad]

    # Captured with valid delta-V
    ok = ~bad
    dv = np.clip(dv_total[ok], 0, 1e4)
    g_penalty = g_load_weight * np.maximum((g_max[ok] - g_load_limit) / g_load_limit, 0) ** 2
    q_penalty = heat_flux_weight * np.maximum((q_max[ok] - heat_flux_limit) / heat_flux_limit, 0) ** 2
    costs[ok] = dv + g_penalty + q_penalty

    return float(np.sqrt(np.mean(costs**2)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cost.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py tests/test_cost.py
git commit -m "feat: rewrite cost function — delta-V primary with normalized constraint penalties"
```

---

## Chunk 3: Thread Cost Parameters Through GA Pipeline

### Task 5: Add `cost_kwargs` to `evaluate_chromosome`

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:506-572`

- [ ] **Step 1: Update `evaluate_chromosome` signature and forwarding**

Change the function signature at line 506-512 to add `cost_kwargs`:

```python
def evaluate_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | Path | None = None,
    mc_seed: int | None = None,
    cost_kwargs: dict[str, float] | None = None,
) -> tuple[float, npt.NDArray[np.float64] | None]:
```

And change line 571 from:
```python
    cost = compute_cost(final)
```
To:
```python
    cost = compute_cost(final, **(cost_kwargs or {}))
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/test_cost.py tests/ -k "cost or chromosome" -v`
Expected: All pass — new parameter has a default of `None` so no callers break.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "feat: add cost_kwargs parameter to evaluate_chromosome"
```

### Task 6: Parse `[cost_function]` from TOML in `train.py`

**Files:**
- Modify: `src/python/aerocapture/training/train.py:247-255` (TOML parsing section)

- [ ] **Step 1: Add cost function config parsing and consolidate TOML loading**

Currently `train.py` has *two separate* TOML loading blocks — one inside `if config.ga.rotate_seeds:` (lines 243-255) and another inside `if config.ga.adaptive_seeds:` (lines 259-269). Both independently call `tomllib.load()`. Refactor into a single shared TOML load that runs unconditionally when a TOML config exists, before both conditional blocks.

Replace lines 241-269 with:

```python
    # Load TOML config once (used for cost function params, seed rotation, adaptive seeds)
    import tomllib

    _toml: dict = {}
    cost_kwargs: dict[str, float] = {}
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        with open(toml_path, "rb") as f:
            _toml = tomllib.load(f)

        # Parse cost function config (with defaults)
        cost_cfg = _toml.get("cost_function", {})
        cost_kwargs = {
            "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
            "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
            "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
            "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
        }

    # Read base MC seed from TOML for seed rotation
    base_mc_seed: int | None = None
    if config.ga.rotate_seeds:
        if not config.sim.toml_config:
            msg = "rotate_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        base_mc_seed = _toml.get("monte_carlo", {}).get("seed")
        if base_mc_seed is None:
            msg = "rotate_seeds requires [monte_carlo].seed in the TOML config"
            raise ValueError(msg)

    # Initialize adaptive seed pool
    seed_pool: SeedPool | None = None
    if config.ga.adaptive_seeds:
        if not config.sim.toml_config:
            msg = "adaptive_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        pool_base_seed = _toml.get("monte_carlo", {}).get("seed")
        if pool_base_seed is None:
            # ... (rest of existing adaptive_seeds init unchanged)
```

This eliminates the duplicate TOML reads and makes `_toml` and `cost_kwargs` available in scope for all subsequent code including closures.

- [ ] **Step 2: Thread `cost_kwargs` to all `evaluate_chromosome` calls in train.py**

Update every `evaluate_chromosome(...)` call to include `cost_kwargs=cost_kwargs`:

- Line ~380 (`_pool_evaluator` closure — the scalar fallback for adaptive seed pool): `cost, _ = evaluate_chromosome(chrom, base_network, config, cwd=cwd, mc_seed=mc_seed, cost_kwargs=cost_kwargs)`. This closure is called by `seed_pool.evaluate_population()` — the seed pool itself doesn't need changes since it delegates to this closure.
- Line ~480 (offspring evaluation in non-adaptive path): add `cost_kwargs=cost_kwargs`
- Line ~492 (parent re-evaluation when rotating seeds): add `cost_kwargs=cost_kwargs`

- [ ] **Step 3: Thread `cost_kwargs` to batch evaluator's `compute_cost` calls**

In the `_batch_eval` closure (line 417), update:
```python
costs: npt.NDArray[np.float64] = np.array([compute_cost(final_records[i : i + 1], **cost_kwargs) for i in range(final_records.shape[0])])
```

The closure captures `cost_kwargs` from the enclosing scope. Since `_make_batch_eval` is a factory function, pass `cost_kwargs` as an argument:

Update `_make_batch_eval` signature (line 388):
```python
    def _make_batch_eval(
        base_net: npt.NDArray[np.float64],
        cfg: TrainingConfig,
        working_dir: str | Path | None,
        cost_kw: dict[str, float],
    ) -> Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]]:
```

And in the inner function, use `cost_kw`:
```python
            costs: npt.NDArray[np.float64] = np.array([compute_cost(final_records[i : i + 1], **cost_kw) for i in range(final_records.shape[0])])
```

Update the call at line 422:
```python
    _batch_evaluator = _make_batch_eval(base_network, config, cwd, cost_kwargs)
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: parse [cost_function] from TOML and thread cost_kwargs through GA loop"
```

### Task 7: Thread `cost_kwargs` through `population.py` and `local_search.py`

**Files:**
- Modify: `src/python/aerocapture/training/population.py:159`
- Modify: `src/python/aerocapture/training/local_search.py:17-49,62`

- [ ] **Step 1: Update `population.py`**

Find the function that contains line 159's `evaluate_chromosome` call (likely `initialize_population` or similar). Add `cost_kwargs: dict[str, float] | None = None` to its signature, then forward:

```python
        cost, _ = evaluate_chromosome(candidates[i], base_network, config, cwd=cwd, cost_kwargs=cost_kwargs)
```

- [ ] **Step 2: Update `local_search.py`**

Add `cost_kwargs: dict[str, float] | None = None` to `improve_chromosome` signature (line 17-24):

```python
def improve_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    mode: int = 0,
    cwd: str | Path | None = None,
    rng: np.random.Generator | None = None,
    cost_kwargs: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.int8], float, float]:
```

Update calls at lines 49 and 62:
```python
    current_cost, _ = evaluate_chromosome(current, base_network, config, cwd=cwd, cost_kwargs=cost_kwargs)
    # ...
            new_cost, _ = evaluate_chromosome(current, base_network, config, cwd=cwd, cost_kwargs=cost_kwargs)
```

- [ ] **Step 3: Update callers in train.py**

Search `train.py` for calls to `initialize_population` and `improve_chromosome`, and add `cost_kwargs=cost_kwargs` to them.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/population.py src/python/aerocapture/training/local_search.py src/python/aerocapture/training/train.py
git commit -m "feat: thread cost_kwargs through population init and local search"
```

### Task 8: Thread `cost_kwargs` through `compare_guidance.py`

**Files:**
- Modify: `src/python/aerocapture/training/compare_guidance.py:143`

- [ ] **Step 1: Update `compute_cost` call**

The function containing line 143 needs to accept and forward cost function parameters. Add `cost_kwargs: dict[str, float] | None = None` to its signature and update:

```python
    "cost": compute_cost(final, **(cost_kwargs or {})),
```

Also update the CLI entry point to parse `[cost_function]` from the `--base-toml` if present, and pass it through.

- [ ] **Step 2: Commit**

```bash
git add src/python/aerocapture/training/compare_guidance.py
git commit -m "feat: thread cost_kwargs through guidance comparison"
```

---

## Chunk 4: TOML Config + Linting + Final Verification

### Task 9: Add `[cost_function]` section to training TOMLs

**Files:**
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`
- Modify: `configs/training/msr_aller_eqglide_train.toml`
- Modify: `configs/training/msr_aller_energy_controller_train.toml`
- Modify: `configs/training/msr_aller_pred_guid_train.toml`
- Modify: `configs/training/msr_aller_fnpag_train.toml`
- Modify: `configs/training/msr_aller_ftc_train.toml`

- [ ] **Step 1: Add `[cost_function]` to all 6 training TOMLs**

Append to each file:

```toml

[cost_function]
g_load_limit = 15.0          # g (Earth g's)
heat_flux_limit = 200.0      # kW/m²
g_load_weight = 1000.0       # penalty weight on normalized squared exceedance
heat_flux_weight = 1000.0    # penalty weight on normalized squared exceedance
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/*.toml
git commit -m "config: add [cost_function] section to all training TOMLs"
```

### Task 10: Lint, type-check, and full test suite

- [ ] **Step 1: Run Python linting and type checks**

Run: `./lint_code.sh`
Expected: No errors from ruff or mypy. Fix any issues introduced by the changes.

- [ ] **Step 2: Run Rust checks**

Run: `./check_all.sh`
Expected: All Rust tests pass, clippy clean, fmt clean, release build succeeds.

- [ ] **Step 3: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Rebuild PyO3 bindings and run PyO3 tests**

Run:
```bash
cd src/rust/aerocapture-py && uv run maturin develop --release && cd ../../..
uv run pytest tests/test_pyo3.py -v
```
Expected: PyO3 bindings compile and tests pass — the updated Rust code now populates columns 16-24.

- [ ] **Step 5: Fix any issues found, then commit**

```bash
git add -u
git commit -m "fix: lint and type-check fixes"
```

(Skip this commit if no fixes were needed.)

### Task 11: Smart commit — sync docs and commit the whole branch

- [ ] **Step 1: Invoke `/smart-commit` taking the whole git branch into account**

Use the `smart-commit` skill, telling it to review all commits on the `feature/cost-function-rework` branch (not just staged changes) and update CLAUDE.md / README.md accordingly before the final commit.
