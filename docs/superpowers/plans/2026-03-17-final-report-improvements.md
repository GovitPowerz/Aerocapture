# Final Report Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Rust trajectory output (8→12 cols), expose dispersion draws via PyO3, and overhaul the final evaluation report with corridor plots, dispersion correlations, proper entry/exit labels, and a performance summary table.

**Architecture:** Rust data model changes (trajectory cols, dispersions array, API plumbing) flow through PyO3 bindings into a rewritten Python `final_report.py`. The report grows from 8 panels to ~35+ panels (8 distribution/scatter + 1 table + 3 corridor + ~24 dispersion correlations).

**Tech Stack:** Rust (nalgebra, rayon, serde), PyO3/maturin, Python (numpy, plotly, scipy.stats)

**Spec:** `docs/superpowers/specs/2026-03-17-final-report-improvements-design.md`

---

## Task 1: Add `to_array()` on `DispersionDraw`

**Files:**
- Modify: `src/rust/src/data/dispersions.rs:330-370`
- Test: inline `#[cfg(test)]` in same file

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)]` module at bottom of `dispersions.rs`:

```rust
#[test]
fn dispersion_draw_to_array_roundtrip() {
    let draw = DispersionDraw {
        altitude: 1.0,
        longitude: 2.0,
        latitude: 3.0,
        velocity: 4.0,
        flight_path: 5.0,
        azimuth: 6.0,
        density: 7.0,
        drag_coeff: 8.0,
        lift_coeff: 9.0,
        incidence: 10.0,
        nav_altitude: 11.0,
        nav_longitude: 12.0,
        nav_latitude: 13.0,
        nav_velocity: 14.0,
        nav_flight_path: 15.0,
        nav_azimuth: 16.0,
        nav_drag_accel: 17.0,
        mass: 18.0,
        ref_area: 19.0,
        max_bank_rate: 20.0,
        pilot_tau: 21.0,
        pilot_damping: 22.0,
        pilot_frequency: 23.0,
        filter_gain: 24.0,
    };
    let arr = draw.to_array();
    assert_eq!(arr.len(), 24);
    for i in 0..24 {
        assert_eq!(arr[i], (i + 1) as f64);
    }
}

#[test]
fn dispersion_draw_default_to_array_all_zeros() {
    let arr = DispersionDraw::default().to_array();
    assert_eq!(arr.len(), 24);
    assert!(arr.iter().all(|&v| v == 0.0));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/rust && cargo test dispersion_draw_to_array -- --nocapture`
Expected: FAIL — `to_array` method not found.

- [ ] **Step 3: Implement `to_array()` and compile-time assertion**

Add to `DispersionDraw` impl block (or create one if none exists) in `dispersions.rs`:

```rust
/// Number of fields in DispersionDraw — keep in sync with to_array().
pub const DISPERSION_DRAW_LEN: usize = 24;

impl DispersionDraw {
    /// Serialize all fields to a flat array in struct field order.
    pub fn to_array(&self) -> [f64; DISPERSION_DRAW_LEN] {
        [
            self.altitude,
            self.longitude,
            self.latitude,
            self.velocity,
            self.flight_path,
            self.azimuth,
            self.density,
            self.drag_coeff,
            self.lift_coeff,
            self.incidence,
            self.nav_altitude,
            self.nav_longitude,
            self.nav_latitude,
            self.nav_velocity,
            self.nav_flight_path,
            self.nav_azimuth,
            self.nav_drag_accel,
            self.mass,
            self.ref_area,
            self.max_bank_rate,
            self.pilot_tau,
            self.pilot_damping,
            self.pilot_frequency,
            self.filter_gain,
        ]
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/rust && cargo test dispersion_draw_to_array -- --nocapture`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/dispersions.rs
git commit -m "feat: add to_array() on DispersionDraw for flat serialization"
```

---

## Task 2: Expand `RunOutput` data model

**Files:**
- Modify: `src/rust/src/lib.rs:10-18`
- Modify: `src/rust/src/simulation/runner.rs:183-198` (run_for_api)
- Modify: `src/rust/src/simulation/runner.rs:80-159` (run_core)
- Modify: `src/rust/src/simulation/runner.rs:644-712` (build_photo_values)

- [ ] **Step 1: Update `RunOutput` struct in `lib.rs`**

Change the struct at lines 10-18:

```rust
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Per-timestep state: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, flux, time, energy_MJkg, pdyn_kPa, bank_deg, incl_deg]
    pub trajectory: Vec<[f64; 12]>,
    /// Full 52-column final record (same layout as CSV file output)
    pub final_record: [f64; 52],
    /// True if orbit is bound (ecc < 1 && energy < 0)
    pub captured: bool,
    /// Dispersion draws for this simulation (24 fields from DispersionDraw::to_array)
    pub dispersions: [f64; data::dispersions::DISPERSION_DRAW_LEN],
}
```

- [ ] **Step 2: Add `include_trajectories` parameter to `run_core()`**

Change signature at line 80 from:
```rust
fn run_core(config: &SimInput, data: &SimData, write_photo: bool) -> Result<Vec<SimResult>, SimError>
```
to:
```rust
fn run_core(config: &SimInput, data: &SimData, write_photo: bool, include_trajectories: bool) -> Result<Vec<SimResult>, SimError>
```

In the MC parallel section (lines 129-136), change the `do_photo` logic:
```rust
let results: Vec<SimResult> = run_states
    .par_iter()
    .enumerate()
    .map(|(idx, (run_state, disp_array))| {
        let do_photo = (write_photo && idx as i32 == photo_sim_idx) || include_trajectories;
        let mut result = run_single(config, data, run_state, idx as i32, do_photo)?;
        result.dispersions = *disp_array;
        Ok(result)
    })
    .collect::<Result<Vec<_>, _>>()?;
```

In the single-sim section (lines 147-158), update similarly:
```rust
let (run_state, disp_array) = &run_states[0];
// ... existing single-sim code ...
let mut result = run_single(config, data, run_state, 0, write_photo)?;
result.dispersions = *disp_array;
Ok(vec![result])
```

- [ ] **Step 3: Thread `DispersionDraw` to output**

In `run_core()`, change the `run_states` construction (lines 104-112) to store draws alongside:

```rust
let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = (0..n_sims)
    .map(|sim_idx| {
        let draw = if let Some(ref d) = draws {
            &d[sim_idx as usize]
        } else {
            &crate::data::dispersions::DispersionDraw::default()
        };
        (init::init_run_from_draw(data, draw), draw.to_array())
    })
    .collect();
```

The internal `SimResult` struct needs a new field with a default of all zeros (populated by the caller after `run_single()` returns):

```rust
struct SimResult {
    sim_idx: i32,
    final_line: [f64; 52],
    photo_lines: Vec<[f64; 24]>,
    dispersions: [f64; DISPERSION_DRAW_LEN],
}
```

In `run_single()`, initialize `dispersions` to `[0.0; DISPERSION_DRAW_LEN]` in the returned `SimResult`. The actual values are set by the caller (in `run_core()`) from the `disp_array` stored alongside the `RunState`. This avoids changing `run_single()`'s signature.

- [ ] **Step 4: Update `run_for_api()` to accept and forward `include_trajectories`**

Change signature from:
```rust
pub fn run_for_api(config: &SimInput, data: &SimData) -> Result<Vec<crate::RunOutput>, SimError>
```
to:
```rust
pub fn run_for_api(config: &SimInput, data: &SimData, include_trajectories: bool) -> Result<Vec<crate::RunOutput>, SimError>
```

Update the body to forward the flag and build 12-column trajectories:

```rust
pub fn run_for_api(config: &SimInput, data: &SimData, include_trajectories: bool) -> Result<Vec<crate::RunOutput>, SimError> {
    let results = run_core(config, data, false, include_trajectories)?;

    Ok(results
        .into_iter()
        .map(|r| {
            let energy = r.final_line[7];
            let ecc = r.final_line[9];
            let trajectory = if include_trajectories {
                r.photo_lines
                    .iter()
                    .map(|p| [
                        p[1],                // [0] alt_km
                        p[2],                // [1] lon_deg
                        p[3],                // [2] lat_deg
                        p[4],                // [3] vel_m_s
                        p[5],                // [4] fpa_deg
                        p[6],                // [5] heading_deg
                        0.0,                 // [6] heat flux — not available in photo row, placeholder
                        p[0],                // [7] time_s
                        p[18] / 1e6,         // [8] energy J/kg → MJ/kg
                        p[19] / 1e3,         // [9] pdyn Pa → kPa
                        p[14],               // [10] bank_angle deg
                        p[9],                // [11] inclination deg
                    ])
                    .collect()
            } else {
                Vec::new()
            };
            crate::RunOutput {
                trajectory,
                final_record: r.final_line,
                captured: ecc < 1.0 && energy < 0.0,
                dispersions: r.dispersions,
            }
        })
        .collect())
}
```

**Trajectory column mapping notes:**
- The photo row has 24 columns (see `build_photo_values()` at runner.rs:686-711).
- Photo indices: [0]=time, [1]=alt_km, [2]=lon_deg, [3]=lat_deg, [4]=vel_m_s, [5]=fpa_deg, [6]=heading_deg, [9]=incl_deg, [14]=bank_deg, [18]=energy_J_kg, [19]=pdyn_Pa.
- Heat flux (trajectory col 6) is set to 0.0 as placeholder. The original `sim.state[6]` (cumulative integrated heat flux) is not in the photo row. This is acceptable since the corridor plots only use cols 8-11 and the heat flux column was already rarely used. Update the `RunOutput` doc comment to note this.
- The key new columns (8-11) are: `p[18]/1e6` (MJ/kg), `p[19]/1e3` (kPa), `p[14]` (deg), `p[9]` (deg).

- [ ] **Step 5: Update `run()` call site**

In `run()` at line 174, update the call:
```rust
let results = run_core(config, data, true, false)?;
```

- [ ] **Step 6: Document `final_record` slots in runner.rs**

Add a comment block above the `final_record` population section (around line 594):

```rust
// final_record layout (52 slots):
//   0  altitude (km)           16 max heat flux (kW/m²)     32-36 UNUSED
//   1  longitude (deg)         17 max g-load (g)             37 dv1 (m/s)
//   2  latitude (deg)          18 max pdyn (kPa)             38 dv2 (m/s)
//   3  velocity (m/s)          19 alt at max flux (km)       39 dv3 (m/s)
//   4  FPA (deg)               20 alt at max load (km)       40 dv1+dv2 (m/s)
//   5  heading (deg)           21 alt at max pdyn (km)       41 dv total (m/s)
//   6  radial velocity (m/s)   22 time at max flux (s)       42-44 UNUSED
//   7  energy (MJ/kg)          23 time at max load (s)       45 bank consumption (deg)
//   8  SMA (km)                24 time at max pdyn (s)       46-47 UNUSED
//   9  eccentricity            25 bounce alt (km)            48 n_reversals
//  10  inclination (deg)       26 bounce time (s)            49-51 UNUSED
//  11  RAAN (deg)              27 sim time (s)
//  12  arg periapsis (deg)     28 cumulative flux (MJ/m²)
//  13  true anomaly (deg)      29 periapsis error (km)
//  14  periapsis alt (km)      30 apoapsis error (km)
//  15  apoapsis alt (km)       31 final phase
```

- [ ] **Step 7: Run Rust tests**

Run: `cd src/rust && cargo test`
Expected: All existing tests pass. Some may need minor fixes for the new `RunOutput` field (add `dispersions: [0.0; DISPERSION_DRAW_LEN]` where `RunOutput` is constructed in tests).

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/lib.rs src/rust/src/simulation/runner.rs
git commit -m "feat: expand RunOutput with 12-col trajectory, dispersions array, include_trajectories plumbing"
```

---

## Task 3: Update PyO3 bindings

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs:98-155`
- Modify: `src/rust/aerocapture-py/src/results.rs:12-163`
- Modify: `src/rust/aerocapture-py/src/batch.rs:79`

- [ ] **Step 1: Forward `include_trajectories` in `run_mc()`**

In `src/rust/aerocapture-py/src/lib.rs`, update the `run_mc()` function body. Find the `run_for_api` call (around line 110) and change:

```rust
let outputs = aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data)
```
to:
```rust
let outputs = aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data, include_trajectories)
```

- [ ] **Step 2: Forward `include_trajectories` in `run()` (single-sim)**

In `run()` function (around line 67), update:
```rust
let outputs = aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data, false)
```

- [ ] **Step 3: Update `batch.rs`**

In `src/rust/aerocapture-py/src/batch.rs`, update the `run_batch()` function signature to accept `include_trajectories: bool` and forward it:

```rust
pub fn run_batch(
    toml_path: &Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
    include_trajectories: bool,
) -> Result<Vec<RunOutput>, String> {
```

And update the `run_for_api` call (around line 79):
```rust
let outputs = aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data, include_trajectories)
```

Also update the PyO3 `run_batch()` wrapper in `lib.rs` to forward the parameter to `batch::run_batch()`.

- [ ] **Step 4: Add `.dispersions` getter on `SimResult`**

In `src/rust/aerocapture-py/src/results.rs`, add to the `SimResult` pymethods:

```rust
/// Dispersion draws as a 1D NumPy array (24 elements).
#[getter]
fn dispersions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, &self.output.dispersions)
}
```

- [ ] **Step 5: Add `.dispersions` getter on `BatchResults`**

Add to `BatchResults` pymethods:

```rust
/// Dispersion draws as an (N, 24) NumPy array — always populated.
#[getter]
fn dispersions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = self.outputs.iter().map(|o| o.dispersions.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).unwrap()
}
```

- [ ] **Step 6: Update trajectory getter column count in docstring/comment**

In `SimResult.trajectory` and `BatchResults.trajectories`, update any comments that say `(N, 8)` to `(N, 12)`.

- [ ] **Step 7: Build and test**

Run: `cd src/rust/aerocapture-py && maturin develop --release`
Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_pyo3.py -v`
Expected: All existing PyO3 tests pass (they don't assert on trajectory column count since trajectories were empty).

- [ ] **Step 8: Commit**

```bash
git add src/rust/aerocapture-py/
git commit -m "feat: expose dispersions array and forward include_trajectories in PyO3 bindings"
```

---

## Task 4: Rewrite `run_final_evaluation()` to return rich data

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:79-123`
- Test: `tests/test_final_report.py`

- [ ] **Step 1: Write failing tests for new return type**

Add to `tests/test_final_report.py`:

```python
from aerocapture.training.final_report import FinalEvalData

def test_final_eval_data_is_namedtuple():
    """FinalEvalData has the expected fields."""
    data = FinalEvalData(
        final_array=np.zeros((10, 52)),
        trajectories=[np.zeros((100, 12)) for _ in range(10)],
        dispersions=np.zeros((10, 24)),
    )
    assert data.final_array.shape == (10, 52)
    assert len(data.trajectories) == 10
    assert data.dispersions.shape == (10, 24)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_final_report.py::test_final_eval_data_is_namedtuple -v`
Expected: FAIL — `FinalEvalData` not found.

- [ ] **Step 3: Define `FinalEvalData` and update `run_final_evaluation()`**

At the top of `final_report.py` (after imports), add:

```python
from collections import namedtuple

FinalEvalData = namedtuple("FinalEvalData", ["final_array", "trajectories", "dispersions"])
```

Update `run_final_evaluation()` return type and body:

```python
def run_final_evaluation(
    cfg: TrainingConfig,
    n_sims: int = 1000,
    seed: int | None = None,
    cwd: Path | None = None,
) -> FinalEvalData | None:
    """Run large-MC re-evaluation of best solution.

    Returns FinalEvalData(final_array, trajectories, dispersions) or None on failure.
    """
    from aerocapture.training.evaluate import _HAS_PYO3, _aero_rs

    if cfg.sim.toml_config is None:
        return None

    cwd_path = Path(cwd) if cwd else Path(".")
    base_toml = cwd_path / cfg.sim.toml_config

    patched_toml = _patch_toml_for_final_eval(base_toml, n_sims, 0 if seed is None else seed)
    orig_toml = cfg.sim.toml_config
    try:
        if _HAS_PYO3:
            assert _aero_rs is not None
            toml_path = str(patched_toml.resolve())
            results = _aero_rs.run_mc(toml_path=toml_path, include_trajectories=True)
            arr: npt.NDArray[np.float64] = results.final_records
            trajectories: list[npt.NDArray[np.float64]] = results.trajectories
            dispersions: npt.NDArray[np.float64] = results.dispersions
            return FinalEvalData(final_array=arr, trajectories=trajectories, dispersions=dispersions)
        else:
            from aerocapture.training.evaluate import run_simulation
            cfg.sim.toml_config = str(patched_toml)
            arr = run_simulation(cfg, cwd=cwd)
            if arr is None:
                return None
            return FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    finally:
        cfg.sim.toml_config = orig_toml
        patched_toml.unlink(missing_ok=True)
```

- [ ] **Step 4: Update all callers of `run_final_evaluation()`**

In `main()` at the bottom of `final_report.py` (around line 397), update:

```python
eval_data = run_final_evaluation(cfg, n_sims=args.n_sims, seed=args.seed)
if eval_data is None:
    print("ERROR: Simulation failed")
    sys.exit(1)

output_path = scheme_dir / "final_report.html"
generate_final_report(eval_data, scheme, target_incl, output_path)
```

Also update `generate_final_report()` signature to accept `FinalEvalData` instead of bare array (done in Task 5).

- [ ] **Step 5: Update existing tests**

Update `tests/test_final_report.py` helper functions and tests that call `generate_final_report()` to pass `FinalEvalData` instead of bare arrays. For backward compat in tests that don't need trajectories/dispersions, pass `None`:

```python
eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
generate_final_report(eval_data, "test_scheme", 50.0, output_path)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/final_report.py tests/test_final_report.py
git commit -m "feat: return FinalEvalData from run_final_evaluation with trajectories and dispersions"
```

---

## Task 5: Rewrite `generate_final_report()` — panels 1-5

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py:125-344`
- Test: `tests/test_final_report.py`

This task rewrites the report layout for rows 1-5 (distributions, entry/exit conditions, performance table). Corridor and dispersion plots are separate tasks.

- [ ] **Step 1: Add column constants for new metrics**

Add to the column constants section at top of `final_report.py`:

```python
_COL_MAX_HEAT_FLUX = 16
_COL_MAX_G_LOAD = 17
_COL_BANK_CONSUMPTION = 45
```

- [ ] **Step 2: Rewrite `generate_final_report()` signature and layout**

Change signature to accept `FinalEvalData`:

```python
def generate_final_report(
    eval_data: FinalEvalData,
    scheme: str,
    target_inclination: float,
    output_path: Path,
    ref_trajectory_path: Path | None = None,
) -> Path:
```

Extract fields:
```python
final_array = eval_data.final_array
trajectories = eval_data.trajectories
dispersions_array = eval_data.dispersions
```

Set up the grid. The total number of rows depends on whether corridors and dispersions are available:

```python
# Base panels: rows 1-5 (5 rows × 2 cols = 10 panel slots)
# Corridor panels: rows 6-7 (2 rows × 2 cols, but row 7 right is empty)
# Dispersion panels: row 8+ (ceil(24/4) = 6 rows × 4 cols)
has_trajectories = trajectories is not None and len(trajectories) > 0
has_dispersions = dispersions_array is not None

n_base_rows = 5
n_corridor_rows = 2 if has_trajectories else 0
n_disp_cols = 4
n_disp_fields = dispersions_array.shape[1] if has_dispersions else 0
n_disp_rows = math.ceil(n_disp_fields / n_disp_cols) if n_disp_fields > 0 else 0
n_total_rows = n_base_rows + n_corridor_rows + n_disp_rows
```

Build the subplot specs dynamically. Row 5 is a table spanning 2 columns. Dispersion rows use 4 columns (separate `make_subplots` call or nested figure).

**Simplification:** Use two separate figures if the 4-column dispersion grid is hard to merge with the 2-column main layout. Concatenate the HTML at the end. Alternatively, use a single 4-column grid where main panels span 2 columns each. Prefer the single-figure approach for scroll continuity.

- [ ] **Step 3: Implement rows 1-3 (distributions + DV vs error)**

Rows 1-3 are mostly unchanged from current code, just reorganized:
- Row 1: Total Delta-V histogram+CDF (left), Individual burns overlaid (right)
- Row 2: Apoapsis error histogram+CDF (left), Periapsis error histogram+CDF (right)
- Row 3: Inclination error histogram+CDF (left), Delta-V vs Orbital Error scatter (right, moved from row 4)

The existing `_add_hist_cdf()` helper is reused.

- [ ] **Step 4: Implement row 4 (entry + exit conditions)**

**Left panel — Entry Conditions:** Use first row of each trajectory for actual dispersed entry state:

```python
if has_trajectories:
    entry_vel = np.array([t[0, 3] for t in trajectories])  # vel_m_s from first timestep
    entry_fpa = np.array([t[0, 4] for t in trajectories])  # fpa_deg from first timestep
else:
    # Fallback: show nothing or use dispersion deltas
    entry_vel = None
```

Plot captured in green, hyperbolic in red.

**Right panel — Exit Conditions:** Current "Entry Conditions" panel, relabeled. Uses `final_record[3]` (exit velocity) and `final_record[4]` (exit FPA), with marker size proportional to delta-V.

- [ ] **Step 5: Implement row 5 (performance summary table)**

Replace current `_add_summary_table()` with new version:

```python
def _add_performance_table(
    fig: object,
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
    target_inclination: float,
    row: int,
    col: int,
) -> None:
    import plotly.graph_objects as go

    n_total = len(final_array)
    n_captured = int(captured.sum())

    header = ["Parameter", "Mean", "Std", "Min", "p5", "p25", "p50", "p75", "p95", "Max"]
    rows: list[list[str]] = []

    if n_captured > 0:
        cap = final_array[captured]
        metrics = {
            "Max g-load (g)": cap[:, _COL_MAX_G_LOAD],
            "Max heat flux (kW/m²)": cap[:, _COL_MAX_HEAT_FLUX],
            "Bank angle consumption (deg)": cap[:, _COL_BANK_CONSUMPTION],
            "Apoapsis error (km)": cap[:, _COL_APO_ERR],
            "Periapsis error (km)": cap[:, _COL_PERI_ERR],
            "Inclination error (deg)": cap[:, _COL_INCL] - target_inclination,
            "Correction cost ΔV (m/s)": cap[:, _COL_DV_TOTAL],
        }
        for name, data in metrics.items():
            pcts = np.percentile(data, _PERCENTILES)
            rows.append([
                name,
                f"{data.mean():.2f}",
                f"{data.std():.2f}",
                f"{data.min():.2f}",
                *[f"{p:.2f}" for p in pcts],
                f"{data.max():.2f}",
            ])

    rows.insert(0, [
        f"Capture rate: {n_captured}/{n_total} ({n_captured / n_total * 100:.1f}%)",
        "", "", "", "", "", "", "", "", "",
    ])

    cells_transposed = list(zip(*rows, strict=False)) if rows else [[] for _ in header]
    fig.add_trace(
        go.Table(
            header={"values": header, "fill_color": _COLOR_PRIMARY, "font_color": "white", "align": "center"},
            cells={"values": cells_transposed, "align": "center"},
        ),
        row=row,
        col=col,
    )
```

- [ ] **Step 6: Write tests for new panels**

Add tests in `tests/test_final_report.py`:

```python
def test_report_has_exit_conditions_label(tmp_path):
    arr = _make_captured_array()
    eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    assert "Exit Conditions" in html
    assert "Entry Conditions" in html or "Entry" in html  # entry panel present

def test_performance_table_has_min_max(tmp_path):
    arr = _make_captured_array()
    eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    assert "Max g-load" in html
    assert "Max heat flux" in html
    assert "Bank angle consumption" in html
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/final_report.py tests/test_final_report.py
git commit -m "feat: rewrite final report rows 1-5 with entry/exit fix and performance table"
```

---

## Task 6: Add energy corridor panels (rows 6-7)

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py`
- Test: `tests/test_final_report.py`

- [ ] **Step 1: Add reference trajectory loader**

Add a helper function to `final_report.py`:

```python
def _load_reference_trajectory(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load reference trajectory .dat file (7 whitespace-separated columns).

    Returns dict with keys: energy_MJkg, pdyn_kPa, inclination_deg, bank_deg.
    """
    try:
        data = np.loadtxt(path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 7:
        return None
    return {
        "energy_MJkg": data[:, 0],           # col 0: energy (MJ/kg)
        "pdyn_kPa": data[:, 1] / 1e3,        # col 1: pdyn (Pa → kPa)
        "inclination_deg": np.degrees(data[:, 4]),  # col 4: inclination (rad → deg)
        "bank_deg": np.degrees(np.arccos(np.clip(data[:, 6], -1, 1))),  # col 6: cos(bank) → deg
    }
```

- [ ] **Step 2: Add `_read_ref_trajectory_path()` helper**

```python
def _read_ref_trajectory_path(toml_path: Path) -> Path | None:
    """Read reference trajectory file path from TOML [data] section."""
    from aerocapture.training.toml_utils import load_toml_with_bases
    data = load_toml_with_bases(toml_path)
    ref_path = data.get("data", {}).get("reference_trajectory")
    if ref_path is None:
        return None
    return Path(ref_path)
```

- [ ] **Step 3: Implement corridor plotting helper**

```python
def _add_corridor_panel(
    fig: object,
    trajectories: list[npt.NDArray[np.float64]],
    captured: npt.NDArray[np.bool_],
    y_col: int,
    y_label: str,
    ref_x: npt.NDArray[np.float64] | None,
    ref_y: npt.NDArray[np.float64] | None,
    row: int,
    col: int,
    n_sims: int,
) -> None:
    """Add energy corridor panel with spaghetti + envelope + reference."""
    import plotly.graph_objects as go

    opacity = max(0.02, min(0.15, 10.0 / n_sims))

    # Spaghetti traces — concatenate all trajectories into TWO traces (captured + hyperbolic)
    # with None separators between segments. This avoids 1000+ individual Plotly traces
    # which would produce multi-hundred-MB HTML and freeze the browser.
    for is_captured, color_base, trace_name in [
        (True, "33, 150, 243", "Captured"),
        (False, "244, 67, 54", "Hyperbolic"),
    ]:
        mask = captured if is_captured else ~captured
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        segments_x: list[float | None] = []
        segments_y: list[float | None] = []
        for i in indices:
            traj = trajectories[i]
            segments_x.extend(traj[:, 8].tolist())
            segments_y.extend(traj[:, y_col].tolist())
            segments_x.append(None)  # segment separator
            segments_y.append(None)
        fig.add_trace(
            go.Scattergl(
                x=segments_x, y=segments_y, mode="lines",
                line={"color": f"rgba({color_base}, {opacity:.3f})", "width": 0.5},
                showlegend=False,
            ),
            row=row, col=col,
        )

    # Envelope — bin captured trajectories by energy and take min/max
    cap_indices = np.where(captured)[0]
    if len(cap_indices) > 10:
        all_e = np.concatenate([trajectories[i][:, 8] for i in cap_indices])
        all_y = np.concatenate([trajectories[i][:, y_col] for i in cap_indices])
        e_min, e_max = np.nanmin(all_e), np.nanmax(all_e)
        n_bins = 100
        edges = np.linspace(e_min, e_max, n_bins + 1)
        bin_idx = np.digitize(all_e, edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        y_lo = np.full(n_bins, np.nan)
        y_hi = np.full(n_bins, np.nan)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                y_lo[b] = np.nanmin(all_y[mask])
                y_hi[b] = np.nanmax(all_y[mask])
        e_centers = (edges[:-1] + edges[1:]) / 2
        valid = ~np.isnan(y_lo)
        fig.add_trace(
            go.Scatter(x=e_centers[valid], y=y_hi[valid], mode="lines", line={"color": "rgba(33, 150, 243, 0.3)", "width": 0}, showlegend=False),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=e_centers[valid], y=y_lo[valid], mode="lines", line={"color": "rgba(33, 150, 243, 0.3)", "width": 0}, fill="tonexty", fillcolor="rgba(33, 150, 243, 0.15)", showlegend=False),
            row=row, col=col,
        )

    # Reference trajectory
    if ref_x is not None and ref_y is not None:
        fig.add_trace(
            go.Scatter(x=ref_x, y=ref_y, mode="lines", name="Reference", line={"color": "red", "width": 2.5, "dash": "dash"}, showlegend=(row == 6 and col == 1)),
            row=row, col=col,
        )

    fig.update_xaxes(title_text="Orbital Energy (MJ/kg)", row=row, col=col)
    fig.update_yaxes(title_text=y_label, row=row, col=col)
```

- [ ] **Step 4: Wire corridor panels into `generate_final_report()`**

After rows 1-5, add corridor panels if trajectories are available:

```python
if has_trajectories:
    ref_traj = _load_reference_trajectory(ref_trajectory_path) if ref_trajectory_path else None

    # Row 6 left: Energy vs Dynamic Pressure
    _add_corridor_panel(
        fig, trajectories, captured, y_col=9, y_label="Dynamic Pressure (kPa)",
        ref_x=ref_traj["energy_MJkg"] if ref_traj else None,
        ref_y=ref_traj["pdyn_kPa"] if ref_traj else None,
        row=6, col=1, n_sims=n_total,
    )
    # Row 6 right: Energy vs Inclination
    _add_corridor_panel(
        fig, trajectories, captured, y_col=11, y_label="Inclination (deg)",
        ref_x=ref_traj["energy_MJkg"] if ref_traj else None,
        ref_y=ref_traj["inclination_deg"] if ref_traj else None,
        row=6, col=2, n_sims=n_total,
    )
    # Row 7 left: Energy vs Bank Angle
    _add_corridor_panel(
        fig, trajectories, captured, y_col=10, y_label="Bank Angle (deg)",
        ref_x=ref_traj["energy_MJkg"] if ref_traj else None,
        ref_y=ref_traj["bank_deg"] if ref_traj else None,
        row=7, col=1, n_sims=n_total,
    )
```

- [ ] **Step 5: Write tests**

```python
def test_corridor_panels_present_with_trajectories(tmp_path):
    n = 20
    arr = _make_captured_array(n=n)
    trajs = [np.random.default_rng(i).random((50, 12)) for i in range(n)]
    eval_data = FinalEvalData(final_array=arr, trajectories=trajs, dispersions=None)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    assert "Dynamic Pressure" in html
    assert "Bank Angle" in html

def test_corridor_panels_absent_without_trajectories(tmp_path):
    arr = _make_captured_array()
    eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    # Should still produce valid HTML without corridor panels
    assert "Final Evaluation" in html
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/final_report.py tests/test_final_report.py
git commit -m "feat: add energy corridor panels (pdyn, inclination, bank angle) to final report"
```

---

## Task 7: Add dispersion correlation grid (row 8+)

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py`
- Test: `tests/test_final_report.py`

- [ ] **Step 1: Define dispersion field labels**

Add to `final_report.py`:

```python
_DISPERSION_LABELS: list[tuple[str, str]] = [
    ("Entry Altitude", "m"),
    ("Entry Longitude", "rad"),
    ("Entry Latitude", "rad"),
    ("Entry Velocity", "m/s"),
    ("Entry FPA", "rad"),
    ("Entry Azimuth", "rad"),
    ("Density Error", "frac"),
    ("Drag Coeff Error", "frac"),
    ("Lift Coeff Error", "frac"),
    ("Incidence Error", "rad"),
    ("Nav Altitude Error", "m"),
    ("Nav Longitude Error", "rad"),
    ("Nav Latitude Error", "rad"),
    ("Nav Velocity Error", "m/s"),
    ("Nav FPA Error", "rad"),
    ("Nav Azimuth Error", "rad"),
    ("Nav Drag Accel Error", "m/s²"),
    ("Mass Error", "frac"),
    ("Ref Area Error", "frac"),
    ("Max Bank Rate Error", "frac"),
    ("Pilot Tau Error", "frac"),
    ("Pilot Damping Error", "frac"),
    ("Pilot Frequency Error", "frac"),
    ("Filter Gain Error", "abs"),
]
```

- [ ] **Step 2: Implement dispersion correlation helper**

```python
def _add_dispersion_correlations(
    fig: object,
    dispersions: npt.NDArray[np.float64],
    dv_total: npt.NDArray[np.float64],
    start_row: int,
    n_cols: int,
) -> None:
    """Add scatter + regression subplots for each dispersion field vs delta-V."""
    from scipy.stats import linregress

    import plotly.graph_objects as go

    n_fields = dispersions.shape[1]
    for i in range(n_fields):
        row = start_row + i // n_cols
        col = (i % n_cols) + 1
        x = dispersions[:, i]
        label, unit = _DISPERSION_LABELS[i]

        # Skip fields with zero variance (not dispersed)
        if np.std(x) < 1e-12:
            fig.add_annotation(
                text=f"{label}: not dispersed",
                x=0.5, y=0.5, showarrow=False,
                font={"size": 10, "color": "gray"},
                row=row, col=col,
            )
            continue

        fig.add_trace(
            go.Scattergl(x=x, y=dv_total, mode="markers", marker={"size": 3, "opacity": 0.4, "color": _COLOR_PRIMARY}, showlegend=False),
            row=row, col=col,
        )

        # Linear regression
        result = linregress(x, dv_total)
        x_line = np.array([x.min(), x.max()])
        y_line = result.slope * x_line + result.intercept
        fig.add_trace(
            go.Scatter(x=x_line, y=y_line, mode="lines", line={"color": "red", "width": 2}, showlegend=False),
            row=row, col=col,
        )

        # Use subplot title for R²/p-value since this is a dedicated 4-column figure
        fig.layout.annotations[i].text = f"{label} (R²={result.rvalue**2:.3f}, p={result.pvalue:.1e})"

        fig.update_xaxes(title_text=f"{label} ({unit})", row=row, col=col)
        fig.update_yaxes(title_text="ΔV (m/s)", row=row, col=col)
```

**Implementation approach:** Use a **separate Plotly figure** for the dispersion grid (4-column `make_subplots`) and concatenate its HTML with the main figure's HTML. This avoids the fragile axis indexing that comes from mixing 2-column and 4-column layouts in one `make_subplots` call. The `fig.add_annotation(row=row, col=col)` form works reliably within a single `make_subplots` grid, so the xref/yref hacks are not needed.

- [ ] **Step 3: Wire into `generate_final_report()`**

After corridor panels, create a **separate figure** for the dispersion grid and write both to the same HTML file:

```python
if has_dispersions and n_captured > 0:
    from plotly.subplots import make_subplots as make_sub

    cap_dispersions = dispersions_array[captured]
    cap_dv = final_array[captured][:, _COL_DV_TOTAL]
    n_disp_fields = cap_dispersions.shape[1]
    n_disp_cols = 4
    n_disp_rows = math.ceil(n_disp_fields / n_disp_cols)

    disp_titles = [f"{label} ({unit})" for label, unit in _DISPERSION_LABELS[:n_disp_fields]]
    disp_fig = make_sub(rows=n_disp_rows, cols=n_disp_cols, subplot_titles=disp_titles)
    _add_dispersion_correlations(disp_fig, cap_dispersions, cap_dv, start_row=1, n_cols=n_disp_cols)
    disp_fig.update_layout(
        height=n_disp_rows * 300,
        title_text="Dispersion Correlation Analysis",
        showlegend=False,
    )

    # Write main fig + dispersion fig to single HTML
    main_html = fig.to_html(include_plotlyjs=True, full_html=False)
    disp_html = disp_fig.to_html(include_plotlyjs=False, full_html=False)
    with open(str(output_path), "w") as f:
        f.write(f"<html><body>{main_html}<hr>{disp_html}</body></html>")
else:
    fig.write_html(str(output_path), include_plotlyjs=True)
```

- [ ] **Step 4: Write tests**

```python
def test_dispersion_grid_present_with_data(tmp_path):
    n = 50
    arr = _make_captured_array(n=n)
    disps = np.random.default_rng(42).standard_normal((n, 24))
    eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=disps)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    assert "Entry Velocity" in html
    assert "R²=" in html

def test_dispersion_grid_absent_without_data(tmp_path):
    arr = _make_captured_array()
    eval_data = FinalEvalData(final_array=arr, trajectories=None, dispersions=None)
    out = tmp_path / "report.html"
    generate_final_report(eval_data, "test", 50.0, out)
    html = out.read_text()
    assert "Final Evaluation" in html  # still valid
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/final_report.py tests/test_final_report.py
git commit -m "feat: add dispersion correlation grid with regressions to final report"
```

---

## Task 8: Update `main()` CLI to pass reference trajectory path

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py` (main function)
- Modify: `src/python/aerocapture/training/train.py` (where final report is called at end of training)

- [ ] **Step 1: Update `main()` in `final_report.py`**

In the CLI `main()`, after loading the TOML, extract the reference trajectory path:

```python
ref_traj_path = _read_ref_trajectory_path(Path(args.toml))
```

Pass it to `generate_final_report()`:

```python
generate_final_report(eval_data, scheme, target_incl, output_path, ref_trajectory_path=ref_traj_path)
```

- [ ] **Step 2: Update the call in `train.py`**

Find where `train.py` calls `generate_final_report` or `run_final_evaluation` at end of training and update to pass the reference trajectory path from the TOML config.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/final_report.py src/python/aerocapture/training/train.py
git commit -m "feat: pass reference trajectory path through CLI and training pipeline"
```

---

## Task 9: Full integration test

**Files:**
- Test: `tests/test_pyo3.py`
- Test: `tests/test_final_report.py`

- [ ] **Step 1: Build PyO3 bindings**

Run: `cd src/rust/aerocapture-py && maturin develop --release`

- [ ] **Step 2: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All pass.

- [ ] **Step 3: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 4: Run linters**

Run: `./lint_code.sh`
Expected: Clean (ruff + mypy).

- [ ] **Step 5: Run Rust checks**

Run: `./check_all.sh`
Expected: Clean (fmt + clippy + test + release build).

- [ ] **Step 6: Manual smoke test**

Generate a final report using an existing training output to visually verify the new panels:

```bash
uv run python -m aerocapture.training.final_report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-sims 100 --seed 42
```

Open the generated `final_report.html` in a browser and verify:
- Entry Conditions and Exit Conditions are separate panels with correct labels
- Performance table shows all 7 metrics with Mean/Std/Min/p5-p95/Max
- Corridor panels show energy vs pdyn/inclination/bank with spaghetti + envelope + reference
- Dispersion correlation grid shows ~24 scatter plots with regression lines and R² values

- [ ] **Step 7: Commit any fixes from smoke test**

Stage specific changed files (do NOT use `git add -A`), then:
```bash
git commit -m "fix: address issues found during integration smoke test"
```

---

## Task 10: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
