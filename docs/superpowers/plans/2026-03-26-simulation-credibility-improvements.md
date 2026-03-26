# Simulation Credibility Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three credibility gaps — integrated heat load exposure, altitude-dependent wind model, and EKF navigation — to make MC results defensible for engineering reviews.

**Architecture:** Three independent, backward-compatible additions sequenced by effort. Heat load leverages existing RK4-accumulated `state[6]` (already in `final_record[28]`), just exposing it through all output channels and GA constraints. Wind model replaces the zero-returning stub with table-loaded altitude profiles and MC dispersions. EKF navigation is a new subsystem with IMU sensor model, star tracker, and drag-derived altitude updates, orchestrated by a mode switch in the existing estimator.

**Tech Stack:** Rust (nalgebra for EKF matrix ops), Python (numpy, matplotlib/seaborn for charts), TOML config, PyO3 bindings.

**Key discovery:** `state[6]` already accumulates heat load via RK4 integration of `dflux = cq * sqrt(rho) * V^3.05`. The value is stored in `final_record[28]` as `integrated_flux_mj_m2`. The IMPROVEMENTS.md claim that "it stores instantaneous, not cumulative" is **incorrect**. The work is exposing this existing value, not computing it.

---

## Part 1: Integrated Heat Load Exposure

### Task 1: Expose heat load in trajectory data and photo CSV

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (lines 232-242 trajectory mapping, lines 820-863 photo builder)
- Modify: `src/rust/src/simulation/output.rs` (line 9 PHOTO_CSV_COLUMNS)
- Modify: `src/rust/src/lib.rs` (line 11 RunOutput trajectory doc comment)

- [ ] **Step 1: Add heat load to trajectory output**

In `runner.rs`, the trajectory mapping at line 242 has `0.0, // [15] reserved`. Replace with the cumulative heat load from `state[6]`:

```rust
// In run_for_api(), inside the photo_lines mapping (around line 242):
// Change:
//     0.0,         // [15] reserved
// To:
    p[18].signum() * sim_state_flux_at_step, // [15] — BUT we don't have state[6] here
```

Actually, the photo_lines array (28 elements) doesn't carry `state[6]`. We need to add it. In `build_photo_values()` (line 795), the function computes photo values from SimState. We need to pass `state[6]` through.

In `build_photo_values()`, change the return array to include cumulative heat load. The array has 28 elements but element indices after 27 don't exist — we need to **use an unused slot** or **grow the array**. Looking at the array, slots are tightly packed. The simplest approach: pass `cumulative_flux` as a parameter and store it in the `0.0` reserved slot of the trajectory output.

In `runner.rs`, modify `build_photo_values()` signature to accept `cumulative_flux_w_s_per_m2: f64`:

```rust
fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &Planet,
    dynamic_pressure_nav: f64,
    density_estimate: f64,
    sim_index: usize,
    cumulative_bank_change: f64,
    data: &SimData,
    density_gain: f64,
    run_state: &init::RunState,
    cumulative_flux: f64,  // NEW: state[6] in W·s/m² (= J/m²)
) -> [f64; 28] {
```

At the end of the returned array, we currently don't have a slot. But the trajectory mapping picks specific indices from the 28-element photo array. We need to carry cumulative_flux through. The cleanest approach: **grow the photo array to 29 elements** by appending `cumulative_flux / 1e3` (kJ/m²) at index [28].

Change the return type to `[f64; 29]` and append:

```rust
    // At the end of the array, after rho_truth (index [27]):
    cumulative_flux / 1e3, // [28] cumulative_heat_load_kj_m2
]
```

Then in the trajectory mapping (line 242), replace `0.0` with `p[28]`:

```rust
    p[28],       // [15] heat_load_kj_m2
```

Update the `photo_lines` type from `Vec<[f64; 28]>` to `Vec<[f64; 29]>` in `run_single()`.

At the call site (line 556), pass `sim.state[6]`:

```rust
photo_lines.push(build_photo_values(
    &sim,
    sim_time,
    planet,
    dynamic_pressure_for_photo,
    density_estimate_for_photo,
    sim_idx + 1,
    cumulative_bank_change_deg * DEG_TO_RAD,
    data,
    nav_state.density_gain,
    run_state,
    sim.state[6],  // cumulative flux
));
```

- [ ] **Step 2: Update RunOutput trajectory doc comment**

In `lib.rs` line 11, update the trajectory column documentation:

```rust
    /// Per-timestep state: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg,
    /// heat_flux_kw_m2, time_s, energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg,
    /// g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2]
    pub trajectory: Vec<[f64; 16]>,
```

- [ ] **Step 3: Add heat load to photo CSV**

In `output.rs`, add `"heat_load_kj_m2"` to `PHOTO_CSV_COLUMNS` (after `"dynamic_pressure_onboard_kpa"`):

```rust
pub const PHOTO_CSV_COLUMNS: &[&str] = &[
    "time_s",
    // ... existing 21 columns ...
    "dynamic_pressure_onboard_kpa",
    "heat_load_kj_m2",  // NEW: column 22 (index 21)
];
```

In `extract_photo_csv_values()` in `runner.rs` (around line 310), add the new column extraction. This function maps the 29-element photo array to the CSV column array. Add `p[28]` (cumulative heat load in kJ/m²) at the end.

- [ ] **Step 4: Build and verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release`

Expected: Successful build. Fix any type mismatches from the `[f64; 28]` → `[f64; 29]` change.

- [ ] **Step 5: Run existing tests to verify backward compatibility**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test`

Expected: All existing tests pass. The photo array size change may require updating test assertions.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/simulation/output.rs src/rust/src/lib.rs
git commit -m "feat: expose cumulative heat load in trajectory data and photo CSV"
```

### Task 2: Expose heat load in PyO3 bindings

**Files:**
- Modify: `src/rust/aerocapture-py/src/results.rs`

- [ ] **Step 1: Add `integrated_heat_load` getter to SimResult**

In `results.rs`, add a new Python getter method to `SimResult`:

```rust
/// Integrated heat load (kJ/m²) — from final_record[28]
#[getter]
fn integrated_heat_load(&self) -> f64 {
    self.output.final_record[28] * 1e3  // MJ/m² → kJ/m² (final_record stores MJ/m²)
}
```

- [ ] **Step 2: Build PyO3 bindings**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust/aerocapture-py && maturin develop --release`

Expected: Successful build.

- [ ] **Step 3: Verify in Python**

```bash
uv run python -c "
import aerocapture_rs
r = aerocapture_rs.run('configs/test/test_ref_orig.toml')
print(f'integrated_heat_load = {r.integrated_heat_load:.2f} kJ/m²')
assert r.integrated_heat_load >= 0, 'heat load must be non-negative'
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/rust/aerocapture-py/src/results.rs
git commit -m "feat: expose integrated_heat_load in PyO3 SimResult"
```

### Task 3: Add heat load constraint to GA cost function

**Files:**
- Modify: `configs/missions/mars.toml`
- Modify: `configs/missions/earth.toml` (if exists)
- Modify: `src/python/aerocapture/training/evaluate.py`
- Modify: `src/python/aerocapture/training/train.py`
- Modify: `src/python/aerocapture/training/charts.py`

- [ ] **Step 1: Add max_heat_load to mission TOML constraints**

In `configs/missions/mars.toml`, add to `[flight.constraints]`:

```toml
[flight.constraints]
max_heat_flux = 200.0
max_load_factor = 4.0
max_dynamic_pressure = 1.081
max_heat_load = 50000.0  # kJ/m² — conservative initial limit, to be tuned
```

- [ ] **Step 2: Add heat load penalty to compute_cost()**

In `evaluate.py`, add `heat_load_limit` and `heat_load_weight` parameters to `compute_cost()`:

```python
def compute_cost(
    final_conditions: npt.NDArray[np.float64],
    *,
    dv_threshold: float = 1000.0,
    g_load_limit: float = 15.0,
    heat_flux_limit: float = 200.0,
    heat_load_limit: float = 50000.0,  # NEW: kJ/m²
    g_load_weight: float = 1000.0,
    heat_flux_weight: float = 1000.0,
    heat_load_weight: float = 1000.0,  # NEW
) -> float:
```

Add the penalty after existing penalties. The integrated heat load is in `final_record[28]` as MJ/m², so convert:

```python
    heat_load = final_conditions[:, 28] * 1e3  # MJ/m² → kJ/m²
    hl_penalty = heat_load_weight * np.maximum((heat_load - heat_load_limit) / heat_load_limit, 0) ** 2
    costs = costs + g_penalty + q_penalty + hl_penalty
```

- [ ] **Step 3: Wire heat_load_limit into cost_kwargs in train.py**

In `train.py` (around line 278), add to the `cost_kwargs` dict:

```python
    cost_kwargs = {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(constraints.get("max_load_factor", 15.0)),
        "heat_flux_limit": float(constraints.get("max_heat_flux", 200.0)),
        "heat_load_limit": float(constraints.get("max_heat_load", 50000.0)),  # NEW
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
        "heat_load_weight": float(cost_cfg.get("heat_load_weight", 1000.0)),  # NEW
    }
```

- [ ] **Step 4: Add heat load column constant to charts.py**

In `charts.py` (around line 75), add the final record index constant:

```python
_FR_INTEGRATED_FLUX = 28  # integrated_flux_mj_m2 (need to convert to kJ/m²)
```

- [ ] **Step 5: Add heat load chart function**

In `charts.py`, add a new chart function after `chart_heat_flux_time()`:

```python
def chart_heat_load_time(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    output: Path,
    limit_kj_m2: float | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Heat load (cumulative) vs time MC spaghetti with optional constraint line."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH, dpi=DPI)
    _draw_spaghetti(ax, trajectories, traj_class, x_col=7, y_col=15)  # col 15 = heat_load_kj_m2
    _draw_time_nominals(ax, y_col=15, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    if limit_kj_m2 is not None:
        ax.axhline(limit_kj_m2, color=COLOR_WORST, linestyle="--", linewidth=1.0, label=f"Limit ({limit_kj_m2:.0f} kJ/m\u00b2)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat load (kJ/m\u00b2)")
    ax.set_title("Cumulative Heat Load vs Time")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize="small")
    sns.despine(fig=fig)
    _save_svg(fig, output)
```

- [ ] **Step 6: Update heat load constraint in classify_trajectories**

In `charts.py`, `classify_trajectories()` (around line 506), add heat load violation checking alongside heat flux:

```python
    if heat_load_limit is not None:
        hl = final_records[:, _FR_INTEGRATED_FLUX] * 1e3  # MJ/m² → kJ/m²
        hl_exceed = hl > heat_load_limit
        constrained = constrained | hl_exceed
```

Add `heat_load_limit: float | None = None` parameter to the function signature.

- [ ] **Step 7: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_cost.py -v`

Expected: Existing cost tests pass. The new parameters have defaults that don't affect existing behavior.

- [ ] **Step 8: Write test for heat load penalty**

Add to `tests/test_cost.py`:

```python
def test_heat_load_penalty_applied_when_exceeded():
    """Cost increases when integrated heat load exceeds limit."""
    fc = factory_final_conditions(n=10)
    fc[:, 28] = 60.0  # 60 MJ/m² = 60000 kJ/m²
    cost_under = compute_cost(fc, heat_load_limit=100000.0)  # well under
    cost_over = compute_cost(fc, heat_load_limit=10000.0)   # well over (10 kJ/m² limit)
    assert cost_over > cost_under, f"Heat load penalty not applied: {cost_over} <= {cost_under}"


def test_heat_load_penalty_zero_when_under_limit():
    """No penalty when heat load is under limit."""
    fc = factory_final_conditions(n=10)
    fc[:, 28] = 10.0  # 10 MJ/m² = 10000 kJ/m²
    cost_no_hl = compute_cost(fc, heat_load_weight=0.0)
    cost_with_hl = compute_cost(fc, heat_load_limit=50000.0, heat_load_weight=1000.0)
    assert cost_no_hl == cost_with_hl, "Penalty applied when heat load is under limit"
```

- [ ] **Step 9: Run test**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_cost.py::test_heat_load_penalty_applied_when_exceeded tests/test_cost.py::test_heat_load_penalty_zero_when_under_limit -v`

Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add configs/missions/mars.toml src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/train.py src/python/aerocapture/training/charts.py tests/test_cost.py
git commit -m "feat: add integrated heat load as GA constraint with chart and cost penalty"
```

### Task 4: Add Rust tests for heat load in trajectory

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (test module)

- [ ] **Step 1: Add property test for heat load monotonicity**

In the `run_output_tests` module at the bottom of `runner.rs`, add:

```rust
#[test]
fn heat_load_in_trajectory_is_monotonically_nondecreasing() {
    let config = SimInput::from_toml_file("../../configs/test/test_ref_orig.toml").unwrap();
    let data = SimData::from_toml(&config.toml, &config).unwrap();
    let results = crate::simulation::runner::run_for_api(&config, &data, true).unwrap();
    let traj = &results[0].trajectory;
    assert!(!traj.is_empty(), "trajectory should not be empty");
    for i in 1..traj.len() {
        assert!(
            traj[i][15] >= traj[i - 1][15],
            "heat load must be monotonically non-decreasing at step {}: {} < {}",
            i, traj[i][15], traj[i - 1][15]
        );
    }
}

#[test]
fn heat_load_final_matches_final_record() {
    let config = SimInput::from_toml_file("../../configs/test/test_ref_orig.toml").unwrap();
    let data = SimData::from_toml(&config.toml, &config).unwrap();
    let results = crate::simulation::runner::run_for_api(&config, &data, true).unwrap();
    let r = &results[0];
    let last_traj_heat_load = r.trajectory.last().unwrap()[15]; // kJ/m²
    let final_record_heat_load = r.final_record[28] * 1e3; // MJ/m² → kJ/m²
    let diff = (last_traj_heat_load - final_record_heat_load).abs();
    assert!(
        diff < 1.0, // allow 1 kJ/m² tolerance (photo cadence vs final state)
        "trajectory last heat load ({:.2}) should match final_record ({:.2}), diff={:.4}",
        last_traj_heat_load, final_record_heat_load, diff
    );
}
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test heat_load -- --nocapture`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "test: add heat load monotonicity and final-record consistency tests"
```

---

## Part 2: Wind Model

### Task 5: Create wind data files

**Files:**
- Create: `data/atmosphere/mars_winds.dat`
- Create: `data/atmosphere/earth_winds.dat`

- [ ] **Step 1: Create Mars wind profile data file**

Based on Forget et al. (1999) Mars General Circulation Model results, create a parametric zonal/meridional wind profile. Values represent equatorial zonal winds (strongest case for aerocapture corridor impact).

```
# Mars wind profile (altitude-dependent)
# Source: Parametric model based on Forget et al. 1999 / Millour et al. 2015
# Columns: altitude_km  zonal_m_s  meridional_m_s
# Note: Zonal winds are eastward (positive = eastward), meridional (positive = northward)
18
  0.0     5.0    2.0
  5.0    10.0    3.0
 10.0    15.0    4.0
 15.0    20.0    5.0
 20.0    30.0    7.0
 25.0    45.0    8.0
 30.0    60.0   10.0
 35.0    75.0   12.0
 40.0    85.0   13.0
 45.0    95.0   14.0
 50.0   100.0   15.0
 60.0    90.0   12.0
 70.0    70.0    8.0
 80.0    50.0    5.0
 90.0    30.0    3.0
100.0    15.0    2.0
120.0     5.0    1.0
150.0     0.0    0.0
```

Write this to `data/atmosphere/mars_winds.dat`.

- [ ] **Step 2: Create Earth wind profile data file**

```
# Earth wind profile (altitude-dependent)
# Source: Parametric model based on standard atmosphere + jet stream data
# Columns: altitude_km  zonal_m_s  meridional_m_s
14
  0.0    10.0    3.0
  5.0    20.0    5.0
 10.0    40.0    8.0
 12.0    60.0   10.0
 15.0    50.0    8.0
 20.0    30.0    5.0
 30.0    50.0    7.0
 40.0    40.0    5.0
 50.0    60.0    8.0
 60.0    70.0   10.0
 70.0    50.0    7.0
 80.0    30.0    4.0
100.0    10.0    2.0
130.0     0.0    0.0
```

Write this to `data/atmosphere/earth_winds.dat`.

- [ ] **Step 3: Commit**

```bash
git add data/atmosphere/mars_winds.dat data/atmosphere/earth_winds.dat
git commit -m "data: add Mars and Earth parametric wind profiles"
```

### Task 6: Implement wind table loader in Rust

**Files:**
- Modify: `src/rust/src/physics/winds.rs`

- [ ] **Step 1: Write tests for wind table loading and interpolation**

Replace the existing test module in `winds.rs` with comprehensive tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_table() -> WindTable {
        WindTable {
            n_points: 4,
            altitudes_m: vec![0.0, 10_000.0, 50_000.0, 100_000.0],
            zonal_m_s: vec![5.0, 50.0, 100.0, 10.0],
            meridional_m_s: vec![2.0, 10.0, 15.0, 1.0],
        }
    }

    #[test]
    fn interpolation_at_table_points() {
        let table = test_table();
        let w = table.wind_at(0.0, 0.0);
        assert!((w.east - 5.0).abs() < 1e-10);
        assert!((w.north - 2.0).abs() < 1e-10);
    }

    #[test]
    fn interpolation_between_points() {
        let table = test_table();
        let w = table.wind_at(5_000.0, 0.0);
        assert!((w.east - 27.5).abs() < 1e-10, "expected 27.5, got {}", w.east);
        assert!((w.north - 6.0).abs() < 1e-10, "expected 6.0, got {}", w.north);
    }

    #[test]
    fn above_table_returns_last() {
        let table = test_table();
        let w = table.wind_at(150_000.0, 0.0);
        assert!((w.east - 10.0).abs() < 1e-10);
    }

    #[test]
    fn below_table_returns_first() {
        let table = test_table();
        let w = table.wind_at(-100.0, 0.0);
        assert!((w.east - 5.0).abs() < 1e-10);
    }

    #[test]
    fn latitude_cosine_scaling() {
        let table = test_table();
        let w_equator = table.wind_at(50_000.0, 0.0);
        let w_pole = table.wind_at(50_000.0, std::f64::consts::FRAC_PI_2);
        // At pole, zonal wind should be ~0 (cos(90°)=0)
        assert!(w_pole.east.abs() < 1e-10, "zonal wind at pole should be ~0, got {}", w_pole.east);
        // Meridional wind is NOT latitude-scaled
        assert!((w_pole.north - w_equator.north).abs() < 1e-10);
    }

    #[test]
    fn vertical_always_zero() {
        let table = test_table();
        let w = table.wind_at(50_000.0, 0.5);
        assert_eq!(w.vertical, 0.0);
    }

    #[test]
    fn disabled_returns_zero() {
        let w = wind_velocity(40_000.0, 0.3, 1.2, None);
        assert_eq!(w.north, 0.0);
        assert_eq!(w.east, 0.0);
        assert_eq!(w.vertical, 0.0);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test winds -- --nocapture`

Expected: FAIL — `WindTable` struct and methods don't exist yet.

- [ ] **Step 3: Implement WindTable and wind_velocity**

Replace the full content of `winds.rs`:

```rust
//! Wind model.
//!
//! Altitude-dependent zonal and meridional wind profiles loaded from data files.
//! Zonal wind is cosine-scaled with latitude (strongest at equator).

use crate::data::DataError;

/// Wind velocity components (m/s) in local horizontal frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindVelocity {
    pub north: f64,    // meridional, positive = northward
    pub east: f64,     // zonal, positive = eastward
    pub vertical: f64, // always 0 for this model
}

/// Tabulated wind profile (altitude-dependent).
#[derive(Debug, Clone, Default)]
pub struct WindTable {
    pub n_points: usize,
    pub altitudes_m: Vec<f64>,
    pub zonal_m_s: Vec<f64>,
    pub meridional_m_s: Vec<f64>,
}

impl WindTable {
    /// Load wind table from a data file.
    ///
    /// Format: comment lines starting with '#', then count N,
    /// then N lines of: altitude_km  zonal_m_s  meridional_m_s
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DataError(format!("Cannot read wind file {}: {}", path, e)))?;

        let mut data_lines: Vec<Vec<f64>> = Vec::new();
        let mut n_points: Option<usize> = None;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if n_points.is_none() {
                // First non-comment line is the count
                n_points = Some(trimmed.parse::<usize>().map_err(|e| {
                    DataError(format!("Wind file {}: bad count '{}': {}", path, trimmed, e))
                })?);
                continue;
            }
            let vals: Vec<f64> = trimmed
                .split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| DataError(format!("Wind file {}: parse error: {}", path, e)))?;
            if vals.len() < 3 {
                return Err(DataError(format!(
                    "Wind file {}: expected 3 columns, got {}",
                    path,
                    vals.len()
                )));
            }
            data_lines.push(vals);
        }

        let n = n_points.unwrap_or(0);
        if data_lines.len() < n {
            return Err(DataError(format!(
                "Wind file {}: expected {} rows, got {}",
                path,
                n,
                data_lines.len()
            )));
        }

        let mut altitudes_m = Vec::with_capacity(n);
        let mut zonal = Vec::with_capacity(n);
        let mut meridional = Vec::with_capacity(n);

        for row in &data_lines[..n] {
            altitudes_m.push(row[0] * 1e3); // km → m
            zonal.push(row[1]);
            meridional.push(row[2]);
        }

        Ok(WindTable {
            n_points: n,
            altitudes_m,
            zonal_m_s: zonal,
            meridional_m_s: meridional,
        })
    }

    /// Interpolate wind at a given altitude (m) and latitude (rad).
    ///
    /// Zonal wind is cosine-scaled with latitude (strongest at equator).
    /// Meridional wind is returned without latitude scaling.
    /// Below/above the table, clamps to the boundary value.
    pub fn wind_at(&self, altitude_m: f64, latitude_rad: f64) -> WindVelocity {
        if self.n_points == 0 {
            return WindVelocity::default();
        }

        let (zonal, merid) = if altitude_m <= self.altitudes_m[0] {
            (self.zonal_m_s[0], self.meridional_m_s[0])
        } else if altitude_m >= self.altitudes_m[self.n_points - 1] {
            (
                self.zonal_m_s[self.n_points - 1],
                self.meridional_m_s[self.n_points - 1],
            )
        } else {
            // Find bracketing interval
            let mut i = 0;
            while i < self.n_points - 1 && self.altitudes_m[i + 1] < altitude_m {
                i += 1;
            }
            let frac = (altitude_m - self.altitudes_m[i])
                / (self.altitudes_m[i + 1] - self.altitudes_m[i]);
            let z = self.zonal_m_s[i] + frac * (self.zonal_m_s[i + 1] - self.zonal_m_s[i]);
            let m = self.meridional_m_s[i]
                + frac * (self.meridional_m_s[i + 1] - self.meridional_m_s[i]);
            (z, m)
        };

        WindVelocity {
            east: zonal * latitude_rad.cos(),
            north: merid,
            vertical: 0.0,
        }
    }
}

/// Compute wind velocity at a given position.
///
/// If no wind table is provided, returns zero wind.
pub fn wind_velocity(
    altitude_m: f64,
    latitude_rad: f64,
    _longitude_rad: f64,
    table: Option<&WindTable>,
) -> WindVelocity {
    match table {
        Some(t) => t.wind_at(altitude_m, latitude_rad),
        None => WindVelocity::default(),
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test winds -- --nocapture`

Expected: All wind tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/physics/winds.rs
git commit -m "feat: implement WindTable loader with altitude interpolation and latitude scaling"
```

### Task 7: Wire wind model into simulation

**Files:**
- Modify: `src/rust/src/data/mod.rs` (SimData: add wind table)
- Modify: `src/rust/src/config.rs` (TOML: add wind_table path)
- Modify: `src/rust/src/simulation/runner.rs` (equations of motion: subtract wind from V_rel)
- Modify: `src/rust/src/data/dispersions.rs` (wind dispersions)
- Modify: `src/rust/src/simulation/init.rs` (RunState: wind biases)

- [ ] **Step 1: Add wind_table to TOML config**

In `config.rs`, add `wind_table: Option<String>` to `TomlData`:

```rust
pub struct TomlData {
    pub base_dir: String,
    pub output_dir: String,
    pub atmosphere: Option<String>,
    pub reference_trajectory: Option<String>,
    pub neural_network: Option<String>,
    pub results_suffix: Option<String>,
    pub wind_table: Option<String>,  // NEW
}
```

- [ ] **Step 2: Add WindTable to SimData**

In `data/mod.rs`, add the wind table to `SimData`:

```rust
use crate::physics::winds::WindTable;

pub struct SimData {
    // ... existing fields ...
    pub wind_enabled: bool,
    pub wind_table: Option<WindTable>,  // NEW
    // ... rest ...
}
```

In `from_toml()`, load the wind table when the path is provided:

```rust
    let wind_table = if let Some(ref path) = data_section.wind_table {
        let resolved = resolve_data_path(path, &config.base_dir);
        Some(WindTable::load(&resolved)?)
    } else {
        None
    };
```

And include in the `SimData` construction:

```rust
    wind_table,
```

- [ ] **Step 3: Add wind dispersions to DispersionDraw**

In `dispersions.rs`, add wind dispersion fields to `DispersionDraw`:

```rust
pub struct DispersionDraw {
    // ... existing 24 fields ...
    pub wind_scale: f64,          // multiplicative scale factor (1.0 = no change)
    pub wind_direction_bias: f64, // rotation angle in radians
}
```

Update `DISPERSION_DRAW_LEN` from 24 to 26.

Update `to_array()` to include the two new fields.

Add `WindSigmas` struct and parsing in `build_dispersion_config()`:

```rust
pub struct WindSigmas {
    pub scale_min: f64,  // e.g. 0.5
    pub scale_max: f64,  // e.g. 1.5
    pub direction_bias_rad: f64, // max rotation in radians
}
```

In the draw generation, when wind dispersions are configured:

```rust
    wind_scale: if let Some(ref ws) = config.wind {
        rng.gen_range(ws.scale_min..=ws.scale_max)
    } else {
        1.0
    },
    wind_direction_bias: if let Some(ref ws) = config.wind {
        uniform.sample(&mut rng) * ws.direction_bias_rad
    } else {
        0.0
    },
```

Add TOML parsing for `[monte_carlo.wind]` with `scale_min`, `scale_max`, `direction_bias_deg`.

- [ ] **Step 4: Add wind biases to RunState**

In `init.rs`, add wind fields to `RunState`:

```rust
pub struct RunState {
    // ... existing fields ...
    pub wind_scale: f64,          // multiplicative wind scale
    pub wind_direction_bias: f64, // wind rotation angle (rad)
}
```

In `init_run_from_draw()`:

```rust
    wind_scale: draw.wind_scale,
    wind_direction_bias: draw.wind_direction_bias,
```

- [ ] **Step 5: Integrate wind into equations of motion**

In `runner.rs`, modify `compute_derivatives()` to account for wind. The wind modifies the relative velocity used for aerodynamic force computation. The state velocity `v = state[3]` is already relative (atmosphere-fixed). Wind velocity changes the effective relative velocity.

The key insight: in the current equations, `v` (state[3]) is the speed relative to the atmosphere (which rotates with the planet). Wind adds a velocity perturbation to the atmosphere itself, so the effective relative velocity for aero forces becomes `V_rel = V_atm_relative - V_wind`.

Since the equations use speed + direction (not Cartesian), the cleanest approach is to modify the aero force magnitudes by adjusting the effective velocity and dynamic pressure. For small wind speeds relative to entry velocity (100 m/s wind vs 5000+ m/s entry), a first-order correction is acceptable:

```rust
fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64,
    aoa: f64,
    planet: &Planet,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 8] {
    let r = state[0];
    let lat = state[2];
    let v = state[3];
    let gamma = state[4];
    let psi = state[5];

    let (gravtl, gravtr) = gravity::gravity(r, lat, planet);
    let (altitude, _lat_geo) = geodetic_from_spherical(r, state[1], lat, planet);
    let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias);

    // Wind contribution (NEW)
    let (v_eff, dv_wind, dheading_wind) = if let Some(ref wt) = data.wind_table {
        if data.wind_enabled {
            let w = wt.wind_at(altitude, lat);
            // Apply MC dispersions: scale + rotate
            let scale = run_state.wind_scale;
            let rot = run_state.wind_direction_bias;
            let we = scale * (w.east * rot.cos() - w.north * rot.sin());
            let wn = scale * (w.east * rot.sin() + w.north * rot.cos());

            // Project wind into trajectory frame
            // V_rel components in local frame: V*cos(gamma)*sin(psi) [east], V*cos(gamma)*cos(psi) [north]
            let cos_g = gamma.cos();
            let v_east = v * cos_g * psi.sin() - we;
            let v_north = v * cos_g * psi.cos() - wn;
            let v_vert = v * gamma.sin(); // no vertical wind

            let v_eff = (v_east * v_east + v_north * v_north + v_vert * v_vert).sqrt();

            // Perturbation to derivatives (first-order)
            let dv = v_eff - v;
            let dheading = if cos_g.abs() > 1e-10 {
                (v_east.atan2(v_north) - psi).rem_euclid(2.0 * std::f64::consts::PI)
            } else {
                0.0
            };
            (v_eff, dv, dheading)
        } else {
            (v, 0.0, 0.0)
        }
    } else {
        (v, 0.0, 0.0)
    };

    // Use v_eff for aero forces instead of v
    let aoa_dispersed = aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);

    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_factor = rho * ref_area / (2.0 * mass);
    let acdrag = aero_factor * cx * v_eff * v_eff;
    let aclift = aero_factor * cz * v_eff * v_eff;

    // ... rest of equations unchanged, but use v_eff for heat flux too:
    let dflux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    // ... (keep remaining derivative equations using v for the kinematic terms,
    //      v_eff only for aero forces and heat flux)
```

**Important:** The kinematic terms (dr, dlon, dlat, dgamma, dpsi) still use the original `v`, `gamma`, `psi` since those describe the vehicle's motion relative to the planet surface. Only the aero forces use `v_eff` (velocity relative to the air mass).

Also update `track_peak_values()` to use wind-corrected velocity for heat flux:

```rust
fn track_peak_values(
    sim: &mut SimState,
    altitude: f64,
    sim_time: f64,
    data: &SimData,
    run_state: &init::RunState,
) {
    let v = sim.state[3];
    let rho = data.atmosphere.density_at(altitude) * (1.0 + run_state.density_bias);

    // Compute effective velocity with wind
    let v_eff = if let Some(ref wt) = data.wind_table {
        if data.wind_enabled {
            let w = wt.wind_at(altitude, sim.state[2]);
            let scale = run_state.wind_scale;
            let rot = run_state.wind_direction_bias;
            let we = scale * (w.east * rot.cos() - w.north * rot.sin());
            let wn = scale * (w.east * rot.sin() + w.north * rot.cos());
            let gamma = sim.state[4];
            let psi = sim.state[5];
            let cos_g = gamma.cos();
            let v_east = v * cos_g * psi.sin() - we;
            let v_north = v * cos_g * psi.cos() - wn;
            let v_vert = v * gamma.sin();
            (v_east * v_east + v_north * v_north + v_vert * v_vert).sqrt()
        } else { v }
    } else { v };

    let heat_flux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    let pdyn = 0.5 * rho * v_eff * v_eff;
    // ... rest unchanged but use v_eff for load factor too
```

- [ ] **Step 6: Build and run tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release && cargo test`

Expected: All existing tests pass (wind is disabled in all test configs). Fix any compilation errors.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/mod.rs src/rust/src/data/dispersions.rs src/rust/src/simulation/init.rs src/rust/src/simulation/runner.rs
git commit -m "feat: wire wind model into simulation with MC dispersions"
```

### Task 8: Add wind integration tests

**Files:**
- Create: `configs/test/test_wind_mars.toml`
- Modify: `src/rust/tests/e2e.rs` or create new integration test file

- [ ] **Step 1: Create test config with winds enabled**

```toml
base = ["../missions/mars.toml"]

[simulation]
n_sims = 1

[flight]
wind = true

[data]
wind_table = "data/atmosphere/mars_winds.dat"

[guidance]
type = "reference"
bank_angle = 64.77026
```

Write to `configs/test/test_wind_mars.toml`.

- [ ] **Step 2: Write integration test comparing wind vs no-wind**

Add to Rust integration tests (e.g., `src/rust/tests/e2e.rs`):

```rust
#[test]
fn wind_model_affects_trajectory() {
    ensure_release_build();
    // Run without wind
    let output_no_wind = run_config("test_ref_orig");
    // Run with wind
    let output_wind = run_config("test_wind_mars");

    // Trajectories should differ
    let final_no_wind = parse_final_csv(&output_no_wind);
    let final_wind = parse_final_csv(&output_wind);

    // Velocity and FPA at exit should differ
    let vel_diff = (final_no_wind[3] - final_wind[3]).abs();
    assert!(vel_diff > 0.1, "Wind should affect exit velocity, diff={}", vel_diff);
}
```

- [ ] **Step 3: Write Rust unit test for wind file loading**

In `winds.rs` tests:

```rust
#[test]
fn load_mars_wind_file() {
    let table = WindTable::load("../../data/atmosphere/mars_winds.dat").unwrap();
    assert_eq!(table.n_points, 18);
    assert!(table.altitudes_m[0] == 0.0);
    assert!(table.zonal_m_s[10] > 90.0); // peak ~100 m/s around 50 km
}
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/test/test_wind_mars.toml src/rust/tests/ src/rust/src/physics/winds.rs
git commit -m "test: add wind model integration tests and data file loading test"
```

---

## Part 3: EKF Navigation

This is the largest part. It introduces a 13-state Extended Kalman Filter with IMU sensor model, star tracker updates, and drag-derived altitude updates.

### Task 9: EKF core — state vector, predict, update

**Files:**
- Create: `src/rust/src/gnc/navigation/ekf.rs`
- Modify: `src/rust/src/gnc/navigation/mod.rs`

- [ ] **Step 1: Write EKF unit tests**

Create `ekf.rs` with a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_zeros_with_identity_covariance() {
        let ekf = EkfState::new(&EkfConfig::default());
        assert_eq!(ekf.state.len(), 13);
        assert!(ekf.state.iter().all(|&x| x == 0.0));
        // Diagonal elements of covariance should be positive
        for i in 0..13 {
            assert!(ekf.covariance[(i, i)] > 0.0);
        }
    }

    #[test]
    fn predict_step_preserves_covariance_symmetry() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        ekf.predict(1.0, &[0.0; 3], &[0.0; 3], &config);
        // Check symmetry
        for i in 0..13 {
            for j in 0..13 {
                let diff = (ekf.covariance[(i, j)] - ekf.covariance[(j, i)]).abs();
                assert!(diff < 1e-12, "P[{},{}] != P[{},{}]: {} vs {}", i, j, j, i,
                    ekf.covariance[(i, j)], ekf.covariance[(j, i)]);
            }
        }
    }

    #[test]
    fn update_step_reduces_covariance() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        let p_before = ekf.covariance[(0, 0)];

        // Position measurement update
        let h = nalgebra::SMatrix::<f64, 3, 13>::zeros(); // will be filled properly
        let z = nalgebra::SVector::<f64, 3>::zeros();
        let r = nalgebra::SMatrix::<f64, 3, 3>::identity() * 100.0;
        ekf.update_position(&z, &r);

        let p_after = ekf.covariance[(0, 0)];
        assert!(p_after <= p_before, "Update should reduce covariance");
    }

    #[test]
    fn density_state_stays_positive() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        ekf.state[12] = 1.0; // density correction factor starts at 1
        // After many predict steps, density state should stay reasonable
        for _ in 0..100 {
            ekf.predict(1.0, &[0.0; 3], &[0.0; 3], &config);
        }
        assert!(ekf.state[12] > 0.0, "Density correction must stay positive");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test ekf -- --nocapture`

Expected: FAIL — `EkfState`, `EkfConfig` don't exist.

- [ ] **Step 3: Implement EKF core**

```rust
//! Extended Kalman Filter for navigation state estimation.
//!
//! 13-state EKF: [r, lon, lat, V, gamma, psi, accel_bias(3), gyro_bias(3), density_correction]

use nalgebra::{SMatrix, SVector};

/// Number of states in the EKF.
pub const N_STATES: usize = 13;

/// EKF configuration parameters.
#[derive(Debug, Clone)]
pub struct EkfConfig {
    /// Initial position covariance (m², rad², rad²)
    pub p0_pos: [f64; 3],
    /// Initial velocity covariance (m²/s², rad², rad²)
    pub p0_vel: [f64; 3],
    /// Initial accelerometer bias covariance (m²/s⁴)
    pub p0_accel_bias: f64,
    /// Initial gyro bias covariance (rad²/s²)
    pub p0_gyro_bias: f64,
    /// Initial density correction covariance
    pub p0_density: f64,
    /// Process noise for accelerometer bias (m²/s⁵)
    pub q_accel_bias: f64,
    /// Process noise for gyro bias (rad²/s³)
    pub q_gyro_bias: f64,
    /// Process noise for density correction
    pub q_density: f64,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            p0_pos: [2500.0, 1e-8, 1e-8],     // 50m altitude, small angle
            p0_vel: [1.0, 1e-6, 1e-6],         // 1 m/s velocity
            p0_accel_bias: 1e-8,                // 1e-4 m/s²
            p0_gyro_bias: 2.5e-11,              // 5e-6 rad/s
            p0_density: 0.01,                   // 10% initial uncertainty
            q_accel_bias: 1e-12,
            q_gyro_bias: 1e-14,
            q_density: 1e-4,
        }
    }
}

/// EKF state and covariance.
#[derive(Debug, Clone)]
pub struct EkfState {
    /// State vector: [r, lon, lat, V, gamma, psi, ax_bias, ay_bias, az_bias,
    ///                wx_bias, wy_bias, wz_bias, density_correction]
    /// States 0-5 are ERROR states (deviation from nav solution).
    /// States 6-11 are sensor biases.
    /// State 12 is density correction factor (multiplicative, centered at 1.0).
    pub state: SVector<f64, N_STATES>,
    /// Covariance matrix (symmetric positive definite).
    pub covariance: SMatrix<f64, N_STATES, N_STATES>,
}

impl EkfState {
    /// Create a new EKF with initial covariance from config.
    pub fn new(config: &EkfConfig) -> Self {
        let mut p = SMatrix::<f64, N_STATES, N_STATES>::zeros();
        // Position
        p[(0, 0)] = config.p0_pos[0];
        p[(1, 1)] = config.p0_pos[1];
        p[(2, 2)] = config.p0_pos[2];
        // Velocity
        p[(3, 3)] = config.p0_vel[0];
        p[(4, 4)] = config.p0_vel[1];
        p[(5, 5)] = config.p0_vel[2];
        // Accel bias
        for i in 6..9 {
            p[(i, i)] = config.p0_accel_bias;
        }
        // Gyro bias
        for i in 9..12 {
            p[(i, i)] = config.p0_gyro_bias;
        }
        // Density
        p[(12, 12)] = config.p0_density;

        Self {
            state: SVector::zeros(),
            covariance: p,
        }
    }

    /// Prediction step: propagate state and covariance forward by dt.
    ///
    /// `accel_meas`: measured specific force in body frame [ax, ay, az] (m/s²)
    /// `gyro_meas`: measured angular rate in body frame [wx, wy, wz] (rad/s)
    pub fn predict(
        &mut self,
        dt: f64,
        accel_meas: &[f64; 3],
        gyro_meas: &[f64; 3],
        config: &EkfConfig,
    ) {
        // State transition matrix (linearized, simplified)
        // For error-state EKF, F relates error state propagation
        let mut f = SMatrix::<f64, N_STATES, N_STATES>::identity();

        // Position errors grow with velocity errors
        f[(0, 3)] = dt; // dr_err += dV_err * dt (simplified)

        // Velocity errors grow with accel bias
        f[(3, 6)] = -dt; // dV_err += accel_bias * dt
        f[(4, 7)] = -dt;
        f[(5, 8)] = -dt;

        // Attitude errors grow with gyro bias
        // (simplified: directly affects heading/FPA via integrated rate)

        // Propagate: P = F * P * F^T + Q
        let q = self.process_noise(dt, config);
        self.covariance = &f * &self.covariance * f.transpose() + q;

        // Enforce symmetry
        self.covariance = (&self.covariance + self.covariance.transpose()) * 0.5;

        // Biases are modeled as random walks — state unchanged in predict
        // (bias states propagate as identity: bias_k+1 = bias_k + noise)

        // Suppress unused warnings for now — will be used in full strapdown
        let _ = accel_meas;
        let _ = gyro_meas;
    }

    /// Build process noise matrix Q for timestep dt.
    fn process_noise(&self, dt: f64, config: &EkfConfig) -> SMatrix<f64, N_STATES, N_STATES> {
        let mut q = SMatrix::<f64, N_STATES, N_STATES>::zeros();

        // Position process noise (driven by velocity uncertainty)
        q[(0, 0)] = 1.0 * dt;
        q[(1, 1)] = 1e-10 * dt;
        q[(2, 2)] = 1e-10 * dt;

        // Velocity process noise (driven by accel uncertainty)
        q[(3, 3)] = 0.01 * dt;
        q[(4, 4)] = 1e-6 * dt;
        q[(5, 5)] = 1e-6 * dt;

        // Accel bias random walk
        for i in 6..9 {
            q[(i, i)] = config.q_accel_bias * dt;
        }
        // Gyro bias random walk
        for i in 9..12 {
            q[(i, i)] = config.q_gyro_bias * dt;
        }
        // Density correction process noise
        q[(12, 12)] = config.q_density * dt;

        q
    }

    /// Position measurement update (star tracker).
    ///
    /// `innovation`: measurement - predicted [dr, dlon, dlat]
    /// `r_meas`: measurement noise covariance (3x3)
    pub fn update_position(
        &mut self,
        innovation: &SVector<f64, 3>,
        r_meas: &SMatrix<f64, 3, 3>,
    ) {
        // H maps state to position observation: H = [I_3x3 | 0_3x10]
        let mut h = SMatrix::<f64, 3, N_STATES>::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        self.kalman_update(&h, innovation, r_meas);
    }

    /// Drag-derived density update.
    ///
    /// `innovation`: (rho_measured / rho_model) - density_correction_state
    /// `r_meas`: scalar measurement variance
    pub fn update_density(&mut self, innovation: f64, r_meas: f64) {
        // H maps state to density observation: H = [0...0, 1] (only state 12)
        let mut h = SMatrix::<f64, 1, N_STATES>::zeros();
        h[(0, 12)] = 1.0;

        let z = SVector::<f64, 1>::new(innovation);
        let r = SMatrix::<f64, 1, 1>::new(r_meas);

        self.kalman_update(&h, &z, &r);

        // Clamp density correction to reasonable range
        self.state[12] = self.state[12].clamp(-0.9, 9.0); // 0.1x to 10x
    }

    /// Generic Kalman update for M-dimensional measurement.
    fn kalman_update<const M: usize>(
        &mut self,
        h: &SMatrix<f64, M, N_STATES>,
        innovation: &SVector<f64, M>,
        r: &SMatrix<f64, M, M>,
    ) {
        let p_ht = &self.covariance * h.transpose();
        let s = h * &p_ht + r; // Innovation covariance
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return, // Singular — skip update
        };
        let k = &p_ht * s_inv; // Kalman gain

        // State update
        self.state += &k * innovation;

        // Covariance update (Joseph form for numerical stability)
        let i_kh = SMatrix::<f64, N_STATES, N_STATES>::identity() - &k * h;
        self.covariance =
            &i_kh * &self.covariance * i_kh.transpose() + &k * r * k.transpose();

        // Enforce symmetry
        self.covariance = (&self.covariance + self.covariance.transpose()) * 0.5;
    }

    /// Get the density correction factor (centered at 1.0).
    pub fn density_correction(&self) -> f64 {
        1.0 + self.state[12]
    }
}
```

- [ ] **Step 4: Register module**

In `gnc/navigation/mod.rs`, add:

```rust
pub mod ekf;
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test ekf -- --nocapture`

Expected: All EKF tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/navigation/ekf.rs src/rust/src/gnc/navigation/mod.rs
git commit -m "feat: implement EKF core with predict, position update, and density update"
```

### Task 10: IMU sensor model

**Files:**
- Create: `src/rust/src/gnc/navigation/imu.rs`
- Modify: `src/rust/src/gnc/navigation/mod.rs`

- [ ] **Step 1: Write IMU tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn zero_true_accel_returns_bias_plus_noise() {
        let config = ImuConfig::default();
        let mut imu = ImuState::new(&config, 42);
        let meas = imu.measure_accel(&[0.0; 3]);
        // With zero true accel, measurement should be small (just bias + noise)
        for i in 0..3 {
            assert!(meas[i].abs() < 0.01, "accel[{}] = {} (should be near zero)", i, meas[i]);
        }
    }

    #[test]
    fn noise_statistics_are_reasonable() {
        let config = ImuConfig::default();
        let mut imu = ImuState::new(&config, 123);
        let n = 10000;
        let mut sum = [0.0; 3];
        for _ in 0..n {
            let m = imu.measure_accel(&[0.0; 3]);
            for i in 0..3 { sum[i] += m[i]; }
        }
        for i in 0..3 {
            let mean = sum[i] / n as f64;
            // Mean should be near the bias (which is small)
            assert!(mean.abs() < 0.01, "mean accel[{}] = {} (should be near 0)", i, mean);
        }
    }
}
```

- [ ] **Step 2: Implement ImuState**

```rust
//! IMU sensor model (accelerometers + gyroscopes).
//!
//! Models bias, scale factor error, and white noise.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// IMU configuration.
#[derive(Debug, Clone)]
pub struct ImuConfig {
    pub accel_bias_sigma: f64,        // m/s² (1-sigma initial bias)
    pub accel_noise_sigma: f64,       // m/s²/√Hz → per-sample at 1 Hz
    pub accel_scale_factor_sigma: f64,
    pub gyro_bias_sigma: f64,         // rad/s (1-sigma initial bias)
    pub gyro_noise_sigma: f64,        // rad/s/√Hz
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            accel_bias_sigma: 1e-4,
            accel_noise_sigma: 5e-4,
            accel_scale_factor_sigma: 1e-4,
            gyro_bias_sigma: 5e-6,
            gyro_noise_sigma: 1e-5,
        }
    }
}

/// IMU sensor state (persistent biases + RNG).
pub struct ImuState {
    accel_bias: [f64; 3],
    accel_scale_factor: [f64; 3],
    gyro_bias: [f64; 3],
    rng: StdRng,
    accel_noise: Normal<f64>,
    gyro_noise: Normal<f64>,
}

impl ImuState {
    pub fn new(config: &ImuConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let bias_dist = Normal::new(0.0, config.accel_bias_sigma).unwrap();
        let sf_dist = Normal::new(0.0, config.accel_scale_factor_sigma).unwrap();
        let gbias_dist = Normal::new(0.0, config.gyro_bias_sigma).unwrap();

        let accel_bias = [
            bias_dist.sample(&mut rng),
            bias_dist.sample(&mut rng),
            bias_dist.sample(&mut rng),
        ];
        let accel_scale_factor = [
            sf_dist.sample(&mut rng),
            sf_dist.sample(&mut rng),
            sf_dist.sample(&mut rng),
        ];
        let gyro_bias = [
            gbias_dist.sample(&mut rng),
            gbias_dist.sample(&mut rng),
            gbias_dist.sample(&mut rng),
        ];

        Self {
            accel_bias,
            accel_scale_factor,
            gyro_bias,
            rng,
            accel_noise: Normal::new(0.0, config.accel_noise_sigma).unwrap(),
            gyro_noise: Normal::new(0.0, config.gyro_noise_sigma).unwrap(),
        }
    }

    /// Measure specific force (accelerometer output).
    pub fn measure_accel(&mut self, true_accel: &[f64; 3]) -> [f64; 3] {
        [
            (1.0 + self.accel_scale_factor[0]) * true_accel[0]
                + self.accel_bias[0]
                + self.accel_noise.sample(&mut self.rng),
            (1.0 + self.accel_scale_factor[1]) * true_accel[1]
                + self.accel_bias[1]
                + self.accel_noise.sample(&mut self.rng),
            (1.0 + self.accel_scale_factor[2]) * true_accel[2]
                + self.accel_bias[2]
                + self.accel_noise.sample(&mut self.rng),
        ]
    }

    /// Measure angular rate (gyroscope output).
    pub fn measure_gyro(&mut self, true_rate: &[f64; 3]) -> [f64; 3] {
        [
            true_rate[0] + self.gyro_bias[0] + self.gyro_noise.sample(&mut self.rng),
            true_rate[1] + self.gyro_bias[1] + self.gyro_noise.sample(&mut self.rng),
            true_rate[2] + self.gyro_bias[2] + self.gyro_noise.sample(&mut self.rng),
        ]
    }
}
```

- [ ] **Step 3: Register module and run tests**

Add `pub mod imu;` to `gnc/navigation/mod.rs`.

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test imu -- --nocapture`

Expected: All IMU tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/navigation/imu.rs src/rust/src/gnc/navigation/mod.rs
git commit -m "feat: implement IMU sensor model with bias, scale factor, and noise"
```

### Task 11: Star tracker model

**Files:**
- Create: `src/rust/src/gnc/navigation/star_tracker.rs`
- Modify: `src/rust/src/gnc/navigation/mod.rs`

- [ ] **Step 1: Write star tracker tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_when_qdyn_below_threshold() {
        let st = StarTrackerConfig::default();
        assert!(st.is_available(50.0, 15.0)); // qdyn=50 < threshold=100
    }

    #[test]
    fn blacked_out_when_qdyn_above_threshold() {
        let st = StarTrackerConfig::default();
        assert!(!st.is_available(200.0, 15.0)); // qdyn=200 > threshold=100
    }

    #[test]
    fn update_due_at_correct_cadence() {
        let st = StarTrackerConfig { update_period: 10.0, ..Default::default() };
        assert!(st.is_update_due(0.0, 10.0));   // exactly at period
        assert!(st.is_update_due(5.0, 15.0));   // 10s elapsed
        assert!(!st.is_update_due(5.0, 12.0));  // only 7s elapsed
    }
}
```

- [ ] **Step 2: Implement StarTrackerConfig**

```rust
//! Star tracker sensor model.
//!
//! Provides position + attitude updates with dynamic pressure blackout.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Star tracker configuration.
#[derive(Debug, Clone)]
pub struct StarTrackerConfig {
    pub position_sigma: f64,          // m (1-sigma position accuracy)
    pub attitude_sigma: f64,          // rad (1-sigma attitude accuracy)
    pub update_period: f64,           // s (update cadence)
    pub blackout_qdyn_threshold: f64, // Pa (no updates above this)
}

impl Default for StarTrackerConfig {
    fn default() -> Self {
        Self {
            position_sigma: 50.0,
            attitude_sigma: 3e-4,
            update_period: 10.0,
            blackout_qdyn_threshold: 100.0,
        }
    }
}

impl StarTrackerConfig {
    /// Check if the star tracker is available (not blacked out).
    pub fn is_available(&self, dynamic_pressure_pa: f64, _sim_time: f64) -> bool {
        dynamic_pressure_pa < self.blackout_qdyn_threshold
    }

    /// Check if an update is due based on elapsed time.
    pub fn is_update_due(&self, last_update_time: f64, current_time: f64) -> bool {
        (current_time - last_update_time) >= self.update_period
    }
}

/// Star tracker measurement state.
pub struct StarTrackerState {
    last_update_time: f64,
    rng: StdRng,
    pos_noise: Normal<f64>,
}

impl StarTrackerState {
    pub fn new(config: &StarTrackerConfig, seed: u64) -> Self {
        Self {
            last_update_time: -1e10, // ensure first update fires
            rng: StdRng::seed_from_u64(seed),
            pos_noise: Normal::new(0.0, config.position_sigma).unwrap(),
        }
    }

    /// Generate a position measurement (true position + noise).
    /// Returns None if not available or not due.
    pub fn measure(
        &mut self,
        true_position: &[f64; 3],
        dynamic_pressure_pa: f64,
        sim_time: f64,
        config: &StarTrackerConfig,
    ) -> Option<[f64; 3]> {
        if !config.is_available(dynamic_pressure_pa, sim_time) {
            return None;
        }
        if !config.is_update_due(self.last_update_time, sim_time) {
            return None;
        }
        self.last_update_time = sim_time;
        Some([
            true_position[0] + self.pos_noise.sample(&mut self.rng),
            true_position[1] + self.pos_noise.sample(&mut self.rng) / true_position[0], // convert m to rad
            true_position[2] + self.pos_noise.sample(&mut self.rng) / true_position[0],
        ])
    }
}
```

- [ ] **Step 3: Register and test**

Add `pub mod star_tracker;` to `mod.rs`.

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test star_tracker -- --nocapture`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/navigation/star_tracker.rs src/rust/src/gnc/navigation/mod.rs
git commit -m "feat: implement star tracker model with dynamic pressure blackout"
```

### Task 12: TOML configuration for EKF navigation

**Files:**
- Modify: `src/rust/src/config.rs` (add TomlNavigation struct)
- Modify: `src/rust/src/data/mod.rs` (add NavConfig to SimData)

- [ ] **Step 1: Add navigation TOML structs**

In `config.rs`, add:

```rust
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TomlNavigation {
    #[serde(default = "default_nav_mode")]
    pub mode: String,  // "bias" or "ekf"
    pub imu: Option<TomlImu>,
    pub star_tracker: Option<TomlStarTracker>,
    pub ekf: Option<TomlEkf>,
}

fn default_nav_mode() -> String { "bias".to_string() }

#[derive(Debug, Clone, Deserialize)]
pub struct TomlImu {
    #[serde(default = "default_accel_bias_sigma")]
    pub accel_bias_sigma: f64,
    #[serde(default = "default_accel_noise_sigma")]
    pub accel_noise_sigma: f64,
    #[serde(default = "default_accel_sf_sigma")]
    pub accel_scale_factor_sigma: f64,
    #[serde(default = "default_gyro_bias_sigma")]
    pub gyro_bias_sigma: f64,
    #[serde(default = "default_gyro_noise_sigma")]
    pub gyro_noise_sigma: f64,
}

fn default_accel_bias_sigma() -> f64 { 1e-4 }
fn default_accel_noise_sigma() -> f64 { 5e-4 }
fn default_accel_sf_sigma() -> f64 { 1e-4 }
fn default_gyro_bias_sigma() -> f64 { 5e-6 }
fn default_gyro_noise_sigma() -> f64 { 1e-5 }

#[derive(Debug, Clone, Deserialize)]
pub struct TomlStarTracker {
    #[serde(default = "default_st_pos_sigma")]
    pub position_sigma: f64,
    #[serde(default = "default_st_att_sigma")]
    pub attitude_sigma: f64,
    #[serde(default = "default_st_period")]
    pub update_period: f64,
    #[serde(default = "default_st_blackout")]
    pub blackout_qdyn_threshold: f64,
}

fn default_st_pos_sigma() -> f64 { 50.0 }
fn default_st_att_sigma() -> f64 { 3e-4 }
fn default_st_period() -> f64 { 10.0 }
fn default_st_blackout() -> f64 { 100.0 }

#[derive(Debug, Clone, Deserialize)]
pub struct TomlEkf {
    #[serde(default = "default_q_density")]
    pub process_noise_density: f64,
}

fn default_q_density() -> f64 { 0.1 }
```

Add `navigation: Option<TomlNavigation>` to `TomlConfig`.

- [ ] **Step 2: Add NavMode enum and config to SimData**

In `data/mod.rs`:

```rust
/// Navigation mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavMode {
    Bias, // legacy: true state + constant bias
    Ekf,  // Extended Kalman Filter
}

// Add to SimData:
pub nav_mode: NavMode,
```

In `from_toml()`, parse:

```rust
let nav_mode = match toml.navigation.as_ref().map(|n| n.mode.as_str()) {
    Some("ekf") => NavMode::Ekf,
    _ => NavMode::Bias,
};
```

- [ ] **Step 3: Build and test**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release && cargo test`

Expected: All existing tests pass. Absent `[navigation]` section defaults to `mode = "bias"`.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat: add EKF navigation TOML configuration with mode switch"
```

### Task 13: Integrate EKF into estimator with mode switch

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs`
- Modify: `src/rust/src/simulation/runner.rs`

- [ ] **Step 1: Add EKF mode to navigate()**

Modify `estimator.rs` to accept `NavMode` and dispatch:

```rust
use crate::data::NavMode;
use super::ekf::{EkfState, EkfConfig};
use super::imu::ImuState;
use super::star_tracker::{StarTrackerConfig, StarTrackerState};

/// Full navigation state (covers both modes).
pub enum NavigationFilter {
    Bias(NavigationState),
    Ekf {
        ekf: EkfState,
        imu: ImuState,
        star_tracker: StarTrackerState,
        st_config: StarTrackerConfig,
        legacy: NavigationState, // still need bounce/phase tracking
    },
}
```

Add a new `navigate_ekf()` function that:
1. Runs the IMU model to get measured accel/gyro
2. Calls `ekf.predict(dt, accel, gyro, config)`
3. Checks star tracker availability and calls `ekf.update_position()` if available
4. Extracts density from drag and calls `ekf.update_density()`
5. Populates `NavigationOutput` from EKF-corrected state
6. Delegates bounce/phase management to the legacy `NavigationState`

The existing `navigate()` function becomes the `Bias` branch. A new dispatcher function routes based on mode.

- [ ] **Step 2: Wire into runner.rs**

In `run_single()`, initialize `NavigationFilter` based on `data.nav_mode`:

```rust
let mut nav_filter = match data.nav_mode {
    NavMode::Bias => NavigationFilter::Bias(NavigationState::new()),
    NavMode::Ekf => {
        // Build config from TOML (or defaults)
        let ekf_config = EkfConfig::default(); // populated from SimData
        NavigationFilter::Ekf {
            ekf: EkfState::new(&ekf_config),
            imu: ImuState::new(&ImuConfig::default(), mc_seed + 1000),
            star_tracker: StarTrackerState::new(&StarTrackerConfig::default(), mc_seed + 2000),
            st_config: StarTrackerConfig::default(),
            legacy: NavigationState::new(),
        }
    }
};
```

Call the appropriate navigate function based on the filter variant.

- [ ] **Step 3: Build and test backward compatibility**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo build --release && cargo test`

Expected: All existing tests pass (all use `mode = "bias"` by default).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs src/rust/src/simulation/runner.rs
git commit -m "feat: integrate EKF into navigation with mode dispatch"
```

### Task 14: EKF integration test

**Files:**
- Create: `configs/test/test_ekf_mars.toml`
- Modify: `src/rust/tests/e2e.rs`

- [ ] **Step 1: Create EKF test config**

```toml
base = ["../missions/mars.toml"]

[simulation]
n_sims = 1

[navigation]
mode = "ekf"

[navigation.imu]
accel_bias_sigma = 1e-4
accel_noise_sigma = 5e-4
gyro_bias_sigma = 5e-6
gyro_noise_sigma = 1e-5

[navigation.star_tracker]
position_sigma = 50.0
update_period = 10.0
blackout_qdyn_threshold = 100.0

[guidance]
type = "ftc"
```

Write to `configs/test/test_ekf_mars.toml`.

- [ ] **Step 2: Write integration test**

```rust
#[test]
fn ekf_navigation_produces_valid_trajectory() {
    ensure_release_build();
    let output = run_config("test_ekf_mars");
    let final_csv = parse_final_csv(&output);

    // Should complete without crashing
    assert!(final_csv[27] > 0.0, "sim_time should be > 0");

    // ifinal should be a valid termination code (1-5)
    let ifinal = final_csv[31] as i32;
    assert!((1..=5).contains(&ifinal), "ifinal={} is invalid", ifinal);
}

#[test]
fn ekf_and_bias_produce_different_results() {
    ensure_release_build();
    let output_bias = run_config("test_guided_orig");  // uses bias mode
    let output_ekf = run_config("test_ekf_mars");

    let final_bias = parse_final_csv(&output_bias);
    let final_ekf = parse_final_csv(&output_ekf);

    // Results should differ (different nav model)
    let vel_diff = (final_bias[3] - final_ekf[3]).abs();
    assert!(vel_diff > 0.01, "EKF and bias should produce different results");
}
```

- [ ] **Step 3: Run integration tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test ekf -- --nocapture`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/test/test_ekf_mars.toml src/rust/tests/e2e.rs
git commit -m "test: add EKF integration test config and e2e tests"
```

### Task 15: Update PyO3 bindings for EKF

**Files:**
- Modify: `src/rust/aerocapture-py/src/config.rs` (support navigation.mode override)

- [ ] **Step 1: Verify EKF mode works through PyO3**

The PyO3 bindings already pass the full TOML config to the Rust core. If `[navigation]` is in the TOML, it should work. Verify:

```bash
uv run python -c "
import aerocapture_rs
r = aerocapture_rs.run('configs/test/test_ekf_mars.toml')
print(f'captured={r.captured}, dv={r.delta_v:.1f}, ecc={r.ecc:.4f}')
print('EKF mode works through PyO3')
"
```

- [ ] **Step 2: Verify override works**

```bash
uv run python -c "
import aerocapture_rs
# Override a bias config to use EKF
r = aerocapture_rs.run(
    'configs/test/test_guided_orig.toml',
    overrides={'navigation.mode': 'ekf'}
)
print(f'captured={r.captured}, dv={r.delta_v:.1f}')
print('EKF override works')
"
```

- [ ] **Step 3: Commit (if changes needed)**

```bash
git add src/rust/aerocapture-py/
git commit -m "feat: verify EKF navigation works through PyO3 bindings"
```

### Task 16: Run full test suite and fix any issues

- [ ] **Step 1: Run Rust tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test`

Expected: All pass.

- [ ] **Step 2: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v`

Expected: All pass.

- [ ] **Step 3: Run linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh`

Expected: Clean.

- [ ] **Step 4: Run clippy**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo clippy -- -D warnings`

Expected: Clean.

- [ ] **Step 5: Fix any failures discovered above**

Address issues iteratively until all checks pass.

- [ ] **Step 6: Commit any fixes**

```bash
git add -u
git commit -m "fix: address test and lint issues from simulation credibility improvements"
```

### Task 17: Update IMPROVEMENTS.md and TODO.md

**Files:**
- Modify: `IMPROVEMENTS.md`
- Modify: `TODO.md`

- [ ] **Step 1: Mark completed items in IMPROVEMENTS.md**

Update the following sections:
- §4.2: Change `[NEW]` to `[DONE]` for integrated heat load tracking
- §1.4: Change `[NEW]` to `[DONE]` for wind model
- §5.1: Change to `[DONE]` for EKF navigation (note: basic EKF, UKF remains future work)

- [ ] **Step 2: Update TODO.md**

Mark the corresponding items as done:
- `[x] Implement wind model`
- `[x] Implement integrated heat load tracking`
- `[x] Replace bias-only navigation with EKF/UKF` (note: EKF done, UKF future)

Fix the incorrect statement about state[6] being instantaneous.

- [ ] **Step 3: Commit**

```bash
git add IMPROVEMENTS.md TODO.md
git commit -m "docs: mark heat load, wind model, and EKF navigation as completed"
```

### Task 18: Smart commit (final)

- [ ] **Invoke smart-commit skill**

Run the `smart-commit` skill to sync CLAUDE.md and README.md with the codebase changes, then commit everything on the branch.

---

## File Map Summary

### New Files
| File | Purpose |
|------|---------|
| `data/atmosphere/mars_winds.dat` | Mars parametric wind profile |
| `data/atmosphere/earth_winds.dat` | Earth parametric wind profile |
| `src/rust/src/gnc/navigation/ekf.rs` | EKF core: state, covariance, predict, update |
| `src/rust/src/gnc/navigation/imu.rs` | IMU sensor model |
| `src/rust/src/gnc/navigation/star_tracker.rs` | Star tracker model |
| `configs/test/test_wind_mars.toml` | Wind integration test config |
| `configs/test/test_ekf_mars.toml` | EKF integration test config |

### Modified Files
| File | Changes |
|------|---------|
| `src/rust/src/simulation/runner.rs` | Heat load in trajectory, wind in EoM, EKF dispatch |
| `src/rust/src/simulation/output.rs` | Photo CSV heat load column |
| `src/rust/src/lib.rs` | RunOutput doc comment |
| `src/rust/src/physics/winds.rs` | Full wind table implementation (replaces stub) |
| `src/rust/src/config.rs` | Wind table path, navigation TOML structs |
| `src/rust/src/data/mod.rs` | SimData: wind table, nav mode |
| `src/rust/src/data/dispersions.rs` | Wind dispersion fields |
| `src/rust/src/simulation/init.rs` | RunState: wind biases |
| `src/rust/src/gnc/navigation/mod.rs` | Register ekf, imu, star_tracker modules |
| `src/rust/src/gnc/navigation/estimator.rs` | NavigationFilter enum, mode dispatch |
| `src/rust/aerocapture-py/src/results.rs` | integrated_heat_load getter |
| `src/python/aerocapture/training/evaluate.py` | Heat load penalty in cost function |
| `src/python/aerocapture/training/train.py` | heat_load_limit in cost_kwargs |
| `src/python/aerocapture/training/charts.py` | Heat load chart, constraint constant |
| `configs/missions/mars.toml` | max_heat_load constraint |
| `IMPROVEMENTS.md` | Mark items [DONE] |
| `TODO.md` | Mark items completed |
