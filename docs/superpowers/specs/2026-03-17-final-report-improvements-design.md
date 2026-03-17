# Final Report Improvements — Design Spec

## Problem

The current `final_report.py` has three shortcomings:

1. **Mislabeled scatter plot** — "Entry Conditions" actually plots exit-state columns (`final_record[3]` velocity, `final_record[4]` FPA), which are atmospheric exit values, not entry.
2. **No dispersion correlation visibility** — dispersion draws (`DispersionDraw`, ~24 fields) aren't exposed through the PyO3 API, so there's no way to see which dispersions drive delta-V variance.
3. **No trajectory corridor plots** — the classic energy-corridor visualization (energy vs pdyn/inclination/bank spaghetti with envelope) requires per-timestep data that the trajectory output doesn't currently carry (missing energy, pdyn, bank angle, inclination).

## Scope

### Rust Plumbing Changes (API Path)

Currently, `run_for_api()` calls `run_core(config, data, false)` — the `false` means `write_photo=false`, so **no trajectory data is captured** and `RunOutput.trajectory` is always `Vec::new()`. Additionally, in MC mode, `run_core` only captures photo output for a single sim (`photo_sim_idx`), not all sims.

To support corridor plots and dispersion export, three plumbing changes are needed:

1. **Add `include_trajectories: bool` parameter to `run_for_api()` and `run_core()`** — as a direct function parameter (NOT on `SimInput`, which is TOML-deserialized). When true, `run_core()` passes `write_photo=true` AND overrides the `photo_sim_idx` logic so ALL sims capture photo rows (not just one). When false, behavior is unchanged (empty trajectory vecs). The CLI `run()` path continues to call `run_core(config, data, true)` with its existing single-sim photo behavior — the new parameter only affects the API path.

2. **Thread `DispersionDraw` through to `RunOutput`** — currently `DispersionDraw` is consumed by `init::init_run_from_draw()` and discarded. Store the `DispersionDraw` (or its flat `[f64; 24]` representation) alongside the `RunState` in the run_states vec (e.g., `Vec<(RunState, [f64; 24])>`), then serialize it into `RunOutput.dispersions` at result assembly time. For single-sim (no MC), dispersions is all zeros (default draw).

3. **Capture trajectory for all sims when requested** — in `run_core()`, when `include_trajectories=true`, set `do_photo=true` for ALL sims in the `par_iter` map, not just `photo_sim_idx`. The existing `write_photo` parameter continues to control file I/O and screen output independently.

The PyO3 `run_mc()` already has an `include_trajectories: bool` parameter — it just needs to forward it to `run_for_api()` so Rust actually captures the data.

### Rust Data Model Changes

#### Trajectory expansion: 8 → 12 columns

`RunOutput.trajectory` grows from `[f64; 8]` to `[f64; 12]`:

| Index | Field | Unit |
|-------|-------|------|
| 0 | altitude | km |
| 1 | longitude | deg |
| 2 | latitude | deg |
| 3 | velocity | m/s |
| 4 | flight path angle | deg |
| 5 | heading | deg |
| 6 | heat flux | W/m² |
| 7 | time | s |
| 8 | orbital energy | MJ/kg |
| 9 | dynamic pressure | kPa |
| 10 | bank angle | deg |
| 11 | inclination | deg |

These values are already computed in `build_photo_values()` (photo output row) at different indices — energy at photo[18] (J/kg, needs ÷1e6 → MJ/kg), pdyn at photo[19] (Pa, needs ÷1e3 → kPa), bank angle at photo[14] (already deg), inclination at photo[9] (already deg). The new trajectory row extraction just picks these 4 values from the existing photo row and converts units.

**Note on pdyn source:** photo[19] is the truth-model dynamic pressure (from navigator output), which is correct for the physical corridor visualization. The onboard-estimated pdyn (photo[21]) is a separate quantity and should NOT be used here.

**Note on memory:** 1000 sims × ~700 timesteps × 12 columns × 8 bytes ≈ 67 MB. Acceptable for a final evaluation report; gated behind `include_trajectories=True` so normal training/evaluation is unaffected.

#### Dispersions array: new field on `RunOutput`

```rust
pub dispersions: [f64; 24],
```

Flat array, one element per `DispersionDraw` field in struct order. Add a `to_array(&self) -> [f64; 24]` method on `DispersionDraw` for clean serialization. Add a compile-time `const` assertion that the array size matches the struct field count to catch future field additions.

| Index | Field | Unit/Type |
|-------|-------|-----------|
| 0 | altitude | meters (Gaussian × sigma) |
| 1 | longitude | radians |
| 2 | latitude | radians |
| 3 | velocity | m/s |
| 4 | flight_path | radians |
| 5 | azimuth | radians |
| 6 | density | fractional |
| 7 | drag_coeff | fractional |
| 8 | lift_coeff | fractional |
| 9 | incidence | radians |
| 10 | nav_altitude | meters |
| 11 | nav_longitude | radians |
| 12 | nav_latitude | radians |
| 13 | nav_velocity | m/s |
| 14 | nav_flight_path | radians |
| 15 | nav_azimuth | radians |
| 16 | nav_drag_accel | m/s² |
| 17 | mass | fractional |
| 18 | ref_area | fractional |
| 19 | max_bank_rate | fractional |
| 20 | pilot_tau | fractional |
| 21 | pilot_damping | fractional |
| 22 | pilot_frequency | fractional |
| 23 | filter_gain | absolute delta |

For single-sim (no MC), dispersions is all zeros.

#### Clean unused `final_record` slots

Slots 32-36, 42-44, 46-47, 49-51 remain zero-initialized. Add a comment block in `runner.rs` documenting which of the 52 slots are used and which are reserved/unused. No backfilling.

### PyO3 Bindings Changes

- `SimResult` gains `.dispersions` property → `PyArray1<f64>` (24 elements)
- `SimResult.trajectory` column count grows from 8 → 12 automatically (no API change)
- `BatchResults` gains `.dispersions` property → `PyArray2<f64>` shape `(n_sims, 24)`, always populated (not gated by `include_trajectories`)
- `BatchResults.trajectories` behavior unchanged (gated by `include_trajectories=True`)

### Python `final_report.py` Changes

#### Data flow

`run_final_evaluation` return type changes from `ndarray | None` to a named tuple:

```python
FinalEvalData = namedtuple("FinalEvalData", ["final_array", "trajectories", "dispersions"])
```

- `final_array`: `(N, 52)` ndarray
- `trajectories`: list of `(T_i, 12)` ndarrays (variable timestep count per sim), or `None`
- `dispersions`: `(N, 24)` ndarray

Calls `run_mc(include_trajectories=True)` to get trajectory data.

#### Report panel layout

| Row | Left | Right |
|-----|------|-------|
| 1 | Total Delta-V Distribution (histogram + CDF) | Individual Correction Burns (overlaid histograms) |
| 2 | Apoapsis Error histogram + CDF (km) | Periapsis Error histogram + CDF (km) |
| 3 | Inclination Error histogram + CDF (deg) | Delta-V vs Orbital Error (scatter) |
| 4 | Entry Conditions (dispersed V vs FPA, colored by outcome) | Exit Conditions (final V vs FPA, colored by outcome, marker size ∝ DV) |
| 5 | Performance Summary Table (colspan 2) | |
| 6 | Energy vs Dynamic Pressure (corridor) | Energy vs Inclination (corridor) |
| 7 | Energy vs Bank Angle (corridor) | *(empty)* |
| 8+ | Dispersion Correlation Grid (~24 scatter subplots) | |

#### Row 4: Entry & Exit Conditions

**Entry Conditions (left):** Scatter of actual entry velocity vs entry FPA, colored green (captured) / red (hyperbolic). Since `dispersions[:, 3]` and `dispersions[:, 4]` are perturbation deltas (not absolute values), compute actual entry state as `nominal + delta` where nominal entry velocity/FPA come from the TOML config `[entry]` section (loaded alongside target inclination). Alternatively, use the first row of each trajectory (trajectory[0][3] for velocity, trajectory[0][4] for FPA) which is the actual dispersed entry state — this is simpler and avoids TOML parsing.

**Exit Conditions (right):** The current panel, relabeled. Uses `final_record[3]` (exit velocity) and `final_record[4]` (exit FPA), with marker size proportional to delta-V.

#### Row 5: Performance Summary Table

Full-width table matching the format from the 2009 article, with extended percentile columns:

| Parameter | Mean | Std | Min | p5 | p25 | p50 | p75 | p95 | Max |
|-----------|------|-----|-----|----|-----|-----|-----|-----|-----|
| Max g-load (g) | | | | | | | | | |
| Max heat flux (kW/m²) | | | | | | | | | |
| Bank angle consumption (deg) | | | | | | | | | |
| Apoapsis error (km) | | | | | | | | | |
| Periapsis error (km) | | | | | | | | | |
| Inclination error (deg) | | | | | | | | | |
| Correction cost ΔV (m/s) | | | | | | | | | |

Data sources from `final_record` columns (captured trajectories only):
- Max g-load: col 17
- Max heat flux: col 16
- Bank angle consumption: col 45
- Apoapsis error: col 30
- Periapsis error: col 29
- Inclination error: col 10 - target_inclination
- Correction cost ΔV: col 41

Capture rate displayed as a header annotation above the table.

#### Rows 6-7: Energy Corridor Panels

Three corridor panels combining the MC spaghetti style with the envelope visualization:

**Common elements:**
- X-axis: orbital energy (MJ/kg) — trajectory column 8
- MC captured trajectories as translucent blue lines (opacity ~0.05-0.1 scaled by n_sims)
- Hyperbolic trajectories as translucent red lines
- Corridor envelope: filled blue region between min/max y-value at each energy bin across captured trajectories (`fill="tonexty"`)
- Reference trajectory as bold dashed red line (loaded from `.dat` file via `ReferenceTrajectory` format: col 0 = energy MJ/kg, col 1 = pdyn Pa)

**Panel-specific:**
- **Energy vs Dynamic Pressure:** Y-axis pdyn (kPa) — trajectory column 9. Reference trajectory overlay from `.dat` file (energy vs pdyn columns).
- **Energy vs Inclination:** Y-axis inclination (deg) — trajectory column 11. Reference trajectory overlay from `.dat` file (energy vs inclination columns).
- **Energy vs Bank Angle:** Y-axis bank angle (deg) — trajectory column 10. Reference trajectory overlay from `.dat` file (energy vs cos_bank → acos → deg).

#### Row 8+: Dispersion Correlation Grid

~24 scatter subplots arranged in a grid (e.g., 6×4 or 8×3), each showing:
- X-axis: one dispersion field value
- Y-axis: total delta-V (m/s)
- Captured trajectories only
- Linear regression line via `scipy.stats.linregress`
- Annotation with R² and p-value
- Human-readable axis labels (e.g., "Entry Velocity (m/s)", "Density Error (%)", "Drag Coeff Error (%)")

These panels sit below all other content. The total figure height grows to accommodate them.

### Reference Trajectory Loading

The reference trajectory `.dat` file path is available in the TOML config under `[data] reference_trajectory`. The `_read_target_inclination` pattern already loads TOML data — extend this to also extract the reference trajectory path and load it via a Python reader (simple whitespace-separated 7-column format matching `ReferenceTrajectory::load`).

**Unit conversions for reference trajectory overlay:**
- Column 0 (energy): MJ/kg — matches trajectory column 8 directly
- Column 1 (pdyn): Pa — divide by 1e3 for kPa to match trajectory column 9
- Column 4 (inclination): radians — convert to degrees to match trajectory column 11
- Column 6 (cos_bank): apply `acos` then convert to degrees to match trajectory column 10

### Files Modified

| File | Change |
|------|--------|
| `src/rust/src/lib.rs` | `RunOutput`: trajectory `[f64; 8]` → `[f64; 12]`, add `dispersions: [f64; 24]` |
| `src/rust/src/data/dispersions.rs` | Add `to_array(&self) -> [f64; 24]` method on `DispersionDraw` |
| `src/rust/src/simulation/runner.rs` | Add `include_trajectories: bool` param to `run_for_api()` and `run_core()`, capture trajectory for ALL sims when requested, populate trajectory cols 8-11 from photo values with unit conversion (energy: photo[18]/1e6 → MJ/kg, pdyn: photo[19]/1e3 → kPa, bank: photo[14] deg, incl: photo[9] deg), thread `DispersionDraw` through to `RunOutput.dispersions`, document final_record slots |
| `src/rust/aerocapture-py/src/lib.rs` | Forward `include_trajectories` from `run_mc()` and `run_batch()` to `run_for_api()` |
| `src/rust/aerocapture-py/src/batch.rs` | Update `run_for_api()` call to pass `include_trajectories` parameter |
| `src/rust/aerocapture-py/src/results.rs` | Add `.dispersions` getters on `SimResult` and `BatchResults` |
| `src/python/aerocapture/training/final_report.py` | New layout, corridor panels, dispersion grid, entry/exit fix, performance table, reference trajectory loader |
| Tests (Rust + Python) | Update trajectory column count assertions (8→12), add dispersions array tests, update `run_for_api` call sites |

### What's NOT Changing

- `report.py` (training convergence report) — untouched
- `evaluate.py` cost function — untouched
- CLI interface for `final_report.py` — same arguments
- The 52-column `final_record` format — no new columns added, just documentation of unused slots
