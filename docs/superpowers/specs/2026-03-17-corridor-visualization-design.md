# Corridor Visualization Redesign

## Problem

The current corridor visualization in `final_report.py` uses a statistical envelope of randomly-sampled MC trajectories with a broken zone-rendering hack (`_draw_pdyn_zones` splits at a horizontal midpoint). The result does not produce the standard aerocapture corridor plot with distinct crash, undershoot, overshoot, and hyperbolic zones separated by physically meaningful boundary curves.

## Goal

Produce a corridor plot (orbital energy vs dynamic pressure) with 5 visually distinct regions:

1. **Crash zone** (red) — above the envelope of all non-crashing trajectories
2. **Undershoot zone** (grey) — between crash boundary and restricted corridor (apoapsis error >= -delta_za)
3. **Viable corridor** (white) — the region where guided trajectories should fly
4. **Overshoot zone** (grey) — between restricted corridor and hyperbolic boundary (apoapsis error <= +delta_za)
5. **Hyperbolic exit zone** (red) — below the envelope of all captured trajectories

On top of these zones: blue MC spaghetti (1000 guided trajectories), red nominal constant-bank trajectory, green nominal guided trajectory.

## Design

### Corridor Computation (`corridor.py` rewrite)

#### Phase 1: Full Dispersed MC (10k sims)

Run a large MC with constant bank angle uniformly dispersed over [0deg, 180deg] **plus all mission dispersions** (atmosphere, entry state, aero). Every sim uses `guidance.reference_trajectory=True` with its sampled bank angle.

`include_trajectories=True` to capture per-timestep energy/pdyn curves.

Classify each sim outcome using `final_record` fields:

| Classification | Condition |
|---------------|-----------|
| Crash | `final_record[31] == 1` (TermReason::Crash) |
| Captured, undershoot | `captured == True` and `final_record[30] < -delta_za` |
| Captured, in corridor | `captured == True` and `-delta_za <= final_record[30] <= +delta_za` |
| Captured, overshoot | `captured == True` and `final_record[30] > +delta_za` |
| Hyperbolic | `captured == False` and `final_record[31] == 3` (AtmosphereExit) |
| Timeout | `final_record[31] == 2` (ignored) |

Where `final_record[30]` is apoapsis error (km) and `delta_za` comes from the TOML `[corridor]` section (default 200 km).

**Classification priority:** Apply in order: crash first (ifinal==1), then timeout (ifinal==2, discard), then among atmosphere-exit sims (ifinal==3): hyperbolic if `captured==False`, else captured sub-categories by apoapsis error. A trajectory that crashes is always classified as crash regardless of its orbital elements.

#### Envelope Extraction

Bin all trajectories along the energy axis (~200 bins). For each classification group, compute percentile boundaries of dynamic pressure per energy bin:

- **Envelope A (undershoot boundary):** p99 of pdyn across all trajectories with `apo_error >= -delta_za` (i.e., captured in-corridor + overshoot + undershoot-but-not-too-far). This is the upper edge of the restricted corridor.
  - More precisely: trajectories that are NOT crash AND have `apo_error >= -delta_za`. This includes all captured sims with apo_error above the undershoot threshold.
- **Envelope B (crash boundary):** p99 of pdyn across all non-crashing trajectories. Upper edge above which everything crashes.
- **Envelope C (overshoot boundary):** p1 of pdyn across all trajectories with `apo_error <= +delta_za`. Lower edge of the restricted corridor.
  - More precisely: trajectories that are captured AND have `apo_error <= +delta_za`.
- **Envelope D (hyperbolic boundary):** p1 of pdyn across all captured trajectories. Lower edge below which everything escapes.

Using p99/p1 instead of true max/min provides robustness against outlier trajectories.

**Energy binning:** Use a shared energy axis computed from the union of all trajectory energy ranges, with 200 bins. Each envelope is computed on this shared axis but may have NaN/gaps in bins where its classification group has no data. Apply `scipy.ndimage.uniform_filter1d` smoothing (window ~5 bins) after percentile extraction to avoid jagged edges from sparse bins. Bins with fewer than 3 contributing trajectories are set to NaN and interpolated from neighbors.

**Empty classification groups:** If a group has zero trajectories (e.g., zero crashes in 10k sims), its envelope is not computed and its fill layer is skipped entirely. Log a note (e.g., "No crashes observed — crash zone not drawn").

#### Phase 2: Bank-Only MC (10k sims, no dispersions)

Same bank angle sweep [0deg, 180deg] but with `dispersion_level="none"`. Among viable captures (same criteria as existing `_viable_capture_mask`: `ecc < 1.0`, `energy < 0.0`, `peri_alt > 0.0`, `dv < 1e10`), select the trajectory that minimizes `final_record[40]` = `|dv1| + |dv2|` (periapsis + apoapsis correction only, excluding inclination correction dv3). This becomes the **nominal constant-bank trajectory**.

**Memory note:** 10k sims with `include_trajectories=True` at ~700 timesteps x 12 floats x 8 bytes = ~670 MB for trajectory data. This is acceptable on modern machines but the `[corridor].n_sims` parameter allows reducing if needed.

### Plot Rendering (`final_report.py` changes)

Replace `_draw_pdyn_zones` with a 4-layer fill approach, painted back-to-front:

1. **Grey `fill_between`** from Envelope A (undershoot boundary) up to y_max — undershoot zone
2. **Red `fill_between`** from Envelope B (crash boundary) up to y_max — crash zone (overpaints grey above crash line)
3. **Grey `fill_between`** from y=0 up to Envelope C (overshoot boundary) — overshoot zone
4. **Red `fill_between`** from y=0 up to Envelope D (hyperbolic boundary) — hyperbolic zone (overpaints grey below hyperbolic line)
5. **Blue MC spaghetti** — 1000 guided trajectories from final evaluation (drawn in the white corridor)
6. **Red line** — nominal constant-bank trajectory (from corridor cache)
7. **Green line** — nominal guided trajectory (min `final_record[41]` (total DV including inclination) among captured trajectories from the 1000-sim guided MC, NOT first-by-index)

Each fill layer is only drawn if its envelope data exists (i.e., the classification group was non-empty). The `fill_between` calls use the shared energy axis; envelope arrays with NaN gaps are handled by matplotlib naturally (gaps in fill).

The layering order ensures: red zones are outermost, grey restricted-corridor bands are between red and white, and the white viable corridor naturally emerges as the unfilled region between Envelopes A and C.

### TOML Configuration

Add `[corridor]` section to mission TOMLs (`configs/missions/mars.toml`, `configs/missions/earth.toml`):

```toml
[corridor]
delta_za = 200.0  # km, apoapsis error tolerance for restricted corridor boundaries
n_sims = 10000    # number of MC sims for corridor boundary computation
```

`corridor.py` reads these from the loaded config. Falls back to `delta_za=200.0` and `n_sims=10000` if the section is absent.

### Cache Format (`.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `envelope_undershoot_energy` | `(B,)` | Energy bin centers for undershoot boundary |
| `envelope_undershoot_pdyn` | `(B,)` | p99 pdyn of trajectories with apo_error >= -delta_za |
| `envelope_crash_energy` | `(B,)` | Energy bin centers for crash boundary |
| `envelope_crash_pdyn` | `(B,)` | p99 pdyn of non-crashing trajectories |
| `envelope_overshoot_energy` | `(B,)` | Energy bin centers for overshoot boundary |
| `envelope_overshoot_pdyn` | `(B,)` | p1 pdyn of trajectories with apo_error <= +delta_za |
| `envelope_hyperbolic_energy` | `(B,)` | Energy bin centers for hyperbolic boundary |
| `envelope_hyperbolic_pdyn` | `(B,)` | p1 pdyn of all captured trajectories |
| `nominal` | `(T, 12)` | Nominal constant-bank trajectory (min |dv1|+|dv2|) |
| `nominal_bank_deg` | `(1,)` | Bank angle of nominal trajectory |
| `nominal_dv` | `(1,)` | |dv1|+|dv2| of nominal (excludes inclination) |
| `nominal_dv_total` | `(1,)` | Total DV including inclination correction |
| `schema_version` | `(1,)` | Cache format version (set to 2; old format is version 1) |
| `target_apoapsis_km` | `(1,)` | From TOML `[flight.target_orbit]` |
| `delta_za_km` | `(1,)` | From TOML `[corridor]` |
| `n_sims` | `(1,)` | MC size used |
| `classification_counts` | `(5,)` | [crash, undershoot, corridor, overshoot, hyperbolic] |

Cached per mission in `training_output/<mission>/corridor_boundaries.npz` (same location as current).

**Backward compatibility:** The cache schema is completely different from the current format. On load, check for `schema_version == 2`; if absent or mismatched, log a warning and recompute. The old `traj_lengths`/`traj_data` packed trajectory arrays and `load_corridor_trajectories()` helper are superseded — the new format stores only the extracted envelopes, not raw trajectories.

### Bug Fixes Included

1. **`_draw_pdyn_zones` midpoint-split hack** replaced with proper 4-layer envelope fill
2. **Guided nominal selection** changed from first-captured-by-index to min-total-DV captured trajectory
3. **PyO3 trajectory docstring** (`results.rs`) corrected for columns 8-11 (actual: energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg; docstring incorrectly says: bank_cmd_deg, bank_actual_deg, g_load, dynamic_pressure_pa)

## Files Modified

- `src/python/aerocapture/training/corridor.py` — Rewrite MC classification, envelope extraction, cache format
- `src/python/aerocapture/training/final_report.py` — Replace `_draw_pdyn_zones`, fix guided nominal selection, update corridor data loading
- `configs/missions/mars.toml` — Add `[corridor]` section
- `configs/missions/earth.toml` — Add `[corridor]` section (if exists)
- `src/rust/aerocapture-py/src/results.rs` — Fix trajectory column docstring (minor)

## Files NOT Modified

- Rust simulator — all required data already available via `final_record` and trajectory arrays
- PyO3 binding logic — no functional changes needed
- Other Python training modules — no interface changes
