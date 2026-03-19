# Piecewise-Constant Bank Guidance & Corridor Optimization

**Date:** 2026-03-19
**Status:** Approved

## Problem

The current corridor visualization and reference trajectory have two weaknesses:

1. **Legacy reference trajectory** — Energy Controller, PredGuid, FNPAG, and FTC all track a pre-recorded constant-bank trajectory (`msr_aller.dat`) from the original Fortran simulator. This reference was hand-picked, not optimized, and is static across training runs.

2. **Narrow restricted corridor** — The ±δZa restricted corridor is computed from constant-bank trajectories that span only ~2-3° of bank angle, producing a corridor ~0.04 kPa wide in the energy-pdyn plane — nearly invisible on plots.

## Solution

Add a new `piecewise_constant` guidance scheme: 10 constant bank angle segments uniformly distributed over orbital energy, GA-optimized for minimum total DV (including inclination correction). This serves two purposes:

1. **Produces an optimized reference trajectory** for other guidance schemes to track.
2. **Builds rich corridor envelopes** from the GA population history across all generations.

## Design

### 1. New Guidance Scheme — `piecewise_constant`

**Concept:** The orbital energy range is divided into 10 uniform segments. Each segment has a constant bank angle in [-180°, +180°]. Negative bank angles implicitly encode roll reversals without any lateral guidance logic. The trajectory is determined by the 10 bank values — open-loop guidance, no navigation feedback, but with pilot dynamics for realistic bank rate transitions.

**Segment ordering:** Segment 0 covers the highest energy (entry, near `energy_range[1]`) and segment 9 covers the lowest energy (deepest capture, near `energy_range[0]`). Energy decreases during atmospheric flight.

**Out-of-range clamping:** If current energy falls outside `energy_range`, clamp to the first segment (above range, early entry) or last segment (below range, deep capture).

**Rust side** (`config.rs`, `runner.rs`):

The `piecewise_constant` scheme does NOT use `reference_trajectory = true`. Instead, it runs through the normal GNC path but with its own guidance step that replaces the FTC/EqGlide/etc. dispatch. This ensures pilot dynamics (bank rate limiting) are active, making transitions between segments physically realistic.

In the guidance dispatch:

1. Compute current orbital energy using `total_energy()` (absolute/inertial velocity via `to_absolute_cartesian`, same as everywhere else in the codebase).
2. Determine which of the 10 energy segments it falls in (clamp if outside range).
3. Return the segment's bank angle as `bank_angle_commanded` (in radians, sign included).

The pilot model then rate-limits the actual bank angle change (max 15°/s), so a segment transition from +70° to -65° takes ~9 seconds to execute. This means the GA naturally avoids profiles with too many rapid reversals, and the resulting reference trajectory is more representative of what a real vehicle would fly.

Navigation runs but its output is only used for the energy computation (state estimation). No lateral guidance / roll reversal logic — the bank angle sign comes directly from the piecewise-constant profile.

TOML config:
```toml
[guidance]
type = "piecewise_constant"
bank_angles = [60.0, 65.0, 70.0, -65.0, -60.0, 55.0, 50.0, 45.0, -40.0, 35.0]
energy_range = [-6.0, 5.0]  # MJ/kg, required for piecewise_constant
```

`energy_range` is **required** in the TOML for `piecewise_constant` (no auto-computation — removes ambiguity about velocity convention and entry conditions). The Python evaluate path sets it from the training config.

**Python side** (`param_spaces.py`):

```python
"piecewise_constant": [
    ParamSpec("bank_angle_0", -180.0, 180.0, 65.0),
    ParamSpec("bank_angle_1", -180.0, 180.0, 65.0),
    ...
    ParamSpec("bank_angle_9", -180.0, 180.0, 65.0),
]
```

10 parameters, each in [-180°, +180°]. Same cost function as other schemes: total DV (including inclination) + TOML-configurable g-load/heat flux soft penalties.

**Evaluate** (`evaluate.py`): The `piecewise_constant` scheme needs a special branch in `evaluate_chromosome` (analogous to the NN branch) because the chromosome encodes a **list** (`guidance.bank_angles`), not flat scalar overrides via `GUIDANCE_TOML_SECTIONS`. The branch decodes the 10 chromosome values into a list and passes them as a PyO3 override: `{"guidance.bank_angles": [v0, v1, ..., v9], "guidance.energy_range": [e_lo, e_hi]}`. The energy range is read once from the training TOML at startup.

### 2. Training Output Artifacts

When `--guidance piecewise_constant` training completes:

**Standard per-scheme outputs** (same as other schemes):
- `training_output/piecewise_constant/best_params.json` — the 10 bank angles
- `training_output/piecewise_constant/checkpoint.pkl`
- `training_output/piecewise_constant/training_log.jsonl`
- `training_output/piecewise_constant/convergence_report.html`
- `training_output/piecewise_constant/final_report.html`
- `training_output/piecewise_constant/final_report_corridors.png`

**Mission-level outputs** (shared across schemes):
- `training_output/<mission>/ref_trajectory.dat` — best individual's trajectory in the legacy 7-column format.
- `training_output/<mission>/corridor_boundaries.npz` — 4 envelopes (crash, restricted upper, restricted lower, capture) built incrementally from all GA generations, plus the best individual's trajectory as the nominal.

**Generating `ref_trajectory.dat`:** Re-run the best individual with `include_trajectories=True` via PyO3. The 12-column trajectory format from `run_for_api` does NOT contain all 7 required columns (missing `radial_vel` and `altitude_rate`). Instead, run the best individual via the **CLI binary** with CSV output enabled, which produces the full 24-column photo CSV. Extract the 7 columns from the 24-column photo format:

| .dat column | Photo column | Field |
|-------------|-------------|-------|
| 0 | 18 | energy (J/kg, already in J/kg in photo) |
| 1 | 19 | dynamic_pressure (Pa) |
| 2 | 15 | radial_velocity (m/s) |
| 3 | 15 | radial_velocity (m/s) (altitude_rate = radial_vel for small FPA) |
| 4 | 9 | inclination (deg → rad) |
| 5 | 0 | time (s) |
| 6 | 14 | cos(bank_angle) (compute from bank_angle deg) |

Alternatively, extend the 12-column PyO3 trajectory to include radial_velocity (it's already computed in `build_photo_values` as `p[15]`). This is the cleaner approach — add `p[15]` as column 12 in a future 13-column format, or compute `radial_vel = V * sin(FPA)` from existing columns 3 and 4 in Python.

### 3. Reference Trajectory Dependency

Schemes that track the reference trajectory require it to exist before training:

```python
REQUIRES_REF_TRAJECTORY = {"energy_controller", "pred_guid", "fnpag", "ftc"}
```

At `train.py` startup, if `guidance_type in REQUIRES_REF_TRAJECTORY`:
- Derive mission name from the base TOML path (existing logic in `train.py` lines 840-843: first base config containing "missions/" in its path, stem extracted)
- Check for `training_output/<mission>/ref_trajectory.dat`
- If missing, exit with explicit error:
  ```
  ERROR: No reference trajectory found for mission 'mars'.
  Run piecewise_constant training first:
    uv run python -m aerocapture.training.train --guidance piecewise_constant --toml <config>
  ```

The ref trajectory path is passed to the Rust sim via TOML override: `data.reference_trajectory = "training_output/<mission>/ref_trajectory.dat"`.

Schemes that don't need a reference (`equilibrium_glide`, `neural_network`, `piecewise_constant`) proceed without this check.

### 4. Incremental Corridor Envelope from GA History

During piecewise_constant training, each generation's evaluated trajectories feed an incremental envelope accumulator.

**`CorridorAccumulator` class** (in `corridor.py`):

```python
class CorridorAccumulator:
    energy_bins: np.ndarray          # (B,) fixed bin centers, set at init
    crash_max_pdyn: np.ndarray       # (B,) max pdyn of non-crash trajectories; area ABOVE = crash zone
    restricted_max_pdyn: np.ndarray  # (B,) max pdyn of corridor-classified (|apo_err| < delta_za_restricted)
    restricted_min_pdyn: np.ndarray  # (B,) min pdyn of corridor-classified (|apo_err| < delta_za_restricted)
    capture_min_pdyn: np.ndarray     # (B,) min pdyn of all captured; area BELOW = hyperbolic zone
```

Note on naming: `crash_max_pdyn` is the max pdyn of **non-crash** trajectories (i.e., the ceiling of the safe region). Above this value, only crash trajectories exist. Similarly, `capture_min_pdyn` is the min pdyn of **captured** trajectories (the floor of the capture region). Below this, only hyperbolic/escape trajectories exist.

**Integration point in `train.py`:** After the offspring evaluation loop (before `save_checkpoint`), if `guidance_type == "piecewise_constant"`:
1. Decode the entire current population's chromosomes into override dicts.
2. Call `run_batch(toml_path, overrides_list, include_trajectories=True)` for the full population.
3. Classify trajectories using `classify_trajectories(final_records, delta_za=delta_za_restricted)`.
4. Call `accumulator.update(trajectories, labels)` to update running max/min envelopes.
5. Discard raw trajectory data.

This replaces the per-chromosome `evaluate_chromosome` calls for this scheme — the batch call serves both fitness evaluation and corridor accumulation.

**Energy range bootstrap:** Fixed at initialization from the `energy_range` in the training TOML config. 200 bins.

**Smoothing:** Applied once at save time (not during accumulation) using `uniform_filter1d`.

**Checkpoint integration:** The 4 envelope arrays are serialized into the checkpoint `.npz` file (alongside the existing population/halloffame data). On resume, the accumulator is restored from the checkpoint and continues accumulating.

### 5. Corridor Save

After piecewise_constant training completes, the accumulated envelopes and the best individual's trajectory are saved to `training_output/<mission>/corridor_boundaries.npz` with **schema version 4** (bumped from v3 to accommodate the new restricted envelope arrays). No separate Phase 2 MC sweep is needed — the GA population history provides all corridor data.

### 6. Visualization — 4-Layer Fill

The corridor panel (a) in `final_report.py` uses 4-layer fills:

**Fill layers** (back to front):
1. Grey (`#BDBDBD`, alpha=0.5) above `restricted_max_pdyn` — transition zone between restricted corridor and crash boundary
2. Red (`#E57373`, alpha=0.5) above `crash_max_pdyn` — crash zone (overpaints grey)
3. Grey (`#BDBDBD`, alpha=0.5) below `restricted_min_pdyn` — transition zone between restricted corridor and capture boundary
4. Red (`#E57373`, alpha=0.5) below `capture_min_pdyn` — hyperbolic exit zone (overpaints grey)

**On top:**
5. Blue spaghetti — MC guided trajectories (final evaluation)
6. Red line — nominal piecewise-constant (best GA individual)
7. Green line — nominal guidance scheme (min-DV from final eval)

White corridor between layers 1 and 3 = the restricted corridor where captures with |apo_err| < δZa_restricted happen. The grey transition zones show where capture is possible but outside the restricted corridor bounds.

**TOML config** (`[corridor]` section in mission TOML):
```toml
[corridor]
delta_za_restricted = 200.0  # km, for restricted corridor envelopes (GA)
```

**Fallback behavior:**
- Corridor `.npz` exists with schema v4 → full 4-layer visualization
- Corridor `.npz` missing or old schema → spaghetti only, no zones

### 7. Changes Summary

| Component | Change |
|-----------|--------|
| **Rust** `config.rs` | Parse `guidance.bank_angles` array + `guidance.energy_range` |
| **Rust** `runner.rs` | New guidance dispatch for piecewise_constant: energy → segment → bank_angle_commanded; runs through normal GNC path with pilot dynamics |
| **Python** `param_spaces.py` | `piecewise_constant`: 10 params [-180, 180]; `REQUIRES_REF_TRAJECTORY` set |
| **Python** `evaluate.py` | Special branch for `piecewise_constant`: chromosome → `guidance.bank_angles` list override (analogous to NN branch) |
| **Python** `corridor.py` | `CorridorAccumulator` class with `update()` and checkpoint serialization; remove Phase 2 constant-bank sweep |
| **Python** `train.py` | Ref trajectory check at startup; during piecewise_constant training: batch eval + corridor accumulation per generation; at end: save ref trajectory + corridor `.npz` |
| **Python** `final_report.py` | 4-layer fill (grey/red/grey/red) using restricted + full envelopes |
| **TOML** `mars.toml` | Add `delta_za_restricted = 200.0` to `[corridor]` |
| **TOML** new config | `configs/training/msr_aller_piecewise_constant_train.toml` |
| **Tests** | Piecewise_constant guidance unit test (Rust: segment lookup, clamping, sign), `CorridorAccumulator` tests (Python: update, checkpoint round-trip, empty input), ref trajectory check test, schema v4 cache test |

**Files NOT modified:**
- Other guidance scheme implementations — they read whatever ref trajectory is loaded
- PyO3 bindings — no API change
- Existing training configs — unchanged
