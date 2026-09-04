# Advanced Sampling Methods & Sensitivity Analysis

**Date:** 2026-04-05
**Branch:** `feature/advanced-sampling-sensitivity`
**Approach:** Hybrid -- LHS + Sobol sampling in Rust, sensitivity analysis (SALib) in Python

## Problem

Current Monte Carlo uses plain pseudorandom draws (Gaussian + Uniform) in `dispersions.rs`. This provides no space-filling guarantees -- samples can cluster, leaving gaps in the 26-dimensional dispersion space. Typical LHS achieves equivalent coverage with ~10x fewer samples. Additionally, there is no built-in way to identify which of the 26 dispersion parameters dominate DV cost variance.

## Goals

1. **Better coverage with fewer runs** -- LHS and Sobol quasi-random sampling in Rust's `generate_draws()`, selectable via TOML config.
2. **Identify dominant uncertainties** -- Morris screening + Sobol variance decomposition in Python, using DV cost as the scalar response metric.

## Non-Goals

- Importance sampling for rare-event probability estimation (future work).
- Modifying the adaptive seed pool or seed rotation mechanisms (they evaluate 1 sim per seed; stratification needs N>1).
- Changing the OU density perturbation process (time-varying, driven per-timestep, not part of the 26 static draws).

---

## Part 1: Rust Sampling Infrastructure

### TOML Configuration

```toml
[monte_carlo]
seed = 42
sampling = "random"  # "random" | "lhs" | "sobol" (default: "random")
```

Absent key defaults to `"random"` -- all existing configs work unchanged.

### Core Refactor: Two-Stage Draw Generation

Refactor `generate_draws()` into two stages:

**Stage 1 -- Unit sample generation:** Produce N points in [0,1]^26 using the selected method.

- `Random`: current RNG-based approach (each dimension sampled independently via `StdRng`).
- `Lhs`: stratified sampling -- divide [0,1] into N equal strata per dimension, random permutation within each. For sample i in dimension d: `(permutation_d[i] + U(0,1)) / N`. Uses seeded `StdRng` for permutations and intra-stratum jitter.
- `Sobol`: Owen-scrambled quasi-random via `sobol_burley::sample(i as u32, d as u32, seed as u32)`. Returns f32 in [0,1] per dimension. Different seeds produce independent scrambled sequences.

**Stage 2 -- Distribution transform:** Map each [0,1] value to the target distribution per dimension.

- Gaussian dimensions (initial state: 6, navigation: 7, nav filter: 1 = 14 total): inverse normal CDF (erfinv-based), then scale by sigma. `value = erfinv(2 * u - 1) * sqrt(2) * sigma`.
- Uniform dimensions (atmosphere: 1, aero: 3, mass: 1, vehicle: 2, pilot: 3, wind: 2 = 12 total): linear scale. `value = (2 * u - 1) * sigma`.

This separation is clean because all three methods produce unit-uniform samples, and the distribution transform logic consolidates what is currently split across `Normal::sample()` and `Uniform::sample()` calls.

### New Rust Types

```rust
#[derive(Debug, Clone, Copy, Default)]
enum SamplingMethod {
    #[default]
    Random,
    Lhs,
    Sobol,
}
```

Added as a field on `DispersionConfig`, parsed from the `sampling` string in `[monte_carlo]`.

### LHS Implementation

~50 lines in `dispersions.rs`. For N samples, 26 dimensions:

1. For each dimension d, create a permutation of [0, 1, ..., N-1] via Fisher-Yates shuffle using the seeded RNG.
2. For each sample i, dimension d: `u = (perm_d[i] as f64 + rng.gen::<f64>()) / N as f64`.
3. Produces an (N, 26) matrix of values in [0, 1].

### Sobol Implementation

~20 lines in `dispersions.rs`. For N samples, 26 dimensions:

1. For each sample i, dimension d: `u = sobol_burley::sample(i as u32, d as u32, seed as u32) as f64`.
2. Owen scrambling is automatic (seed controls the scramble).
3. Maximum 65,536 samples (2^16 limit of sobol_burley). Training uses ~1000, final evaluations rarely exceed 10,000 -- well within bounds. Rust returns an error at parse time if `sampling = "sobol"` and `n_sims > 65536`.

### Dependency

`sobol_burley = "0.5"` in `src/rust/Cargo.toml`. Lightweight crate, no transitive dependencies. Provides Owen-scrambled Sobol sequences up to 256 dimensions.

---

## Part 2: PyO3 API Extension

### New Function: `run_with_draws()`

Sensitivity analysis requires SALib to control exact dispersion values (Saltelli cross-sampling matrix). The current API only accepts seeds. New PyO3 function:

```python
aerocapture_rs.run_with_draws(
    toml_path: str,
    draws: np.ndarray,              # (N, 26) float64 -- raw dispersion draws
    overrides: dict | None = None,
    include_trajectories: bool = False,
    sim_timeout_secs: float | None = None,
) -> BatchResults
```

Each row maps 1:1 to `DispersionDraw`'s 26 fields in declaration order. Rust bypasses `generate_draws()` and constructs `DispersionDraw` structs directly from the array. Rayon parallelism same as `run_batch()`.

### Column Order Contract

The 26 columns follow `DispersionDraw` field order (same as `.dispersions` output):

```python
DISPERSION_COLUMNS = [
    "altitude", "longitude", "latitude", "velocity", "flight_path", "azimuth",
    "density", "drag_coeff", "lift_coeff", "incidence",
    "nav_altitude", "nav_longitude", "nav_latitude", "nav_velocity",
    "nav_flight_path", "nav_azimuth", "nav_drag_accel",
    "mass", "ref_area", "max_bank_rate",
    "pilot_tau", "pilot_damping", "pilot_frequency",
    "filter_gain", "wind_scale", "wind_direction_bias",
]
```

Documented as a constant in both Rust (`dispersions.rs`) and Python (`sensitivity.py`).

### Why a Dedicated Function

Dispersion draws are runtime-generated, not TOML values. Serializing 26 floats per row into dot-path override strings and re-parsing them would be both awkward and slow for the 50K+ rows typical of Saltelli sampling.

---

## Part 3: Python Sensitivity Analysis

### Module: `src/python/aerocapture/training/sensitivity.py`

Standalone CLI + importable module for variance-based sensitivity analysis using SALib.

### Two-Stage Workflow: Morris then Sobol

**Stage 1 -- Morris screening (cheap):**
- SALib generates Morris trajectories for all 26 dispersion parameters.
- Run sims via `run_with_draws()`.
- Compute mu_star (importance ranking) and sigma (nonlinearity/interaction indicator) per parameter.
- Cost: N * (k+1) = 1000 * 27 = 27,000 runs.

**Stage 2 -- Sobol decomposition (expensive, focused):**
- Run only on the top-k influential parameters identified by Morris (default k=10).
- SALib generates Saltelli matrix: N * (2k+2) = 1024 * 22 = 22,528 runs.
- Computes first-order (S1), total-order (ST), and second-order (S2) indices with bootstrap confidence intervals.
- S1: fraction of variance from parameter alone. ST: fraction including all interactions. S2: pairwise interaction strength.

Non-influential parameters (below Morris threshold) are fixed at their nominal values (zero dispersion) during Sobol analysis.

### Problem Definition Builder

Reads `[monte_carlo.*]` sections from the training TOML and maps to SALib's problem dict:

```python
problem = {
    'num_vars': 26,
    'names': DISPERSION_COLUMNS,
    'bounds': [...],   # from TOML sigma values per domain/level
    'dists': [...]     # 'norm' for Gaussian domains, 'unif' for Uniform
}
```

Gaussian domains use `bounds = [0.0, sigma]` with `dists = 'norm'` (SALib interprets as mean, std). Uniform domains use `bounds = [-sigma, sigma]` with `dists = 'unif'`.

### CLI

```bash
# Full pipeline (Morris screen + Sobol on top parameters)
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-n 1000 --sobol-n 1024 --top-k 10

# Morris screening only (quick)
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-only --morris-n 1000

# Sobol only on all 26 params (expensive, no screening)
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --sobol-only --sobol-n 1024
```

Loads best parameters from `training_output/<scheme>/best_params.json` when available, so analysis runs against the optimized guidance configuration.

### Output

**Charts** (SVG, added to `charts.py`):
- Morris mu_star vs sigma scatter plot (importance vs nonlinearity)
- Sobol S1/ST grouped bar chart with confidence intervals
- S2 interaction heatmap
- Convergence plot (indices vs N to verify stability)

**Data:** `sensitivity_results.json` with all indices, confidence intervals, and parameter rankings.

**Directory:** `training_output/<scheme>/sensitivity/`

### Report Integration

Optional `--sensitivity` flag on `report.py`:

```bash
uv run python -m aerocapture.training.report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --sensitivity
```

Adds "Part 3: Sensitivity Analysis" to the PDF with Morris and Sobol charts. Only renders if pre-computed sensitivity data exists in the output directory -- does not trigger a fresh campaign during report generation.

---

## Part 4: Integration & Compatibility

### Where Advanced Sampling Helps

- `run_mc()` with n_sims > 1: final evaluation, compare_guidance, report MC campaigns.
- Per-generation GA evaluation when using fixed seeds with n_sims > 1.

### Where It Does Not Help

- Adaptive seed pool: evaluates 1 sim per seed. Stratification is a batch property.
- Seed rotation: same -- 1 sim per seed per generation.

No changes to seed pool or seed rotation logic.

### Backward Compatibility

- `sampling` key absent or `"random"`: identical behavior to current code.
- All existing TOML configs work unchanged.
- `run()`, `run_mc()`, `run_batch()` PyO3 APIs unchanged.
- `run_with_draws()` is purely additive.
- `DispersionDraw` struct layout and `.dispersions` output array unchanged.

### New Dependencies

**Rust:** `sobol_burley = "0.5"` (lightweight, no transitive deps).

**Python:** `SALib >= 1.5` in `pyproject.toml` main dependencies (pulls in numpy/scipy/matplotlib which we already have; adds `multiprocess`).

### Files Changed/Added

| File | Change |
|------|--------|
| `src/rust/src/data/dispersions.rs` | `SamplingMethod` enum, two-stage `generate_draws()`, LHS impl, Sobol impl |
| `src/rust/src/config.rs` | Parse `sampling` from `[monte_carlo]` |
| `src/rust/Cargo.toml` | Add `sobol_burley = "0.5"` |
| `src/rust/aerocapture-py/src/lib.rs` | Add `run_with_draws()` |
| `src/rust/aerocapture-py/src/batch.rs` | Support pre-computed draws path |
| `src/python/aerocapture/training/sensitivity.py` | New: Morris + Sobol pipeline, CLI |
| `src/python/aerocapture/training/charts.py` | New chart functions (4 charts) |
| `src/python/aerocapture/training/report.py` | Optional Part 3 sensitivity section |
| `src/typst/report.typ` | Sensitivity charts layout |
| `pyproject.toml` | Add `SALib` dependency |

### Testing

**Rust:**
- LHS: verify stratification (each stratum has exactly 1 sample per dim), bounds [0,1], deterministic given seed.
- Sobol: verify low discrepancy (star discrepancy < random for same N), bounds [0,1], deterministic given seed.
- Inverse CDF transform: Gaussian and Uniform paths produce correct distributions (statistical tests on large N).
- Proptest: all sampling methods produce finite, in-range draws for arbitrary seeds and n_sims.
- Integration: `run_for_api()` with each sampling method produces valid sim results.
- `run_with_draws()` path: pre-computed draws produce identical results to equivalent `generate_draws()` output.

**Python:**
- Problem definition builder: TOML config -> SALib problem dict (bounds, distributions, names).
- Draw column mapping: SALib matrix rows -> correct DispersionDraw field order.
- Integration: Morris + Sobol on small N with test config, verify index shapes and ranges [0,1].
- Chart tests: SVG output for each new chart function.
- Report: `--sensitivity` flag renders Part 3 when data exists, skips when absent.
