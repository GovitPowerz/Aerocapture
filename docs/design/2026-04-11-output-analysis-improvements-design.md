# Output & Analysis Improvements Design

**Date**: 2026-04-11
**Branch**: `feature/output-analysis-improvements`
**Status**: Approved

## Overview

Add Parquet output for large MC campaigns, embed full config metadata for reproducibility, and fix dispersion scatter charts to use consistent three-way trajectory classification.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Binary format | Parquet (not HDF5) | Columnar queries, lighter dep (pyarrow), native Pandas support |
| Where Parquet is written | Python-side only | MC campaigns always driven from Python; keeps Rust lean |
| Dispersions in CSV | No | CSV stays at 39 columns for readability; Parquet has 65 (39 + 26 dispersions) |
| Config metadata | Full resolved TOML in Parquet schema metadata | Single-file reproducibility without needing original config files |
| Dispersion chart markers | Three-way (blue/orange/red) | Consistency with all other end-condition plots |

## 1. Parquet Output Module

**New file**: `src/python/aerocapture/training/parquet_output.py`

### write_parquet()

```python
def write_parquet(
    path: str | Path,
    batch_results: BatchResults,
    config: dict,
    toml_path: str | None = None,
) -> None
```

**Column schema** (65 total):
- Columns 0-38: Existing 39 final-record fields (same names as CSV headers)
- Columns 39-64: 26 dispersion fields using names from `DISPERSION_COLUMNS` in `sensitivity.py`

**Parquet schema-level metadata**:
- `aerocapture.config`: JSON-serialized resolved TOML config (after base inheritance)
- `aerocapture.toml_path`: original config file path
- `aerocapture.timestamp`: ISO 8601 write time
- `aerocapture.guidance_scheme`: extracted from config
- `aerocapture.n_sims`: number of runs

### read_parquet()

```python
def read_parquet(path: str | Path) -> tuple[pd.DataFrame, dict]:
```

Returns DataFrame (65 columns) and metadata dict (with `config` deserialized back to dict). Keeps read/write symmetry in one place.

### Dependency

`pyarrow` added to main dependencies in `pyproject.toml`.

## 2. Integration Points

**Training pipeline** (`train.py`): After the final evaluation (`--final-n-sims` run), auto-write `training_output/<scheme>/final_eval.parquet`. No new CLI flag -- writes automatically if `pyarrow` is installed, degrades gracefully if not.

**Compare guidance** (`compare_guidance.py`): After running all schemes, write one Parquet per scheme to the comparison output directory.

**Report** (`report.py`): When running standalone MC re-evaluation, write Parquet alongside the PDF.

**Manual use**: Users can call `write_parquet()` directly after any `run_mc()` / `run_batch()` / `run_with_draws()` call.

**Not integrated**: Rust CLI binary stays CSV-only.

## 3. Dispersion Chart Fix

**File**: `src/python/aerocapture/training/charts.py`, function `chart_dispersion_grid()`

**Current**: All points plotted uniformly (single color scatter + regression line).

**New**: Three-way classification per subplot:
- Blue dots: captured + constraints OK (`TRAJ_OK`)
- Orange dots: captured + constraint violation (`TRAJ_CONSTRAINED`)
- Red crosses (`marker='x'`): crash/hyperbolic/timeout (`TRAJ_FAILED`)

Implementation:
- Reuse existing `classify_trajectories()` for per-run classification
- Plot each class as a separate scatter call with appropriate color/marker
- Regression line computed on captured points only (blue + orange) -- virtual DV from crash/hyperbolic would distort the fit
- Single legend (not per subplot)

## 4. Non-Goals

- No Rust changes
- No CSV format changes (stays at 39 columns)
- No HDF5 support
- No new CLI flags
- No changes to chart functions other than `chart_dispersion_grid()`
- No Parquet for single-sim `run()` results (batch/MC only)

## 5. Tests

- `write_parquet` / `read_parquet` roundtrip: verify schema (65 columns), metadata keys, column names, data integrity
- Dispersion grid classification: mock data, verify marker types per trajectory class
- Graceful degradation when `pyarrow` is not installed

## 6. Smart Commit

Final step: invoke `smart-commit` skill taking the whole branch into account.
