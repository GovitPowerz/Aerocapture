# Replace Plotly Reports with Typst PDF Reports

**Date:** 2026-03-24
**Status:** Approved
**Branch:** `feature/plotly_to_typst`

## Goal

Consolidate the current 5 separate report files (3 Plotly HTML + 2 matplotlib PNG) into a single PDF per training run. Drop the Plotly dependency. Use matplotlib + seaborn for charts (SVG output) and Typst for document layout and tables.

## Motivation

- Single PDF combining all reports is easier to archive and analyse
- Plotly is a heavy dependency that provides interactivity we don't need
- Seaborn provides clean, publication-style aesthetics with minimal code
- Typst compiles in milliseconds and produces professional layouts with a readable template language

## Report Structure

### Cover Page

- Scheme name, mission, date, config hash
- Key metrics: best cost, capture rate, total generations, n_sims for final eval

### Part 1: Training Convergence

| # | Panel | Chart Type | Width |
|---|-------|-----------|-------|
| 1 | Convergence (best/mean/worst cost, log y) | Line plot | Full |
| 2 | Capture Rate + Constraint Violation Rate | Dual-axis line | Full |
| 3 | Population Diversity vs Best Cost | Dual-axis line | Half |
| 4 | Cost Distribution (box plots, sampled generations) | Box plot | Half |
| 5 | Parameter Evolution | Multi-line | Full |
| 6 | Seed Pool Evolution (conditional: adaptive seeds only) | Dual-axis line | Full |

Resume markers (vertical dashed lines) on panels 1-3, 5-6.

### Part 2: Mission Performance

| # | Panel | Chart Type | Width |
|---|-------|-----------|-------|
| 7 | Energy vs Pdyn (corridor zones + 3 nominals + MC spaghetti) | Fill + line | Full |
| 8 | Energy vs Inclination (MC spaghetti + envelope) | Fill + line | Half |
| 9 | Energy vs Bank Angle (MC spaghetti + envelope) | Fill + line | Half |
| 10 | Altitude vs Time (MC spaghetti, best-case highlighted) | Line | Full |
| 11 | Heat Flux vs Time (MC spaghetti + limit line) | Line | Half |
| 12 | G-Load vs Time (MC spaghetti + limit line) | Line | Half |
| 13 | Bank Angle vs Time (MC spaghetti) | Line | Full |
| 14 | Nav Filter: Density Ratio estimated/truth (MC spaghetti) | Line | Full |
| 15 | Total DV Distribution (histogram + CDF + percentile lines) | Histogram | Half |
| 16 | Individual Burns dv1/dv2/dv3 overlaid | Histogram | Half |
| 17 | Entry Conditions (V vs FPA, captured/hyperbolic markers) | Scatter | Half |
| 18 | Exit Conditions (V vs FPA, size proportional to DV) | Scatter | Half |
| 19 | Performance Summary Table | Native Typst table | Full |
| 20 | Dispersion Correlation Grid (~24 scatter subplots with R^2) | Scatter grid | Full page |

### Part 3: Cross-Scheme Comparison (separate PDF)

| # | Panel | Chart Type |
|---|-------|-----------|
| 1 | Cross-Scheme Convergence | Multi-line |
| 2 | Final Metrics Table | Native Typst table |

## Panels Dropped vs Current Reports

| Dropped Panel | Reason |
|---------------|--------|
| MC Seed Trace | Just a sequence of integers, never actionable |
| Correction Cost histogram (corridor PNG) | Redundant with Total DV Distribution (panel 15) |
| DV vs Orbital Error scatter | Individual error histograms + DV histogram already tell this story |
| Summary text panel | Absorbed into cover page metadata |

## New Panels Added

| New Panel | Rationale |
|-----------|-----------|
| Altitude vs Time (10) | Most intuitive view of trajectory shape, previously absent |
| Heat Flux vs Time (11) | See when thermal constraints bind, not just peak values |
| G-Load vs Time (12) | See when acceleration constraints bind |
| Bank Angle vs Time (13) | Control profile in time domain, complements energy-domain view |
| Constraint Violation Rate (in panel 2) | Track if GA learns to respect constraints over generations |
| Nav Filter Density Ratio (14) | Verify density filter performance (estimated vs truth) |

## Architecture

### Pipeline

```
report.generate_report(scheme_dir, toml_path)
  |
  +-- 1. Load data
  |     +-- JSONL logs -> training metrics (pd.DataFrame)
  |     +-- run_mc(include_trajectories=True) -> final eval data
  |     +-- corridor_boundaries.npz (if available)
  |     +-- TOML config -> metadata + limits (g-load, heat flux)
  |
  +-- 2. Generate charts (matplotlib + seaborn)
  |     +-- charts.py -> writes SVGs to temp dir
  |     +-- data.py -> writes metadata.json + summary_table.json to temp dir
  |
  +-- 3. Compile PDF
  |     +-- typst compile report.typ --input dir=<temp_dir> -> report.pdf
  |
  +-- 4. Cleanup temp dir (unless --keep-artifacts)
```

### File Layout

```
src/python/aerocapture/training/
  report.py          <- REWRITE: orchestrator (load data, call charts, invoke typst)
  charts.py          <- NEW: all matplotlib/seaborn chart functions, one per panel
  final_report.py    <- DELETE (absorbed into report.py + charts.py)
  plot_comparison.py <- REWRITE: comparison PDF via same pipeline

src/typst/
  report.typ         <- NEW: main template (cover + Part 1 + Part 2)
  comparison.typ     <- NEW: cross-scheme comparison template
  lib.typ            <- NEW: shared helpers (page style, colors, heading format)
```

### Chart Module Design (`charts.py`)

- One function per panel, each takes data (DataFrame/array) and output path, writes an SVG
- Seaborn theme set once at module level: `sns.set_theme(style="whitegrid", palette="muted")`
- Specific color overrides for trajectory types: captured (blue), hyperbolic (red), nominals (red/orange/green for piecewise-constant/undispersed/best-case)
- Spaghetti plot opacity scales with `1/sqrt(n_trajectories)` for readability

### Typst Template Design

- Receives a directory path as input via `--input dir=<path>`
- Loads SVGs with `image()`, loads JSON data with `json()`
- No string templating or Jinja -- Typst has native data loading
- Summary table (panel 19) rendered natively in Typst for clean formatting
- Cover page metadata from `metadata.json`
- Half-width panels placed side-by-side using Typst `grid()` layout

## CLI Interface

### Single-scheme report

```bash
# Full report (training convergence + mission performance)
uv run python -m aerocapture.training.report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml

# Debug: keep SVGs and JSON after compile
uv run python -m aerocapture.training.report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --keep-artifacts
```

**Output:** `training_output/equilibrium_glide/report.pdf`

### Cross-scheme comparison

```bash
uv run python -m aerocapture.training.report \
    --compare training_output/ \
    --schemes equilibrium_glide energy_controller pred_guid
```

**Output:** `training_output/comparison_report.pdf`

### Skip report during training

```bash
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20 --skip-report
```

### Integration with `train.py`

End-of-training calls:

```python
# Replaces generate_single_report() + generate_final_report()
if not args.skip_report:
    generate_report(scheme_dir, toml_path)
```

`--skip-final-report` renamed to `--skip-report`.

## Configuration

- `n_sims` and `seed` for final MC evaluation read from TOML config (not CLI args)
- Corridor data path derived from training output directory structure
- Config hash computed from TOML for cover page metadata

## Dependencies

### Added

- `seaborn` — core dependency in `pyproject.toml`
- `typst` CLI — external binary, checked at report generation time

### Removed

- `plotly` — removed from `pyproject.toml`

### Typst installation

Checked at report time with a clear error:

> `typst not found -- install with 'brew install typst' or 'cargo install typst-cli'. Skipping PDF generation.`

Training results are still saved; only the PDF is skipped.

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Missing `typst` CLI | Warning printed, PDF skipped, training results saved |
| Missing corridor data | Corridor panels show placeholder text in PDF |
| Missing trajectories | Time-domain panels (10-14) omitted with note; histograms/scatters still render |
| No captures (n_captured=0) | Orbital error panels show "No captured trajectories" annotation |

## Files Deleted

- `final_report.py` -- logic moves to `report.py` + `charts.py`
- All Plotly imports and `include_plotlyjs` machinery in `report.py`

## Files Unchanged

- `logger.py`, `metrics.py`, `display.py` -- training instrumentation untouched
- `corridor.py` -- still generates `.npz` during piecewise_constant training
- `compare_guidance.py` -- still runs MC comparisons and saves JSON
