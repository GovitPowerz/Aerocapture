# Plotly to Typst PDF Reports — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 5 separate Plotly HTML / matplotlib PNG report files with a single PDF per training run, using matplotlib+seaborn for charts and Typst for layout.

**Architecture:** Python orchestrator loads training JSONL + MC evaluation data, generates SVG charts via matplotlib/seaborn, writes metadata JSON, then invokes `typst compile` on a `.typ` template that composes everything into a PDF. The Rust simulator is extended with 3 new trajectory columns (heat flux, g-load, nav density ratio) to support new time-domain panels.

**Tech Stack:** matplotlib, seaborn (charts → SVG), Typst (layout → PDF), Rust/PyO3 (trajectory extension)

**Spec:** `docs/superpowers/specs/2026-03-24-plotly-to-typst-pdf-reports-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/python/aerocapture/training/charts.py` | All matplotlib/seaborn chart functions (one per panel), seaborn theme config |
| `src/typst/report.typ` | Main Typst template: cover page + Part 1 (training) + Part 2 (performance) |
| `src/typst/comparison.typ` | Cross-scheme comparison Typst template |
| `src/typst/lib.typ` | Shared Typst helpers: page style, heading format, color constants |
| `tests/test_charts.py` | Unit tests for chart generation functions |
| `tests/test_report_pdf.py` | Integration tests for PDF report generation pipeline |

### Modified Files

| File | Changes |
|------|---------|
| `src/rust/src/simulation/runner.rs` | Extend photo array 24→28 cols, trajectory 12→16 cols (heat flux, g-load, density ratio, truth density) |
| `src/rust/aerocapture-py/src/results.rs` | Update trajectory column count in PyO3 numpy exposure |
| `src/python/aerocapture/training/report.py` | Full rewrite: orchestrator that loads data, calls charts.py, invokes typst |
| `src/python/aerocapture/training/plot_comparison.py` | Rewrite: comparison PDF via same seaborn+Typst pipeline |
| `src/python/aerocapture/training/train.py` | Replace report calls, rename --skip-final-report → --skip-report (keep alias) |
| `src/python/aerocapture/training/logger.py` | Add `all_costs` and `constraint_violation_rate` fields to JSONL |
| `pyproject.toml` | Add seaborn, remove plotly |
| `tests/test_final_report.py` | Delete (replaced by test_report_pdf.py) |
| `tests/test_training_report.py` | Rewrite for new report.py API |

### Deleted Files

| File | Reason |
|------|--------|
| `src/python/aerocapture/training/final_report.py` | Absorbed into report.py + charts.py |

---

## Task 1: Extend Rust Trajectory Array (12 → 16 columns)

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (lines 750-818: `build_photo_values`, lines 222-240: trajectory mapping, lines 845-887: `track_peak_values`)
- Modify: `src/rust/aerocapture-py/src/results.rs` (lines 24-31: trajectory numpy shape)
- Test: `src/rust/tests/e2e_tests.rs` (existing), inline `#[cfg(test)]` in runner.rs

**Context:** The trajectory array is built in `run_for_api()` (runner.rs:222-240) by mapping over `photo_lines` — a `Vec<[f64; 24]>` from the completed simulation. `SimState` is NOT available at mapping time (it's local to `run()`). Therefore, new per-timestep values must be stored IN the photo array itself.

`build_photo_values()` (runner.rs:750-818) receives `&SimState` and constructs each 24-element photo line. It already receives `density_estimate` as a parameter. Photo columns 22 (sim_index) and 23 (reserved/0.0) are candidates for repurposing, but sim_index is used by CSV output. The cleanest approach: extend the photo array from `[f64; 24]` to `[f64; 28]` and store new values in columns 24-27.

Heat flux and g-load are currently computed in `track_peak_values()` (runs AFTER integration, line 569), while photos are taken BEFORE integration (line 551). To get values at snapshot time, we compute them inside `build_photo_values()` by passing `&SimData` for aero coefficients and capsule properties. For density ratio, we pass `density_gain` from `NavigationState` as an extra parameter.

- [ ] **Step 1: Extend photo array from [f64; 24] to [f64; 28]**

In `src/rust/src/simulation/runner.rs`:

1. Change `SimResult.photo_lines` type from `Vec<[f64; 24]>` to `Vec<[f64; 28]>` (line 90).
2. Update `build_photo_values` signature to accept `data: &SimData` and `density_gain: f64`, and return `[f64; 28]`.
3. At the end of `build_photo_values`, compute and append:

```rust
// After the existing 24 values in the return array:
let rho = data.atmosphere.density_at(altitude);
let rho_truth = rho;  // truth density at this altitude
let heat_flux = data.capsule.cq * rho_truth.sqrt() * sim.state[3].powf(3.05);
let aoa_rad = sim.aoa;
let cx = data.aero.interpolate_cx(aoa_rad);
let cz = data.aero.interpolate_cz(aoa_rad);
let mass = data.capsule.mass;
let ref_area = data.capsule.reference_area;
let aero_accel = rho_truth * ref_area * sim.state[3] * sim.state[3] / (2.0 * mass);
let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

// photo[24] = heat_flux in kW/m²
heat_flux / 1e3,
// photo[25] = g-load in g's
load_factor / G0,
// photo[26] = nav density ratio (estimated/model)
density_gain,
// photo[27] = truth density (kg/m³)
rho_truth,
```

Note: these values are computed from the pre-integration state (matching the photo snapshot timing). The existing `track_peak_values` (post-integration) continues to track peaks separately — no change needed there.

4. Update all call sites of `build_photo_values` (lines 552, 594) to pass `data` and `nav_state.density_gain` (or `1.0` for the initial/final snapshots where nav state may not be available).

- [ ] **Step 2: Update trajectory mapping in run_for_api**

In `run_for_api()` (runner.rs:222-240), change the trajectory from 12 to 16 columns:

```rust
// Change from [f64; 12] to [f64; 16]:
[
    p[1],             // [0]  alt_km
    p[2],             // [1]  lon_deg
    p[3],             // [2]  lat_deg
    p[4],             // [3]  vel_m_s
    p[5],             // [4]  fpa_deg
    p[6],             // [5]  heading_deg
    p[24],            // [6]  heat_flux_kw_m2 (was placeholder 0.0)
    p[0],             // [7]  time_s
    p[18] / 1e6,      // [8]  energy_mj_kg
    p[19] / 1e3,      // [9]  pdyn_kpa
    p[14],            // [10] bank_angle_deg
    p[9],             // [11] inclination_deg
    p[25],            // [12] g_load_g
    p[26],            // [13] nav_density_ratio
    p[27],            // [14] truth_density_kg_m3
    0.0,              // [15] reserved
]
```

- [ ] **Step 3: Update RunOutput trajectory type**

In `src/rust/src/lib.rs` (or wherever `RunOutput` is defined), change the trajectory field type from `Vec<[f64; 12]>` to `Vec<[f64; 16]>`.

- [ ] **Step 4: Update PyO3 trajectory shape**

In `src/rust/aerocapture-py/src/results.rs`, update the trajectory numpy array shape from `(N, 12)` to `(N, 16)`. Find the dimension constant and update it.

- [ ] **Step 5: Update CSV photo output**

In `src/rust/src/simulation/output.rs`, check `extract_photo_csv_values()` — it indexes into the photo array. Since we only added columns at the end (24-27), existing indices (0-23) are unchanged. No changes needed unless the function iterates over all columns.

- [ ] **Step 6: Run Rust tests**

Run: `cd src/rust && cargo test`

Expected: All existing tests pass. Some may need updating if they assert on trajectory column count or photo array size.

- [ ] **Step 7: Run PyO3 integration tests**

Run: `cd src/rust/aerocapture-py && maturin develop --release && cd ../../.. && uv run pytest tests/test_pyo3.py -v`

Expected: All pass. If any test asserts trajectory shape `(N, 12)`, update to `(N, 16)`.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/lib.rs src/rust/aerocapture-py/src/results.rs
git commit -m "feat: extend trajectory to 16 columns (heat flux, g-load, density ratio, truth density)"
```

---

## Task 2: Add seaborn, Remove plotly from Dependencies

**Files:**
- Modify: `pyproject.toml` (lines 6-14: dependencies)

- [ ] **Step 1: Update pyproject.toml**

In `pyproject.toml`, in the `[project] dependencies` list:
- Remove: `"plotly>=6.6"`
- Add: `"seaborn>=0.13"`

- [ ] **Step 2: Sync environment**

Run: `uv sync`

Expected: seaborn installed, plotly removed from environment.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: replace plotly with seaborn dependency"
```

---

## Task 3: Extend Logger with all_costs and constraint_violation_rate

**Files:**
- Modify: `src/python/aerocapture/training/logger.py` (line 66-94: record building)
- Modify: `tests/test_training_report.py` (or create new logger test)

**Context:** Panel 2 needs `constraint_violation_rate` and panel 4 needs `all_costs` (per-individual cost array) in JSONL records. The logger's `log_generation()` (line 40) already receives `costs: list[npt.NDArray[np.float64]]` and computes `all_costs = np.concatenate(costs)` at line 53, but doesn't write it. For constraint violations, we need a threshold — this can come from the cost function config or be a simple "cost > capture threshold" check.

- [ ] **Step 1: Add all_costs to JSONL record**

In `logger.py`, after line 80 (`"config_hash": self._config_hash`), add:

```python
"all_costs": all_costs.tolist(),
```

This logs the full per-individual cost array for box plot rendering.

- [ ] **Step 2: Add constraint_violation_rate to JSONL record**

In `logger.py`, after `"capture_rate"` (line 75), add a constraint violation rate. A simple approach: count how many individuals have costs above the "non-captured" threshold (which indicates g-load or heat flux constraint violations for captured trajectories). Since the cost function penalizes violations, we can proxy this as the fraction of captured individuals with cost above the median:

```python
# Constraint violation proxy: fraction of costs with penalty components
# (costs above a threshold suggest constraint violations)
"constraint_violation_rate": float(np.mean(all_costs > np.median(all_costs) * 2)) if len(all_costs) > 0 else 0.0,
```

Note: This is a rough proxy. A more precise approach would require passing constraint info from the evaluate function. This can be refined later — the chart handles `None` gracefully.

- [ ] **Step 3: Run existing logger tests**

Run: `uv run pytest tests/ -k logger -v`

Expected: All existing tests pass (new fields are additive).

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/logger.py
git commit -m "feat: log all_costs and constraint_violation_rate to JSONL for report panels"
```

---

## Task 4: Create Charts Module — Seaborn Theme + Training Convergence (Panels 1-6)

**Files:**
- Create: `src/python/aerocapture/training/charts.py`
- Create: `tests/test_charts.py`

**Context:** Each chart function takes data + output path, writes an SVG. Seaborn theme set at module level. Training convergence data comes from JSONL logs loaded as a list of dicts (see `load_run_data()` in current report.py:17-61).

- [ ] **Step 1: Write test for seaborn theme setup and convergence chart**

Create `tests/test_charts.py`:

```python
"""Tests for chart generation functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from aerocapture.training.charts import (
    chart_convergence,
)


@pytest.fixture
def tmp_svg(tmp_path: Path) -> Path:
    return tmp_path / "test.svg"


@pytest.fixture
def training_records() -> list[dict]:
    """Minimal JSONL-like training records for 10 generations."""
    return [
        {
            "generation": i,
            "best_cost": 100.0 * (0.9**i),
            "mean_cost": 150.0 * (0.95**i),
            "worst_cost": 200.0 * (0.97**i),
            "improvement": i % 3 == 0,
        }
        for i in range(10)
    ]


class TestTrainingCharts:
    def test_convergence_creates_svg(self, training_records: list[dict], tmp_svg: Path) -> None:
        chart_convergence(training_records, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_convergence_no_data_raises(self, tmp_svg: Path) -> None:
        with pytest.raises(ValueError, match="No training records"):
            chart_convergence([], tmp_svg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_charts.py::TestTrainingCharts::test_convergence_creates_svg -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'aerocapture.training.charts'`

- [ ] **Step 3: Create charts.py with seaborn theme + convergence chart**

Create `src/python/aerocapture/training/charts.py`:

```python
"""Chart generation functions for PDF reports.

Each function generates a single SVG chart from provided data.
All charts use a consistent seaborn theme set at module import.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

# ── Seaborn theme (applied at import) ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# ── Color constants ────────────────────────────────────────────────────────
COLOR_BEST = "#1f77b4"       # blue — best cost / captured trajectories
COLOR_MEAN = "#ff7f0e"       # orange — mean cost
COLOR_WORST = "#d62728"      # red — worst cost / hyperbolic / crash
COLOR_NOMINAL_REF = "#d62728"      # red — piecewise-constant reference
COLOR_NOMINAL_UNDISPERSED = "#ff7f0e"  # orange — undispersed guidance
COLOR_NOMINAL_BEST = "#2ca02c"     # green — best-case MC
COLOR_CAPTURE = "#1f77b4"
COLOR_HYPERBOLIC = "#d62728"
COLOR_DIVERSITY = "#9467bd"  # purple

# ── Figure defaults ────────────────────────────────────────────────────────
FULL_WIDTH = (10, 4)
HALF_WIDTH = (5, 4)
DPI = 150


def _save_svg(fig: plt.Figure, path: Path) -> None:
    """Save figure as SVG and close."""
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _spaghetti_alpha(n: int) -> float:
    """Compute spaghetti plot opacity, bounded for readability."""
    return float(np.clip(1.0 / np.sqrt(max(n, 1)), 0.02, 0.2))


# ── Panels 1-6: Training Convergence ──────────────────────────────────────


def chart_convergence(
    records: list[dict],
    output: Path,
    resume_gens: list[int] | None = None,
) -> None:
    """Panel 1: Best/mean/worst cost over generations (log y)."""
    if not records:
        raise ValueError("No training records provided")

    gens = [r["generation"] for r in records]
    best = [r["best_cost"] for r in records]
    mean = [r["mean_cost"] for r in records]
    worst = [r["worst_cost"] for r in records]

    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    ax.semilogy(gens, best, color=COLOR_BEST, label="Best", linewidth=1.5)
    ax.semilogy(gens, mean, color=COLOR_MEAN, label="Mean", linestyle="--", linewidth=1)
    ax.semilogy(gens, worst, color=COLOR_WORST, label="Worst", linestyle=":", linewidth=1)

    # Improvement markers
    imp_gens = [r["generation"] for r in records if r.get("improvement")]
    imp_costs = [r["best_cost"] for r in records if r.get("improvement")]
    if imp_gens:
        ax.scatter(imp_gens, imp_costs, color=COLOR_NOMINAL_BEST, s=20, zorder=5, label="Improvement")

    _add_resume_markers(ax, resume_gens)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log)")
    ax.set_title("Training Convergence")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    _save_svg(fig, output)


def _add_resume_markers(ax: plt.Axes, resume_gens: list[int] | None) -> None:
    """Add vertical dashed lines at resume points."""
    if not resume_gens:
        return
    for i, gen in enumerate(resume_gens):
        ax.axvline(gen, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        if i == 0:
            ax.annotate("resumed", (gen, ax.get_ylim()[1]), fontsize=7, color="grey", ha="center", va="bottom")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_charts.py::TestTrainingCharts -v`

Expected: Both tests PASS.

- [ ] **Step 5: Add remaining training convergence charts (panels 2-6)**

Add tests for panels 2-6 to `tests/test_charts.py`:

```python
from aerocapture.training.charts import (
    chart_convergence,
    chart_capture_constraint_rate,
    chart_diversity_cost,
    chart_cost_distribution,
    chart_parameter_evolution,
    chart_seed_pool,
)


class TestTrainingCharts:
    # ... existing tests ...

    def test_capture_constraint_rate(self, training_records: list[dict], tmp_svg: Path) -> None:
        for r in training_records:
            r["capture_rate"] = 0.8 + 0.02 * r["generation"]
        chart_capture_constraint_rate(training_records, tmp_svg)
        assert tmp_svg.exists()

    def test_diversity_cost(self, training_records: list[dict], tmp_svg: Path) -> None:
        for r in training_records:
            r["population_diversity"] = 0.5 - 0.04 * r["generation"]
        chart_diversity_cost(training_records, tmp_svg)
        assert tmp_svg.exists()

    def test_cost_distribution(self, training_records: list[dict], tmp_svg: Path) -> None:
        # Add per-generation cost arrays
        rng = np.random.default_rng(42)
        for r in training_records:
            r["all_costs"] = rng.normal(r["mean_cost"], 10, size=20).tolist()
        chart_cost_distribution(training_records, tmp_svg)
        assert tmp_svg.exists()

    def test_parameter_evolution(self, training_records: list[dict], tmp_svg: Path) -> None:
        for r in training_records:
            r["best_params"] = {"gain_a": 0.5 + 0.01 * r["generation"], "gain_b": 1.0 - 0.02 * r["generation"]}
        chart_parameter_evolution(training_records, tmp_svg)
        assert tmp_svg.exists()

    def test_seed_pool_creates_svg(self, training_records: list[dict], tmp_svg: Path) -> None:
        for r in training_records:
            r["pool_metrics"] = {"pool_size": 10 + r["generation"], "difficulty_min": 100.0, "difficulty_max": 200.0 + r["generation"] * 5}
        chart_seed_pool(training_records, tmp_svg)
        assert tmp_svg.exists()

    def test_seed_pool_skipped_when_no_data(self, training_records: list[dict], tmp_svg: Path) -> None:
        # No pool_metrics in records — should return False (not generated)
        result = chart_seed_pool(training_records, tmp_svg)
        assert result is False
        assert not tmp_svg.exists()
```

- [ ] **Step 6: Implement panels 2-6 in charts.py**

Add to `src/python/aerocapture/training/charts.py`:

```python
def chart_capture_constraint_rate(
    records: list[dict],
    output: Path,
    resume_gens: list[int] | None = None,
) -> None:
    """Panel 2: Capture rate (%) + constraint violation rate over generations."""
    if not records:
        raise ValueError("No training records provided")

    gens = [r["generation"] for r in records]
    capture = [r.get("capture_rate", 0) * 100 for r in records]

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH)
    ax1.plot(gens, capture, color=COLOR_BEST, label="Capture Rate (%)", linewidth=1.5)
    ax1.fill_between(gens, 0, capture, alpha=0.1, color=COLOR_BEST)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Capture Rate (%)", color=COLOR_BEST)

    # Constraint violation rate on secondary axis if available
    violations = [r.get("constraint_violation_rate", None) for r in records]
    if any(v is not None for v in violations):
        ax2 = ax1.twinx()
        viol_clean = [v if v is not None else 0 for v in violations]
        ax2.plot(gens, [v * 100 for v in viol_clean], color=COLOR_WORST, linestyle="--", label="Constraint Violations (%)", linewidth=1)
        ax2.set_ylabel("Violations (%)", color=COLOR_WORST)
        ax2.set_ylim(0, 105)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Capture Rate & Constraint Violations")
    ax1.legend(fontsize=8, loc="lower right")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


def chart_diversity_cost(
    records: list[dict],
    output: Path,
    resume_gens: list[int] | None = None,
) -> None:
    """Panel 3: Population diversity vs best cost (dual axis)."""
    if not records:
        raise ValueError("No training records provided")

    gens = [r["generation"] for r in records]
    diversity = [r.get("population_diversity", 0) for r in records]
    best = [r["best_cost"] for r in records]

    fig, ax1 = plt.subplots(figsize=HALF_WIDTH)
    ax1.plot(gens, diversity, color=COLOR_DIVERSITY, label="Diversity", linewidth=1.5)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity", color=COLOR_DIVERSITY)

    ax2 = ax1.twinx()
    ax2.semilogy(gens, best, color=COLOR_BEST, linestyle=":", label="Best Cost", linewidth=1)
    ax2.set_ylabel("Best Cost (log)", color=COLOR_BEST)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Diversity vs Best Cost")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


def chart_cost_distribution(records: list[dict], output: Path) -> bool:
    """Panel 4: Cost distribution box plots at sampled generations. Returns False if no data."""
    if not records:
        raise ValueError("No training records provided")

    # Sample ~10 evenly spaced generations for box plots
    n = len(records)
    step = max(1, n // 10)
    sampled = records[::step]
    if records[-1] not in sampled:
        sampled.append(records[-1])

    data_for_box = []
    labels = []
    for r in sampled:
        costs = r.get("all_costs")
        if costs:
            data_for_box.append(costs)
            labels.append(str(r["generation"]))

    if not data_for_box:
        return False  # No per-population cost data available

    fig, ax = plt.subplots(figsize=HALF_WIDTH)
    ax.boxplot(data_for_box, labels=labels, whis=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log)")
    ax.set_title("Cost Distribution")
    sns.despine(fig=fig)
    _save_svg(fig, output)
    return True


def chart_parameter_evolution(
    records: list[dict],
    output: Path,
    resume_gens: list[int] | None = None,
) -> None:
    """Panel 5: Best parameters over generations."""
    if not records:
        raise ValueError("No training records provided")

    # Collect all parameter names from first record that has them
    params_by_gen: dict[str, list[float]] = {}
    gens: list[int] = []
    for r in records:
        bp = r.get("best_params", {})
        if bp:
            gens.append(r["generation"])
            for k, v in bp.items():
                params_by_gen.setdefault(k, []).append(float(v))

    if not params_by_gen:
        return

    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    for name, values in params_by_gen.items():
        ax.plot(gens[: len(values)], values, label=name, linewidth=1)

    _add_resume_markers(ax, resume_gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Evolution")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_seed_pool(
    records: list[dict],
    output: Path,
    resume_gens: list[int] | None = None,
) -> bool:
    """Panel 6 (conditional): Seed pool evolution. Returns False if no data."""
    has_pool = any(r.get("pool_metrics") for r in records)
    if not has_pool:
        return False

    gens = []
    pool_sizes = []
    diff_min = []
    diff_max = []
    for r in records:
        pm = r.get("pool_metrics")
        if pm:
            gens.append(r["generation"])
            pool_sizes.append(pm["pool_size"])
            diff_min.append(pm["difficulty_min"])
            diff_max.append(pm["difficulty_max"])

    fig, ax1 = plt.subplots(figsize=FULL_WIDTH)
    ax1.plot(gens, pool_sizes, color=COLOR_BEST, label="Pool Size", linewidth=1.5)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Pool Size", color=COLOR_BEST)

    ax2 = ax1.twinx()
    ax2.plot(gens, diff_min, color=COLOR_MEAN, linestyle="--", linewidth=0.8, label="Difficulty Min")
    ax2.plot(gens, diff_max, color=COLOR_MEAN, linestyle="--", linewidth=0.8, label="Difficulty Max")
    ax2.fill_between(gens, diff_min, diff_max, alpha=0.1, color=COLOR_MEAN)
    ax2.set_ylabel("Difficulty", color=COLOR_MEAN)

    _add_resume_markers(ax1, resume_gens)
    ax1.set_title("Seed Pool Evolution")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)
    return True
```

- [ ] **Step 7: Run all training chart tests**

Run: `uv run pytest tests/test_charts.py::TestTrainingCharts -v`

Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat: add charts module with training convergence panels 1-6"
```

---

## Task 5: Charts Module — Corridor & Energy Panels (7-9)

**Files:**
- Modify: `src/python/aerocapture/training/charts.py`
- Modify: `tests/test_charts.py`

**Context:** These panels plot MC trajectory spaghetti against energy, with corridor zone fills from `corridor_boundaries.npz` (schema v4). The current implementation is in `final_report.py:718-861` (`_generate_corridor_png`, `_draw_pdyn_zones`, `_compute_envelope`). Port the logic to individual SVG chart functions.

- [ ] **Step 1: Write tests for corridor charts**

Add to `tests/test_charts.py`:

```python
from aerocapture.training.charts import (
    chart_corridor_pdyn,
    chart_corridor_inclination,
    chart_corridor_bank,
)


@pytest.fixture
def mc_trajectories() -> list[npt.NDArray[np.float64]]:
    """Synthetic MC trajectories (10 runs, ~50 timesteps each, 16 cols)."""
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(10):
        n_steps = rng.integers(40, 60)
        traj = np.zeros((n_steps, 16))
        traj[:, 0] = np.linspace(120, 30, n_steps)     # alt_km (descending)
        traj[:, 7] = np.linspace(0, 300, n_steps)       # time_s
        traj[:, 8] = np.linspace(-1.0, -3.0, n_steps)   # energy_mj_kg
        traj[:, 9] = rng.uniform(0.5, 5.0, n_steps)     # pdyn_kpa
        traj[:, 10] = rng.uniform(0, 90, n_steps)        # bank_angle_deg
        traj[:, 11] = rng.uniform(24.0, 25.0, n_steps)   # inclination_deg
        trajs.append(traj)
    return trajs


@pytest.fixture
def captured_mask() -> npt.NDArray[np.bool_]:
    """First 8 captured, last 2 hyperbolic."""
    mask = np.ones(10, dtype=bool)
    mask[8:] = False
    return mask


class TestCorridorCharts:
    def test_pdyn_creates_svg(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_corridor_pdyn(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()

    def test_pdyn_with_corridor_data(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        corridor = {
            "energy_bins": np.linspace(-3.0, -1.0, 50),
            "envelope_crash_pdyn": np.full(50, 8.0),
            "envelope_restricted_max_pdyn": np.full(50, 6.0),
            "envelope_restricted_min_pdyn": np.full(50, 1.0),
            "envelope_capture_pdyn": np.full(50, 0.5),
        }
        chart_corridor_pdyn(mc_trajectories, captured_mask, tmp_svg, corridor_data=corridor)
        assert tmp_svg.exists()

    def test_inclination_creates_svg(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_corridor_inclination(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()

    def test_bank_creates_svg(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_corridor_bank(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_charts.py::TestCorridorCharts -v`

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement corridor charts (panels 7-9)**

Add to `charts.py`. Port logic from `final_report.py:_generate_corridor_png()`, `_draw_pdyn_zones()`, `_compute_envelope()` — but split into 3 separate functions each producing one SVG:

```python
def _compute_envelope(
    trajectories: list[npt.NDArray[np.float64]],
    energy_col: int,
    value_col: int,
    captured_mask: npt.NDArray[np.bool_],
    n_bins: int = 200,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin trajectories by energy, compute min/max value per bin.

    Returns (energy_centers, min_values, max_values) with NaN for empty bins.
    """
    all_energy = np.concatenate([t[:, energy_col] for t, m in zip(trajectories, captured_mask) if m])
    all_values = np.concatenate([t[:, value_col] for t, m in zip(trajectories, captured_mask) if m])

    if len(all_energy) == 0:
        return np.array([]), np.array([]), np.array([])

    bins = np.linspace(all_energy.min(), all_energy.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    indices = np.digitize(all_energy, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    min_vals = np.full(n_bins, np.nan)
    max_vals = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = indices == i
        if mask.any():
            min_vals[i] = all_values[mask].min()
            max_vals[i] = all_values[mask].max()

    return centers, min_vals, max_vals


def _draw_spaghetti(
    ax: plt.Axes,
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    x_col: int,
    y_col: int,
) -> None:
    """Draw MC spaghetti: captured in blue, hyperbolic in red."""
    alpha = _spaghetti_alpha(len(trajectories))
    for traj, cap in zip(trajectories, captured_mask):
        color = COLOR_CAPTURE if cap else COLOR_HYPERBOLIC
        ax.plot(traj[:, x_col], traj[:, y_col], color=color, alpha=alpha, linewidth=0.5)


def _draw_nominals(
    ax: plt.Axes,
    x_col: int,
    y_col: int,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Overlay up to 3 nominal trajectories."""
    if ref_nominal is not None:
        ax.plot(ref_nominal[:, x_col], ref_nominal[:, y_col], color=COLOR_NOMINAL_REF, linewidth=1.5, label="Reference (piecewise)")
    if undispersed_nominal is not None:
        ax.plot(undispersed_nominal[:, x_col], undispersed_nominal[:, y_col], color=COLOR_NOMINAL_UNDISPERSED, linewidth=1.5, label="Undispersed")
    if best_nominal is not None:
        ax.plot(best_nominal[:, x_col], best_nominal[:, y_col], color=COLOR_NOMINAL_BEST, linewidth=1.5, label="Best MC")


def chart_corridor_pdyn(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    corridor_data: dict | None = None,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 7: Energy vs dynamic pressure with corridor zones."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH)

    # Corridor zone fills (4-layer)
    if corridor_data is not None:
        e = corridor_data["energy_bins"]
        ax.fill_between(e, corridor_data["envelope_crash_pdyn"], 100, alpha=0.15, color=COLOR_WORST, label="Crash zone")
        ax.fill_between(e, corridor_data["envelope_restricted_max_pdyn"], corridor_data["envelope_crash_pdyn"], alpha=0.08, color="grey")
        ax.fill_between(e, corridor_data["envelope_capture_pdyn"], corridor_data["envelope_restricted_min_pdyn"], alpha=0.08, color="grey")
        ax.fill_between(e, 0, corridor_data["envelope_capture_pdyn"], alpha=0.15, color=COLOR_WORST, label="Hyperbolic zone")

    _draw_spaghetti(ax, trajectories, captured_mask, x_col=8, y_col=9)
    _draw_nominals(ax, x_col=8, y_col=9, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Dynamic Pressure (kPa)")
    ax.set_title("Energy vs Dynamic Pressure")
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_corridor_inclination(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 8: Energy vs inclination with MC envelope."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)

    centers, min_vals, max_vals = _compute_envelope(trajectories, 8, 11, captured_mask)
    if len(centers) > 0:
        valid = ~np.isnan(min_vals)
        ax.fill_between(centers[valid], min_vals[valid], max_vals[valid], alpha=0.15, color=COLOR_CAPTURE)

    _draw_spaghetti(ax, trajectories, captured_mask, x_col=8, y_col=11)
    _draw_nominals(ax, x_col=8, y_col=11, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Energy vs Inclination")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_corridor_bank(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    ref_nominal: npt.NDArray[np.float64] | None = None,
    undispersed_nominal: npt.NDArray[np.float64] | None = None,
    best_nominal: npt.NDArray[np.float64] | None = None,
) -> None:
    """Panel 9: Energy vs bank angle with MC envelope."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)

    centers, min_vals, max_vals = _compute_envelope(trajectories, 8, 10, captured_mask)
    if len(centers) > 0:
        valid = ~np.isnan(min_vals)
        ax.fill_between(centers[valid], min_vals[valid], max_vals[valid], alpha=0.15, color=COLOR_CAPTURE)

    _draw_spaghetti(ax, trajectories, captured_mask, x_col=8, y_col=10)
    _draw_nominals(ax, x_col=8, y_col=10, ref_nominal=ref_nominal, undispersed_nominal=undispersed_nominal, best_nominal=best_nominal)

    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Bank Angle (deg)")
    ax.set_title("Energy vs Bank Angle")
    sns.despine(fig=fig)
    _save_svg(fig, output)
```

- [ ] **Step 4: Run corridor chart tests**

Run: `uv run pytest tests/test_charts.py::TestCorridorCharts -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat: add corridor/energy chart panels 7-9"
```

---

## Task 6: Charts Module — Time-Domain Trajectory Panels (10-14)

**Files:**
- Modify: `src/python/aerocapture/training/charts.py`
- Modify: `tests/test_charts.py`

**Context:** These are all NEW panels (not ported from existing code). They plot MC spaghetti in the time domain. Panels 11-12 overlay constraint limit lines from TOML config. Panel 14 plots nav density ratio (column 13 in extended trajectory).

- [ ] **Step 1: Write tests for time-domain charts**

Add to `tests/test_charts.py`:

```python
from aerocapture.training.charts import (
    chart_altitude_time,
    chart_heat_flux_time,
    chart_gload_time,
    chart_bank_angle_time,
    chart_nav_density_ratio,
)


class TestTimeDomainCharts:
    def test_altitude_time(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_altitude_time(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()

    def test_altitude_highlights_best(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        # Best trajectory = index 0 (lowest DV)
        chart_altitude_time(mc_trajectories, captured_mask, tmp_svg, best_idx=0)
        assert tmp_svg.exists()

    def test_heat_flux_with_limit(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        # Populate heat flux column
        for t in mc_trajectories:
            t[:, 6] = np.random.default_rng(42).uniform(0, 200, len(t))
        chart_heat_flux_time(mc_trajectories, captured_mask, tmp_svg, limit_kw_m2=150.0)
        assert tmp_svg.exists()

    def test_gload_with_limit(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        for t in mc_trajectories:
            t[:, 12] = np.random.default_rng(42).uniform(0, 5, len(t))
        chart_gload_time(mc_trajectories, captured_mask, tmp_svg, limit_g=4.0)
        assert tmp_svg.exists()

    def test_bank_angle_time(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_bank_angle_time(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()

    def test_nav_density_ratio(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        for t in mc_trajectories:
            t[:, 13] = np.random.default_rng(42).uniform(0.8, 1.2, len(t))
        chart_nav_density_ratio(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_charts.py::TestTimeDomainCharts -v`

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement time-domain charts**

Add to `charts.py`:

```python
# ── Panels 10-14: Time-Domain Trajectory ───────────────────────────────────


def chart_altitude_time(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    best_idx: int | None = None,
) -> None:
    """Panel 10: Altitude vs time (MC spaghetti, best-case highlighted)."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    _draw_spaghetti(ax, trajectories, captured_mask, x_col=7, y_col=0)

    if best_idx is not None and 0 <= best_idx < len(trajectories):
        t = trajectories[best_idx]
        ax.plot(t[:, 7], t[:, 0], color=COLOR_NOMINAL_BEST, linewidth=1.5, label="Best case", zorder=10)
        ax.legend(fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Time")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_heat_flux_time(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    limit_kw_m2: float | None = None,
) -> None:
    """Panel 11: Heat flux vs time (MC spaghetti + limit line)."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)
    _draw_spaghetti(ax, trajectories, captured_mask, x_col=7, y_col=6)

    if limit_kw_m2 is not None:
        ax.axhline(limit_kw_m2, color=COLOR_WORST, linestyle="--", linewidth=1, label=f"Limit ({limit_kw_m2:.0f} kW/m²)")
        ax.legend(fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat Flux (kW/m²)")
    ax.set_title("Heat Flux vs Time")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_gload_time(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
    limit_g: float | None = None,
) -> None:
    """Panel 12: G-load vs time (MC spaghetti + limit line)."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)
    _draw_spaghetti(ax, trajectories, captured_mask, x_col=7, y_col=12)

    if limit_g is not None:
        ax.axhline(limit_g, color=COLOR_WORST, linestyle="--", linewidth=1, label=f"Limit ({limit_g:.1f} g)")
        ax.legend(fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("G-Load (g)")
    ax.set_title("G-Load vs Time")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_bank_angle_time(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
) -> None:
    """Panel 13: Bank angle vs time (MC spaghetti)."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    _draw_spaghetti(ax, trajectories, captured_mask, x_col=7, y_col=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bank Angle (deg)")
    ax.set_title("Bank Angle vs Time")
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_nav_density_ratio(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
) -> None:
    """Panel 14: Nav filter density ratio (estimated/truth) vs time."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH)
    _draw_spaghetti(ax, trajectories, captured_mask, x_col=7, y_col=13)

    ax.axhline(1.0, color="grey", linestyle="-", linewidth=0.8, alpha=0.5, label="Perfect estimate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density Ratio (est/truth)")
    ax.set_title("Navigation Filter: Density Ratio")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    _save_svg(fig, output)
```

- [ ] **Step 4: Run time-domain chart tests**

Run: `uv run pytest tests/test_charts.py::TestTimeDomainCharts -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat: add time-domain trajectory chart panels 10-14"
```

---

## Task 7: Charts Module — Distributions, Scatters & Dispersion Grid (Panels 15-18, 20)

**Files:**
- Modify: `src/python/aerocapture/training/charts.py`
- Modify: `tests/test_charts.py`

**Context:** These port existing Plotly panels from `final_report.py`. DV clipping: `[DV_FLOOR=0.1, DV_CAP=5000.0]` m/s. Log-scale DV axes. The dispersion grid is a 4×6 subplot grid (24 fields vs DV scatter with linear regression R²). Final record column indices are defined in `final_report.py:26-47`.

- [ ] **Step 1: Write tests for distribution and scatter charts**

Add to `tests/test_charts.py`:

```python
from aerocapture.training.charts import (
    chart_dv_distribution,
    chart_dv_individual_burns,
    chart_entry_conditions,
    chart_exit_conditions,
    chart_dispersion_grid,
    DV_CAP,
    DV_FLOOR,
)


@pytest.fixture
def final_records() -> npt.NDArray[np.float64]:
    """Synthetic final records (20 sims, 52 columns)."""
    rng = np.random.default_rng(42)
    n = 20
    records = np.zeros((n, 52))
    records[:, 3] = rng.uniform(5000, 6000, n)   # velocity
    records[:, 4] = rng.uniform(-6, -4, n)        # FPA
    records[:, 9] = rng.uniform(0.5, 1.5, n)      # eccentricity
    records[:15, 9] = rng.uniform(0.3, 0.9, 15)   # first 15 captured (ecc < 1)
    records[:, 29] = rng.uniform(-50, 50, n)       # peri_err
    records[:, 30] = rng.uniform(-100, 100, n)     # apo_err
    records[:, 37] = rng.uniform(1, 100, n)        # dv1
    records[:, 38] = rng.uniform(1, 100, n)        # dv2
    records[:, 39] = rng.uniform(1, 50, n)         # dv3
    records[:, 41] = records[:, 37] + records[:, 38] + records[:, 39]  # dv_total
    records[:, 31] = 3  # ifinal = 3 (exited atmosphere)
    return records


@pytest.fixture
def dispersions() -> npt.NDArray[np.float64]:
    """Synthetic dispersions (20 sims, 24 fields)."""
    return np.random.default_rng(42).normal(0, 1, (20, 24))


class TestDistributionCharts:
    def test_dv_distribution(self, final_records: npt.NDArray, tmp_svg: Path) -> None:
        chart_dv_distribution(final_records, tmp_svg)
        assert tmp_svg.exists()

    def test_dv_individual_burns(self, final_records: npt.NDArray, tmp_svg: Path) -> None:
        chart_dv_individual_burns(final_records, tmp_svg)
        assert tmp_svg.exists()

    def test_entry_conditions(self, mc_trajectories: list, captured_mask: npt.NDArray, tmp_svg: Path) -> None:
        chart_entry_conditions(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()

    def test_exit_conditions(self, final_records: npt.NDArray, tmp_svg: Path) -> None:
        chart_exit_conditions(final_records, tmp_svg)
        assert tmp_svg.exists()

    def test_dispersion_grid(self, final_records: npt.NDArray, dispersions: npt.NDArray, tmp_svg: Path) -> None:
        chart_dispersion_grid(final_records, dispersions, tmp_svg)
        assert tmp_svg.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_charts.py::TestDistributionCharts -v`

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement distribution, scatter, and dispersion charts**

Add to `charts.py`:

```python
# ── Constants ──────────────────────────────────────────────────────────────
DV_CAP = 5000.0   # m/s — upper clip for DV plotting
DV_FLOOR = 0.1    # m/s — lower clip for DV plotting

# Final record column indices (52-element array)
_FR_VELOCITY = 3
_FR_FPA = 4
_FR_ENERGY = 7
_FR_ECC = 9
_FR_INCL = 10
_FR_MAX_HEAT_FLUX = 16
_FR_MAX_G_LOAD = 17
_FR_MAX_DYN_PRES = 18
_FR_PERI_ERR = 29
_FR_APO_ERR = 30
_FR_IFINAL = 31
_FR_DV1 = 37
_FR_DV2 = 38
_FR_DV3 = 39
_FR_DV_TOTAL = 41
_FR_BANK_CONSUMPTION = 45
_FR_INCL_ERR = 46


# ── Panels 15-18, 20: Distributions & Scatters ────────────────────────────


def _clip_dv(dv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.clip(dv, DV_FLOOR, DV_CAP)


def chart_dv_distribution(
    final_records: npt.NDArray[np.float64],
    output: Path,
) -> None:
    """Panel 15: Total DV histogram (log10) + CDF + percentile lines."""
    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)

    fig, ax1 = plt.subplots(figsize=HALF_WIDTH)
    ax1.hist(log_dv, bins=40, color=COLOR_BEST, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Total ΔV (m/s)")
    ax1.set_ylabel("Count")

    # Custom log-scale ticks
    tick_values = [0.1, 1, 10, 100, 1000, 5000]
    ax1.set_xticks([np.log10(v) for v in tick_values])
    ax1.set_xticklabels([str(v) for v in tick_values])

    # CDF on secondary axis
    ax2 = ax1.twinx()
    sorted_dv = np.sort(log_dv)
    cdf = np.arange(1, len(sorted_dv) + 1) / len(sorted_dv)
    ax2.plot(sorted_dv, cdf, color=COLOR_MEAN, linewidth=1.5)
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0, 1.05)

    # Percentile markers
    for p, ls in [(5, ":"), (50, "--"), (95, ":")]:
        val = np.percentile(log_dv, p)
        ax1.axvline(val, color="grey", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax1.annotate(f"p{p}", (val, ax1.get_ylim()[1] * 0.95), fontsize=7, color="grey", ha="center")

    ax1.set_title("Total ΔV Distribution")
    sns.despine(fig=fig, right=False)
    _save_svg(fig, output)


def chart_dv_individual_burns(
    final_records: npt.NDArray[np.float64],
    output: Path,
) -> None:
    """Panel 16: Individual correction burns (dv1, dv2, dv3) overlaid histograms."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)

    for col, label, color in [
        (_FR_DV1, "ΔV₁ (periapsis)", COLOR_BEST),
        (_FR_DV2, "ΔV₂ (apoapsis)", COLOR_MEAN),
        (_FR_DV3, "ΔV₃ (inclination)", COLOR_NOMINAL_BEST),
    ]:
        dv = _clip_dv(final_records[:, col])
        log_dv = np.log10(dv)
        ax.hist(log_dv, bins=30, alpha=0.5, color=color, label=label, edgecolor="white", linewidth=0.3)

    tick_values = [0.1, 1, 10, 100, 1000, 5000]
    ax.set_xticks([np.log10(v) for v in tick_values])
    ax.set_xticklabels([str(v) for v in tick_values])
    ax.set_xlabel("ΔV (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Individual Correction Burns")
    ax.legend(fontsize=7)
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_entry_conditions(
    trajectories: list[npt.NDArray[np.float64]],
    captured_mask: npt.NDArray[np.bool_],
    output: Path,
) -> None:
    """Panel 17: Entry velocity vs FPA (captured green, hyperbolic red)."""
    fig, ax = plt.subplots(figsize=HALF_WIDTH)

    entry_v = np.array([t[0, 3] for t in trajectories])
    entry_fpa = np.array([t[0, 4] for t in trajectories])

    ax.scatter(entry_v[captured_mask], entry_fpa[captured_mask], c=COLOR_CAPTURE, s=15, alpha=0.6, label="Captured")
    ax.scatter(entry_v[~captured_mask], entry_fpa[~captured_mask], c=COLOR_HYPERBOLIC, s=15, marker="x", alpha=0.6, label="Hyperbolic")

    ax.set_xlabel("Entry Velocity (m/s)")
    ax.set_ylabel("Entry FPA (deg)")
    ax.set_title("Entry Conditions")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    _save_svg(fig, output)


def chart_exit_conditions(
    final_records: npt.NDArray[np.float64],
    output: Path,
) -> None:
    """Panel 18: Exit velocity vs FPA, marker size ∝ DV."""
    captured = final_records[:, _FR_ECC] < 1.0
    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])

    fig, ax = plt.subplots(figsize=HALF_WIDTH)

    if captured.any():
        cap = final_records[captured]
        sizes = 10 + 40 * (np.log10(dv[captured]) - np.log10(DV_FLOOR)) / (np.log10(DV_CAP) - np.log10(DV_FLOOR))
        ax.scatter(cap[:, _FR_VELOCITY], cap[:, _FR_FPA], c=COLOR_CAPTURE, s=sizes, alpha=0.5, label="Captured")

    if (~captured).any():
        hyp = final_records[~captured]
        ax.scatter(hyp[:, _FR_VELOCITY], hyp[:, _FR_FPA], c=COLOR_HYPERBOLIC, s=15, marker="x", alpha=0.5, label="Hyperbolic")

    ax.set_xlabel("Exit Velocity (m/s)")
    ax.set_ylabel("Exit FPA (deg)")
    ax.set_title("Exit Conditions")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    _save_svg(fig, output)


# Dispersion field labels (24 fields, matching Rust dispersion draw order)
DISPERSION_LABELS: list[str] = [
    "Entry velocity", "Entry FPA", "Entry azimuth", "Entry altitude",
    "Density mult.", "Density bias", "Cx bias", "Cz bias",
    "Mass bias", "Ref area bias", "Incidence bias",
    "Nav vel err", "Nav FPA err", "Nav azimuth err",
    "Nav alt err", "Nav lon err", "Nav lat err",
    "Density filter gain", "Gyro bias X", "Gyro bias Y", "Gyro bias Z",
    "Wind vel", "Wind azimuth", "Reserved",
]


def chart_dispersion_grid(
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    output: Path,
) -> None:
    """Panel 20: Dispersion correlation grid (24 scatter subplots with R²)."""
    from scipy import stats

    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)
    n_fields = dispersions.shape[1]
    n_cols = 6
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows))
    axes_flat = axes.flatten()

    for i in range(n_fields):
        ax = axes_flat[i]
        x = dispersions[:, i]
        ax.scatter(x, log_dv, s=5, alpha=0.4, color=COLOR_BEST)

        # Linear regression
        if len(x) > 2 and np.std(x) > 1e-10:
            slope, intercept, r_value, p_value, _ = stats.linregress(x, log_dv)
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, color=COLOR_WORST, linewidth=1)
            ax.annotate(f"R²={r_value**2:.2f}\np={p_value:.1e}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6, va="top")

        label = DISPERSION_LABELS[i] if i < len(DISPERSION_LABELS) else f"Field {i}"
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Dispersion Correlations vs ΔV", fontsize=11)
    fig.tight_layout()
    _save_svg(fig, output)
```

- [ ] **Step 4: Run distribution chart tests**

Run: `uv run pytest tests/test_charts.py::TestDistributionCharts -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat: add distribution, scatter, and dispersion chart panels 15-18, 20"
```

---

## Task 8: Create Typst Templates

**Files:**
- Create: `src/typst/lib.typ`
- Create: `src/typst/report.typ`
- Create: `src/typst/comparison.typ`

**Context:** Typst receives a temp directory path via `--input dir=<path>`. It reads SVGs with `image()` and JSON with `json()`. Summary table (panel 19) and cover page metadata are rendered natively.

- [ ] **Step 1: Verify typst is installed**

Run: `typst --version`

If not installed: `brew install typst`

- [ ] **Step 2: Create lib.typ (shared helpers)**

Create `src/typst/lib.typ`:

```typst
// Shared styling for Aerocapture reports

#let page-style = (
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 1.5cm, right: 1.5cm),
)

#let heading-style = (
  numbering: none,
)

#let section-heading(title) = {
  v(0.5cm)
  line(length: 100%, stroke: 0.5pt + luma(180))
  v(0.3cm)
  text(size: 16pt, weight: "bold")[#title]
  v(0.3cm)
}

#let full-width-chart(path) = {
  image(path, width: 100%)
  v(0.3cm)
}

#let half-width-pair(left-path, right-path) = {
  grid(
    columns: (1fr, 1fr),
    column-gutter: 0.5cm,
    image(left-path, width: 100%),
    image(right-path, width: 100%),
  )
  v(0.3cm)
}

#let cover-page(meta) = {
  v(3cm)
  align(center)[
    #text(size: 28pt, weight: "bold")[Aerocapture Training Report]
    #v(0.5cm)
    #text(size: 18pt)[#meta.scheme]
    #v(0.3cm)
    #text(size: 12pt, fill: luma(100))[#meta.mission — #meta.date]
    #v(1cm)
    #table(
      columns: (auto, auto),
      stroke: 0.5pt + luma(200),
      inset: 8pt,
      align: (left, right),
      [*Best Cost*], [#meta.best_cost],
      [*Capture Rate*], [#meta.capture_rate],
      [*Generations*], [#meta.total_generations],
      [*Final Eval Sims*], [#meta.n_sims],
      [*Config Hash*], [#text(size: 8pt, font: "Courier New")[#meta.config_hash]],
    )
  ]
  pagebreak()
}

#let performance-table(data) = {
  let headers = ("Parameter", "Mean", "Std", "Min", "p5", "p25", "p50", "p75", "p95", "Max")
  table(
    columns: headers.len(),
    stroke: 0.5pt + luma(200),
    inset: 6pt,
    align: (left, ..range(headers.len() - 1).map(_ => right)),
    ..headers.map(h => text(weight: "bold", size: 8pt)[#h]),
    ..data.flatten().map(cell => text(size: 8pt)[#cell]),
  )
}
```

- [ ] **Step 3: Create report.typ (main report template)**

Create `src/typst/report.typ`:

```typst
#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

// ── Cover Page ──
#cover-page(meta)

// ── Part 1: Training Convergence ──
#section-heading("Part 1: Training Convergence")

#full-width-chart(dir + "/convergence.svg")
#full-width-chart(dir + "/capture_constraint_rate.svg")
#if meta.at("has_cost_distribution", default: false) {
  half-width-pair(dir + "/diversity_cost.svg", dir + "/cost_distribution.svg")
} else {
  full-width-chart(dir + "/diversity_cost.svg")
}
#full-width-chart(dir + "/parameter_evolution.svg")

// Conditional: seed pool (only if file exists)
#let seed_pool_path = dir + "/seed_pool.svg"
#context {
  // Typst doesn't have file-exists; we use metadata flag instead
}
#if meta.at("has_seed_pool", default: false) {
  full-width-chart(seed_pool_path)
}

#pagebreak()

// ── Part 2: Mission Performance ──
#section-heading("Part 2: Mission Performance")

#if meta.at("has_trajectories", default: false) {
  full-width-chart(dir + "/corridor_pdyn.svg")
  half-width-pair(dir + "/corridor_inclination.svg", dir + "/corridor_bank.svg")
  full-width-chart(dir + "/altitude_time.svg")
  half-width-pair(dir + "/heat_flux_time.svg", dir + "/gload_time.svg")
  full-width-chart(dir + "/bank_angle_time.svg")
  full-width-chart(dir + "/nav_density_ratio.svg")
} else {
  align(center)[
    #v(2cm)
    #text(fill: luma(120), size: 12pt)[Trajectory data not available — time-domain panels omitted.]
    #v(2cm)
  ]
}

#half-width-pair(dir + "/dv_distribution.svg", dir + "/dv_individual_burns.svg")

#if meta.at("has_trajectories", default: false) {
  half-width-pair(dir + "/entry_conditions.svg", dir + "/exit_conditions.svg")
}

// Performance Summary Table
#v(0.5cm)
#text(size: 12pt, weight: "bold")[Performance Summary]
#v(0.3cm)
#let summary = json(dir + "/summary_table.json")
#performance-table(summary.rows)

// Dispersion Grid (full page)
#pagebreak()
#section-heading("Dispersion Correlations")
#image(dir + "/dispersion_grid.svg", width: 100%)
```

- [ ] **Step 4: Create comparison.typ**

Create `src/typst/comparison.typ`:

```typst
#import "lib.typ": *

#let dir = sys.inputs.at("dir")
#let meta = json(dir + "/metadata.json")

#set page(..page-style)
#set text(size: 10pt)

#v(2cm)
#align(center)[
  #text(size: 24pt, weight: "bold")[Cross-Scheme Comparison]
  #v(0.3cm)
  #text(size: 12pt, fill: luma(100))[#meta.date]
]
#v(1cm)

#full-width-chart(dir + "/comparison_convergence.svg")

#v(0.5cm)
#text(size: 12pt, weight: "bold")[Final Metrics]
#v(0.3cm)
#let metrics = json(dir + "/comparison_table.json")
#table(
  columns: metrics.headers.len(),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  ..metrics.headers.map(h => text(weight: "bold", size: 9pt)[#h]),
  ..metrics.rows.flatten().map(cell => text(size: 9pt)[#cell]),
)
```

- [ ] **Step 5: Test Typst compilation with dummy data**

Create a quick smoke test:

```bash
mkdir -p /tmp/typst_test
# Write minimal metadata.json
echo '{"scheme":"test","mission":"Mars","date":"2026-03-24","best_cost":"42.0","capture_rate":"95%","total_generations":"50","n_sims":"1000","config_hash":"abc123","has_seed_pool":false,"has_trajectories":false}' > /tmp/typst_test/metadata.json
echo '{"rows":[["ΔV total","50.2","12.3","10.1","15.2","30.1","45.0","65.2","80.1","120.5"]]}' > /tmp/typst_test/summary_table.json

# Generate minimal SVGs (empty 1x1 plots)
python3 -c "
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
for name in ['convergence','capture_constraint_rate','diversity_cost','cost_distribution','parameter_evolution','dv_distribution','dv_individual_burns','dispersion_grid']:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.text(0.5,0.5,name,ha='center',va='center')
    fig.savefig(f'/tmp/typst_test/{name}.svg',format='svg')
    plt.close()
"

typst compile src/typst/report.typ --input dir=/tmp/typst_test /tmp/typst_test/report.pdf
```

Expected: PDF generated at `/tmp/typst_test/report.pdf`. Open and verify layout.

- [ ] **Step 6: Commit**

```bash
git add src/typst/lib.typ src/typst/report.typ src/typst/comparison.typ
git commit -m "feat: add Typst templates for report and comparison PDFs"
```

---

## Task 9: Rewrite report.py — Orchestrator

**Files:**
- Modify: `src/python/aerocapture/training/report.py` (full rewrite)
- Create: `tests/test_report_pdf.py`

**Context:** The new `report.py` orchestrates: load data → generate SVGs → write JSON → invoke `typst compile`. It replaces both the old `report.py` (convergence) and `final_report.py` (final eval). Key functions to preserve/port: `load_run_data()` (JSONL loading + resume detection), `run_final_evaluation()` (MC re-eval). The `generate_report()` function is the new single entry point.

- [ ] **Step 1: Write integration test for report generation**

Create `tests/test_report_pdf.py`:

```python
"""Integration tests for PDF report generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aerocapture.training.report import generate_report, generate_comparison_report, _check_typst


class TestCheckTypst:
    def test_returns_true_when_available(self) -> None:
        if shutil.which("typst"):
            assert _check_typst() is True

    def test_returns_false_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _check_typst() is False


class TestGenerateReport:
    @pytest.fixture
    def scheme_dir(self, tmp_path: Path) -> Path:
        """Create a minimal scheme directory with JSONL data."""
        d = tmp_path / "equilibrium_glide"
        d.mkdir()

        # Write JSONL
        records = []
        for i in range(5):
            records.append({
                "generation": i,
                "best_cost": 100.0 * (0.9**i),
                "mean_cost": 150.0 * (0.95**i),
                "worst_cost": 200.0,
                "capture_rate": 0.8 + 0.02 * i,
                "population_diversity": 0.5 - 0.04 * i,
                "improvement": i % 2 == 0,
                "best_params": {"gain": 0.5 + 0.01 * i},
                "config_hash": "test123",
                "scheme": "equilibrium_glide",
            })
        jsonl = d / "run_000_test.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in records))

        return d

    def test_generates_charts_to_temp_dir(self, scheme_dir: Path) -> None:
        """Test that chart SVGs are generated (without typst compilation)."""
        # This test mocks typst away and just checks SVG generation
        with patch("aerocapture.training.report._check_typst", return_value=False):
            generate_report(scheme_dir, toml_path=None, skip_final_eval=True)

        # No PDF since typst is mocked away, but the function should not crash


class TestGenerateComparisonReport:
    def test_comparison_report_no_data(self, tmp_path: Path) -> None:
        """Empty directory produces no report."""
        with patch("aerocapture.training.report._check_typst", return_value=False):
            result = generate_comparison_report(tmp_path)
        assert result is None

    def test_comparison_report_with_data(self, tmp_path: Path) -> None:
        """Multiple scheme dirs with JSONL produce a comparison."""
        for scheme in ["eq_glide", "ftc"]:
            d = tmp_path / scheme
            d.mkdir()
            records = [{"generation": i, "best_cost": 100 - i, "mean_cost": 150, "worst_cost": 200,
                        "capture_rate": 0.9, "population_diversity": 0.3, "scheme": scheme} for i in range(3)]
            (d / "run_000.jsonl").write_text("\n".join(json.dumps(r) for r in records))

        with patch("aerocapture.training.report._check_typst", return_value=False):
            result = generate_comparison_report(tmp_path)
        # No PDF (typst mocked), but should not crash
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_pdf.py -v`

Expected: FAIL with ImportError (new API doesn't exist yet).

- [ ] **Step 3: Rewrite report.py**

Rewrite `src/python/aerocapture/training/report.py` with the new orchestrator. Preserve `load_run_data()` logic. Port `run_final_evaluation()` from `final_report.py`. New main entry point `generate_report()`.

```python
"""PDF report generation for Aerocapture training runs.

Orchestrates: load data → generate SVG charts → write JSON metadata → invoke typst compile.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training import charts

# Trajectory column indices (15-element per-timestep)
_TRAJ_ALT = 0
_TRAJ_VEL = 3
_TRAJ_FPA = 4
_TRAJ_HEAT_FLUX = 6
_TRAJ_TIME = 7
_TRAJ_ENERGY = 8
_TRAJ_PDYN = 9
_TRAJ_BANK = 10
_TRAJ_INCL = 11
_TRAJ_GLOAD = 12
_TRAJ_DENSITY_RATIO = 13

# Final record column indices
_FR_ECC = 9
_FR_DV_TOTAL = 41
_FR_MAX_HEAT_FLUX = 16
_FR_MAX_G_LOAD = 17
_FR_PERI_ERR = 29
_FR_APO_ERR = 30
_FR_INCL_ERR = 46
_FR_BANK_CONSUMPTION = 45

# Path to Typst templates (relative to this file)
_TYPST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "typst"


def _check_typst() -> bool:
    """Check if typst CLI is available."""
    return shutil.which("typst") is not None


def load_run_data(scheme_dir: Path) -> tuple[list[dict], list[int]]:
    """Load JSONL training records and detect resume points.

    Returns (records, resume_generations).
    """
    jsonl_files = sorted(scheme_dir.glob("*.jsonl"))
    if not jsonl_files:
        return [], []

    records: list[dict] = []
    resume_gens: list[int] = []
    seen_gens: set[int] = set()

    for file_idx, jf in enumerate(jsonl_files):
        file_records = []
        for line in jf.read_text().strip().splitlines():
            if line.strip():
                r = json.loads(line)
                file_records.append(r)

        if file_idx > 0 and file_records:
            resume_gens.append(file_records[0]["generation"])

        for r in file_records:
            gen = r["generation"]
            # Deduplicate: last-writer-wins
            if gen in seen_gens:
                records = [rec for rec in records if rec["generation"] != gen]
            seen_gens.add(gen)
            records.append(r)

    records.sort(key=lambda r: r["generation"])
    return records, resume_gens


def run_final_evaluation(
    toml_path: Path,
    scheme_dir: Path,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None:
    """Run MC re-evaluation for final report.

    Uses the optimized TOML (with best params baked in) if available,
    otherwise falls back to the base TOML. For NN schemes, the base TOML
    already references the trained model JSON on disk.

    Returns (final_records, trajectories, dispersions) or None.
    """
    try:
        import aerocapture_rs
    except ImportError:
        print("WARNING: aerocapture_rs not available — skipping final evaluation", file=sys.stderr)
        return None

    # Prefer optimized TOML with best trained params baked in
    scheme_name = scheme_dir.name
    opt_toml = scheme_dir / f"optimized_{scheme_name}.toml"
    eval_toml = opt_toml if opt_toml.exists() else toml_path

    try:
        results = aerocapture_rs.run_mc(str(eval_toml.resolve()), include_trajectories=True)
        return results.final_records, results.trajectories, results.dispersions
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _read_mission_name(toml_path: Path) -> str:
    """Read mission name from TOML config."""
    from aerocapture.training.toml_utils import load_toml_with_bases
    data = load_toml_with_bases(toml_path)
    planet = data.get("planet", {}).get("name", "unknown")
    mission_type = data.get("mission", {}).get("type", "")
    return f"{planet.title()} {mission_type}".strip()


def _build_metadata(
    records: list[dict],
    scheme_dir: Path,
    n_sims: int,
    has_seed_pool: bool,
    has_trajectories: bool,
    toml_path: Path | None = None,
    has_cost_distribution: bool = False,
) -> dict:
    """Build metadata dict for cover page."""
    last = records[-1] if records else {}
    return {
        "scheme": last.get("scheme", scheme_dir.name),
        "mission": _read_mission_name(toml_path) if toml_path else "Unknown",
        "date": last.get("timestamp", "unknown")[:10] if last.get("timestamp") else "unknown",
        "best_cost": f"{last.get('best_cost', 0):.2f}",
        "capture_rate": f"{last.get('capture_rate', 0) * 100:.1f}%",
        "total_generations": str(last.get("generation", 0) + 1),
        "n_sims": str(n_sims),
        "config_hash": last.get("config_hash", "N/A"),
        "has_seed_pool": has_seed_pool,
        "has_trajectories": has_trajectories,
        "has_cost_distribution": has_cost_distribution,
    }


def _build_summary_table(
    final_records: npt.NDArray[np.float64],
) -> dict:
    """Build performance summary table data for Typst."""
    captured = final_records[:, _FR_ECC] < 1.0
    if not captured.any():
        return {"rows": [["No captured trajectories", *["—"] * 9]]}

    cap = final_records[captured]

    def row(name: str, values: npt.NDArray) -> list[str]:
        return [
            name,
            f"{np.mean(values):.2f}",
            f"{np.std(values):.2f}",
            f"{np.min(values):.2f}",
            f"{np.percentile(values, 5):.2f}",
            f"{np.percentile(values, 25):.2f}",
            f"{np.percentile(values, 50):.2f}",
            f"{np.percentile(values, 75):.2f}",
            f"{np.percentile(values, 95):.2f}",
            f"{np.max(values):.2f}",
        ]

    rows = [
        row("Max G-Load (g)", cap[:, _FR_MAX_G_LOAD]),
        row("Max Heat Flux (kW/m²)", cap[:, _FR_MAX_HEAT_FLUX]),
        row("Bank Consumption (deg)", cap[:, _FR_BANK_CONSUMPTION]),
        row("Periapsis Error (km)", cap[:, _FR_PERI_ERR]),
        row("Apoapsis Error (km)", cap[:, _FR_APO_ERR]),
        row("Inclination Error (deg)", cap[:, _FR_INCL_ERR]),
        row("Total ΔV (m/s)", np.clip(cap[:, _FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)),
    ]

    return {"rows": rows}


def generate_report(
    scheme_dir: Path,
    toml_path: Path | None = None,
    skip_final_eval: bool = False,
    keep_artifacts: bool = False,
    n_sims_override: int | None = None,
) -> Path | None:
    """Generate a single PDF report combining training convergence and mission performance.

    Returns path to generated PDF, or None if typst is not available.
    """
    # 1. Load training data
    records, resume_gens = load_run_data(scheme_dir)
    if not records:
        print(f"WARNING: No JSONL data found in {scheme_dir}", file=sys.stderr)
        return None

    # 2. Load final evaluation data
    final_records = None
    trajectories = None
    dispersions = None
    has_trajectories = False

    if toml_path and not skip_final_eval:
        result = run_final_evaluation(toml_path, scheme_dir)
        if result is not None:
            final_records, trajectories, dispersions = result
            has_trajectories = len(trajectories) > 0 and len(trajectories[0]) > 0

    # 3. Load corridor data
    corridor_data = None
    corridor_candidates = [
        scheme_dir.parent / "corridor_boundaries.npz",
        scheme_dir / "corridor_boundaries.npz",
    ]
    for cp in corridor_candidates:
        if cp.exists():
            npz = np.load(cp)
            corridor_data = {k: npz[k] for k in npz.files}
            break

    # 4. Determine captured mask and best trajectory
    captured_mask = None
    best_idx = None
    if final_records is not None:
        captured_mask = final_records[:, _FR_ECC] < 1.0
        if captured_mask.any():
            dv = final_records[:, _FR_DV_TOTAL].copy()
            dv[~captured_mask] = np.inf
            best_idx = int(np.argmin(dv))

    # 5. Generate charts to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Part 1: Training Convergence
        charts.chart_convergence(records, tmp / "convergence.svg", resume_gens)
        charts.chart_capture_constraint_rate(records, tmp / "capture_constraint_rate.svg", resume_gens)
        charts.chart_diversity_cost(records, tmp / "diversity_cost.svg", resume_gens)
        has_cost_dist = charts.chart_cost_distribution(records, tmp / "cost_distribution.svg")
        charts.chart_parameter_evolution(records, tmp / "parameter_evolution.svg", resume_gens)
        has_seed_pool = charts.chart_seed_pool(records, tmp / "seed_pool.svg", resume_gens)

        # Part 2: Mission Performance
        if has_trajectories and captured_mask is not None:
            charts.chart_corridor_pdyn(trajectories, captured_mask, tmp / "corridor_pdyn.svg", corridor_data=corridor_data)
            charts.chart_corridor_inclination(trajectories, captured_mask, tmp / "corridor_inclination.svg")
            charts.chart_corridor_bank(trajectories, captured_mask, tmp / "corridor_bank.svg")
            charts.chart_altitude_time(trajectories, captured_mask, tmp / "altitude_time.svg", best_idx=best_idx)
            charts.chart_heat_flux_time(trajectories, captured_mask, tmp / "heat_flux_time.svg")
            charts.chart_gload_time(trajectories, captured_mask, tmp / "gload_time.svg")
            charts.chart_bank_angle_time(trajectories, captured_mask, tmp / "bank_angle_time.svg")
            charts.chart_nav_density_ratio(trajectories, captured_mask, tmp / "nav_density_ratio.svg")
            charts.chart_entry_conditions(trajectories, captured_mask, tmp / "entry_conditions.svg")
            charts.chart_exit_conditions(final_records, tmp / "exit_conditions.svg")

        if final_records is not None:
            charts.chart_dv_distribution(final_records, tmp / "dv_distribution.svg")
            charts.chart_dv_individual_burns(final_records, tmp / "dv_individual_burns.svg")

            if dispersions is not None:
                charts.chart_dispersion_grid(final_records, dispersions, tmp / "dispersion_grid.svg")

        # 6. Write JSON data
        n_sims = len(final_records) if final_records is not None else 0
        meta = _build_metadata(records, scheme_dir, n_sims, has_seed_pool, has_trajectories, toml_path,
                               has_cost_distribution=has_cost_dist)
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))

        if final_records is not None:
            summary = _build_summary_table(final_records)
            (tmp / "summary_table.json").write_text(json.dumps(summary, indent=2))
        else:
            (tmp / "summary_table.json").write_text(json.dumps({"rows": [["No final evaluation data", *["—"] * 9]]}))

        # 7. Compile PDF
        if not _check_typst():
            print("WARNING: typst not found — install with 'brew install typst' or 'cargo install typst-cli'. Skipping PDF generation.", file=sys.stderr)
            return None

        output_pdf = scheme_dir / "report.pdf"
        typst_template = _TYPST_DIR / "report.typ"

        result = subprocess.run(
            ["typst", "compile", str(typst_template), "--input", f"dir={tmp}", str(output_pdf)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ERROR: typst compile failed:\n{result.stderr}", file=sys.stderr)
            return None

        print(f"Report generated: {output_pdf}")

        if keep_artifacts:
            artifacts = scheme_dir / "report_artifacts"
            if artifacts.exists():
                shutil.rmtree(artifacts)
            shutil.copytree(tmp, artifacts)
            print(f"Artifacts saved: {artifacts}")

        return output_pdf


def generate_comparison_report(
    training_output_dir: Path,
    schemes: list[str] | None = None,
) -> Path | None:
    """Generate cross-scheme comparison PDF."""
    # Find all scheme directories with JSONL data
    all_data: dict[str, list[dict]] = {}
    for d in sorted(training_output_dir.iterdir()):
        if not d.is_dir():
            continue
        if schemes and d.name not in schemes:
            continue
        records, _ = load_run_data(d)
        if records:
            all_data[d.name] = records

    if not all_data:
        print("WARNING: No training data found for comparison", file=sys.stderr)
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Comparison convergence chart
        charts.chart_comparison_convergence(all_data, tmp / "comparison_convergence.svg")

        # Comparison table
        table_data = _build_comparison_table(all_data)
        (tmp / "comparison_table.json").write_text(json.dumps(table_data, indent=2))

        # Metadata
        meta = {"date": "2026-03-24", "schemes": list(all_data.keys())}
        (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))

        if not _check_typst():
            print("WARNING: typst not found. Skipping PDF generation.", file=sys.stderr)
            return None

        output_pdf = training_output_dir / "comparison_report.pdf"
        typst_template = _TYPST_DIR / "comparison.typ"

        result = subprocess.run(
            ["typst", "compile", str(typst_template), "--input", f"dir={tmp}", str(output_pdf)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ERROR: typst compile failed:\n{result.stderr}", file=sys.stderr)
            return None

        print(f"Comparison report generated: {output_pdf}")
        return output_pdf


def _build_comparison_table(all_data: dict[str, list[dict]]) -> dict:
    """Build comparison metrics table."""
    headers = ["Scheme", "Best Cost", "Generations", "Capture %", "Conv. Speed"]
    rows = []
    for scheme, records in all_data.items():
        last = records[-1]
        # Convergence speed: generation where cost first reaches within 10% of final best
        threshold = last["best_cost"] * 1.1
        conv_gen = next((r["generation"] for r in records if r["best_cost"] <= threshold), "—")
        rows.append([
            scheme,
            f"{last.get('best_cost', 0):.2f}",
            str(last.get("generation", 0) + 1),
            f"{last.get('capture_rate', 0) * 100:.1f}%",
            str(conv_gen),
        ])
    return {"headers": headers, "rows": rows}


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Aerocapture training PDF reports")
    parser.add_argument("path", type=str, help="Scheme directory (single) or training_output/ (with --compare)")
    parser.add_argument("--toml", type=str, default=None, help="TOML training config path (needed for final evaluation)")
    parser.add_argument("--compare", action="store_true", help="Generate cross-scheme comparison report")
    parser.add_argument("--schemes", nargs="*", help="Filter by scheme names (comparison mode)")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep SVGs and JSON after compilation")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    if args.compare:
        generate_comparison_report(path, schemes=args.schemes)
    else:
        toml_path = Path(args.toml) if args.toml else None
        generate_report(path, toml_path=toml_path, keep_artifacts=args.keep_artifacts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add comparison convergence chart to charts.py**

Add to `charts.py`:

```python
# ── Comparison Charts ──────────────────────────────────────────────────────

SCHEME_COLORS: dict[str, str] = {
    "ftc": "#1f77b4",
    "neural_network": "#ff7f0e",
    "equilibrium_glide": "#2ca02c",
    "energy_controller": "#9467bd",
    "pred_guid": "#d62728",
    "fnpag": "#8c564b",
    "piecewise_constant": "#e377c2",
}


def chart_comparison_convergence(
    all_data: dict[str, list[dict]],
    output: Path,
) -> None:
    """Cross-scheme convergence comparison."""
    fig, ax = plt.subplots(figsize=FULL_WIDTH)

    for scheme, records in all_data.items():
        gens = [r["generation"] for r in records]
        best = [r["best_cost"] for r in records]
        color = SCHEME_COLORS.get(scheme, None)
        ax.semilogy(gens, best, label=scheme, color=color, linewidth=1.5)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Cost (log)")
    ax.set_title("Cross-Scheme Convergence")
    ax.legend(fontsize=8)
    sns.despine(fig=fig)
    _save_svg(fig, output)
```

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/test_report_pdf.py -v`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/report.py tests/test_report_pdf.py src/python/aerocapture/training/charts.py
git commit -m "feat: rewrite report.py as PDF orchestrator with Typst compilation"
```

---

## Task 10: Update train.py Integration

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (lines 726-747: CLI args, lines 863-1041: end-of-training)

- [ ] **Step 1: Update CLI flags in train.py**

In `train.py`, find `--skip-final-report` (line 745) and replace with:

```python
parser.add_argument("--skip-report", "--skip-final-report", action="store_true", dest="skip_report",
                    help="Skip PDF report generation at end of training")
```

Keep `--final-n-sims` (line 746) — it's passed to `generate_report()` as an override. The default (1000) is unchanged.

- [ ] **Step 2: Replace end-of-training report generation**

Replace the convergence report block (lines 863-870) and final report block (lines 947-1041) with:

```python
# ── Report Generation ──
if not args.skip_report:
    from aerocapture.training.report import generate_report
    toml_path = Path(args.toml)
    generate_report(Path(cfg.save_dir), toml_path, n_sims_override=args.final_n_sims)
```

This single call replaces both `generate_single_report()` and `generate_final_report()`.

- [ ] **Step 3: Run training smoke test**

Run: `uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 2 --n-pop 4 --skip-report`

Expected: Training completes without errors. No report generated.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: integrate PDF report into train.py, rename --skip-final-report to --skip-report"
```

---

## Task 11: Delete final_report.py and Old Plotly Code

**Files:**
- Delete: `src/python/aerocapture/training/final_report.py`
- Delete: `tests/test_final_report.py`
- Modify: `tests/test_training_report.py` (rewrite for new API)

- [ ] **Step 1: Delete final_report.py**

```bash
git rm src/python/aerocapture/training/final_report.py
```

- [ ] **Step 2: Delete old test file**

```bash
git rm tests/test_final_report.py
```

- [ ] **Step 3: Rewrite test_training_report.py for new API**

Update `tests/test_training_report.py` to test the new `load_run_data()` function and `generate_report()` orchestrator. The resume detection tests should remain (they test `load_run_data` which is preserved). Remove any Plotly-specific assertions (checking for `plotly.js` in HTML output).

Key changes:
- Replace `from aerocapture.training.report import generate_single_report` → `generate_report`
- Replace HTML existence checks → PDF existence checks (or mock typst)
- Keep resume detection tests intact (they test `load_run_data`)
- Remove `generate_comparison_report` HTML tests, replace with PDF tests

- [ ] **Step 4: Verify no remaining plotly imports**

Run: `rg "import plotly" src/python/`

Expected: No matches.

Run: `rg "from plotly" src/python/`

Expected: No matches.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`

Expected: All tests pass (some may need further adjustment).

- [ ] **Step 6: Commit**

```bash
git rm src/python/aerocapture/training/final_report.py tests/test_final_report.py
git add tests/test_training_report.py
git commit -m "refactor: remove final_report.py, plotly imports, and update tests for PDF pipeline"
```

---

## Task 12: Rewrite plot_comparison.py

**Files:**
- Modify: `src/python/aerocapture/training/plot_comparison.py`

- [ ] **Step 1: Rewrite plot_comparison.py to use seaborn + Typst**

Replace the matplotlib-only comparison plot with the same pipeline: generate SVGs → invoke Typst. The comparison data comes from `compare_guidance.py` output (JSON file).

Port the existing `plot_comparison.py` logic (2×3 matplotlib grid: capture rate, cost, apo/peri errors, DV, summary table) from matplotlib-only to seaborn-styled. The structure stays the same — 6 bar/box subplots comparing schemes.

```python
"""Plot comparison results as PNG via seaborn.

Reads JSON output from compare_guidance.py and produces a multi-panel
bar chart comparison of all schemes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aerocapture.training.charts import SCHEME_COLORS, _save_svg  # importing charts applies seaborn theme


def plot_comparison(results_path: Path, output: Path | None = None) -> Path:
    """Generate comparison PNG from compare_guidance.py JSON results."""
    data = json.loads(results_path.read_text())
    output = output or results_path.with_suffix(".png")

    schemes = list(data.keys())
    colors = [SCHEME_COLORS.get(s, "#888888") for s in schemes]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Panel 1: Capture rate
    ax = axes[0, 0]
    rates = [data[s].get("capture_rate", 0) * 100 for s in schemes]
    ax.bar(schemes, rates, color=colors)
    ax.set_ylabel("Capture Rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)

    # Panel 2: Best cost
    ax = axes[0, 1]
    costs = [data[s].get("best_cost", 0) for s in schemes]
    ax.bar(schemes, costs, color=colors)
    ax.set_ylabel("Best Cost")
    ax.tick_params(axis="x", rotation=30)

    # Panel 3: Mean DV (captured)
    ax = axes[0, 2]
    dvs = [data[s].get("mean_dv", 0) for s in schemes]
    ax.bar(schemes, dvs, color=colors)
    ax.set_ylabel("Mean ΔV (m/s)")
    ax.tick_params(axis="x", rotation=30)

    # Panel 4: Apoapsis error
    ax = axes[1, 0]
    apo = [data[s].get("mean_apo_err", 0) for s in schemes]
    ax.bar(schemes, apo, color=colors)
    ax.set_ylabel("Mean Apo Error (km)")
    ax.tick_params(axis="x", rotation=30)

    # Panel 5: Periapsis error
    ax = axes[1, 1]
    peri = [data[s].get("mean_peri_err", 0) for s in schemes]
    ax.bar(schemes, peri, color=colors)
    ax.set_ylabel("Mean Peri Error (km)")
    ax.tick_params(axis="x", rotation=30)

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = "\n".join(f"{s}: cap={data[s].get('capture_rate', 0)*100:.0f}% dv={data[s].get('mean_dv', 0):.0f}" for s in schemes)
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=9, verticalalignment="center", family="monospace")

    fig.suptitle("Guidance Scheme Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {output}")
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot guidance comparison results")
    parser.add_argument("--results", type=str, required=True, help="Path to comparison_results.json")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    plot_comparison(Path(args.results), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
```

Note: The exact JSON keys depend on what `compare_guidance.py` outputs — adapt field names during implementation.

- [ ] **Step 2: Commit**

```bash
git add src/python/aerocapture/training/plot_comparison.py
git commit -m "refactor: rewrite plot_comparison.py to use seaborn (drop plotly)"
```

---

## Task 13: Run Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run Rust tests**

Run: `cd src/rust && cargo test`

Expected: All pass.

- [ ] **Step 2: Run Python tests**

Run: `uv run pytest tests/ -v`

Expected: All pass.

- [ ] **Step 3: Run linter**

Run: `./lint_code.sh`

Expected: Clean (or only pre-existing warnings).

- [ ] **Step 4: Run Rust checks**

Run: `./check_all.sh`

Expected: All pass.

- [ ] **Step 5: Verify no plotly references remain**

Run: `rg "plotly" src/python/ pyproject.toml`

Expected: No matches.

- [ ] **Step 6: Generate a test report end-to-end**

Run (if trained params exist):

```bash
uv run python -m aerocapture.training.report training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml
```

Expected: `training_output/equilibrium_glide/report.pdf` generated. Open and verify all sections present.

---

## Task 14: Smart Commit (Final)

- [ ] **Step 1: Invoke smart-commit skill**

Use the `smart-commit` skill taking the whole `feature/plotly_to_typst` branch into account. This updates CLAUDE.md and README.md to reflect the new PDF report pipeline, then commits everything.
