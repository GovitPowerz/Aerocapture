# Final Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 1000-sim final re-evaluation with Plotly HTML report showing statistical distributions of delta-V, orbital corrections, errors, and entry conditions after GA training.

**Architecture:** New module `final_report.py` with two pure functions (`run_final_evaluation` for sim execution, `generate_final_report` for Plotly HTML) plus CLI. Integrated into `train.py` end-of-loop. Follows patterns from existing `report.py`.

**Tech Stack:** Python 3.14, Plotly (core dep), numpy, tomllib, argparse

**Spec:** `docs/superpowers/specs/2026-03-13-final-evaluation-report-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/python/aerocapture/training/final_report.py` | **New.** `run_final_evaluation()` (TOML patching + sim), `generate_final_report()` (Plotly HTML), CLI `main()` |
| `src/python/aerocapture/training/train.py` | **Modify.** Add `--skip-final-report` and `--final-n-sims` args; call final report after GA loop |
| `tests/test_final_report.py` | **New.** Unit tests for report generation + config override logic |

---

## Chunk 1: Core report generation

### Task 1: `generate_final_report` — tests

**Files:**
- Create: `tests/test_final_report.py`

- [ ] **Step 1: Write tests for `generate_final_report`**

Create `tests/test_final_report.py` with synthetic final arrays. The legacy array is shape `(n, 53)`.

Key column indices (from spec):
- 8: energy, 10: eccentricity, 11: inclination
- 4: velocity, 5: FPA, 28: sim_time
- 30: periapsis_err, 31: apoapsis_err
- 38: dv1, 39: dv2, 40: dv3, 42: dv_total

```python
"""Tests for final evaluation report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_captured_array(n: int = 100, seed: int = 42) -> np.ndarray:
    """Create a synthetic final conditions array with all captured trajectories."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((n, 53))
    arr[:, 0] = np.arange(n)  # sim_number
    arr[:, 4] = rng.normal(5500, 50, n)  # velocity_m_s
    arr[:, 5] = rng.normal(-12.0, 0.5, n)  # flight_path_deg
    arr[:, 8] = rng.uniform(-2.0, -0.5, n)  # energy < 0 (captured)
    arr[:, 10] = rng.uniform(0.3, 0.9, n)  # ecc < 1 (captured)
    arr[:, 11] = rng.normal(50.0, 1.0, n)  # inclination_deg
    arr[:, 28] = rng.uniform(300, 600, n)  # sim_time_s
    arr[:, 30] = rng.normal(0, 10, n)  # periapsis_err_km
    arr[:, 31] = rng.normal(0, 15, n)  # apoapsis_err_km
    arr[:, 38] = rng.exponential(20, n)  # dv1
    arr[:, 39] = rng.exponential(50, n)  # dv2
    arr[:, 40] = rng.exponential(10, n)  # dv3
    arr[:, 42] = arr[:, 38] + arr[:, 39] + arr[:, 40]  # dv_total
    return arr


def _make_mixed_array(n_captured: int = 80, n_hyper: int = 20, seed: int = 42) -> np.ndarray:
    """Create array with both captured and hyperbolic trajectories."""
    arr = _make_captured_array(n_captured + n_hyper, seed)
    # Make last n_hyper trajectories hyperbolic
    arr[n_captured:, 8] = np.abs(arr[n_captured:, 8])  # energy > 0
    arr[n_captured:, 10] = 1.0 + np.abs(arr[n_captured:, 10])  # ecc > 1
    return arr


def _make_all_hyperbolic(n: int = 50, seed: int = 42) -> np.ndarray:
    """Create array with zero captured trajectories."""
    arr = _make_captured_array(n, seed)
    arr[:, 8] = np.abs(arr[:, 8])  # energy > 0
    arr[:, 10] = 1.0 + np.abs(arr[:, 10])  # ecc > 1
    return arr


class TestGenerateFinalReport:
    def test_produces_html_file(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        arr = _make_captured_array(100)
        output = tmp_path / "report.html"
        result = generate_final_report(arr, "equilibrium_glide", 50.0, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "plotly" in content.lower()

    def test_html_contains_expected_panels(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        arr = _make_captured_array(100)
        output = tmp_path / "report.html"
        generate_final_report(arr, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Delta-V" in content
        assert "Apoapsis" in content
        assert "Periapsis" in content
        assert "Inclination" in content

    def test_mixed_captured_and_hyperbolic(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        arr = _make_mixed_array(80, 20)
        output = tmp_path / "report.html"
        result = generate_final_report(arr, "ftc", 50.0, output)
        assert result == output
        assert output.exists()

    def test_zero_captures_does_not_crash(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        arr = _make_all_hyperbolic(50)
        output = tmp_path / "report.html"
        result = generate_final_report(arr, "fnpag", 50.0, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "No captured trajectories" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerocapture.training.final_report'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_final_report.py
git commit -m "test: add failing tests for final evaluation report"
```

---

### Task 2: `generate_final_report` — implementation

**Files:**
- Create: `src/python/aerocapture/training/final_report.py`

- [ ] **Step 4: Write `generate_final_report` function**

Create `src/python/aerocapture/training/final_report.py`. Follow `report.py` patterns: lazy plotly import, `fig.write_html(str(path), include_plotlyjs=True)`.

```python
"""Final evaluation report — statistical distributions from large-MC re-evaluation.

Usage (standalone):
    uv run python -m aerocapture.training.final_report \\
        training_output/equilibrium_glide/ \\
        --toml configs/training/msr_aller_eqglide_train.toml \\
        --n-sims 1000 --seed 42
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

# Legacy array column indices (53-column format from evaluate._parse_final_to_legacy_array)
_COL_VELOCITY = 4
_COL_FPA = 5
_COL_ENERGY = 8
_COL_ECC = 10
_COL_INCL = 11
_COL_PERI_ERR = 30
_COL_APO_ERR = 31
_COL_DV1 = 38
_COL_DV2 = 39
_COL_DV3 = 40
_COL_DV_TOTAL = 42

_PERCENTILES = [5, 25, 50, 75, 95]

# Colors consistent with report.py palette
_COLOR_PRIMARY = "#2196F3"
_COLOR_SECONDARY = "#FF9800"
_COLOR_TERTIARY = "#4CAF50"
_COLOR_DV1 = "#2196F3"
_COLOR_DV2 = "#FF9800"
_COLOR_DV3 = "#4CAF50"
_COLOR_CAPTURED = "#4CAF50"
_COLOR_HYPERBOLIC = "#F44336"
_COLOR_CDF = "#9C27B0"


def generate_final_report(
    final_array: npt.NDArray[np.float64],
    scheme: str,
    target_inclination: float,
    output_path: Path,
) -> Path:
    """Generate self-contained Plotly HTML report with statistical distributions.

    Returns path to generated HTML file.
    Handles 0% capture rate gracefully (empty distribution panels with annotation).
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    energy = final_array[:, _COL_ENERGY]
    ecc = final_array[:, _COL_ECC]
    captured = (ecc < 1.0) & (energy < 0)
    n_total = len(final_array)
    n_captured = int(captured.sum())
    capture_rate = n_captured / n_total * 100 if n_total > 0 else 0.0

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Total Delta-V Distribution",
            "Individual Correction Burns",
            "Apoapsis Error (km)",
            "Periapsis Error (km)",
            "Inclination Error (deg)",
            "Entry Conditions",
            "Delta-V vs Orbital Error",
            "Summary Statistics",
        ),
        specs=[
            [{"secondary_y": True}, {}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {}],
            [{}, {"type": "table"}],
        ],
    )

    if n_captured == 0:
        # Add "No captured trajectories" annotation to all distribution panels
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]:
            fig.add_annotation(
                text="No captured trajectories",
                xref=f"x{(row - 1) * 2 + col} domain",
                yref=f"y{(row - 1) * 2 + col} domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14, "color": "#F44336"},
            )
    else:
        cap = final_array[captured]
        dv_total = cap[:, _COL_DV_TOTAL]
        dv1 = cap[:, _COL_DV1]
        dv2 = cap[:, _COL_DV2]
        dv3 = cap[:, _COL_DV3]
        apo_err = cap[:, _COL_APO_ERR]
        peri_err = cap[:, _COL_PERI_ERR]
        incl_err = cap[:, _COL_INCL] - target_inclination

        # Panel 1: Total Delta-V histogram + CDF
        _add_hist_cdf(fig, dv_total, "Delta-V (m/s)", _COLOR_PRIMARY, row=1, col=1)

        # Panel 2: Individual corrections overlaid
        fig.add_trace(go.Histogram(x=dv1, name="dv1 (incl.)", opacity=0.5, marker_color=_COLOR_DV1, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv2, name="dv2 (SMA/ecc)", opacity=0.5, marker_color=_COLOR_DV2, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv3, name="dv3 (RAAN)", opacity=0.5, marker_color=_COLOR_DV3, nbinsx=30), row=1, col=2)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="m/s", row=1, col=2)

        # Panel 3: Apoapsis error
        _add_hist_cdf(fig, apo_err, "km", _COLOR_PRIMARY, row=2, col=1)

        # Panel 4: Periapsis error
        _add_hist_cdf(fig, peri_err, "km", _COLOR_SECONDARY, row=2, col=2)

        # Panel 5: Inclination error
        _add_hist_cdf(fig, incl_err, "deg", _COLOR_TERTIARY, row=3, col=1)

    # Panel 6: Entry conditions scatter (all trajectories, colored by outcome)
    velocity = final_array[:, _COL_VELOCITY]
    fpa = final_array[:, _COL_FPA]
    dv_all = final_array[:, _COL_DV_TOTAL]

    if n_captured > 0:
        fig.add_trace(
            go.Scatter(
                x=velocity[captured],
                y=fpa[captured],
                mode="markers",
                name="Captured",
                marker={"color": _COLOR_CAPTURED, "size": np.clip(dv_all[captured] / 20, 3, 15), "opacity": 0.6},
            ),
            row=3,
            col=2,
        )
    hyper = ~captured
    if hyper.any():
        fig.add_trace(
            go.Scatter(
                x=velocity[hyper],
                y=fpa[hyper],
                mode="markers",
                name="Hyperbolic",
                marker={"color": _COLOR_HYPERBOLIC, "size": 5, "opacity": 0.6, "symbol": "x"},
            ),
            row=3,
            col=2,
        )
    fig.update_xaxes(title_text="Entry Velocity (m/s)", row=3, col=2)
    fig.update_yaxes(title_text="Entry FPA (deg)", row=3, col=2)

    # Panel 7: Delta-V vs orbital error scatter (captured only)
    if n_captured > 0:
        cap = final_array[captured]
        orbital_err = np.sqrt(cap[:, _COL_APO_ERR] ** 2 + cap[:, _COL_PERI_ERR] ** 2)
        fig.add_trace(
            go.Scatter(
                x=orbital_err,
                y=cap[:, _COL_DV_TOTAL],
                mode="markers",
                name="DV vs Error",
                marker={"color": _COLOR_PRIMARY, "opacity": 0.5},
            ),
            row=4,
            col=1,
        )
    fig.update_xaxes(title_text="Orbital Error (km)", row=4, col=1)
    fig.update_yaxes(title_text="Delta-V (m/s)", row=4, col=1)

    # Panel 8: Summary statistics table
    _add_summary_table(fig, final_array, captured, target_inclination, row=4, col=2)

    fig.update_layout(
        height=1600,
        title_text=f"Final Evaluation — {scheme} ({n_captured}/{n_total} captured, {capture_rate:.1f}%)",
        showlegend=True,
    )

    fig.write_html(str(output_path), include_plotlyjs=True)
    return output_path


def _add_hist_cdf(
    fig: object,
    data: npt.NDArray[np.float64],
    xaxis_label: str,
    color: str,
    row: int,
    col: int,
) -> None:
    """Add histogram + CDF overlay with percentile lines to a subplot."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    fig.add_trace(go.Histogram(x=data, name=xaxis_label, marker_color=color, opacity=0.7, nbinsx=40, showlegend=False), row=row, col=col)  # type: ignore[union-attr]

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(go.Scatter(x=sorted_data, y=cdf, name="CDF", line={"color": _COLOR_CDF, "width": 2}, showlegend=False), row=row, col=col, secondary_y=True)  # type: ignore[union-attr]

    # Percentile lines
    for p in _PERCENTILES:
        val = float(np.percentile(data, p))
        fig.add_vline(x=val, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col, annotation_text=f"p{p}")  # type: ignore[union-attr]

    fig.update_xaxes(title_text=xaxis_label, row=row, col=col)  # type: ignore[union-attr]
    fig.update_yaxes(title_text="Count", row=row, col=col, secondary_y=False)  # type: ignore[union-attr]
    fig.update_yaxes(title_text="CDF", row=row, col=col, secondary_y=True)  # type: ignore[union-attr]


def _add_summary_table(
    fig: object,
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
    target_inclination: float,
    row: int,
    col: int,
) -> None:
    """Add summary statistics table to a subplot."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    n_total = len(final_array)
    n_captured = int(captured.sum())

    header = ["Metric", "Mean", "Std", "p5", "p25", "p50", "p75", "p95"]
    rows: list[list[str]] = []

    if n_captured > 0:
        cap = final_array[captured]
        metrics = {
            "Delta-V total (m/s)": cap[:, _COL_DV_TOTAL],
            "dv1 incl. (m/s)": cap[:, _COL_DV1],
            "dv2 SMA/ecc (m/s)": cap[:, _COL_DV2],
            "dv3 RAAN (m/s)": cap[:, _COL_DV3],
            "Apoapsis err (km)": cap[:, _COL_APO_ERR],
            "Periapsis err (km)": cap[:, _COL_PERI_ERR],
            "Inclination err (deg)": cap[:, _COL_INCL] - target_inclination,
        }
        for name, data in metrics.items():
            pcts = np.percentile(data, _PERCENTILES)
            rows.append([name, f"{data.mean():.2f}", f"{data.std():.2f}", *[f"{p:.2f}" for p in pcts]])

    # Add capture rate as first row
    rows.insert(0, [f"Capture rate: {n_captured}/{n_total} ({n_captured / n_total * 100:.1f}%)", "", "", "", "", "", "", ""])

    cells_transposed = list(zip(*rows, strict=False)) if rows else [[] for _ in header]
    fig.add_trace(  # type: ignore[union-attr]
        go.Table(
            header={"values": header, "fill_color": _COLOR_PRIMARY, "font_color": "white", "align": "center"},
            cells={"values": cells_transposed, "align": "center"},
        ),
        row=row,
        col=col,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: add generate_final_report with Plotly HTML distributions"
```

---

## Chunk 2: Re-evaluation runner + CLI

### Task 3: `run_final_evaluation` — tests

**Files:**
- Modify: `tests/test_final_report.py`

- [ ] **Step 7: Write tests for `run_final_evaluation` config patching**

These test the TOML patching logic without actually running the simulator. Add to `tests/test_final_report.py`:

```python
class TestRunFinalEvaluation:
    def test_patches_n_sims_and_seed(self, tmp_path: Path) -> None:
        """Verify TOML patching writes correct n_sims and seed."""
        import tomllib

        from aerocapture.training.final_report import _patch_toml_for_final_eval

        # Create a minimal TOML
        toml_content = '[monte_carlo]\nn_sims = 10\nseed = 1\n[guidance]\ntype = "ftc"\n'
        src_toml = tmp_path / "base.toml"
        src_toml.write_text(toml_content)

        patched = _patch_toml_for_final_eval(src_toml, n_sims=1000, seed=9999)
        with open(patched, "rb") as f:
            data = tomllib.load(f)
        assert data["monte_carlo"]["n_sims"] == 1000
        assert data["monte_carlo"]["seed"] == 9999
        patched.unlink()

    def test_reads_target_inclination_from_toml(self, tmp_path: Path) -> None:
        """Verify target inclination extraction from TOML."""
        from aerocapture.training.final_report import _read_target_inclination

        toml_content = '[flight.target_orbit]\napoapsis = 500.0\nperiapsis = 250.0\ninclination = 50.0\n'
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)

        assert _read_target_inclination(toml_file) == 50.0

    def test_target_inclination_missing_returns_zero(self, tmp_path: Path) -> None:
        """Fallback to 0.0 if inclination not in TOML."""
        from aerocapture.training.final_report import _read_target_inclination

        toml_content = '[flight.target_orbit]\napoapsis = 500.0\n'
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)

        assert _read_target_inclination(toml_file) == 0.0

    def test_seed_zero_is_not_replaced(self, tmp_path: Path) -> None:
        """Explicit seed=0 should be preserved, not treated as None."""
        import tomllib

        from aerocapture.training.final_report import _patch_toml_for_final_eval

        toml_content = '[monte_carlo]\nn_sims = 10\nseed = 99\n'
        src_toml = tmp_path / "base.toml"
        src_toml.write_text(toml_content)

        patched = _patch_toml_for_final_eval(src_toml, n_sims=100, seed=0)
        with open(patched, "rb") as f:
            data = tomllib.load(f)
        assert data["monte_carlo"]["seed"] == 0
        patched.unlink()


@pytest.mark.skipif(
    not Path("src/rust/target/release/aerocapture").exists(),
    reason="Rust binary not built",
)
class TestFinalReportCLI:
    def test_cli_produces_html(self, tmp_path: Path) -> None:
        """Integration test: standalone CLI produces an HTML report."""
        import subprocess

        # This test requires a valid training output with best_params.json
        # and a matching TOML config. Skip if prerequisites are missing.
        pytest.importorskip("plotly")

        result = subprocess.run(
            ["uv", "run", "python", "-m", "aerocapture.training.final_report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--n-sims" in result.stdout
```

- [ ] **Step 8: Run tests to verify new ones fail**

Run: `uv run pytest tests/test_final_report.py::TestRunFinalEvaluation -v`
Expected: FAIL — `ImportError: cannot import name '_patch_toml_for_final_eval'`

- [ ] **Step 9: Commit**

```bash
git add tests/test_final_report.py
git commit -m "test: add failing tests for TOML patching and target inclination"
```

---

### Task 4: `run_final_evaluation` + helpers — implementation

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py`

- [ ] **Step 10: Add `_patch_toml_for_final_eval`, `_read_target_inclination`, and `run_final_evaluation`**

Add these functions to `final_report.py` (before `generate_final_report`):

```python
def _read_target_inclination(toml_path: Path) -> float:
    """Read target inclination from TOML [flight.target_orbit] section."""
    import tomllib

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return float(data.get("flight", {}).get("target_orbit", {}).get("inclination", 0.0))


def _patch_toml_for_final_eval(
    base_toml_path: Path,
    n_sims: int,
    seed: int,
) -> Path:
    """Create a temporary TOML with overridden n_sims and mc_seed."""
    import tempfile
    import tomllib

    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    toml_data.setdefault("monte_carlo", {})["n_sims"] = n_sims
    toml_data["monte_carlo"]["seed"] = seed

    from aerocapture.training.evaluate import _write_toml

    fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="final_eval_")
    import os

    os.close(fd)
    output_path = Path(path_str)
    _write_toml(toml_data, output_path)
    return output_path


def run_final_evaluation(
    cfg: TrainingConfig,
    n_sims: int = 1000,
    seed: int | None = None,
    cwd: Path | None = None,
) -> npt.NDArray[np.float64] | None:
    """Run large-MC re-evaluation of best solution.

    Patches the TOML config to override n_sims and mc_seed, then runs
    the simulator. Returns final conditions array (n_sims, 53) in
    legacy format, or None if the simulation fails.
    """
    from aerocapture.training.evaluate import run_simulation

    if cfg.sim.toml_config is None:
        return None

    cwd_path = Path(cwd) if cwd else Path(".")
    base_toml = cwd_path / cfg.sim.toml_config

    patched_toml = _patch_toml_for_final_eval(base_toml, n_sims, 0 if seed is None else seed)
    orig_toml = cfg.sim.toml_config
    try:
        cfg.sim.toml_config = str(patched_toml)
        return run_simulation(cfg, cwd=cwd)
    finally:
        cfg.sim.toml_config = orig_toml
        patched_toml.unlink(missing_ok=True)
```

Also add the import at the top of the file (after the existing imports):

```python
from aerocapture.training.config import TrainingConfig
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All 9 tests PASS

- [ ] **Step 12: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: add run_final_evaluation with TOML patching"
```

---

### Task 5: Standalone CLI

**Files:**
- Modify: `src/python/aerocapture/training/final_report.py`

- [ ] **Step 13: Add `main()` function and `__main__` block**

Add at the bottom of `final_report.py`:

```python
def main() -> None:
    """CLI entry point for standalone final evaluation."""
    import argparse
    import json
    import sys

    from aerocapture.training.config import TrainingConfig
    from aerocapture.training.evaluate import decode_params_from_chromosome, write_guidance_toml

    parser = argparse.ArgumentParser(description="Run final evaluation and generate report")
    parser.add_argument("scheme_dir", type=str, help="Path to scheme output directory (contains best_params.json or best_model.json)")
    parser.add_argument("--toml", type=str, required=True, help="Base TOML config path")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of MC simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="MC seed for re-evaluation")
    args = parser.parse_args()

    scheme_dir = Path(args.scheme_dir)
    if not scheme_dir.exists():
        print(f"ERROR: Directory not found: {scheme_dir}")
        sys.exit(1)

    # Detect scheme from directory name
    scheme = scheme_dir.name

    # Load best params and patch TOML
    params_path = scheme_dir / "best_params.json"
    model_path = scheme_dir / "best_model.json"

    cfg = TrainingConfig()
    cfg.sim.toml_config = args.toml
    cfg.sim.executable = "src/rust/target/release/aerocapture"
    cfg.guidance_type = scheme

    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        opt_toml = scheme_dir / f"optimized_{scheme}.toml"
        if not opt_toml.exists():
            base_toml = Path(args.toml)
            write_guidance_toml(base_toml, scheme, params, opt_toml)
        cfg.sim.toml_config = str(opt_toml)
    elif model_path.exists():
        # NN: model already written, just use existing TOML
        import tomllib

        with open(args.toml, "rb") as f:
            toml_data = tomllib.load(f)
        cfg.sim.nn_param_file = toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
    else:
        print(f"ERROR: No best_params.json or best_model.json found in {scheme_dir}")
        sys.exit(1)

    target_incl = _read_target_inclination(Path(args.toml))

    print(f"Running {args.n_sims}-sim final evaluation for {scheme} (seed={args.seed})...")
    final = run_final_evaluation(cfg, n_sims=args.n_sims, seed=args.seed)

    if final is None:
        print("ERROR: Simulation failed")
        sys.exit(1)

    output_path = scheme_dir / "final_report.html"
    generate_final_report(final, scheme, target_incl, output_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 14: Run linting**

Run: `uv run ruff check src/python/aerocapture/training/final_report.py && uv run ruff format --check src/python/aerocapture/training/final_report.py`
Expected: No errors. Fix any issues.

- [ ] **Step 15: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/test_final_report.py -v`
Expected: All 9 tests PASS

- [ ] **Step 16: Commit**

```bash
git add src/python/aerocapture/training/final_report.py
git commit -m "feat: add standalone CLI for final evaluation report"
```

---

## Chunk 3: train.py integration

### Task 6: Integrate final report into train.py

**Files:**
- Modify: `src/python/aerocapture/training/train.py:448-550`

- [ ] **Step 17: Add CLI flags to train.py argparse**

In `train.py`, after line 468 (`--rotate-seeds`), add:

```python
    parser.add_argument("--skip-final-report", action="store_true", help="Skip final re-evaluation report")
    parser.add_argument("--final-n-sims", type=int, default=1000, help="Number of MC sims for final re-evaluation (default: 1000)")
```

- [ ] **Step 18: Add final report call after GA loop**

In `train.py`, after the existing final re-evaluation block (after line 549), add the final report generation. Replace the block from line 538 to 549 with an expanded version:

```python
        final = run_simulation(cfg, cwd=cwd)
        if final is not None:
            cost = compute_cost(final)
            print(f"Final re-evaluation cost: {cost:.4e}")
            energy = final[:, 8]
            ecc = final[:, 10]
            captured = (ecc < 1.0) & (energy < 0)
            print(f"  Captured: {captured.sum()}/{len(final)}")
            if captured.any():
                print(f"  Apoapsis err (km):  mean={np.abs(final[captured, 31]).mean():.1f}")
                print(f"  Periapsis err (km): mean={np.abs(final[captured, 30]).mean():.1f}")
                print(f"  Delta-V (m/s):      mean={final[captured, 42].mean():.1f}")

        # Final evaluation report (large-MC re-evaluation)
        if not args.skip_final_report:
            from aerocapture.training.final_report import (
                _read_target_inclination,
                generate_final_report,
                run_final_evaluation,
            )

            # For non-NN schemes, use the optimized TOML (contains best guidance params)
            # For NN, the base TOML already references the NN JSON on disk
            if cfg.guidance_type != "neural_network":
                opt_toml = Path(cfg.save_dir) / f"optimized_{cfg.guidance_type}.toml"
                if opt_toml.exists():
                    cfg.sim.toml_config = str(opt_toml)

            # Read target inclination from the base TOML (target_orbit is unchanged by patching)
            target_incl = _read_target_inclination(Path(cwd or ".") / args.toml)

            final_seed = args.seed + 9999
            print(f"\nRunning {args.final_n_sims}-sim final evaluation (seed={final_seed})...")
            final_eval = run_final_evaluation(cfg, n_sims=args.final_n_sims, seed=final_seed, cwd=cwd)
            if final_eval is not None:
                report_path = Path(cfg.save_dir) / "final_report.html"
                generate_final_report(final_eval, cfg.guidance_type, target_incl, report_path)
                print(f"Final report saved to {report_path}")
            else:
                print("WARNING: Final evaluation simulation failed, skipping report")
```

- [ ] **Step 19: Run linting**

Run: `uv run ruff check src/python/aerocapture/training/train.py && uv run ruff format --check src/python/aerocapture/training/train.py`
Expected: No errors

- [ ] **Step 20: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 21: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: integrate final evaluation report into train.py"
```

---

## Chunk 4: Type checking + finalization

### Task 7: Type checking and final cleanup

- [ ] **Step 22: Run mypy**

Run: `uv run mypy src/python/aerocapture/training/final_report.py`
Expected: No errors. Fix any type issues (plotly `type: ignore` comments are expected, follow `report.py` pattern).

- [ ] **Step 23: Run full linting pipeline**

Run: `./lint_code.sh`
Expected: All clean

- [ ] **Step 24: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 25: Commit any fixes**

```bash
git add -u
git commit -m "fix: address mypy/lint issues in final_report"
```

(Skip this commit if no fixes were needed.)

---

### Task 8: Finalize with smart-commit

- [ ] **Step 26: Use smart-commit skill**

Invoke the `smart-commit` skill, taking into account all commits on this branch linked to this plan. The smart-commit should update CLAUDE.md and README.md to reflect the new `final_report.py` module, CLI usage, and `--skip-final-report` / `--final-n-sims` flags.
