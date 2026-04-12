# Output & Analysis Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Parquet output for large MC campaigns with embedded config metadata and dispersions, and fix dispersion scatter charts to use three-way trajectory classification.

**Architecture:** Python-side Parquet module (`parquet_output.py`) wraps `pyarrow` to serialize `BatchResults` (39 final-record + 26 dispersion columns) with schema-level metadata. The dispersion chart fix modifies `chart_dispersion_grid()` to accept a classification array and render blue/orange/red markers consistent with all other charts. Integration hooks added in `report.py` and `train.py`.

**Tech Stack:** Python 3.14+, pyarrow, numpy, pandas, matplotlib, pytest

---

### Task 1: Add pyarrow dependency

**Files:**
- Modify: `pyproject.toml:7` (dependencies list)

- [ ] **Step 1: Add pyarrow to dependencies**

In `pyproject.toml`, add `pyarrow` to the dependencies list:

```toml
dependencies = [
    "numpy>=2.4",
    "pandas>=3.0",
    "matplotlib>=3.10",
    "deap>=1.4",
    "scipy>=1.17.1",
    "rich>=14.3",
    "seaborn>=0.13",
    "SALib>=1.5",
    "pyarrow>=19.0",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs pyarrow

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pyarrow dependency for Parquet output"
```

---

### Task 2: Write Parquet output module -- write_parquet()

**Files:**
- Create: `src/python/aerocapture/training/parquet_output.py`
- Test: `tests/test_parquet_output.py`

- [ ] **Step 1: Write the failing test for write_parquet**

Create `tests/test_parquet_output.py`:

```python
"""Tests for Parquet output module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest


class TestWriteParquet:
    """Tests for write_parquet()."""

    def test_roundtrip_schema(self, tmp_path: Path) -> None:
        """Written Parquet has 65 columns: 39 final + 26 dispersion."""
        from aerocapture.training.parquet_output import write_parquet

        n = 10
        final_records = np.random.default_rng(42).standard_normal((n, 52))
        dispersions = np.random.default_rng(42).standard_normal((n, 26))
        config = {"guidance": {"type": "equilibrium_glide"}, "simulation": {"n_sims": n}}
        out = tmp_path / "test.parquet"

        write_parquet(out, final_records, dispersions, config)

        table = pq.read_table(out)
        assert table.num_columns == 65
        assert table.num_rows == n

    def test_column_names(self, tmp_path: Path) -> None:
        """Column names match FINAL_COLUMNS + DISPERSION_COLUMNS."""
        from aerocapture.training.parquet_output import FINAL_COLUMNS, write_parquet
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS

        n = 5
        final_records = np.random.default_rng(42).standard_normal((n, 52))
        dispersions = np.random.default_rng(42).standard_normal((n, 26))
        config = {"guidance": {"type": "equilibrium_glide"}}
        out = tmp_path / "test.parquet"

        write_parquet(out, final_records, dispersions, config)

        table = pq.read_table(out)
        expected = FINAL_COLUMNS + [f"disp_{c}" for c in DISPERSION_COLUMNS]
        assert table.column_names == expected

    def test_metadata_keys(self, tmp_path: Path) -> None:
        """Parquet metadata contains all required aerocapture keys."""
        from aerocapture.training.parquet_output import write_parquet

        n = 5
        final_records = np.random.default_rng(42).standard_normal((n, 52))
        dispersions = np.random.default_rng(42).standard_normal((n, 26))
        config = {"guidance": {"type": "equilibrium_glide"}, "simulation": {"n_sims": n}}
        out = tmp_path / "test.parquet"

        write_parquet(out, final_records, dispersions, config, toml_path="configs/test.toml")

        meta = pq.read_metadata(out).metadata
        assert b"aerocapture.config" in meta
        assert b"aerocapture.toml_path" in meta
        assert b"aerocapture.timestamp" in meta
        assert b"aerocapture.guidance_scheme" in meta
        assert b"aerocapture.n_sims" in meta

    def test_data_integrity(self, tmp_path: Path) -> None:
        """Round-tripped final record and dispersion values match originals."""
        from aerocapture.training.parquet_output import FINAL_COLUMNS, write_parquet

        rng = np.random.default_rng(42)
        n = 20
        final_records = rng.standard_normal((n, 52))
        dispersions = rng.standard_normal((n, 26))
        config = {"guidance": {"type": "ftc"}}
        out = tmp_path / "test.parquet"

        write_parquet(out, final_records, dispersions, config)

        import pandas as pd

        df = pd.read_parquet(out)
        # Final record columns use the 39 selected indices
        from aerocapture.training.parquet_output import FINAL_RECORD_INDICES

        for i, col in enumerate(FINAL_COLUMNS):
            np.testing.assert_array_almost_equal(df[col].values, final_records[:, FINAL_RECORD_INDICES[i]])

        from aerocapture.training.sensitivity import DISPERSION_COLUMNS

        for i, col in enumerate(DISPERSION_COLUMNS):
            np.testing.assert_array_almost_equal(df[f"disp_{col}"].values, dispersions[:, i])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parquet_output.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'aerocapture.training.parquet_output'`

- [ ] **Step 3: Write parquet_output.py with write_parquet()**

Create `src/python/aerocapture/training/parquet_output.py`:

```python
"""Parquet output for MC campaign results.

Writes BatchResults (39 final-record + 26 dispersion columns) to a single
Parquet file with schema-level metadata embedding the full resolved config
for reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from aerocapture.training.sensitivity import DISPERSION_COLUMNS

# The 39 final-record columns written to CSV (same names as Rust FINAL_CSV_COLUMNS).
# These are the indices into the 52-element internal final_record array.
FINAL_RECORD_INDICES: list[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 45, 48,
]

FINAL_COLUMNS: list[str] = [
    "sim_number",
    "altitude_km",
    "longitude_deg",
    "latitude_deg",
    "velocity_m_s",
    "flight_path_deg",
    "azimuth_deg",
    "radial_velocity_m_s",
    "energy_mj_kg",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_periapsis_deg",
    "true_anomaly_deg",
    "periapsis_alt_km",
    "apoapsis_alt_km",
    "max_heat_flux_kw_m2",
    "max_load_factor_g",
    "max_dyn_pressure_kpa",
    "alt_max_flux_km",
    "alt_max_load_km",
    "alt_max_pdyn_km",
    "time_max_flux_s",
    "time_max_load_s",
    "time_max_pdyn_s",
    "bounce_alt_km",
    "bounce_time_s",
    "sim_time_s",
    "integrated_flux_mj_m2",
    "periapsis_err_km",
    "apoapsis_err_km",
    "ifinal",
    "dv1_m_s",
    "dv2_m_s",
    "dv3_m_s",
    "dv12_m_s",
    "dv_total_m_s",
    "cumulative_bank_change_deg",
    "n_roll_reversals",
]


def write_parquet(
    path: str | Path,
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    config: dict,
    toml_path: str | None = None,
) -> None:
    """Write MC results to a Parquet file with embedded metadata.

    Parameters
    ----------
    path : output file path
    final_records : (N, 52) array of final trajectory records
    dispersions : (N, 26) array of per-run dispersion draws
    config : resolved TOML config dict (after base inheritance)
    toml_path : original config file path (for reference)
    """
    path = Path(path)
    n = final_records.shape[0]

    # Extract the 39 columns from the 52-element array
    final_data = final_records[:, FINAL_RECORD_INDICES]

    # Build column arrays: 39 final + 26 dispersion
    columns: list[pa.Array] = []
    names: list[str] = []

    for i, col_name in enumerate(FINAL_COLUMNS):
        columns.append(pa.array(final_data[:, i], type=pa.float64()))
        names.append(col_name)

    for i, col_name in enumerate(DISPERSION_COLUMNS):
        columns.append(pa.array(dispersions[:, i], type=pa.float64()))
        names.append(f"disp_{col_name}")

    table = pa.table(dict(zip(names, columns)))

    # Guidance scheme: try several TOML paths
    scheme = (
        config.get("guidance", {}).get("type")
        or config.get("guidance", {}).get("scheme")
        or "unknown"
    )

    # Schema-level metadata
    metadata = {
        b"aerocapture.config": json.dumps(config, default=str).encode(),
        b"aerocapture.toml_path": (toml_path or "").encode(),
        b"aerocapture.timestamp": datetime.now(timezone.utc).isoformat().encode(),
        b"aerocapture.guidance_scheme": scheme.encode(),
        b"aerocapture.n_sims": str(n).encode(),
    }

    # Merge with any existing Arrow metadata
    existing = table.schema.metadata or {}
    existing.update(metadata)
    table = table.replace_schema_metadata(existing)

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parquet_output.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/parquet_output.py tests/test_parquet_output.py
git commit -m "feat: add write_parquet for MC campaign output with embedded metadata"
```

---

### Task 3: Write read_parquet()

**Files:**
- Modify: `src/python/aerocapture/training/parquet_output.py`
- Modify: `tests/test_parquet_output.py`

- [ ] **Step 1: Write the failing test for read_parquet**

Append to `tests/test_parquet_output.py`:

```python
class TestReadParquet:
    """Tests for read_parquet()."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """write_parquet -> read_parquet returns same data and metadata."""
        from aerocapture.training.parquet_output import read_parquet, write_parquet

        rng = np.random.default_rng(99)
        n = 15
        final_records = rng.standard_normal((n, 52))
        dispersions = rng.standard_normal((n, 26))
        config = {"guidance": {"type": "pred_guid"}, "planet": {"name": "mars"}}
        out = tmp_path / "rt.parquet"

        write_parquet(out, final_records, dispersions, config, toml_path="configs/test.toml")
        df, meta = read_parquet(out)

        assert len(df) == n
        assert df.shape[1] == 65
        assert meta["guidance_scheme"] == "pred_guid"
        assert meta["toml_path"] == "configs/test.toml"
        assert "config" in meta
        assert meta["config"]["planet"]["name"] == "mars"

    def test_metadata_config_deserialized(self, tmp_path: Path) -> None:
        """Config metadata is deserialized back to a dict."""
        from aerocapture.training.parquet_output import read_parquet, write_parquet

        config = {"guidance": {"type": "fnpag"}, "nested": {"a": {"b": 42}}}
        out = tmp_path / "meta.parquet"
        final_records = np.zeros((3, 52))
        dispersions = np.zeros((3, 26))

        write_parquet(out, final_records, dispersions, config)
        _, meta = read_parquet(out)

        assert meta["config"]["nested"]["a"]["b"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parquet_output.py::TestReadParquet -v`
Expected: FAIL -- `ImportError: cannot import name 'read_parquet'`

- [ ] **Step 3: Add read_parquet() to parquet_output.py**

Append to `src/python/aerocapture/training/parquet_output.py`:

```python
def read_parquet(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read an aerocapture Parquet file, returning data and metadata.

    Parameters
    ----------
    path : path to the Parquet file

    Returns
    -------
    df : DataFrame with 65 columns (39 final + 26 dispersion)
    meta : dict with keys: config (dict), toml_path, timestamp,
           guidance_scheme, n_sims
    """
    import pandas as pd

    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()

    raw_meta = table.schema.metadata or {}
    meta: dict = {
        "config": json.loads(raw_meta.get(b"aerocapture.config", b"{}")),
        "toml_path": raw_meta.get(b"aerocapture.toml_path", b"").decode(),
        "timestamp": raw_meta.get(b"aerocapture.timestamp", b"").decode(),
        "guidance_scheme": raw_meta.get(b"aerocapture.guidance_scheme", b"").decode(),
        "n_sims": raw_meta.get(b"aerocapture.n_sims", b"0").decode(),
    }

    return df, meta
```

Also add the `pd` import -- add `import pandas as pd` at the top of the file (or keep it as a local import inside `read_parquet` to avoid import cost when only writing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parquet_output.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/parquet_output.py tests/test_parquet_output.py
git commit -m "feat: add read_parquet for loading MC results with metadata"
```

---

### Task 4: Fix dispersion chart three-way classification

**Files:**
- Modify: `src/python/aerocapture/training/charts.py:1129-1171` (chart_dispersion_grid)
- Modify: `tests/test_charts.py` (test_dispersion_grid)

- [ ] **Step 1: Write the failing test for classified dispersion grid**

In `tests/test_charts.py`, update the existing `test_dispersion_grid` test (or add a new one) to verify classification is used. Find the existing test and add a new test alongside it:

```python
def test_dispersion_grid_classification(
    self,
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    traj_class: npt.NDArray[np.int8],
    tmp_svg: Path,
) -> None:
    """Panel 20: dispersion grid uses three-way classification markers."""
    chart_dispersion_grid(final_records, dispersions, tmp_svg, traj_class=traj_class)
    assert tmp_svg.exists()
    content = tmp_svg.read_text()
    assert "<svg" in content
    # All three colors should appear (blue, orange, red from _TRAJ_COLORS)
    assert "#1f77b4" in content  # COLOR_CAPTURE (blue)
    assert "#ff7f0e" in content  # COLOR_CONSTRAINED (orange)
    assert "#d62728" in content  # COLOR_HYPERBOLIC (red)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_charts.py::TestCharts::test_dispersion_grid_classification -v`
Expected: FAIL -- `chart_dispersion_grid() got an unexpected keyword argument 'traj_class'`

- [ ] **Step 3: Update chart_dispersion_grid() to accept and use traj_class**

In `src/python/aerocapture/training/charts.py`, replace the existing `chart_dispersion_grid` function (lines 1129-1171):

```python
def chart_dispersion_grid(
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    output: Path,
    traj_class: npt.NDArray[np.int8] | None = None,
) -> None:
    """Panel 20: subplot grid -- each dispersion field vs log10(DV) with linear regression.

    When *traj_class* is provided, points are colored by trajectory outcome:
    blue (OK), orange (constrained), red x (failed).  Regression uses captured
    trajectories only.
    """
    n_fields = dispersions.shape[1]
    n_cols = 4
    n_rows = math.ceil(n_fields / n_cols)

    dv = _clip_dv(final_records[:, _FR_DV_TOTAL])
    log_dv = np.log10(dv)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), dpi=DPI)
    axes_flat = axes.flatten()

    for i in range(n_fields):
        ax = axes_flat[i]
        x = dispersions[:, i]

        if traj_class is not None:
            for cls, color in _TRAJ_COLORS.items():
                mask = traj_class == cls
                if not mask.any():
                    continue
                marker = "x" if cls == TRAJ_FAILED else "o"
                ax.scatter(x[mask], log_dv[mask], s=8, alpha=0.5, color=color, marker=marker)
            # Regression on captured trajectories only (OK + constrained)
            captured = traj_class != TRAJ_FAILED
            reg_x, reg_y = x[captured], log_dv[captured]
        else:
            ax.scatter(x, log_dv, s=8, alpha=0.5, color=COLOR_CAPTURE)
            reg_x, reg_y = x, log_dv

        # Linear regression (skip if all x values are identical)
        finite = np.isfinite(reg_x) & np.isfinite(reg_y)
        if np.sum(finite) > 2 and np.ptp(reg_x[finite]) > 0:
            result = stats.linregress(reg_x[finite], reg_y[finite])
            x_range = np.array([float(np.min(reg_x[finite])), float(np.max(reg_x[finite]))])
            ax.plot(x_range, result.slope * x_range + result.intercept, color=COLOR_WORST, linewidth=1.0)
            label_txt = f"R\u00b2={result.rvalue**2:.2f}\np={result.pvalue:.1e}"
            ax.annotate(label_txt, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6, verticalalignment="top")

        label = DISPERSION_LABELS[i] if i < len(DISPERSION_LABELS) else f"Field {i}"
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supylabel("log10(DV)", fontsize=9)
    fig.suptitle("Dispersion Correlation Grid", fontsize=11)
    fig.tight_layout()
    _save_svg(fig, output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_charts.py::TestCharts::test_dispersion_grid -v && uv run pytest tests/test_charts.py::TestCharts::test_dispersion_grid_classification -v`
Expected: both PASS (old test still works with no traj_class, new test exercises classification)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts.py tests/test_charts.py
git commit -m "feat: three-way trajectory classification in dispersion scatter grid"
```

---

### Task 5: Pass traj_class through report pipeline

**Files:**
- Modify: `src/python/aerocapture/training/report.py:529` (_generate_trajectory_charts)

- [ ] **Step 1: Update the chart_dispersion_grid call in report.py**

In `src/python/aerocapture/training/report.py`, the function `_generate_trajectory_charts` already computes `traj_class` at line 491. Update the call at line 529 to pass it through:

Replace:
```python
    charts.chart_dispersion_grid(final_records, dispersions, out_dir / "dispersion_grid.svg")
```

With:
```python
    charts.chart_dispersion_grid(final_records, dispersions, out_dir / "dispersion_grid.svg", traj_class=traj_class)
```

- [ ] **Step 2: Run existing report tests**

Run: `uv run pytest tests/test_training_report.py -v`
Expected: all PASS (traj_class is keyword-only with default None; existing code paths unaffected)

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/report.py
git commit -m "fix: pass trajectory classification to dispersion grid in report"
```

---

### Task 6: Integrate Parquet write into report.py

**Files:**
- Modify: `src/python/aerocapture/training/report.py:574-588` (generate_report, after eval_result)

- [ ] **Step 1: Add Parquet write after final evaluation in report.py**

In `src/python/aerocapture/training/report.py`, after line 588 (`final_records = final_records_arr`), add the Parquet write:

```python
                final_records = final_records_arr

                # Write Parquet output for analysis
                try:
                    from aerocapture.training.parquet_output import write_parquet
                    from aerocapture.training.toml_utils import load_toml_with_bases

                    resolved_config = load_toml_with_bases(toml_path)
                    parquet_path = scheme_dir / "final_eval.parquet"
                    write_parquet(parquet_path, final_records_arr, dispersions, resolved_config, toml_path=str(toml_path))
                    print(f"Parquet output: {parquet_path}")
                except ImportError:
                    pass  # pyarrow not installed
```

- [ ] **Step 2: Run existing report tests**

Run: `uv run pytest tests/test_training_report.py -v`
Expected: all PASS (Parquet write is inside a try/except, does not affect existing flow)

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/report.py
git commit -m "feat: auto-write Parquet output after final MC evaluation in reports"
```

---

### Task 7: Integrate Parquet write into train.py

**Files:**
- Modify: `src/python/aerocapture/training/train.py:1113-1118` (after best params save, before report generation)

- [ ] **Step 1: Add Parquet write in train.py final section**

In `src/python/aerocapture/training/train.py`, the report generation call at line 1113 delegates to `generate_report()` which now writes Parquet internally (Task 6). No additional integration is needed in `train.py` itself -- the Parquet file is written as part of the report's final evaluation.

Verify this by checking the flow: `train.py` calls `generate_report()` -> `run_final_evaluation()` produces `(final_records, trajectories, dispersions)` -> Parquet write added in Task 6 fires.

This task is a no-op verification step. Move to the next task.

- [ ] **Step 2: Run the full test suite to verify nothing is broken**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: all tests PASS

- [ ] **Step 3: Commit (only if any changes were needed)**

No commit needed if verification passes without changes.

---

### Task 8: Verify FINAL_RECORD_INDICES match Rust output

**Files:**
- Modify: `tests/test_parquet_output.py` (add index validation test)

- [ ] **Step 1: Write a test that validates FINAL_RECORD_INDICES against Rust CSV output**

The FINAL_RECORD_INDICES in `parquet_output.py` must match the indices used by Rust's `write_final_csv` in `output.rs`. The Rust code selects these 39 indices from the 52-element array. Add a test to `tests/test_parquet_output.py`:

```python
class TestFinalRecordIndices:
    """Validate FINAL_RECORD_INDICES consistency."""

    def test_index_count_matches_columns(self) -> None:
        """FINAL_RECORD_INDICES has exactly as many entries as FINAL_COLUMNS."""
        from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES

        assert len(FINAL_RECORD_INDICES) == len(FINAL_COLUMNS)

    def test_indices_within_bounds(self) -> None:
        """All indices are valid for a 52-element final record array."""
        from aerocapture.training.parquet_output import FINAL_RECORD_INDICES

        for idx in FINAL_RECORD_INDICES:
            assert 0 <= idx < 52, f"Index {idx} out of bounds for 52-element array"

    def test_no_duplicate_indices(self) -> None:
        """No duplicate indices in FINAL_RECORD_INDICES."""
        from aerocapture.training.parquet_output import FINAL_RECORD_INDICES

        assert len(FINAL_RECORD_INDICES) == len(set(FINAL_RECORD_INDICES))
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_parquet_output.py::TestFinalRecordIndices -v`
Expected: all 3 PASS

**Important note for implementer:** The `FINAL_RECORD_INDICES` in the plan above are based on exploration of the Rust `output.rs` `FINAL_CSV_INDICES` constant. Before implementing Task 2, read `src/rust/src/simulation/output.rs` and find the `FINAL_CSV_INDICES` (or equivalent array/slice that maps 52 -> 39). Copy those exact indices. If the Rust code does not have an explicit index array (e.g., it skips indices inline), derive the 39 indices by examining which of the 52 elements are written. This is a critical correctness detail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_parquet_output.py
git commit -m "test: validate FINAL_RECORD_INDICES bounds and consistency"
```

---

### Task 9: Lint and full test pass

**Files:**
- All modified files

- [ ] **Step 1: Run linter**

Run: `./lint_code.sh`
Expected: no errors

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 3: Fix any lint or test issues found**

Address any issues and re-run until clean.

- [ ] **Step 4: Commit fixes if any**

```bash
git add -u
git commit -m "fix: lint and test fixes for output analysis improvements"
```

---

### Task 10: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
