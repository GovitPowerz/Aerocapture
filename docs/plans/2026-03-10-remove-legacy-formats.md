# Remove Legacy `.in` Input / Fortran Output Formats — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all legacy Fortran `.in` input format code and D-notation text output format code from both Rust and Python, leaving CSV as the only supported format.

**Architecture:** Bottom-up approach. First regenerate reference data in CSV format, then migrate tests to use CSV, then delete dead Python code, then remove Rust `OutputFormat::Text` and legacy neural loader, then clean up orphaned reference data. Each task is independently committable and the test suite stays green throughout.

**Tech Stack:** Rust (cargo test, cargo clippy), Python (pytest, ruff, mypy), TOML configs

---

## Inventory of Legacy Code

### Rust (to remove)
- `config.rs:83-91` — `OutputFormat::Text` enum variant
- `output.rs:118-174` — `fortran_float()`, `write_photo_text_line()`, `write_final_text_line()` + tests (lines 176-251)
- `runner.rs:165,220-262` — `write_text_output()` function + match branch
- `neural.rs:146-227` — `from_legacy()` Fortran nn_param loader

### Python (to remove)
- `io/parse_fort.py` — entire file (dead, parses `fort.201-204` which Rust never writes)
- `io/parse_initial.py` — entire file (dead, parses `initial.*` which Rust never writes)
- `io/_fortran.py` — entire file (shared utility, all callers removed or migrated)
- `io/__init__.py` — remove `parse_fort`, `parse_initial` exports
- `io/parse_photo.py:119-133` — Fortran text fallback branch
- `io/parse_final.py:14,125-141` — Fortran text fallback branch
- `training/config.py:72` — `init_file` default (`"train_nn.in"`)
- `training/config.py:151-179` — `load_base_network()` Fortran branch
- `training/evaluate.py:79-106` — `write_nn_params()` (dead, Rust reads JSON)
- `training/evaluate.py:174-183` — `_parse_final_to_legacy_array()` Fortran branch
- `training/evaluate.py:211-220` — `run_simulation()` stdin/.in branch
- `plotting/corridor.py:11,14-22` — `_load_fortran_table()` + import (uses missing data files)
- `scripts/plot_mc_comparison.py:9,13-21` — `load_final()` uses `parse_fortran_line` directly
- `tests/compare_results.py` — entire file (standalone script with own Fortran parser, unused)

### TOML configs (to update)
- `configs/test_ref_orig.toml` — `output_format = "text"` → remove (CSV is default)
- `configs/test_high_bank_orig.toml` — same
- `configs/test_guided_orig.toml` — same

### Reference data (to regenerate or delete)
- `tests/reference_data/rust_golden/{ref,high_bank,guided}/` — regenerate as CSV
- `tests/reference_data/{ref_orig,high_bank_orig,guided_orig}/` — Fortran golden data, keep for history or delete
- `tests/reference_data/{guided_nn_ftc,guided_nn2,ref_nn,mc10_orig}/` — orphaned Fortran-only dirs (contain `fort.*`, `sorties/`, `stdout.log`), no code references them

### Bug to fix along the way
- `training/evaluate.py:221` — `except subprocess.TimeoutExpired, FileNotFoundError:` is Python 2 syntax, should be `except (subprocess.TimeoutExpired, FileNotFoundError):`

---

## Task 1: Regenerate Rust Golden Reference Data as CSV

**Purpose:** The 3 test configs (`test_ref_orig`, `test_high_bank_orig`, `test_guided_orig`) currently produce D-notation text output. Switch them to CSV and regenerate golden references so `test_regression.py` can be migrated to CSV parsing.

**Files:**
- Modify: `configs/test_ref_orig.toml:22`
- Modify: `configs/test_high_bank_orig.toml:22`
- Modify: `configs/test_guided_orig.toml:73`
- Regenerate: `tests/reference_data/rust_golden/ref/`
- Regenerate: `tests/reference_data/rust_golden/high_bank/`
- Regenerate: `tests/reference_data/rust_golden/guided/`

**Step 1: Update TOML configs to CSV output**

In each of the 3 TOML files, remove or comment out the `output_format = "text"` line. CSV is the default, so no replacement needed.

`configs/test_ref_orig.toml` — delete line `output_format = "text"`
`configs/test_high_bank_orig.toml` — delete line `output_format = "text"`
`configs/test_guided_orig.toml` — delete line `output_format = "text"`

**Step 2: Build Rust simulator**

Run: `cd src/rust && cargo build --release`
Expected: builds successfully

**Step 3: Regenerate golden reference data**

Run each config and copy output to golden dirs:

```bash
# ref_orig
./src/rust/target/release/aerocapture configs/test_ref_orig.toml
cp output/photo.test_ref_orig.csv tests/reference_data/rust_golden/ref/
cp output/final.test_ref_orig.csv tests/reference_data/rust_golden/ref/

# high_bank_orig
./src/rust/target/release/aerocapture configs/test_high_bank_orig.toml
cp output/photo.test_high_bank_orig.csv tests/reference_data/rust_golden/high_bank/
cp output/final.test_high_bank_orig.csv tests/reference_data/rust_golden/high_bank/

# guided_orig
./src/rust/target/release/aerocapture configs/test_guided_orig.toml
cp output/photo.test_guided_orig.csv tests/reference_data/rust_golden/guided/
cp output/final.test_guided_orig.csv tests/reference_data/rust_golden/guided/
```

**Step 4: Delete old text-format golden files**

```bash
rm -f tests/reference_data/rust_golden/ref/photo.test_ref_orig tests/reference_data/rust_golden/ref/final.test_ref_orig
rm -f tests/reference_data/rust_golden/high_bank/photo.test_high_bank_orig tests/reference_data/rust_golden/high_bank/final.test_high_bank_orig
rm -f tests/reference_data/rust_golden/guided/photo.test_guided_orig tests/reference_data/rust_golden/guided/final.test_guided_orig
```

**Step 5: Commit**

```bash
git add configs/test_ref_orig.toml configs/test_high_bank_orig.toml configs/test_guided_orig.toml
git add tests/reference_data/rust_golden/
git commit -m "refactor: switch 3 test configs from text to CSV output, regenerate golden data"
```

---

## Task 2: Migrate `test_regression.py` to CSV Parsing

**Purpose:** Replace `parse_fortran_line` usage with standard CSV reading. Split Fortran-reference tests (comparing against legacy Fortran output) from Rust-golden tests (comparing against Rust's own CSV output). The Fortran-reference tests become obsolete since the configs now produce CSV.

**Files:**
- Modify: `tests/test_regression.py`

**Step 1: Rewrite test_regression.py**

Replace the Fortran parser with CSV-based comparison. Key changes:
- Remove `from aerocapture.io._fortran import parse_fortran_line`
- Replace `parse_output_file()` with a CSV reader (`pandas.read_csv` or `csv.reader`)
- Update `RUST_GOLDEN_CASES` file names to `.csv` extension
- Remove `FORTRAN_REF_CASES` entirely (those tests compared Rust text output against Fortran text output — both formats are being removed)
- Update `_run_and_compare()` to compare CSV files

```python
#!/usr/bin/env python3
"""Regression tests: run Rust simulator with TOML configs and compare against golden CSV data.

Usage:
    pytest tests/test_regression.py -v
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BINARY = ROOT / "src" / "rust" / "target" / "release" / "aerocapture"
GOLDEN_DIR = ROOT / "tests" / "reference_data" / "rust_golden"
OUTPUT_DIR = ROOT / "output"

# (test_id, toml_config, golden_subdir, output_suffix)
GOLDEN_CASES = [
    ("ref", "configs/test_ref_orig.toml", "ref", "test_ref_orig"),
    ("high_bank", "configs/test_high_bank_orig.toml", "high_bank", "test_high_bank_orig"),
    ("guided", "configs/test_guided_orig.toml", "guided", "test_guided_orig"),
]

ATOL = 1e-10
RTOL = 1e-10


def _parse_csv(filepath: Path) -> list[list[float]]:
    """Parse a CSV file into rows of floats (skipping header)."""
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append([float(v) for v in row])
    return rows


def _compare_values(
    test_rows: list[list[float]],
    ref_rows: list[list[float]],
) -> tuple[bool, list[str]]:
    """Compare two sets of rows within tolerance."""
    errors = []

    if len(test_rows) != len(ref_rows):
        errors.append(f"Row count mismatch: test={len(test_rows)} ref={len(ref_rows)}")
        return False, errors

    for row_idx, (test_row, ref_row) in enumerate(zip(test_rows, ref_rows, strict=False)):
        if len(test_row) != len(ref_row):
            errors.append(f"Col count mismatch at row {row_idx}: test={len(test_row)} ref={len(ref_row)}")
            continue

        for col_idx, (tv, rv) in enumerate(zip(test_row, ref_row, strict=False)):
            abs_dev = abs(tv - rv)
            rel_dev = abs_dev / max(abs(rv), 1e-300) if rv != 0.0 else abs_dev

            if abs_dev > ATOL and rel_dev > RTOL:
                errors.append(f"  row {row_idx}, col {col_idx}: test={tv:.10e} ref={rv:.10e} abs={abs_dev:.3e} rel={rel_dev:.3e}")

    return len(errors) == 0, errors


@pytest.fixture(scope="session", autouse=True)
def _build_rust() -> None:
    """Ensure Rust binary is built before running tests."""
    if not BINARY.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "src" / "rust", check=True)


def _run_and_compare(toml_config: str, golden_dir: Path, suffix: str) -> None:
    """Run the simulator and compare CSV output against golden reference."""
    toml_path = ROOT / toml_config

    result = subprocess.run(
        [str(BINARY), str(toml_path)],
        capture_output=True,
        cwd=str(ROOT),
        timeout=60,
    )
    assert result.returncode == 0, f"Simulator failed:\n{result.stderr.decode()}"

    for prefix in ["photo", "final"]:
        test_file = OUTPUT_DIR / f"{prefix}.{suffix}.csv"
        ref_file = golden_dir / f"{prefix}.{suffix}.csv"

        assert test_file.exists(), f"Output file not found: {test_file}"
        assert ref_file.exists(), f"Reference file not found: {ref_file}"

        test_data = _parse_csv(test_file)
        ref_data = _parse_csv(ref_file)

        passed, errors = _compare_values(test_data, ref_data)

        error_msg = f"\n{prefix}.{suffix}.csv comparison failed:\n" + "\n".join(errors[:20])
        if len(errors) > 20:
            error_msg += f"\n  ... and {len(errors) - 20} more"
        assert passed, error_msg


@pytest.mark.parametrize(
    "name,toml_config,golden_subdir,suffix",
    GOLDEN_CASES,
    ids=[t[0] for t in GOLDEN_CASES],
)
def test_rust_golden(name: str, toml_config: str, golden_subdir: str, suffix: str) -> None:
    """Run simulator and compare CSV output against Rust golden reference."""
    _run_and_compare(toml_config, GOLDEN_DIR / golden_subdir, suffix)
```

**Step 2: Run tests to verify**

Run: `uv run pytest tests/test_regression.py -v`
Expected: all 3 `test_rust_golden` tests PASS

**Step 3: Commit**

```bash
git add tests/test_regression.py
git commit -m "refactor: migrate test_regression.py from Fortran text to CSV parsing"
```

---

## Task 3: Update `test_parsers.py` — Remove Fortran Format Tests

**Purpose:** Remove tests that validate Fortran D-notation parsing, since we're removing that capability from the parsers.

**Files:**
- Modify: `tests/test_parsers.py`

**Step 1: Remove Fortran format test methods**

Remove:
- `TestParsePhoto.test_fortran_text_detection` (lines 37-46) — tests D-notation parsing
- `TestParseFinal.test_fortran_text_detection` (lines 75-83) — tests D-notation parsing
- `TestParsePhoto.test_csv_has_fewer_columns_than_legacy` (lines 48-51) — references `PHOTO_COLUMNS` (legacy 24-col constant being removed)

Update imports — remove `PHOTO_COLUMNS` from the import line:
```python
from aerocapture.io.parse_photo import PHOTO_CSV_COLUMNS, parse_photo
```

The `TestLegacyArrayMapping` and `TestComputeCost` classes stay — they test the CSV→legacy-index mapping used by `compute_cost()`, which is live code.

**Step 2: Run tests**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: PASS (remaining tests still work)

**Step 3: Commit**

```bash
git add tests/test_parsers.py
git commit -m "test: remove Fortran D-notation parser tests from test_parsers.py"
```

---

## Task 4: Remove Dead Python Files and Functions

**Purpose:** Delete entirely dead code that has no live callers.

**Files:**
- Delete: `src/python/aerocapture/io/parse_fort.py`
- Delete: `src/python/aerocapture/io/parse_initial.py`
- Delete: `tests/compare_results.py`
- Modify: `src/python/aerocapture/io/__init__.py`
- Modify: `src/python/aerocapture/training/evaluate.py`
- Modify: `src/python/aerocapture/training/config.py`

**Step 1: Delete dead parser files**

```bash
rm src/python/aerocapture/io/parse_fort.py
rm src/python/aerocapture/io/parse_initial.py
rm tests/compare_results.py
```

**Step 2: Update `io/__init__.py`**

Replace contents with:
```python
"""Parsers for simulation output files."""

from aerocapture.io.parse_final import parse_final
from aerocapture.io.parse_photo import parse_photo

__all__ = ["parse_photo", "parse_final"]
```

**Step 3: Remove `write_nn_params()` from `evaluate.py`**

Delete the function at lines 79-106 (the one that writes Fortran-readable 6-line header format). Keep `write_nn_json()` (lines 108-145) — that's the live JSON writer.

**Step 4: Remove stdin/.in branch from `run_simulation()` in `evaluate.py`**

Replace `run_simulation()` (lines 186-236) — remove the `else` branch that pipes `.in` via stdin, remove the `init_file` reference, fix the Python 2 `except` syntax bug on line 221:

```python
def run_simulation(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run the Rust simulator and parse final conditions.

    Args:
        config: Training configuration.
        cwd: Working directory (defaults to config.sim.exec_dir).

    Returns:
        Array of final conditions, or None if simulation failed.
    """
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)

    executable = (cwd / config.sim.executable).resolve()

    if not config.sim.toml_config:
        return None

    toml_path = (cwd / config.sim.toml_config).resolve()
    try:
        subprocess.run(
            [str(executable), str(toml_path)],
            capture_output=True,
            cwd=str(cwd.resolve()),
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # Parse final conditions — auto-detect CSV vs legacy text
    final_file = cwd / config.sim.final_file
    csv_final = Path(str(final_file) + ".csv")
    if csv_final.exists():
        final_file = csv_final
    elif not final_file.exists():
        return None

    try:
        return _parse_final_to_legacy_array(final_file)
    except Exception:
        return None
```

**Step 5: Remove `init_file` from `SimConfig` in `config.py`**

Delete line 72: `init_file: str = "train_nn.in"`

**Step 6: Remove legacy Fortran branch from `load_base_network()` in `config.py`**

Replace lines 129-185 of `load_base_network()` — keep only the JSON path:

```python
    def load_base_network(self, filepath: str | Path) -> npt.NDArray[np.float64]:
        """Load base network weights from a JSON nn_param file.

        Returns:
            Array of shape (n_coef,) with loaded weights (padded with 1.0).
        """
        import json

        filepath = Path(filepath)
        content = filepath.read_text().strip()

        data = json.loads(content)
        weights = []
        for i in range(len(data["architecture"]["layers"]) - 1):
            layer = data["weights"][f"layer_{i}"]
            for row in layer["w"]:
                weights.extend(row)
            weights.extend(layer["b"])

        n_base = self.network.n_base_coef
        base = np.array(weights[:n_base], dtype=np.float64)
        padded = np.ones(self.network.n_coef, dtype=np.float64)
        padded[:n_base] = base
        return padded
```

**Step 7: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

**Step 8: Run linters**

Run: `./lint_code.sh`
Expected: clean (ruff + mypy pass)

**Step 9: Commit**

```bash
git add -A src/python/aerocapture/io/ src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/config.py
git rm tests/compare_results.py
git commit -m "refactor: remove dead Fortran I/O code — parse_fort, parse_initial, write_nn_params, stdin branch"
```

---

## Task 5: Remove Fortran Fallback from `parse_photo.py` and `parse_final.py`

**Purpose:** Strip Fortran text format auto-detection from the two live parsers. They become CSV-only.

**Files:**
- Modify: `src/python/aerocapture/io/parse_photo.py`
- Modify: `src/python/aerocapture/io/parse_final.py`
- Modify: `src/python/aerocapture/training/evaluate.py` (the `_parse_final_to_legacy_array` legacy branch)

**Step 1: Simplify `parse_photo.py`**

Remove the `parse_fortran_line` import, the legacy `PHOTO_COLUMNS` list, and the Fortran fallback branch. Keep `_CSV_TO_LEGACY_NAMES` for backward-compat column renaming. Keep `PHOTO_CSV_COLUMNS`.

```python
"""Parse trajectory snapshot files (photo.*) into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# CSV column names (21 columns — matches Rust PHOTO_CSV_COLUMNS)
PHOTO_CSV_COLUMNS = [
    "time_s",
    "altitude_km",
    "longitude_deg",
    "latitude_deg",
    "velocity_m_s",
    "flight_path_deg",
    "azimuth_deg",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "periapsis_alt_km",
    "apoapsis_alt_km",
    "phase",
    "bank_angle_deg",
    "radial_velocity_m_s",
    "aoa_deg",
    "cumulative_bank_change_deg",
    "energy_j_kg",
    "dynamic_pressure_pa",
    "dynamic_pressure_onboard_kpa",
]

# Map CSV column names to legacy column names for backward compatibility.
# This ensures plotting modules that access by column name work with both formats.
_CSV_TO_LEGACY_NAMES: dict[str, str] = {
    "time_s": "time",
    "altitude_km": "altitude",
    "longitude_deg": "longitude",
    "latitude_deg": "latitude",
    "velocity_m_s": "velocity",
    "flight_path_deg": "flight_path_angle",
    "azimuth_deg": "azimuth",
    "semi_major_axis_km": "semi_major_axis",
    "eccentricity": "eccentricity",
    "inclination_deg": "inclination",
    "raan_deg": "raan",
    "periapsis_alt_km": "periapsis_alt",
    "apoapsis_alt_km": "apoapsis_alt",
    "phase": "phase",
    "bank_angle_deg": "bank_angle",
    "radial_velocity_m_s": "radial_velocity",
    "aoa_deg": "aoa",
    "cumulative_bank_change_deg": "bank_rate",
    "energy_j_kg": "energy",
    "dynamic_pressure_pa": "dynamic_pressure",
    "dynamic_pressure_onboard_kpa": "dynamic_pressure_rho",
}


def parse_photo(filepath: str | Path) -> pd.DataFrame:
    """Parse a photo trajectory snapshot CSV file into a DataFrame.

    Args:
        filepath: Path to the photo CSV file.

    Returns:
        DataFrame with named columns (CSV names normalized to legacy names).
    """
    filepath = Path(filepath)

    if filepath.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    # Normalize CSV column names to legacy names for backward compatibility
    df = df.rename(columns=_CSV_TO_LEGACY_NAMES)
    return df
```

**Step 2: Simplify `parse_final.py`**

Remove the `parse_fortran_line` import and the Fortran fallback branch. Keep `CSV_TO_LEGACY_INDEX` (used by `_parse_final_to_legacy_array` in evaluate.py). Keep `FINAL_CSV_COLUMNS`.

```python
"""Parse final conditions files (final.*) into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# CSV column names (40 columns — matches Rust FINAL_CSV_COLUMNS)
FINAL_CSV_COLUMNS = [
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

# Mapping from CSV column name to legacy Fortran column index (0-based in xsauve[52]).
# This enables compute_cost and other consumers to access data by name.
CSV_TO_LEGACY_INDEX: dict[str, int] = {
    "altitude_km": 0,
    "longitude_deg": 1,
    "latitude_deg": 2,
    "velocity_m_s": 3,
    "flight_path_deg": 4,
    "azimuth_deg": 5,
    "radial_velocity_m_s": 6,
    "energy_mj_kg": 7,
    "semi_major_axis_km": 8,
    "eccentricity": 9,
    "inclination_deg": 10,
    "raan_deg": 11,
    "arg_periapsis_deg": 12,
    "true_anomaly_deg": 13,
    "periapsis_alt_km": 14,
    "apoapsis_alt_km": 15,
    "max_heat_flux_kw_m2": 16,
    "max_load_factor_g": 17,
    "max_dyn_pressure_kpa": 18,
    "alt_max_flux_km": 19,
    "alt_max_load_km": 20,
    "alt_max_pdyn_km": 21,
    "time_max_flux_s": 22,
    "time_max_load_s": 23,
    "time_max_pdyn_s": 24,
    "bounce_alt_km": 25,
    "bounce_time_s": 26,
    "sim_time_s": 27,
    "integrated_flux_mj_m2": 28,
    "periapsis_err_km": 29,
    "apoapsis_err_km": 30,
    "ifinal": 31,
    "dv1_m_s": 37,
    "dv2_m_s": 38,
    "dv3_m_s": 39,
    "dv12_m_s": 40,
    "dv_total_m_s": 41,
    "cumulative_bank_change_deg": 45,
    "n_roll_reversals": 48,
}


def parse_final(filepath: str | Path) -> pd.DataFrame:
    """Parse a final conditions CSV file into a DataFrame.

    Args:
        filepath: Path to the final CSV file.

    Returns:
        DataFrame with named columns.
    """
    filepath = Path(filepath)

    if filepath.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_csv(filepath)
```

**Step 3: Simplify `_parse_final_to_legacy_array()` in `evaluate.py`**

Remove the Fortran text fallback branch (lines 174-183). CSV is the only path:

```python
def _parse_final_to_legacy_array(filepath: Path) -> npt.NDArray[np.float64] | None:
    """Parse a final conditions CSV file, returning legacy-compatible 53-column array.

    Maps named CSV columns back to the legacy 53-column positions so
    compute_cost() works unchanged.
    """
    import pandas as pd

    from aerocapture.io.parse_final import CSV_TO_LEGACY_INDEX

    df = pd.read_csv(filepath)
    if df.empty:
        return None
    n = len(df)
    result = np.zeros((n, 53))
    result[:, 0] = df["sim_number"].to_numpy()
    for col_name, legacy_idx in CSV_TO_LEGACY_INDEX.items():
        if col_name in df.columns:
            result[:, legacy_idx + 1] = df[col_name].to_numpy()
    return result
```

**Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

**Step 5: Run linters**

Run: `./lint_code.sh`
Expected: clean

**Step 6: Commit**

```bash
git add src/python/aerocapture/io/parse_photo.py src/python/aerocapture/io/parse_final.py src/python/aerocapture/training/evaluate.py
git commit -m "refactor: remove Fortran text fallback from parse_photo, parse_final, evaluate"
```

---

## Task 6: Remove `_fortran.py` and Update Remaining Python Callers

**Purpose:** Delete the Fortran parsing utility now that no parser uses it. Update `corridor.py` and `plot_mc_comparison.py` which directly import it.

**Files:**
- Delete: `src/python/aerocapture/io/_fortran.py`
- Modify: `src/python/aerocapture/plotting/corridor.py`
- Modify: `scripts/plot_mc_comparison.py`

**Step 1: Update `corridor.py`**

The `_load_fortran_table()` function and `load_corridor_boundaries()` load corridor boundary files (`visu.ovr_res`, `visu.udr_res`) that don't exist in the repo. Replace the Fortran parser with a simple whitespace-delimited reader that handles standard scientific notation (no D-notation):

```python
"""Shared corridor-drawing utilities for aerocapture visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def _load_table(path: str | Path) -> npt.NDArray[np.float64]:
    """Load a whitespace-delimited numeric file."""
    return np.loadtxt(path, dtype=np.float64)


def load_corridor_boundaries(
    overshoot_path: str | Path,
    undershoot_path: str | Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load corridor boundary data from files.

    Args:
        overshoot_path: Path to overshoot boundary file.
        undershoot_path: Path to undershoot boundary file.

    Returns:
        (overshoot, undershoot): 2D arrays with columns [energy, pressure].
    """
    overshoot = _load_table(overshoot_path)
    undershoot = _load_table(undershoot_path)
    return overshoot, undershoot


def draw_corridor(
    ax: plt.Axes,
    overshoot: npt.NDArray[np.float64] | None = None,
    undershoot: npt.NDArray[np.float64] | None = None,
    color: str = "0.9",
) -> None:
    """Draw overshoot/undershoot corridor boundaries on an axes.

    Args:
        ax: Matplotlib axes to draw on.
        overshoot: Array with columns [energy (MJ/kg), pressure (kPa)].
        undershoot: Array with columns [energy (MJ/kg), pressure (kPa)].
        color: Fill color for constraint regions.
    """
    if overshoot is not None:
        ax.fill_between(overshoot[:, 0], overshoot[:, 1], 0, color=color, alpha=0.5, label="Overshoot")
    if undershoot is not None:
        ax.fill_between(undershoot[:, 0], undershoot[:, 1], 10, color=color, alpha=0.5, label="Undershoot")


def segment_mc_trajectories(photo_data: npt.NDArray[np.float64], time_col: int = 0) -> list[npt.NDArray[np.float64]]:
    """Split concatenated Monte Carlo photo data into individual trajectories.

    Detects trajectory boundaries where time decreases (restart).

    Args:
        photo_data: Full photo array with all MC runs concatenated.
        time_col: Column index for time.

    Returns:
        List of arrays, one per trajectory.
    """
    time = photo_data[:, time_col]
    restart_indices = np.where(np.diff(time) < 0)[0] + 1
    splits = np.split(photo_data, restart_indices)
    return [s for s in splits if len(s) > 0]
```

**Step 2: Update `scripts/plot_mc_comparison.py`**

Replace `load_final()` function (lines 13-21) to use CSV via pandas:

```python
import pandas as pd

def load_final(path: str | Path) -> np.ndarray:
    """Load a final conditions CSV file into a numpy array."""
    df = pd.read_csv(path)
    return df.to_numpy()
```

Remove the `from aerocapture.io._fortran import parse_fortran_line` import (line 9).

**Step 3: Delete `_fortran.py`**

```bash
rm src/python/aerocapture/io/_fortran.py
```

**Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all PASS

**Step 5: Run linters**

Run: `./lint_code.sh`
Expected: clean

**Step 6: Commit**

```bash
git rm src/python/aerocapture/io/_fortran.py
git add src/python/aerocapture/plotting/corridor.py scripts/plot_mc_comparison.py
git commit -m "refactor: delete _fortran.py, update corridor.py and plot_mc_comparison.py to CSV"
```

---

## Task 7: Remove Rust `OutputFormat::Text` and Legacy Neural Loader

**Purpose:** Remove the D-notation text output path and the legacy Fortran nn_param loader from Rust.

**Files:**
- Modify: `src/rust/src/config.rs`
- Modify: `src/rust/src/simulation/output.rs`
- Modify: `src/rust/src/simulation/runner.rs`
- Modify: `src/rust/src/data/neural.rs`

**Step 1: Remove `OutputFormat` enum entirely from `config.rs`**

Since CSV is the only format, the enum is unnecessary. Delete the enum definition (lines 83-91). Remove the `output_format` field from `SimInput` (line 122). Remove the `output_format` field from `TomlData` (line 222). Remove the deserialization of `output_format` in the TOML parser (search for `output_format` in the `From<TomlConfig>` impl). Replace `config.output_format` usage in runner.rs with a direct call to `write_csv_output`.

In `config.rs`, delete:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Csv,
    Text,
}
```

Remove `pub output_format: OutputFormat` from `SimInput`.
Remove `pub output_format: OutputFormat` from `TomlData` (or equivalent).
Remove assignment of `output_format` in the conversion impl.

**Step 2: Remove text writers from `output.rs`**

Delete everything from line 118 (`// ─── Legacy Fortran text writers ───`) to line 174 (end of `write_final_text_line`). Also delete the Fortran-specific tests (lines 183-229: `zero_formats_correctly`, `d12_5_known_values`, `width_is_respected`, `extreme_values_no_panic`, `photo_text_line_has_correct_column_count`). Keep CSV writers and CSV tests.

**Step 3: Remove `write_text_output()` from `runner.rs`**

Delete `write_text_output()` (lines 220-262). Replace the match statement (lines 163-166):
```rust
// Before:
match config.output_format {
    OutputFormat::Csv => write_csv_output(config, &results, photo_sim_idx)?,
    OutputFormat::Text => write_text_output(config, &results, photo_sim_idx)?,
}

// After:
write_csv_output(config, &results, photo_sim_idx)?;
```

Remove `use crate::config::OutputFormat;` import if present.

**Step 4: Remove `from_legacy()` from `neural.rs`**

Delete the `from_legacy()` method (lines 146-227). Update `load()` to only support JSON:

```rust
pub fn load(path: &str) -> Result<Self, DataError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| DataError(format!("Cannot read {}: {}", path, e)))?;
    Self::from_json(&content, path)
}
```

Remove the format auto-detection logic.

**Step 5: Build and test**

Run: `cd src/rust && cargo build --release && cargo test`
Expected: all pass, no warnings about dead code

Run: `cd src/rust && cargo clippy`
Expected: clean

**Step 6: Run full check**

Run: `./check_all.sh`
Expected: all pass

**Step 7: Commit**

```bash
git add src/rust/
git commit -m "refactor: remove OutputFormat::Text, D-notation writers, and legacy nn_param loader from Rust"
```

---

## Task 8: Delete Orphaned Reference Data

**Purpose:** Remove Fortran-only reference data directories that no code references.

**Files:**
- Delete: `tests/reference_data/ref_orig/` (Fortran golden data — superseded by `rust_golden/ref/`)
- Delete: `tests/reference_data/high_bank_orig/` (superseded by `rust_golden/high_bank/`)
- Delete: `tests/reference_data/guided_orig/` (superseded by `rust_golden/guided/`)
- Delete: `tests/reference_data/guided_nn_ftc/` (orphaned — no code references)
- Delete: `tests/reference_data/guided_nn2/` (orphaned)
- Delete: `tests/reference_data/ref_nn/` (orphaned)
- Delete: `tests/reference_data/mc10_orig/` (orphaned)

**Step 1: Verify no code references these directories**

```bash
rg "ref_orig|high_bank_orig|guided_orig|guided_nn_ftc|guided_nn2|ref_nn|mc10_orig" --type rust --type python --type toml
```

Expected: no matches (after Task 2 removed `FORTRAN_REF_CASES` from `test_regression.py`)

**Step 2: Delete orphaned directories**

```bash
rm -rf tests/reference_data/ref_orig
rm -rf tests/reference_data/high_bank_orig
rm -rf tests/reference_data/guided_orig
rm -rf tests/reference_data/guided_nn_ftc
rm -rf tests/reference_data/guided_nn2
rm -rf tests/reference_data/ref_nn
rm -rf tests/reference_data/mc10_orig
```

**Step 3: Verify remaining structure**

```bash
ls tests/reference_data/
```

Expected: only `rust_golden/` remains

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v && cd src/rust && cargo test`
Expected: all pass

**Step 5: Commit**

```bash
git add -A tests/reference_data/
git commit -m "chore: delete orphaned Fortran reference data directories"
```

---

## Task 9: Update TODO.md and Clean Up Pycache

**Purpose:** Mark the task as done and remove stale bytecode.

**Files:**
- Modify: `TODO.md`
- Clean: `src/python/aerocapture/io/__pycache__/`

**Step 1: Update TODO.md**

Change line 7 from:
```
- [ ] Remove Rust and Python code for legacy `.in` input/output formats
```
to:
```
- [x] Remove Rust and Python code for legacy `.in` input/output formats — done, CSV is now the only supported format
```

**Step 2: Clean stale pycache**

```bash
find src/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

**Step 3: Run full test suite one final time**

```bash
./check_all.sh && uv run pytest tests/ -v
```

Expected: all pass

**Step 4: Commit**

```bash
git add TODO.md
git commit -m "chore: mark legacy format removal as done in TODO.md"
```

---

## Execution Order & Dependencies

```
Task 1 (regen golden CSV)
  └── Task 2 (migrate test_regression.py)
        └── Task 8 (delete orphaned ref data)

Task 3 (update test_parsers.py)
  └── Task 5 (strip Fortran fallback from parsers)
        └── Task 6 (delete _fortran.py)

Task 4 (delete dead Python files/funcs) — independent

Task 7 (Rust OutputFormat::Text removal) — depends on Task 1

Task 9 (TODO.md) — last
```

Parallelizable groups:
- **Group A:** Tasks 1 → 2 → 8 (reference data pipeline)
- **Group B:** Tasks 3 → 5 → 6 (parser cleanup pipeline)
- **Group C:** Task 4 (dead code deletion)
- **Group D:** Task 7 (Rust cleanup, after Task 1)
- **Final:** Task 9

## Lines Removed (estimate)

| Area | Lines |
|------|-------|
| Rust: OutputFormat::Text + D-notation writers + tests | ~170 |
| Rust: from_legacy() neural loader | ~80 |
| Python: parse_fort.py + parse_initial.py + _fortran.py | ~240 |
| Python: evaluate.py dead code | ~50 |
| Python: config.py dead code | ~55 |
| Python: parse_photo.py + parse_final.py Fortran branches | ~40 |
| Python: compare_results.py | ~180 |
| Test: Fortran format tests | ~20 |
| Reference data (Fortran text files) | many files |
| **Total code** | **~835 lines** |
