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

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "src" / "rust" / "target" / "release" / "aerocapture"
GOLDEN_DIR = ROOT / "tests" / "reference_data" / "rust_golden"
OUTPUT_DIR = ROOT / "output"

# (test_id, toml_config, golden_subdir, output_suffix)
GOLDEN_CASES = [
    ("ref", "configs/test/test_ref_orig.toml", "ref", "test_ref_orig"),
    ("high_bank", "configs/test/test_high_bank_orig.toml", "high_bank", "test_high_bank_orig"),
    ("guided", "configs/test/test_guided_orig.toml", "guided", "test_guided_orig"),
]

ATOL = 1e-9
RTOL = 1e-9


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
def _build_rust(rust_binary: Path) -> Path:
    return rust_binary


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
