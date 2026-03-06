#!/usr/bin/env python3
"""Compare simulation output files against reference data.

Parses Fortran output files (fort.*, photo.*, final.*, initial.*) with
D-notation floats and compares against golden reference files with
configurable tolerances.

Usage:
    python compare_results.py <test_dir> <reference_dir> [--atol 1e-10] [--rtol 1e-10]
    python compare_results.py tests/reference_data/ref_orig tests/reference_data/ref_orig  # self-test

Exit code: 0 if all comparisons pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def parse_fortran_float(s: str) -> float:
    """Parse a Fortran D-notation float (e.g. '0.1234D+02') to Python float."""
    return float(s.replace("D", "E").replace("d", "e"))


def parse_fortran_line(line: str) -> list[float]:
    """Parse a line of Fortran output into a list of floats.

    Handles D-notation (0.1234D+02), E-notation, and plain numbers.
    Also handles integer-prefixed lines (e.g. final.* files with sim number).
    """
    # Match Fortran floats: optional sign, digits, optional decimal, optional D/E exponent
    # Also matches plain integers
    tokens = re.findall(r"[+-]?\d+\.?\d*(?:[DEde][+-]?\d+)?", line)
    values = []
    for token in tokens:
        if "D" in token or "d" in token or "E" in token or "e" in token or "." in token:
            values.append(parse_fortran_float(token))
        else:
            values.append(float(token))
    return values


def parse_output_file(filepath: Path) -> list[list[float]]:
    """Parse a Fortran output file into a 2D list of floats."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = parse_fortran_line(line)
            if values:
                rows.append(values)
    return rows


def compare_files(
    test_file: Path,
    ref_file: Path,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Compare two output files within tolerance.

    Returns (passed, messages) where messages contain deviation details.
    """
    messages = []

    if not test_file.exists():
        return False, [f"Test file not found: {test_file}"]
    if not ref_file.exists():
        return False, [f"Reference file not found: {ref_file}"]

    test_data = parse_output_file(test_file)
    ref_data = parse_output_file(ref_file)

    if len(test_data) != len(ref_data):
        messages.append(f"Row count mismatch: test={len(test_data)} ref={len(ref_data)}")
        return False, messages

    max_abs_dev = 0.0
    max_rel_dev = 0.0
    max_dev_loc = ""
    passed = True

    for row_idx, (test_row, ref_row) in enumerate(zip(test_data, ref_data, strict=False)):
        if len(test_row) != len(ref_row):
            messages.append(f"Column count mismatch at row {row_idx}: test={len(test_row)} ref={len(ref_row)}")
            passed = False
            continue

        for col_idx, (tv, rv) in enumerate(zip(test_row, ref_row, strict=False)):
            abs_dev = abs(tv - rv)
            rel_dev = abs_dev / max(abs(rv), 1e-300) if rv != 0.0 else abs_dev

            if abs_dev > atol and rel_dev > rtol:
                passed = False

            if abs_dev > max_abs_dev:
                max_abs_dev = abs_dev
                max_rel_dev = rel_dev
                max_dev_loc = f"row {row_idx}, col {col_idx}"

    if max_abs_dev > 0:
        messages.append(f"Max deviation: abs={max_abs_dev:.6e} rel={max_rel_dev:.6e} at {max_dev_loc}")

    return passed, messages


def compare_directories(
    test_dir: Path,
    ref_dir: Path,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> bool:
    """Compare all output files in test_dir against ref_dir.

    Looks for fort.*, photo.*, final.*, initial.* files recursively.
    """
    all_passed = True
    compared = 0

    # Find all reference files to compare against
    patterns = ["fort.*", "photo.*", "final.*", "initial.*"]
    ref_files: list[Path] = []
    for pattern in patterns:
        ref_files.extend(ref_dir.rglob(pattern))

    # Filter to actual data files (not directories)
    ref_files = [f for f in ref_files if f.is_file() and f.stat().st_size > 0]
    ref_files.sort()

    if not ref_files:
        print(f"WARNING: No reference files found in {ref_dir}")
        return False

    for ref_file in ref_files:
        # Compute relative path to find corresponding test file
        rel_path = ref_file.relative_to(ref_dir)
        test_file = test_dir / rel_path

        passed, messages = compare_files(test_file, ref_file, atol=atol, rtol=rtol)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {rel_path}")
        for msg in messages:
            print(f"         {msg}")

        if not passed:
            all_passed = False
        compared += 1

    print(f"\n  Compared {compared} files: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare simulation outputs against reference data")
    parser.add_argument("test_dir", type=Path, help="Directory containing test outputs")
    parser.add_argument("reference_dir", type=Path, help="Directory containing reference outputs")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance (default: 1e-10)")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance (default: 1e-10)")
    args = parser.parse_args()

    if not args.test_dir.exists():
        print(f"ERROR: Test directory not found: {args.test_dir}")
        return 1
    if not args.reference_dir.exists():
        print(f"ERROR: Reference directory not found: {args.reference_dir}")
        return 1

    print(f"Comparing: {args.test_dir} vs {args.reference_dir}")
    print(f"Tolerances: atol={args.atol:.0e} rtol={args.rtol:.0e}\n")

    passed = compare_directories(args.test_dir, args.reference_dir, atol=args.atol, rtol=args.rtol)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
