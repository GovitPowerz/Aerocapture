#!/usr/bin/env python3
"""Integration tests for domain-based Monte Carlo dispersions.

Verifies that the domain-based MC system (runtime RNG draws from seed)
produces correct, deterministic, multi-run output.

Usage:
    pytest tests/test_mc_domain.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "src" / "rust" / "target" / "release" / "aerocapture"
OUTPUT_DIR = ROOT / "output"
BASE_CONFIG = ROOT / "configs" / "nominal" / "msr_aller_ftc_mc_domain.toml"


@pytest.fixture(scope="session", autouse=True)
def _build_rust(rust_binary: Path) -> Path:
    return rust_binary


def _make_mc_config(n_sims: int, seed: int, suffix: str) -> Path:
    """Create a temporary TOML config for MC testing with given n_sims and seed."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(BASE_CONFIG)
    # Override n_sims, seed, and results_suffix
    data.setdefault("simulation", {})["n_sims"] = n_sims
    data.setdefault("monte_carlo", {})["seed"] = seed
    data.setdefault("data", {})["results_suffix"] = f".{suffix}"

    from aerocapture.training.evaluate import _write_toml

    tmp = Path(tempfile.mktemp(suffix=".toml", prefix="mc_test_"))
    _write_toml(data, tmp)
    return tmp


def _run_sim(config_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the Rust simulator with the given config."""
    return subprocess.run(
        [str(BINARY), str(config_path)],
        capture_output=True,
        cwd=str(ROOT),
        timeout=120,
    )


def _count_final_rows(suffix: str) -> int:
    """Count data rows in the final output file (CSV or text format)."""
    # Try CSV first (consolidated configs), then text (suffix-based configs)
    for ext in [".csv", ""]:
        final_file = OUTPUT_DIR / f"final.{suffix}{ext}"
        if final_file.exists():
            count = 0
            for line in final_file.read_text().splitlines():
                stripped = line.strip()
                # Skip empty, comments, and CSV header
                if stripped and not stripped.startswith("#") and not stripped.startswith("sim_number"):
                    count += 1
            return count
    return 0


def _read_file_bytes(suffix: str, prefix: str = "final") -> bytes:
    """Read an output file as raw bytes for comparison (CSV or text format)."""
    for ext in [".csv", ""]:
        path = OUTPUT_DIR / f"{prefix}.{suffix}{ext}"
        if path.exists():
            return path.read_bytes()
    return b""


class TestDomainMC:
    """Domain-based Monte Carlo integration tests."""

    def test_domain_mc_produces_output(self) -> None:
        """Run domain MC with 3 sims, verify final file has 3 rows."""
        config = _make_mc_config(n_sims=3, seed=42, suffix="mc_test_3")
        try:
            result = _run_sim(config)
            assert result.returncode == 0, f"Simulator failed:\n{result.stderr.decode()}"
            n_rows = _count_final_rows("mc_test_3")
            assert n_rows == 3, f"Expected 3 final rows for 3 sims, got {n_rows}"
        finally:
            config.unlink(missing_ok=True)

    def test_domain_mc_deterministic(self) -> None:
        """Two runs with same seed produce identical output."""
        config = _make_mc_config(n_sims=3, seed=99, suffix="mc_test_det")
        try:
            result1 = _run_sim(config)
            assert result1.returncode == 0, f"Run 1 failed:\n{result1.stderr.decode()}"
            final_1 = _read_file_bytes("mc_test_det", "final")
            photo_1 = _read_file_bytes("mc_test_det", "photo")

            result2 = _run_sim(config)
            assert result2.returncode == 0, f"Run 2 failed:\n{result2.stderr.decode()}"
            final_2 = _read_file_bytes("mc_test_det", "final")
            photo_2 = _read_file_bytes("mc_test_det", "photo")

            assert final_1 == final_2, "Final files differ between identical runs"
            assert photo_1 == photo_2, "Photo files differ between identical runs"
        finally:
            config.unlink(missing_ok=True)

    def test_different_seeds_produce_different_output(self) -> None:
        """Different seeds should produce different trajectories."""
        config_a = _make_mc_config(n_sims=2, seed=42, suffix="mc_test_seed_a")
        config_b = _make_mc_config(n_sims=2, seed=99, suffix="mc_test_seed_b")
        try:
            result_a = _run_sim(config_a)
            assert result_a.returncode == 0, f"Run A failed:\n{result_a.stderr.decode()}"
            final_a = _read_file_bytes("mc_test_seed_a", "final")

            result_b = _run_sim(config_b)
            assert result_b.returncode == 0, f"Run B failed:\n{result_b.stderr.decode()}"
            final_b = _read_file_bytes("mc_test_seed_b", "final")

            assert final_a != final_b, "Different seeds produced identical output"
        finally:
            config_a.unlink(missing_ok=True)
            config_b.unlink(missing_ok=True)

    def test_single_sim_no_mc_config(self) -> None:
        """Single sim with no [monte_carlo] section should produce clean output."""
        # Use a non-MC config that has 1 sim
        config = ROOT / "configs" / "test" / "test_guided_orig.toml"
        result = _run_sim(config)
        assert result.returncode == 0, f"Simulator failed:\n{result.stderr.decode()}"
        n_rows = _count_final_rows("test_guided_orig")
        assert n_rows == 1, f"Expected 1 final row for single sim, got {n_rows}"
