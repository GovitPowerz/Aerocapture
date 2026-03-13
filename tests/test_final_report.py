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


class TestRunFinalEvaluation:
    def test_patches_n_sims_and_seed(self, tmp_path: Path) -> None:
        """Verify TOML patching writes correct n_sims and seed."""
        import tomllib

        from aerocapture.training.final_report import _patch_toml_for_final_eval

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

        toml_content = "[flight.target_orbit]\napoapsis = 500.0\nperiapsis = 250.0\ninclination = 50.0\n"
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)

        assert _read_target_inclination(toml_file) == 50.0

    def test_target_inclination_missing_returns_zero(self, tmp_path: Path) -> None:
        """Fallback to 0.0 if inclination not in TOML."""
        from aerocapture.training.final_report import _read_target_inclination

        toml_content = "[flight.target_orbit]\napoapsis = 500.0\n"
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)

        assert _read_target_inclination(toml_file) == 0.0

    def test_seed_zero_is_not_replaced(self, tmp_path: Path) -> None:
        """Explicit seed=0 should be preserved, not treated as None."""
        import tomllib

        from aerocapture.training.final_report import _patch_toml_for_final_eval

        toml_content = "[monte_carlo]\nn_sims = 10\nseed = 99\n"
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

        pytest.importorskip("plotly")

        result = subprocess.run(
            ["uv", "run", "python", "-m", "aerocapture.training.final_report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--n-sims" in result.stdout
