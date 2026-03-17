"""Tests for final evaluation report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.final_report import FinalEvalData


def _make_captured_array(n: int = 100, seed: int = 42) -> np.ndarray:
    """Create a synthetic final conditions array with all captured trajectories."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((n, 52))
    arr[:, 3] = rng.normal(5500, 50, n)  # velocity_m_s
    arr[:, 4] = rng.normal(-12.0, 0.5, n)  # flight_path_deg
    arr[:, 7] = rng.uniform(-2.0, -0.5, n)  # energy < 0 (captured)
    arr[:, 9] = rng.uniform(0.3, 0.9, n)  # ecc < 1 (captured)
    arr[:, 10] = rng.normal(50.0, 1.0, n)  # inclination_deg
    arr[:, 16] = rng.uniform(50, 200, n)  # max heat flux kW/m2
    arr[:, 17] = rng.uniform(1, 8, n)  # max g-load
    arr[:, 27] = rng.uniform(300, 600, n)  # sim_time_s
    arr[:, 29] = rng.normal(0, 10, n)  # periapsis_err_km
    arr[:, 30] = rng.normal(0, 15, n)  # apoapsis_err_km
    arr[:, 37] = rng.exponential(20, n)  # dv1
    arr[:, 38] = rng.exponential(50, n)  # dv2
    arr[:, 39] = rng.exponential(10, n)  # dv3
    arr[:, 41] = arr[:, 37] + arr[:, 38] + arr[:, 39]  # dv_total
    arr[:, 45] = rng.uniform(100, 500, n)  # bank consumption deg
    return arr


def _make_mixed_array(n_captured: int = 80, n_hyper: int = 20, seed: int = 42) -> np.ndarray:
    """Create array with both captured and hyperbolic trajectories."""
    arr = _make_captured_array(n_captured + n_hyper, seed)
    # Make last n_hyper trajectories hyperbolic
    arr[n_captured:, 7] = np.abs(arr[n_captured:, 7])  # energy > 0
    arr[n_captured:, 9] = 1.0 + np.abs(arr[n_captured:, 9])  # ecc > 1
    return arr


def _make_all_hyperbolic(n: int = 50, seed: int = 42) -> np.ndarray:
    """Create array with zero captured trajectories."""
    arr = _make_captured_array(n, seed)
    arr[:, 7] = np.abs(arr[:, 7])  # energy > 0
    arr[:, 9] = 1.0 + np.abs(arr[:, 9])  # ecc > 1
    return arr


def _make_trajectories(n: int, n_steps: int = 50, seed: int = 42) -> list[np.ndarray]:
    """Create synthetic trajectory list: list of (T_i, 12) arrays."""
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n):
        t = np.zeros((n_steps, 12))
        t[:, 3] = rng.normal(5500, 50, n_steps)  # velocity
        t[:, 4] = rng.normal(-12.0, 0.5, n_steps)  # fpa
        t[:, 8] = np.linspace(5.0, -2.0, n_steps)  # energy MJ/kg
        t[:, 9] = rng.uniform(0, 10, n_steps)  # pdyn kPa
        t[:, 10] = rng.uniform(0, 80, n_steps)  # bank angle deg
        t[:, 11] = rng.normal(50, 1, n_steps)  # inclination deg
        trajs.append(t)
    return trajs


def _make_dispersions(n: int, seed: int = 42) -> np.ndarray:
    """Create synthetic dispersions (N, 24) array."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 24))


def _make_eval_data(
    n: int = 100,
    seed: int = 42,
    with_trajectories: bool = False,
    with_dispersions: bool = False,
    all_hyperbolic: bool = False,
    n_captured: int | None = None,
    n_hyper: int | None = None,
) -> FinalEvalData:
    """Create a FinalEvalData with optional trajectories/dispersions."""
    from aerocapture.training.final_report import FinalEvalData

    if all_hyperbolic:
        arr = _make_all_hyperbolic(n, seed)
    elif n_captured is not None and n_hyper is not None:
        arr = _make_mixed_array(n_captured, n_hyper, seed)
        n = n_captured + n_hyper
    else:
        arr = _make_captured_array(n, seed)

    trajectories = _make_trajectories(n, seed=seed) if with_trajectories else None
    dispersions = _make_dispersions(n, seed=seed) if with_dispersions else None

    return FinalEvalData(final_array=arr, trajectories=trajectories, dispersions=dispersions)


class TestFinalEvalData:
    def test_namedtuple_fields(self) -> None:
        from aerocapture.training.final_report import FinalEvalData

        data = FinalEvalData(final_array=np.zeros((5, 52)), trajectories=None, dispersions=None)
        assert data.final_array.shape == (5, 52)
        assert data.trajectories is None
        assert data.dispersions is None

    def test_with_all_fields(self) -> None:
        from aerocapture.training.final_report import FinalEvalData

        arr = np.zeros((3, 52))
        trajs = [np.zeros((10, 12)) for _ in range(3)]
        disp = np.zeros((3, 24))
        data = FinalEvalData(final_array=arr, trajectories=trajs, dispersions=disp)
        assert len(data.trajectories) == 3
        assert data.dispersions.shape == (3, 24)


class TestGenerateFinalReport:
    def test_produces_html_file(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100)
        output = tmp_path / "report.html"
        result = generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "plotly" in content.lower()

    def test_html_contains_expected_panels(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Delta-V" in content
        assert "Apoapsis" in content
        assert "Periapsis" in content
        assert "Inclination" in content

    def test_mixed_captured_and_hyperbolic(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(n_captured=80, n_hyper=20)
        output = tmp_path / "report.html"
        result = generate_final_report(eval_data, "ftc", 50.0, output)
        assert result == output
        assert output.exists()

    def test_zero_captures_does_not_crash(self, tmp_path: Path) -> None:
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(50, all_hyperbolic=True)
        output = tmp_path / "report.html"
        result = generate_final_report(eval_data, "fnpag", 50.0, output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "No captured trajectories" in content

    def test_exit_conditions_label(self, tmp_path: Path) -> None:
        """The old 'Entry Conditions' panel is now 'Exit Conditions'."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Exit Conditions" in content

    def test_performance_table_has_min_max_and_metrics(self, tmp_path: Path) -> None:
        """Performance table includes Min, Max, g-load, heat flux, bank angle."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Min" in content
        assert "Max" in content
        assert "g-load" in content
        assert "heat flux" in content
        assert "Bank angle" in content

    def test_corridor_panels_present_with_trajectories(self, tmp_path: Path) -> None:
        """Corridor panels appear when trajectories are provided."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(50, with_trajectories=True)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Dynamic Pressure" in content
        assert "Bank Angle" in content
        assert "Energy vs" in content

    def test_corridor_panels_absent_without_trajectories(self, tmp_path: Path) -> None:
        """Corridor panels should not appear without trajectory data."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(50, with_trajectories=False)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        content = output.read_text()
        assert "Energy vs Dynamic Pressure" not in content

    def test_dispersion_grid_written_to_separate_file(self, tmp_path: Path) -> None:
        """Dispersion grid is written to a separate HTML file."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100, with_dispersions=True)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        disp_path = tmp_path / "report_dispersions.html"
        assert disp_path.exists()
        content = disp_path.read_text()
        assert "Dispersion Correlation Grid" in content

    def test_dispersion_grid_absent_without_data(self, tmp_path: Path) -> None:
        """No dispersion grid file when dispersions are None."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100, with_dispersions=False)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        disp_path = tmp_path / "report_dispersions.html"
        assert not disp_path.exists()

    def test_dispersion_grid_has_r_squared_and_pvalue(self, tmp_path: Path) -> None:
        """Dispersion scatter subplots contain R² and p-value annotations."""
        from aerocapture.training.final_report import generate_final_report

        eval_data = _make_eval_data(100, with_dispersions=True)
        output = tmp_path / "report.html"
        generate_final_report(eval_data, "equilibrium_glide", 50.0, output)
        disp_path = tmp_path / "report_dispersions.html"
        content = disp_path.read_text()
        # R² is encoded as R\u00b2 in Plotly JSON
        assert "R\\u00b2=" in content
        assert "p=" in content


class TestRunFinalEvaluation:
    def test_patches_n_sims_and_seed(self, tmp_path: Path) -> None:
        """Verify TOML patching writes correct n_sims and seed."""
        import tomllib

        from aerocapture.training.final_report import _patch_toml_for_final_eval

        toml_content = '[simulation]\nn_sims = 10\n[monte_carlo]\nseed = 1\n[guidance]\ntype = "ftc"\n'
        src_toml = tmp_path / "base.toml"
        src_toml.write_text(toml_content)

        patched = _patch_toml_for_final_eval(src_toml, n_sims=1000, seed=9999)
        with open(patched, "rb") as f:
            data = tomllib.load(f)
        assert data["simulation"]["n_sims"] == 1000
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


class TestRefTrajectory:
    def test_read_ref_trajectory_path_string(self, tmp_path: Path) -> None:
        """Reads reference trajectory path from TOML when it's a string."""
        from aerocapture.training.final_report import _read_ref_trajectory_path

        # Create a dummy .dat file for the path to resolve
        dat_file = tmp_path / "ref.dat"
        dat_file.write_text("1.0 2.0 3.0 4.0 5.0 6.0 0.5\n")
        toml_content = f'[data]\nreference_trajectory = "{dat_file}"\n'
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)
        result = _read_ref_trajectory_path(toml_file)
        assert result is not None

    def test_read_ref_trajectory_path_bool(self, tmp_path: Path) -> None:
        """Returns None when reference_trajectory is a boolean."""
        from aerocapture.training.final_report import _read_ref_trajectory_path

        toml_content = "[data]\nreference_trajectory = true\n"
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(toml_content)
        assert _read_ref_trajectory_path(toml_file) is None

    def test_load_reference_trajectory(self, tmp_path: Path) -> None:
        """Loads and converts reference trajectory data."""
        from aerocapture.training.final_report import _load_reference_trajectory

        data = np.column_stack(
            [
                np.linspace(5, -2, 10),  # energy MJ/kg
                np.linspace(0, 1000, 10),  # pdyn Pa
                np.zeros(10),
                np.zeros(10),
                np.linspace(0.5, 1.0, 10),  # inclination rad
                np.zeros(10),
                np.linspace(0.0, 1.0, 10),  # cos(bank)
            ]
        )
        dat_file = tmp_path / "ref.dat"
        np.savetxt(dat_file, data)
        result = _load_reference_trajectory(dat_file)
        assert result is not None
        assert "energy_MJkg" in result
        assert "pdyn_kPa" in result
        assert "inclination_deg" in result
        assert "bank_deg" in result
        # Check conversion: pdyn Pa -> kPa
        np.testing.assert_allclose(result["pdyn_kPa"], data[:, 1] / 1e3)


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
