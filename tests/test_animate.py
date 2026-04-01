"""Tests for training animation generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


class TestDiscoverCheckpoints:
    @pytest.fixture()
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create a directory with 5 fake checkpoints."""
        d = tmp_path / "piecewise_constant"
        d.mkdir()
        for gen in [0, 10, 20, 30, 40]:
            prefix = f"checkpoint_r000_g{gen:05d}"
            meta = {"run": 0, "generation": gen, "best_cost": 100.0 - gen, "cost_history": [100.0 - g for g in range(0, gen + 1, 10)]}
            (d / f"{prefix}.json").write_text(json.dumps(meta))
            np.savez_compressed(
                d / f"{prefix}.npz",
                pop_0=np.zeros((5, 10), dtype=np.int8),
                costs_0=np.full(5, 100.0 - gen),
                n_subpops=np.array([1]),
                best_chromosome=np.zeros(10, dtype=np.int8),
            )
        return d

    def test_discovers_all_checkpoints_sorted(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=1)
        assert len(result) == 5
        assert [c["generation"] for c in result] == [0, 10, 20, 30, 40]

    def test_every_filters_checkpoints(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=2)
        # Every=2 means take every 2nd: indices 0, 2, 4 -> gens 0, 20, 40
        assert len(result) == 3
        assert [c["generation"] for c in result] == [0, 20, 40]

    def test_always_includes_last_checkpoint(self, checkpoint_dir: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(checkpoint_dir, every=3)
        # Indices 0, 3 -> gens 0, 30. Last (40) always included.
        assert result[-1]["generation"] == 40

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import _discover_checkpoints

        result = _discover_checkpoints(tmp_path, every=1)
        assert result == []


class TestBuildOverrides:
    def test_builds_dot_path_overrides(self) -> None:
        from aerocapture.training.animate import _build_overrides

        params = {"gain_kp": 0.5, "gain_kd": 0.1}
        overrides = _build_overrides("equilibrium_glide", params, n_sims=50)
        assert overrides["guidance.equilibrium_glide.gain_kp"] == 0.5
        assert overrides["guidance.equilibrium_glide.gain_kd"] == 0.1
        assert overrides["guidance.type"] == "equilibrium_glide"
        assert overrides["simulation.n_sims"] == 50

    def test_lateral_params_go_to_lateral_section(self) -> None:
        from aerocapture.training.animate import _build_overrides

        params = {"gain_kp": 0.5, "lateral.corridor_slope": 100.0}
        overrides = _build_overrides("equilibrium_glide", params, n_sims=50)
        assert overrides["guidance.lateral.corridor_slope"] == 100.0
        assert "guidance.equilibrium_glide.lateral.corridor_slope" not in overrides


class TestComputeAxisRanges:
    def test_returns_dict_with_expected_keys(self) -> None:
        from aerocapture.training.animate import _compute_axis_ranges

        # Create fake trajectory data (17-column format matching PyO3 output)
        rng = np.random.default_rng(42)
        trajectories = [rng.standard_normal((50, 17)) for _ in range(10)]
        costs = rng.uniform(50, 200, size=30)

        ranges = _compute_axis_ranges(trajectories, costs)
        for key in ("energy_min", "energy_max", "pdyn_min", "pdyn_max", "incl_min", "incl_max", "bank_min", "bank_max", "cost_max"):
            assert key in ranges

    def test_ranges_have_margin(self) -> None:
        from aerocapture.training.animate import _compute_axis_ranges

        # All trajectories have energy in [0, 1], pdyn in [0, 100]
        traj = np.zeros((50, 17))
        traj[:, 8] = np.linspace(0, 1, 50)  # energy
        traj[:, 9] = np.linspace(0, 100, 50)  # pdyn
        traj[:, 10] = np.linspace(-90, 90, 50)  # bank
        traj[:, 11] = np.linspace(20, 30, 50)  # inclination

        ranges = _compute_axis_ranges([traj], np.array([100.0]))
        # Ranges should be slightly wider than data (5% margin)
        assert ranges["energy_min"] < 0
        assert ranges["energy_max"] > 1


class TestRenderFrame:
    def test_returns_figure_with_4_axes(self) -> None:
        from aerocapture.training.animate import _render_frame
        from aerocapture.training.charts import classify_trajectories

        rng = np.random.default_rng(42)
        trajectories = [rng.standard_normal((50, 17)).astype(np.float64) for _ in range(10)]
        final_records = rng.standard_normal((10, 52)).astype(np.float64)
        # Set ifinal=3 and ecc<1 for some to be "captured"
        final_records[:5, 31] = 3.0  # ifinal
        final_records[:5, 9] = 0.5  # ecc < 1
        final_records[5:, 31] = 1.0  # not captured

        traj_class = classify_trajectories(final_records)

        costs = rng.uniform(50, 200, size=30)
        axis_ranges = {
            "energy_min": -2.0,
            "energy_max": 2.0,
            "pdyn_min": -2.0,
            "pdyn_max": 2.0,
            "incl_min": -2.0,
            "incl_max": 2.0,
            "bank_min": -2.0,
            "bank_max": 2.0,
            "cost_max": 250.0,
        }

        fig = _render_frame(
            generation=42,
            best_cost=55.0,
            capture_rate=0.8,
            trajectories=trajectories,
            traj_class=traj_class,
            costs=costs,
            corridor_data=None,
            axis_ranges=axis_ranges,
        )
        assert fig is not None
        axes = fig.get_axes()
        # 4 main panels + 1 twinx on cost CDF = 5 axes
        assert len(axes) == 5
        plt.close(fig)


class TestGenerateAnimation:
    def test_errors_without_pyo3(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import generate_animation

        with patch.dict("sys.modules", {"aerocapture_rs": None}), pytest.raises(RuntimeError, match="aerocapture_rs"):
            generate_animation(tmp_path, toml_path=tmp_path / "fake.toml")

    def test_errors_on_empty_dir(self, tmp_path: Path) -> None:
        from aerocapture.training.animate import generate_animation

        mock_aero = MagicMock()
        with patch.dict("sys.modules", {"aerocapture_rs": mock_aero}), pytest.raises(FileNotFoundError, match="No checkpoints"):
            generate_animation(tmp_path, toml_path=tmp_path / "fake.toml")

    def test_generates_gif(self, tmp_path: Path) -> None:
        """End-to-end test with mocked PyO3 calls."""
        from aerocapture.training.animate import generate_animation

        # Create 2 fake checkpoints
        d = tmp_path / "scheme"
        d.mkdir()
        rng = np.random.default_rng(42)
        for gen in [0, 10]:
            prefix = f"checkpoint_r000_g{gen:05d}"
            meta = {"run": 0, "generation": gen, "best_cost": 100.0 - gen, "cost_history": [100.0]}
            (d / f"{prefix}.json").write_text(json.dumps(meta))
            np.savez_compressed(
                d / f"{prefix}.npz",
                pop_0=np.zeros((5, 80), dtype=np.int8),
                costs_0=rng.uniform(50, 150, size=5),
                n_subpops=np.array([1]),
                best_chromosome=np.zeros(80, dtype=np.int8),
            )

        # Create a fake TOML
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[guidance]\ntype = "equilibrium_glide"\n')

        # Mock aerocapture_rs.run_mc to return fake data
        n_sims = 10
        mock_results = MagicMock()
        mock_results.final_records = rng.standard_normal((n_sims, 52)).astype(np.float64)
        # Set some as captured (ifinal=3, ecc<1)
        mock_results.final_records[:5, 31] = 3.0
        mock_results.final_records[:5, 9] = 0.5
        mock_results.final_records[5:, 31] = 1.0
        mock_results.trajectories = [rng.standard_normal((50, 17)).astype(np.float64) for _ in range(n_sims)]

        mock_aero = MagicMock()
        mock_aero.run_mc.return_value = mock_results

        with (
            patch.dict("sys.modules", {"aerocapture_rs": mock_aero}),
            patch("aerocapture.training.animate._load_pyo3", return_value=mock_aero),
            patch("aerocapture.training.animate._decode_and_build_overrides", return_value={"guidance.type": "equilibrium_glide", "simulation.n_sims": n_sims}),
        ):
            gif_path = generate_animation(d, toml_path=toml_path, n_sims=n_sims, fps=2)

        assert gif_path.exists()
        assert gif_path.suffix == ".gif"
        assert gif_path.stat().st_size > 0
