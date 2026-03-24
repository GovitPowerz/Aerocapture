"""Tests for aerocapture.training.charts — training convergence panels 1-6 and corridor panels 7-9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.charts import (
    chart_altitude_time,
    chart_bank_angle_time,
    chart_capture_constraint_rate,
    chart_convergence,
    chart_corridor_bank,
    chart_corridor_inclination,
    chart_corridor_pdyn,
    chart_cost_distribution,
    chart_diversity_cost,
    chart_gload_time,
    chart_heat_flux_time,
    chart_nav_density_ratio,
    chart_parameter_evolution,
    chart_seed_pool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_svg(tmp_path: Path) -> Path:
    """Return a temporary SVG output path."""
    return tmp_path / "test.svg"


@pytest.fixture()
def training_records() -> list[dict[str, Any]]:
    """Return 10 generation records with basic convergence data."""
    records: list[dict[str, Any]] = []
    for i in range(10):
        records.append({
            "generation": i,
            "best_cost": 100.0 / (i + 1),
            "mean_cost": 200.0 / (i + 1),
            "worst_cost": 500.0 / (i + 1),
            "improvement": i % 3 == 0,
        })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTrainingCharts:
    """Tests for panels 1-6 chart functions."""

    def test_convergence_creates_svg(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 1: convergence chart creates a valid SVG file."""
        chart_convergence(training_records, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_convergence_no_data_raises(self, tmp_svg: Path) -> None:
        """Panel 1: empty records raises ValueError."""
        with pytest.raises(ValueError, match="No training records provided"):
            chart_convergence([], tmp_svg)

    def test_capture_constraint_rate(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 2: capture + constraint rate chart creates SVG."""
        for r in training_records:
            r["capture_rate"] = 0.5 + 0.05 * r["generation"]
            r["constraint_violation_rate"] = 0.3 - 0.02 * r["generation"]
        chart_capture_constraint_rate(training_records, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_diversity_cost(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 3: diversity vs cost chart creates SVG."""
        for r in training_records:
            r["population_diversity"] = 0.8 - 0.05 * r["generation"]
        chart_diversity_cost(training_records, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_cost_distribution(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 4: cost distribution box plots creates SVG."""
        for r in training_records:
            r["all_costs"] = [r["best_cost"] + j * 10 for j in range(5)]
        result = chart_cost_distribution(training_records, tmp_svg)
        assert result is True
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_parameter_evolution(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 5: parameter evolution chart creates SVG."""
        for r in training_records:
            r["best_params"] = {"alpha": 0.1 * r["generation"], "beta": 1.0 - 0.05 * r["generation"]}
        chart_parameter_evolution(training_records, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_seed_pool_creates_svg(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 6: seed pool chart creates SVG when pool_metrics present."""
        for r in training_records:
            r["pool_metrics"] = {"pool_size": 10 + r["generation"], "mean_difficulty": 0.5 + 0.02 * r["generation"]}
        result = chart_seed_pool(training_records, tmp_svg)
        assert result is True
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_seed_pool_skipped_when_no_data(self, training_records: list[dict[str, Any]], tmp_svg: Path) -> None:
        """Panel 6: returns False and creates no file when pool_metrics absent."""
        result = chart_seed_pool(training_records, tmp_svg)
        assert result is False
        assert not tmp_svg.exists()


# ---------------------------------------------------------------------------
# Corridor chart fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mc_trajectories() -> list[npt.NDArray[np.float64]]:
    """Synthetic MC trajectories (10 runs, ~50 timesteps each, 16 cols)."""
    rng = np.random.default_rng(42)
    trajs: list[npt.NDArray[np.float64]] = []
    for _ in range(10):
        n_steps = int(rng.integers(40, 60))
        traj = np.zeros((n_steps, 16))
        traj[:, 0] = np.linspace(120, 30, n_steps)  # alt_km
        traj[:, 7] = np.linspace(0, 300, n_steps)  # time_s
        traj[:, 8] = np.linspace(-1.0, -3.0, n_steps)  # energy_mj_kg
        traj[:, 9] = rng.uniform(0.5, 5.0, n_steps)  # pdyn_kpa
        traj[:, 10] = rng.uniform(0, 90, n_steps)  # bank_angle_deg
        traj[:, 11] = rng.uniform(24.0, 25.0, n_steps)  # inclination_deg
        trajs.append(traj)
    return trajs


@pytest.fixture()
def captured_mask() -> npt.NDArray[np.bool_]:
    """Capture mask: first 8 captured, last 2 hyperbolic."""
    mask = np.ones(10, dtype=bool)
    mask[8:] = False
    return mask


# ---------------------------------------------------------------------------
# Corridor chart tests
# ---------------------------------------------------------------------------
class TestCorridorCharts:
    """Tests for corridor/energy panels 7-9."""

    def test_pdyn_creates_svg(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 7: pdyn corridor chart creates a valid SVG file."""
        chart_corridor_pdyn(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_pdyn_with_corridor_data(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 7: pdyn corridor chart with 4-layer corridor fill zones."""
        n_bins = 50
        energy_bins = np.linspace(-1.0, -3.0, n_bins)
        corridor_data: dict[str, Any] = {
            "energy_bins": energy_bins,
            "envelope_crash_pdyn": np.full(n_bins, 8.0),
            "envelope_restricted_max_pdyn": np.full(n_bins, 6.0),
            "envelope_restricted_min_pdyn": np.full(n_bins, 2.0),
            "envelope_capture_pdyn": np.full(n_bins, 0.5),
        }
        chart_corridor_pdyn(mc_trajectories, captured_mask, tmp_svg, corridor_data=corridor_data)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_inclination_creates_svg(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 8: inclination corridor chart creates a valid SVG file."""
        chart_corridor_inclination(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_bank_creates_svg(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 9: bank angle corridor chart creates a valid SVG file."""
        chart_corridor_bank(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content


# ---------------------------------------------------------------------------
# Time-domain chart tests
# ---------------------------------------------------------------------------
class TestTimeDomainCharts:
    """Tests for time-domain trajectory panels 10-14."""

    def test_altitude_time(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 10: altitude vs time spaghetti creates a valid SVG file."""
        chart_altitude_time(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_altitude_highlights_best(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 10: altitude chart highlights best trajectory when best_idx provided."""
        chart_altitude_time(mc_trajectories, captured_mask, tmp_svg, best_idx=0)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_heat_flux_with_limit(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 11: heat flux chart with constraint limit line."""
        rng = np.random.default_rng(99)
        for traj in mc_trajectories:
            traj[:, 6] = rng.uniform(50.0, 200.0, traj.shape[0])
        chart_heat_flux_time(mc_trajectories, captured_mask, tmp_svg, limit_kw_m2=150.0)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_gload_with_limit(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 12: g-load chart with constraint limit line."""
        rng = np.random.default_rng(99)
        for traj in mc_trajectories:
            traj[:, 12] = rng.uniform(0.5, 5.0, traj.shape[0])
        chart_gload_time(mc_trajectories, captured_mask, tmp_svg, limit_g=4.0)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_bank_angle_time(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 13: bank angle vs time spaghetti creates a valid SVG file."""
        chart_bank_angle_time(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content

    def test_nav_density_ratio(
        self, mc_trajectories: list[npt.NDArray[np.float64]], captured_mask: npt.NDArray[np.bool_], tmp_svg: Path
    ) -> None:
        """Panel 14: nav density ratio chart with perfect-estimate reference line."""
        rng = np.random.default_rng(99)
        for traj in mc_trajectories:
            traj[:, 13] = rng.uniform(0.8, 1.2, traj.shape[0])
        chart_nav_density_ratio(mc_trajectories, captured_mask, tmp_svg)
        assert tmp_svg.exists()
        content = tmp_svg.read_text()
        assert "<svg" in content
