"""Tests for aerocapture.training.charts — training convergence panels 1-6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aerocapture.training.charts import (
    chart_capture_constraint_rate,
    chart_convergence,
    chart_cost_distribution,
    chart_diversity_cost,
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
