"""Tests for the adaptive seed pool."""

from __future__ import annotations

import numpy as np
import pytest

from aerocapture.training.seed_pool import aggregate_fitness, compute_cvar


class TestComputeCvar:
    """Tests for CVaR (Conditional Value at Risk) computation."""

    def test_cvar_basic(self) -> None:
        """CVaR-20 of [1, 2, 3, 4, 5] = mean of worst 20% = 5.0."""
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_cvar(costs, percentile=20)
        assert result == pytest.approx(5.0)

    def test_cvar_50(self) -> None:
        """CVaR-50 of [1, 2, 3, 4] = mean of worst 50% = (3+4)/2 = 3.5."""
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_cvar(costs, percentile=50)
        assert result == pytest.approx(3.5)

    def test_cvar_floor_single_element(self) -> None:
        """When n_seeds * percentile / 100 < 1, floor to 1 sample (worst)."""
        costs = np.array([1.0, 2.0, 3.0])
        result = compute_cvar(costs, percentile=20)  # 3 * 0.2 = 0.6 -> floor to 1
        assert result == pytest.approx(3.0)

    def test_cvar_single_seed(self) -> None:
        """Single seed: CVaR = that cost."""
        costs = np.array([42.0])
        result = compute_cvar(costs, percentile=20)
        assert result == pytest.approx(42.0)


class TestAggregateFitness:
    """Tests for mean/CVaR blended fitness aggregation."""

    def test_alpha_1_is_pure_mean(self) -> None:
        """alpha=1.0 -> pure mean."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=1.0, cvar_percentile=20)
        assert result[0] == pytest.approx(3.0)

    def test_alpha_0_is_pure_cvar(self) -> None:
        """alpha=0.0 -> pure CVaR."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=0.0, cvar_percentile=20)
        assert result[0] == pytest.approx(5.0)

    def test_default_blend(self) -> None:
        """alpha=0.7 -> 0.7*mean + 0.3*CVaR."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=0.7, cvar_percentile=20)
        expected = 0.7 * 3.0 + 0.3 * 5.0
        assert result[0] == pytest.approx(expected)

    def test_multiple_individuals(self) -> None:
        """Aggregation works row-wise for multiple individuals."""
        cost_matrix = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        )
        result = aggregate_fitness(cost_matrix, alpha=1.0, cvar_percentile=20)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(30.0)
