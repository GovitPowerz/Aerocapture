"""Tests for training metrics pure functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from aerocapture.training.metrics import capture_rate, convergence_speed, cost_stats, population_diversity, stagnation_count
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


class TestCostStats:
    def test_basic_stats(self) -> None:
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["std"] == pytest.approx(np.std(costs), abs=1e-10)

    def test_filters_inf(self) -> None:
        costs = np.array([1.0, np.inf, 3.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 3.0
        assert stats["mean"] == 2.0

    def test_filters_nan(self) -> None:
        costs = np.array([1.0, np.nan, 5.0])
        stats = cost_stats(costs)
        assert stats["best"] == 1.0
        assert stats["worst"] == 5.0

    def test_all_nonfinite_returns_nan(self) -> None:
        costs = np.array([np.inf, np.nan, np.inf])
        stats = cost_stats(costs)
        assert math.isnan(stats["best"])
        assert math.isnan(stats["mean"])

    def test_single_element(self) -> None:
        costs = np.array([42.0])
        stats = cost_stats(costs)
        assert stats["best"] == 42.0
        assert stats["std"] == 0.0


class TestPopulationDiversity:
    def test_identical_population_zero_diversity(self) -> None:
        pop = np.array([[0.5, 0.3, 0.7], [0.5, 0.3, 0.7], [0.5, 0.3, 0.7]], dtype=np.float64)
        assert population_diversity(pop) == 0.0

    def test_maximally_diverse_pair(self) -> None:
        # Corners of unit hypercube: distance = sqrt(4) = 2, max_distance = sqrt(4) = 2
        pop = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        assert population_diversity(pop) == pytest.approx(1.0)

    def test_partial_diversity(self) -> None:
        # Distance = sqrt(1) = 1, max = sqrt(4) = 2, normalized = 0.5
        pop = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        assert population_diversity(pop) == pytest.approx(0.5)

    def test_single_individual(self) -> None:
        pop = np.array([[0.5, 0.3, 0.7]], dtype=np.float64)
        assert population_diversity(pop) == 0.0

    @given(
        arrays(dtype=np.float64, shape=st.tuples(st.integers(2, 20), st.integers(1, 50)), elements=st.floats(0.0, 1.0)),
    )
    @settings(max_examples=50)
    def test_diversity_in_unit_range(self, pop: np.ndarray) -> None:
        d = population_diversity(pop)
        assert 0.0 <= d <= 1.0 + 1e-10

    def test_matches_reference_double_loop(self) -> None:
        """Pin the exact statistic: mean of all unordered pairwise L2 distances,
        normalized by sqrt(n_dims). Reference computed inline with a double loop."""
        rng = np.random.default_rng(0)
        pop = rng.random((5, 3))
        n, n_dims = pop.shape
        total = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(np.sqrt(np.sum((pop[i] - pop[j]) ** 2)))
                n_pairs += 1
        expected = total / (n_pairs * np.sqrt(float(n_dims)))
        assert population_diversity(pop) == pytest.approx(expected, abs=1e-12)

    def test_n_less_than_two_returns_zero(self) -> None:
        assert population_diversity(np.zeros((0, 3), dtype=np.float64)) == 0.0
        assert population_diversity(np.array([[0.1, 0.2, 0.3]], dtype=np.float64)) == 0.0


class TestCaptureRate:
    def test_all_captured(self) -> None:
        costs = np.array([100.0, 200.0, 500.0])
        assert capture_rate(costs) == 1.0

    def test_none_captured(self) -> None:
        costs = np.array([1e6 + 100, 1e6 + 200, 2e6])
        assert capture_rate(costs) == 0.0

    def test_mixed(self) -> None:
        costs = np.array([100.0, 1e6 + 100, 200.0, 2e6])
        assert capture_rate(costs) == 0.5

    def test_custom_threshold(self) -> None:
        costs = np.array([10.0, 50.0, 100.0])
        assert capture_rate(costs, capture_threshold=50.0) == pytest.approx(1 / 3)


class TestConvergenceSpeed:
    def test_instant_convergence(self) -> None:
        history = [100.0, 10.0, 10.0, 10.0, 10.0]
        assert convergence_speed(history) == 1

    def test_gradual_convergence(self) -> None:
        history = [100.0, 80.0, 60.0, 40.0, 20.0, 10.0]
        speed = convergence_speed(history)
        assert 1 <= speed <= len(history)

    def test_no_improvement(self) -> None:
        history = [50.0, 50.0, 50.0]
        assert convergence_speed(history) == 0


class TestStagnationCount:
    def test_no_stagnation(self) -> None:
        history = [100.0, 90.0, 80.0, 70.0]
        assert stagnation_count(history) == 0

    def test_full_stagnation(self) -> None:
        history = [50.0, 50.0, 50.0, 50.0]
        assert stagnation_count(history) == 3

    def test_trailing_stagnation(self) -> None:
        history = [100.0, 50.0, 50.0, 50.0]
        assert stagnation_count(history) == 2
