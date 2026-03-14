"""Tests for the adaptive seed pool."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.seed_pool import SeedPool, aggregate_fitness, compute_cvar


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


class TestSeedPoolGrowth:
    def test_bootstrap_creates_5_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        assert pool.seeds == [100, 101, 102, 103, 104]

    def test_incremental_growth(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        pool.add_seeds(generation=1)
        assert len(pool.seeds) == 6
        assert 105 in pool.seeds

    def test_no_duplicate_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5


class TestSeedPoolEviction:
    def test_eviction_at_cap(self) -> None:
        pool = SeedPool(base_seed=0, max_size=7)
        pool.add_seeds(generation=0)  # 5 seeds
        pool.add_seeds(generation=1)  # 6 seeds
        pool.add_seeds(generation=2)  # 7 seeds
        pool.add_seeds(generation=3)  # 8 seeds -> should evict to 7
        for i, seed in enumerate(pool.seeds):
            pool.difficulty[seed] = float(i * 10)
        pool.evict_redundant()
        assert len(pool.seeds) == 7

    def test_evict_closest_pair_older_one(self) -> None:
        pool = SeedPool(base_seed=0, max_size=3)
        pool.seeds = [10, 20, 30, 40]
        pool.difficulty = {10: 1.0, 20: 1.5, 30: 5.0, 40: 10.0}
        pool.generation_added = {10: 0, 20: 1, 30: 2, 40: 3}
        pool.evict_redundant()
        assert len(pool.seeds) == 3
        assert 10 not in pool.seeds
        assert 20 in pool.seeds

    def test_no_eviction_under_cap(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [1, 2, 3]
        pool.difficulty = {1: 1.0, 2: 2.0, 3: 3.0}
        pool.generation_added = {1: 0, 2: 1, 3: 2}
        pool.evict_redundant()
        assert len(pool.seeds) == 3


class TestSeedPoolScoring:
    def test_score_updates_difficulty(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [10, 20, 30]
        cost_matrix = np.array(
            [
                [100.0, 200.0, 300.0],
                [10.0, 20.0, 30.0],
                [50.0, 60.0, 70.0],
            ]
        )
        best_idx = 1
        pool.score_difficulty(cost_matrix, best_idx)
        assert pool.difficulty[10] == pytest.approx(10.0)
        assert pool.difficulty[20] == pytest.approx(20.0)
        assert pool.difficulty[30] == pytest.approx(30.0)


class TestSeedPoolCheckpoint:
    def test_round_trip(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50, alpha=0.8, cvar_percentile=25)
        pool.seeds = [42, 43, 44]
        pool.difficulty = {42: 1.0, 43: 5.0, 44: 10.0}
        pool.generation_added = {42: 0, 43: 0, 44: 1}
        pool.n_evictions = 2
        data = pool.to_dict()
        restored = SeedPool.from_dict(data)
        assert restored.base_seed == 42
        assert restored.max_size == 50
        assert restored.alpha == 0.8
        assert restored.cvar_percentile == 25
        assert restored.seeds == [42, 43, 44]
        assert restored.difficulty == {42: 1.0, 43: 5.0, 44: 10.0}
        assert restored.generation_added == {42: 0, 43: 0, 44: 1}
        assert restored.n_evictions == 2

    def test_round_trip_json_compatible(self) -> None:
        import json

        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [0, 1, 2]
        pool.difficulty = {0: 1.0, 1: 2.0, 2: 3.0}
        pool.generation_added = {0: 0, 1: 0, 2: 1}
        data = pool.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = SeedPool.from_dict(restored_data)
        assert restored.seeds == [0, 1, 2]


class TestSeedPoolEvaluation:
    """Tests for pool-based population evaluation."""

    def test_evaluate_population_calls_evaluator(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1, 2]
        pool.generation_added = {0: 0, 1: 0, 2: 0}

        population = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

        def evaluator(chrom: npt.NDArray[np.int8], seed: int) -> float:
            return float(seed) + float(chrom[0])

        fitness = pool.evaluate_population(population, evaluator)
        assert fitness.shape == (2,)
        assert fitness[0] == pytest.approx(2.0)  # costs=[1,2,3], mean=2.0
        assert fitness[1] == pytest.approx(1.0)  # costs=[0,1,2], mean=1.0

    def test_evaluate_population_updates_difficulty(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1]
        pool.generation_added = {0: 0, 1: 0}

        population = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)

        def evaluator(chrom: npt.NDArray[np.int8], seed: int) -> float:
            return float(seed) * 10.0

        fitness = pool.evaluate_population(population, evaluator)
        assert pool.difficulty[0] == pytest.approx(0.0)
        assert pool.difficulty[1] == pytest.approx(10.0)

    def test_evaluate_population_with_batch_evaluator(self) -> None:
        """Batch evaluator is used when provided."""
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1, 2]
        pool.generation_added = {0: 0, 1: 0, 2: 0}

        population = np.array([[1, 0], [0, 1]], dtype=np.int8)

        scalar_called = False

        def scalar_eval(chrom: npt.NDArray[np.int8], seed: int) -> float:
            nonlocal scalar_called
            scalar_called = True
            return 0.0

        def batch_eval(chrom: npt.NDArray[np.int8], seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([float(s) + float(chrom[0]) for s in seeds])

        fitness = pool.evaluate_population(population, scalar_eval, batch_evaluator=batch_eval)
        assert not scalar_called  # batch should be used instead
        assert fitness.shape == (2,)


class TestAdaptiveSeedIntegration:
    """Integration test: adaptive seed pool in the GA training loop."""

    def test_pool_grows_and_evicts_during_training(self) -> None:
        """Verify pool grows, evicts, and produces valid fitness across generations."""
        pool = SeedPool(base_seed=0, max_size=8, alpha=0.7, cvar_percentile=20)

        rng = np.random.default_rng(42)
        pop = rng.integers(0, 2, size=(4, 10), dtype=np.int8)

        def evaluator(chrom: npt.NDArray[np.int8], seed: int) -> float:
            quality = float(np.sum(chrom)) / len(chrom)
            return float(seed) * 10.0 + quality * 5.0

        for gen in range(10):
            pool.add_seeds(gen)
            fitness = pool.evaluate_population(pop, evaluator)

            assert fitness.shape == (4,)
            assert all(np.isfinite(fitness))

            pool.evict_redundant()
            assert len(pool.seeds) <= 8

        # After 10 gens: bootstrapped 5, added 9 more = 14 total, evicted to 8
        assert len(pool.seeds) == 8
        assert pool.n_evictions == 6

        # Difficulty should be populated for all active seeds
        assert len(pool.difficulty) == len(pool.seeds)

        # Difficulty range should be non-trivial
        d_min, d_max = pool.difficulty_range
        assert d_max > d_min
