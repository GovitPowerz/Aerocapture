"""Tests for the adaptive seed pool and reserved seed utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, VALIDATION_SEED_OFFSET, make_reserved_seeds
from aerocapture.training.seed_pool import SeedPool, _pool_seed, _stress_seed, aggregate_fitness, compute_cvar


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


class TestPoolSeedHash:
    def test_different_indices_produce_different_seeds(self) -> None:
        s0 = _pool_seed(42, 0)
        s1 = _pool_seed(42, 1)
        assert s0 != s1

    def test_seeds_within_valid_range(self) -> None:
        for i in range(100):
            s = _pool_seed(42, i)
            assert 0 <= s < 2**31

    def test_different_bases_produce_different_seeds(self) -> None:
        assert _pool_seed(42, 0) != _pool_seed(99, 0)

    def test_deterministic(self) -> None:
        assert _pool_seed(42, 7) == _pool_seed(42, 7)


class TestSeedPoolExclusion:
    def test_excluded_seed_never_in_pool(self) -> None:
        excluded = _pool_seed(42, 0)
        pool = SeedPool(base_seed=42, max_size=50, excluded_seeds={excluded})
        pool.add_seeds(generation=0)
        assert excluded not in pool.seeds
        assert len(pool.seeds) == 10

    def test_pool_uses_hash_seeds_not_consecutive(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)
        assert pool.seeds != [42, 43, 44, 45, 46]
        assert len(pool.seeds) == 10


class TestSeedPoolGrowth:
    def test_bootstrap_creates_default_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 10
        assert all(0 <= s < 2**31 for s in pool.seeds)
        assert len(set(pool.seeds)) == 10

    def test_incremental_growth(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 10
        pool.add_seeds(generation=1)
        assert len(pool.seeds) == 11

    def test_no_duplicate_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        pool.add_seeds(generation=0)
        # Bootstrap adds 10, second call adds 1 more (always unique via hash)
        assert len(pool.seeds) == 11
        assert len(set(pool.seeds)) == 11


class TestGapClosureEviction:
    def test_closest_pair_evicted(self) -> None:
        """Seeds with the smallest difficulty gap are evicted first."""
        pool = SeedPool(base_seed=0, max_size=3)
        # Sorted by difficulty: 20(1.0), 40(2.0), 30(50.0), 50(75.0), 10(100.0)
        # Gaps: 1.0, 48.0, 25.0, 25.0 -- tightest pair is (20, 40)
        pool.seeds = [10, 20, 30, 40, 50]
        pool.difficulty = {10: 100.0, 20: 1.0, 30: 50.0, 40: 2.0, 50: 75.0}
        pool.generation_added = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
        pool.evict_redundant()
        assert len(pool.seeds) == 3
        # Endpoints (easiest=20 or hardest=10) should survive because
        # gap-closure targets the tightest pair, not the extremes.
        assert 10 in pool.seeds  # hardest survives
        assert 20 in pool.seeds or 40 in pool.seeds  # one of the tight pair survives

    def test_spectrum_endpoints_survive(self) -> None:
        """Easiest and hardest seeds survive when they're far from neighbors."""
        pool = SeedPool(base_seed=0, max_size=2)
        # Sorted: 2(0.1), 3(0.2), 5(0.3), 4(999.0), 1(1000.0)
        # Gaps: 0.1, 0.1, 998.7, 1.0 -- tightest pairs at the easy end
        pool.seeds = [1, 2, 3, 4, 5]
        pool.difficulty = {1: 1000.0, 2: 0.1, 3: 0.2, 4: 999.0, 5: 0.3}
        pool.generation_added = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        pool.evict_redundant()
        assert len(pool.seeds) == 2
        # The three clustered easy seeds (0.1, 0.2, 0.3) get evicted against
        # each other. The two hard seeds (999, 1000) have a gap of 1.0 which
        # is larger than 0.1, so they survive longer. One easy seed survives.
        assert 1 in pool.seeds or 4 in pool.seeds  # at least one hard seed

    def test_newer_seed_evicted_from_pair(self) -> None:
        """When two seeds have equal difficulty, the newer one is evicted."""
        pool = SeedPool(base_seed=0, max_size=2)
        pool.seeds = [10, 20, 30]
        pool.difficulty = {10: 5.0, 20: 5.0, 30: 100.0}
        pool.generation_added = {10: 0, 20: 5, 30: 3}
        pool.evict_redundant()
        assert len(pool.seeds) == 2
        # Seeds 10 and 20 have gap=0 (tightest pair). Seed 20 is newer -> evicted.
        assert 10 in pool.seeds
        assert 30 in pool.seeds

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
        pool.add_seeds(generation=0)
        seeds_before = pool.seeds.copy()
        for s in pool.seeds:
            pool.difficulty[s] = float(s % 100)
        pool.n_evictions = 2
        data = pool.to_dict()
        restored = SeedPool.from_dict(data)
        assert restored.base_seed == 42
        assert restored.max_size == 50
        assert restored.alpha == 0.8
        assert restored.cvar_percentile == 25
        assert restored.seeds == seeds_before
        assert restored.n_evictions == 2
        assert restored._next_index == pool._next_index

    def test_round_trip_json_compatible(self) -> None:
        import json

        pool = SeedPool(base_seed=0, max_size=10)
        pool.add_seeds(generation=0)
        for s in pool.seeds:
            pool.difficulty[s] = 1.0
        data = pool.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = SeedPool.from_dict(restored_data)
        assert restored.seeds == pool.seeds

    def test_from_dict_with_excluded_seeds(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)
        data = pool.to_dict()
        restored = SeedPool.from_dict(data, excluded_seeds={999})
        assert 999 in restored.excluded_seeds


class TestSeedPoolEvaluation:
    """Tests for pool-based population evaluation."""

    def test_evaluate_population_calls_evaluator(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1, 2]
        pool.generation_added = {0: 0, 1: 0, 2: 0}

        population = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64)

        def evaluator(chrom: npt.NDArray[np.float64], seed: int) -> float:
            return float(seed) + float(chrom[0])

        fitness = pool.evaluate_population(population, evaluator)
        assert fitness.shape == (2,)
        assert fitness[0] == pytest.approx(2.0)  # costs=[1,2,3], mean=2.0
        assert fitness[1] == pytest.approx(1.0)  # costs=[0,1,2], mean=1.0

    def test_evaluate_population_updates_difficulty(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1]
        pool.generation_added = {0: 0, 1: 0}

        population = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float64)

        def evaluator(chrom: npt.NDArray[np.float64], seed: int) -> float:
            return float(seed) * 10.0

        pool.evaluate_population(population, evaluator)
        assert pool.difficulty[0] == pytest.approx(0.0)
        assert pool.difficulty[1] == pytest.approx(10.0)

    def test_evaluate_population_with_batch_evaluator(self) -> None:
        """Batch evaluator is used when provided."""
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1, 2]
        pool.generation_added = {0: 0, 1: 0, 2: 0}

        population = np.array([[1, 0], [0, 1]], dtype=np.float64)

        scalar_called = False

        def scalar_eval(chrom: npt.NDArray[np.float64], seed: int) -> float:
            nonlocal scalar_called
            scalar_called = True
            return 0.0

        def batch_eval(chrom: npt.NDArray[np.float64], seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([float(s) + float(chrom[0]) for s in seeds])

        fitness = pool.evaluate_population(population, scalar_eval, batch_evaluator=batch_eval)
        assert not scalar_called  # batch should be used instead
        assert fitness.shape == (2,)


class TestAdaptiveSeedIntegration:
    """Integration test: adaptive seed pool with stress tests in a GA loop."""

    def test_pool_grows_evicts_and_stress_tests(self) -> None:
        """Verify pool grows, evicts hardest-first, and stress tests inject hard seeds."""
        pool = SeedPool(base_seed=0, max_size=8, alpha=0.7, cvar_percentile=20)

        rng = np.random.default_rng(42)
        pop = rng.random((4, 10))

        def evaluator(chrom: npt.NDArray[np.float64], seed: int) -> float:
            quality = float(np.sum(chrom)) / len(chrom)
            return float(seed % 1000) * 10.0 + quality * 5.0

        stress_ran = False
        for gen in range(10):
            pool.add_seeds(gen)
            fitness = pool.evaluate_population(pop, evaluator)

            assert fitness.shape == (4,)
            assert all(np.isfinite(fitness))

            pool.evict_redundant()
            assert len(pool.seeds) <= 8

            # Run stress test every 5 generations
            if (gen + 1) % 5 == 0:

                def stress_eval(seeds: list[int]) -> npt.NDArray[np.float64]:
                    return np.array([float(s % 1000) * 10.0 for s in seeds])

                metrics = pool.stress_test(gen, stress_eval, n_probes=20, n_inject=3)
                assert metrics["n_injected"] <= 3
                assert metrics["n_probes"] == 20
                stress_ran = True

        assert stress_ran
        assert len(pool.seeds) <= 8
        assert pool.n_evictions > 0
        assert len(pool.difficulty) == len(pool.seeds)

        # Verify difficulty spectrum has spread (gap-closure eviction)
        difficulties = sorted(pool.difficulty.values())
        assert difficulties[-1] > difficulties[0]


class TestStressSeedHash:
    def test_stress_seeds_differ_from_pool_seeds(self) -> None:
        assert _pool_seed(42, 0) != _stress_seed(42, 0, 0)

    def test_stress_seeds_vary_by_generation(self) -> None:
        assert _stress_seed(42, 0, 0) != _stress_seed(42, 1, 0)

    def test_stress_seeds_within_range(self) -> None:
        for i in range(100):
            assert 0 <= _stress_seed(42, 5, i) < 2**31


class TestStressTest:
    def test_injects_worst_seeds(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)
        initial_size = len(pool.seeds)

        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([float(s) for s in seeds])

        metrics = pool.stress_test(generation=5, evaluator=evaluator, n_probes=20, n_inject=5)
        assert len(pool.seeds) == initial_size + 5
        assert metrics["n_probes"] == 20
        assert metrics["n_injected"] == 5
        assert "worst_cost" in metrics
        assert "median_cost" in metrics
        assert "capture_rate" in metrics

    def test_injected_seeds_eviction_preserves_spectrum(self) -> None:
        """After stress injection, gap-closure preserves difficulty coverage."""
        pool = SeedPool(base_seed=42, max_size=8)
        pool.add_seeds(generation=0)
        # Give initial seeds spread-out difficulties
        for i, s in enumerate(pool.seeds):
            pool.difficulty[s] = float(i) * 10.0

        # Inject hard seeds with distinct difficulties so they aren't all redundant
        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([500.0 + i * 50.0 for i in range(len(seeds))])

        pool.stress_test(generation=5, evaluator=evaluator, n_probes=10, n_inject=5)
        pool.evict_redundant()
        assert len(pool.seeds) == 8
        # Spectrum should span from easy (initial) to hard (injected)
        diffs = [pool.difficulty[s] for s in pool.seeds]
        assert min(diffs) < 50.0  # some easy seeds survive
        assert max(diffs) > 400.0  # some hard seeds survive

    def test_stress_test_capture_rate(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)

        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([1.0 if i % 2 == 0 else 50000.0 for i in range(len(seeds))])

        metrics = pool.stress_test(generation=5, evaluator=evaluator, n_probes=20, n_inject=5)
        assert 0.0 <= metrics["capture_rate"] <= 1.0


class TestReservedSeeds:
    """Tests for make_reserved_seeds and seed separation guarantees."""

    def test_deterministic(self) -> None:
        a = make_reserved_seeds(42, 100, 50)
        b = make_reserved_seeds(42, 100, 50)
        assert a == b

    def test_different_offsets_disjoint(self) -> None:
        val = set(make_reserved_seeds(42, VALIDATION_SEED_OFFSET, 1000))
        final = set(make_reserved_seeds(42, FINAL_EVAL_SEED_OFFSET, 1000))
        assert len(val & final) == 0

    @pytest.mark.parametrize("base_seed", [0, 1, 42, 999, 2**20])
    def test_disjoint_across_base_seeds(self, base_seed: int) -> None:
        val = set(make_reserved_seeds(base_seed, VALIDATION_SEED_OFFSET, 1000))
        final = set(make_reserved_seeds(base_seed, FINAL_EVAL_SEED_OFFSET, 1000))
        assert len(val & final) == 0

    def test_prefix_stable(self) -> None:
        """First N seeds of a larger request match a request of size N."""
        small = make_reserved_seeds(42, VALIDATION_SEED_OFFSET, 100)
        large = make_reserved_seeds(42, VALIDATION_SEED_OFFSET, 1000)
        assert small == large[:100]
