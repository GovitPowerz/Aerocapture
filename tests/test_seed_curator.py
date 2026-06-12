"""Tests for SeedCurator -- curated-CDF adaptive seed framework."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.seed_curator import SeedCurator


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestStratifiedPick:
    """Tests for the pure stratified-random selection logic."""

    def test_returns_exactly_n_bins_seeds(self) -> None:
        curator = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0))
        seeds = list(range(1_000, 1_100))
        costs = np.linspace(0.0, 1.0, 100)
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 10

    def test_deterministic_with_same_rng(self) -> None:
        seeds = list(range(1_000, 1_100))
        costs = np.linspace(0.0, 1.0, 100)
        a = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(42))._stratified_pick(seeds, costs)
        b = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(42))._stratified_pick(seeds, costs)
        assert a == b

    def test_different_rng_differs(self) -> None:
        """Different RNGs produce different picks for at least one bin."""
        # Use many bins over many seeds so independent RNG draws MUST diverge on at
        # least one bin -- probability of full collision across 50 independent
        # uniform-over-20 choices is (1/20)^50, vanishingly small.
        seeds = list(range(1_000, 2_000))
        costs = np.linspace(0.0, 1.0, 1000)
        a = SeedCurator(sample_size=1000, n_bins=50, excluded_seeds=set(), rng=_rng(1))._stratified_pick(seeds, costs)
        b = SeedCurator(sample_size=1000, n_bins=50, excluded_seeds=set(), rng=_rng(2))._stratified_pick(seeds, costs)
        assert any(x != y for x, y in zip(a, b, strict=True))

    def test_one_pick_per_quantile_bin(self) -> None:
        """With 100 seeds sorted by cost and 10 bins, picks come from each decile."""
        seeds = list(range(1_000, 1_100))
        costs = np.linspace(0.0, 1.0, 100)
        curator = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0))
        picked = curator._stratified_pick(seeds, costs)
        # Map each picked seed back to its cost, assert one per decile
        seed_to_cost = dict(zip(seeds, costs, strict=True))
        picked_costs = sorted(seed_to_cost[s] for s in picked)
        for i, c in enumerate(picked_costs):
            assert i / 10 <= c < (i + 1) / 10 or (i == 9 and c == 1.0)

    def test_trim_fraction_excludes_extremes(self) -> None:
        """trim_fraction drops the lowest/highest-cost seeds before binning."""
        seeds = list(range(100))
        costs = np.arange(100, dtype=float)  # cost == seed, already sorted ascending
        curator = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0), trim_fraction=0.2)
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 10
        # central 60% kept -> nothing from the trimmed 20% tails (cost < 20 or >= 80)
        assert all(20 <= s < 80 for s in picked)

    def test_trim_fraction_zero_is_legacy(self) -> None:
        """trim_fraction=0.0 (default) keeps the full range incl. extremes."""
        seeds = list(range(100))
        costs = np.arange(100, dtype=float)
        picked = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0), trim_fraction=0.0)._stratified_pick(seeds, costs)
        # bin 0 picks from cost [0,10), bin 9 from [90,100) -- extremes reachable
        assert min(picked) < 10 and max(picked) >= 90

    def test_bucket_selection_max_picks_hardest_per_bin(self) -> None:
        """bucket_selection='max' deterministically picks the highest-cost seed in each bin."""
        seeds = list(range(100))
        costs = np.arange(100, dtype=float)  # cost == seed
        picked = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0), bucket_selection="max")._stratified_pick(seeds, costs)
        assert sorted(picked) == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    def test_bucket_selection_min_picks_easiest_per_bin(self) -> None:
        seeds = list(range(100))
        costs = np.arange(100, dtype=float)
        picked = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0), bucket_selection="min")._stratified_pick(seeds, costs)
        assert sorted(picked) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_bucket_selection_middle_picks_median_per_bin(self) -> None:
        seeds = list(range(100))
        costs = np.arange(100, dtype=float)
        picked = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0), bucket_selection="middle")._stratified_pick(seeds, costs)
        assert sorted(picked) == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    def test_non_finite_seeds_never_picked(self) -> None:
        """Un-simulatable (NaN/inf cost) probe seeds are dropped before binning.

        The old sort-to-tail sentinel made bucket_selection='max' pick a
        non-finite seed DETERMINISTICALLY (b[-1] of the tail bin), poisoning
        every individual's training RMS with an inf contribution."""
        seeds = list(range(1_000, 1_020))
        bad = set(seeds[:3])
        costs = np.array([float("nan"), float("inf"), float("nan")] + list(np.linspace(0.1, 1.0, 17)))
        for mode in ("random", "min", "max", "middle"):
            curator = SeedCurator(sample_size=20, n_bins=4, excluded_seeds=set(), rng=_rng(0), bucket_selection=mode)
            picked = curator._stratified_pick(seeds, costs)
            assert len(picked) == 4, mode
            assert not (set(picked) & bad), f"{mode}: picked a non-finite seed"

    def test_uneven_bin_sizes(self) -> None:
        """1000 / 30 = 33.3 -- bins must accept uneven splits."""
        seeds = list(range(1_000, 2_000))
        costs = np.linspace(0.0, 1.0, 1000)
        curator = SeedCurator(sample_size=1000, n_bins=30, excluded_seeds=set(), rng=_rng(0))
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 30

    def test_raises_when_n_bins_exceeds_seeds(self) -> None:
        curator = SeedCurator(sample_size=100, n_bins=20, excluded_seeds=set(), rng=_rng(0))
        with pytest.raises(ValueError, match="n_bins .* must be <=.*"):
            curator._stratified_pick([1, 2, 3, 4, 5], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


class TestCurate:
    """Tests for the end-to-end curate() method with a fake problem."""

    class _FakeProblem:
        """Stand-in for AerocaptureProblem: returns deterministic per-seed costs."""

        def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
            # Cost is seed-dependent: higher seed -> higher cost, plus small x offset.
            return np.array([float(s) + 0.01 * float(x[0]) for s in seeds])

    def test_returns_n_bins_seeds_disjoint_from_excluded(self) -> None:
        excluded = {1, 2, 3, 42, 999}
        curator = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=excluded, rng=_rng(7))
        top_k_X = np.random.default_rng(0).random((5, 4))
        new_seeds = curator.curate(self._FakeProblem(), top_k_X)
        assert len(new_seeds) == 10
        assert not (set(new_seeds) & excluded)
        assert curator.seed_list == new_seeds

    def test_deterministic_same_rng_and_inputs(self) -> None:
        top_k_X = np.random.default_rng(0).random((5, 4))
        a = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0)).curate(self._FakeProblem(), top_k_X)
        b = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(0)).curate(self._FakeProblem(), top_k_X)
        assert a == b

    def test_k_min_one(self) -> None:
        """K=1 (only one individual) still works: averaging over 1 is identity."""
        top_k_X = np.random.default_rng(0).random((1, 4))
        curator = SeedCurator(sample_size=50, n_bins=5, excluded_seeds=set(), rng=_rng(0))
        new_seeds = curator.curate(self._FakeProblem(), top_k_X)
        assert len(new_seeds) == 5


class TestCheckpointRoundtrip:
    def test_to_dict_from_dict_preserves_state(self) -> None:
        a = SeedCurator(sample_size=100, n_bins=10, excluded_seeds={1, 2}, rng=_rng(0))
        a.seed_list = [10, 20, 30]
        a.last_curation_gen = 42
        d = a.to_dict()
        b = SeedCurator.from_dict(d, excluded_seeds={1, 2}, rng=_rng(0))
        assert b.sample_size == a.sample_size
        assert b.n_bins == a.n_bins
        assert b.seed_list == a.seed_list
        assert b.last_curation_gen == a.last_curation_gen
        assert b.excluded_seeds == {1, 2}

    def test_from_dict_with_empty_state(self) -> None:
        d = {"sample_size": 100, "n_bins": 10, "seed_list": None, "last_curation_gen": -1}
        c = SeedCurator.from_dict(d, excluded_seeds=set(), rng=_rng(0))
        assert c.seed_list is None
        assert c.last_curation_gen == -1


class TestRestoreOnResume:
    def test_toml_knobs_override_checkpoint_state(self) -> None:
        """Resume keeps curator STATE (seed_list, last_curation_gen) from the
        checkpoint but config knobs from the TOML -- a legacy checkpoint must not
        silently reset trim_fraction/bucket_selection to 0.0/'random'."""
        from aerocapture.training.train import _restore_seed_curator

        configured = SeedCurator(
            sample_size=500,
            n_bins=10,
            excluded_seeds={7},
            rng=_rng(0),
            trim_fraction=0.2,
            bucket_selection="middle",
        )
        legacy_state = {"sample_size": 1000, "n_bins": 20, "seed_list": [1, 2, 3], "last_curation_gen": 17}
        restored = _restore_seed_curator(legacy_state, configured, verbose=False)
        assert restored.seed_list == [1, 2, 3]
        assert restored.last_curation_gen == 17
        assert restored.trim_fraction == 0.2
        assert restored.bucket_selection == "middle"
        assert restored.sample_size == 500
        assert restored.n_bins == 10
        assert restored.excluded_seeds == {7}
