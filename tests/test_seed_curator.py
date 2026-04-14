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

    def test_non_finite_costs_sort_to_tail(self) -> None:
        """NaN-cost seeds land in the highest-cost bin, not randomly distributed."""
        # First 5 seeds (indices 0-4) have NaN costs; remaining 15 have ascending finite costs.
        seeds = list(range(1_000, 1_020))
        nan_seeds = set(seeds[:5])
        costs = np.array([float("nan")] * 5 + list(np.linspace(0.1, 1.0, 15)))
        # With 20 seeds and 4 bins, np.array_split yields bin sizes [5, 5, 5, 5].
        # All 5 NaN seeds land in the last bin (sentinel sorts to the tail),
        # so the pick from the last bin is guaranteed to be one of them.
        curator = SeedCurator(sample_size=20, n_bins=4, excluded_seeds=set(), rng=_rng(0))
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 4
        assert picked[-1] in nan_seeds

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
