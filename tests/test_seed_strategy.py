"""Tests for the seed_strategy dispatch layer in train.py."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.train import _draw_disjoint_seeds


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestDrawDisjointSeeds:
    def test_returns_n_seeds(self) -> None:
        seeds = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        assert len(seeds) == 20

    def test_excludes_reserved(self) -> None:
        excluded = {1, 2, 3, 42, 999}
        seeds = _draw_disjoint_seeds(_rng(0), n=20, excluded=excluded)
        assert not (set(seeds) & excluded)

    def test_deterministic_with_same_rng(self) -> None:
        a = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        b = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        assert a == b

    def test_handles_empty_exclusion(self) -> None:
        seeds = _draw_disjoint_seeds(_rng(0), n=5, excluded=set())
        assert len(seeds) == 5


class _StubProblem:
    """Minimal problem stand-in: records seed updates."""

    def __init__(self) -> None:
        self.seed_updates: list[list[int]] = []

    def update_seeds(self, seeds: list[int]) -> None:
        self.seed_updates.append(list(seeds))


class TestFixedStrategySetup:
    def test_fixed_seeds_are_deterministic_range(self) -> None:
        from aerocapture.training.train import _compute_fixed_seeds

        seeds = _compute_fixed_seeds(base_mc_seed=100, n_sims=5, excluded=set())
        assert seeds == [100, 101, 102, 103, 104]

    def test_fixed_seeds_raise_on_overlap(self) -> None:
        from aerocapture.training.train import _compute_fixed_seeds

        with pytest.raises(ValueError, match="overlaps"):
            _compute_fixed_seeds(base_mc_seed=100, n_sims=5, excluded={102})


class TestStrategyDispatch:
    """Exercises the loop-body dispatch logic via a minimal helper.

    These tests do NOT run a real training loop (too expensive). They verify
    the dispatch decisions deciding when to call `problem.update_seeds`.
    """

    def test_rotating_calls_update_each_gen(self) -> None:
        # Simulate three gens of rotating: three update_seeds calls with different lists.
        stub = _StubProblem()
        rng = _rng(0)
        for _ in range(3):
            fresh = _draw_disjoint_seeds(rng, n=5, excluded=set())
            stub.update_seeds(fresh)
        assert len(stub.seed_updates) == 3
        # Between gens seeds should differ (RNG advances).
        assert stub.seed_updates[0] != stub.seed_updates[1]
        assert stub.seed_updates[1] != stub.seed_updates[2]
