"""Tests for the seed_strategy dispatch layer in train.py."""

from __future__ import annotations

import numpy as np
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
