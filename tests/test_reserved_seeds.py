"""Tests for make_reserved_seeds and seed separation guarantees."""

from __future__ import annotations

import pytest
from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, VALIDATION_SEED_OFFSET, make_reserved_seeds


class TestReservedSeeds:
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
