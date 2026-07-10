"""Confirmatory sizing-pool seed generation (R4/R5 revision).

The confirmatory pools must be disjoint BY CONSTRUCTION from every scenario the
project ever touched: all historical seeds (reserved pools, training draws,
curation probes) come from ``default_rng(...).integers(0, 2**31, ...)``, so the
confirmatory seeds live in [2**31, 2**32).
"""

import numpy as np
from aerocapture.training.evaluate import make_confirmatory_pools, make_reserved_seeds


def test_pools_shape_unique_and_range():
    pools = make_confirmatory_pools(42, n_replicates=3, n=500)
    assert len(pools) == 3
    assert all(len(p) == 500 for p in pools)
    flat = np.concatenate(pools)
    assert len(np.unique(flat)) == len(flat)  # unique within AND across replicates
    assert flat.min() >= 2**31
    assert flat.max() < 2**32


def test_pools_deterministic():
    assert make_confirmatory_pools(42, 2, 100) == make_confirmatory_pools(42, 2, 100)


def test_pools_vary_with_base_seed():
    assert make_confirmatory_pools(42, 1, 100) != make_confirmatory_pools(43, 1, 100)


def test_disjoint_from_reserved_streams():
    flat = set(np.concatenate(make_confirmatory_pools(42, 2, 1000)).tolist())
    for offset in (1_000_000, 2_000_000, 8_000_000, 9_000_000):
        assert flat.isdisjoint(make_reserved_seeds(42, offset, 1000))
