"""Curated-CDF adaptive seed framework.

Maintains a fixed-size training seed list, refreshed on trigger by
stratified-random sampling from the cost CDF of the current top-K
individuals. See
``docs/superpowers/specs/2026-04-14-curated-cdf-seed-framework-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SeedCurator:
    """Picks a representative subset of seeds from a larger probe pool.

    ``sample_size`` seeds are drawn fresh each curation; ``n_bins`` seeds are
    picked, one per equal-count cost-quantile bin, via the injected ``rng``.
    """

    sample_size: int
    n_bins: int
    excluded_seeds: set[int]
    rng: np.random.Generator
    seed_list: list[int] | None = None
    last_curation_gen: int = -1

    def _stratified_pick(
        self,
        seeds: list[int],
        costs: npt.NDArray[np.float64],
    ) -> list[int]:
        """Sort seeds by cost, split into n_bins equal-count bins, pick one per bin.

        Non-finite costs are replaced with a large sentinel so their seeds sort
        to the tail bin.
        """
        if self.n_bins > len(seeds):
            msg = f"n_bins ({self.n_bins}) must be <= len(seeds) ({len(seeds)})"
            raise ValueError(msg)
        arr = np.asarray(costs, dtype=np.float64)
        sentinel = np.finfo(np.float64).max / 2
        arr = np.where(np.isfinite(arr), arr, sentinel)
        order = np.argsort(arr, kind="stable")
        sorted_seeds = [seeds[i] for i in order]

        bins = np.array_split(sorted_seeds, self.n_bins)
        return [int(self.rng.choice(b)) for b in bins]
