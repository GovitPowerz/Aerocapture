"""Curated-CDF adaptive seed framework.

Maintains a fixed-size training seed list, refreshed on trigger by
stratified-random sampling from the cost CDF of the current top-K
individuals. See
``docs/superpowers/specs/2026-04-14-curated-cdf-seed-framework-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    trim_fraction: float = 0.0
    bucket_selection: str = "random"

    def _stratified_pick(
        self,
        seeds: list[int],
        costs: npt.NDArray[np.float64],
    ) -> list[int]:
        """Sort seeds by cost, split into n_bins equal-count bins, pick one per bin.

        Non-finite costs are replaced with a large sentinel so their seeds sort
        to the tail bin.
        """
        arr = np.asarray(costs, dtype=np.float64)
        sentinel = np.finfo(np.float64).max / 2
        arr = np.where(np.isfinite(arr), arr, sentinel)
        order = np.argsort(arr, kind="stable")
        sorted_seeds = [seeds[i] for i in order]

        # Trim the non-discriminative extremes before binning: the easiest seeds
        # carry no between-individual signal, the hardest are un-improvable
        # dispersion outliers that destabilize the (moving) objective.
        if self.trim_fraction > 0.0:
            cut = int(len(sorted_seeds) * self.trim_fraction)
            sorted_seeds = sorted_seeds[cut : len(sorted_seeds) - cut]

        if self.n_bins > len(sorted_seeds):
            msg = f"n_bins ({self.n_bins}) must be <= len(seeds) after trim ({len(sorted_seeds)})"
            raise ValueError(msg)
        bins = np.array_split(sorted_seeds, self.n_bins)
        # Representative per cost-quantile bin: random (default) or a deterministic
        # difficulty within the bin (min=easiest, max=hardest, middle=median).
        if self.bucket_selection == "random":
            return [int(self.rng.choice(b)) for b in bins]
        if self.bucket_selection == "min":
            return [int(b[0]) for b in bins]
        if self.bucket_selection == "max":
            return [int(b[-1]) for b in bins]
        if self.bucket_selection == "middle":
            return [int(b[len(b) // 2]) for b in bins]
        msg = f"unknown bucket_selection={self.bucket_selection!r} (expected 'random', 'min', 'max', or 'middle')"
        raise ValueError(msg)

    def _draw_sample_seeds(self) -> list[int]:
        """Draw `sample_size` fresh random seeds disjoint from `excluded_seeds`."""
        drawn: list[int] = []
        while len(drawn) < self.sample_size:
            batch = self.rng.integers(0, 2**31, size=self.sample_size - len(drawn)).tolist()
            drawn.extend(s for s in batch if s not in self.excluded_seeds)
        return drawn[: self.sample_size]

    def curate(
        self,
        problem: Any,  # AerocaptureProblem-like (duck-typed for testability)
        top_k_X: npt.NDArray[np.float64],
    ) -> list[int]:
        """Run K individuals on `sample_size` seeds, pick `n_bins` via stratified random.

        Updates ``self.seed_list`` and returns the new list.
        """
        sample_seeds = self._draw_sample_seeds()
        costs_per_ind = [problem.evaluate_individual_per_seed(top_k_X[i], sample_seeds) for i in range(top_k_X.shape[0])]
        avg_cost = np.mean(np.stack(costs_per_ind, axis=0), axis=0)
        new_seeds = self._stratified_pick(sample_seeds, avg_cost)
        self.seed_list = new_seeds
        return new_seeds

    def to_dict(self) -> dict:
        return {
            "sample_size": self.sample_size,
            "n_bins": self.n_bins,
            "seed_list": self.seed_list,
            "last_curation_gen": self.last_curation_gen,
            "trim_fraction": self.trim_fraction,
            "bucket_selection": self.bucket_selection,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        excluded_seeds: set[int],
        rng: np.random.Generator,
    ) -> SeedCurator:
        return cls(
            sample_size=int(d["sample_size"]),
            n_bins=int(d["n_bins"]),
            excluded_seeds=excluded_seeds,
            rng=rng,
            seed_list=list(d["seed_list"]) if d.get("seed_list") is not None else None,
            last_curation_gen=int(d.get("last_curation_gen", -1)),
            trim_fraction=float(d.get("trim_fraction", 0.0)),
            bucket_selection=str(d.get("bucket_selection", "random")),
        )
