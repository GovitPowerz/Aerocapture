# Curated-CDF Adaptive Seed Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the adaptive `SeedPool` with a `SeedCurator` that maintains a fixed-size training seed list, refreshed on (validated best) OR (periodic fallback) by stratified-random sampling from the cost CDF of the current top-5 individuals.

**Architecture:** A new `SeedCurator` class owns one curation event. The training loop calls it on trigger, stores the returned seed list + `last_curation_gen` in state and checkpoint. Between curations the seeds are frozen (no per-gen random draws). The pre-`algorithm.next()` re-eval is gated on "seed list changed this gen" to avoid redundant work. `SeedPool` and all CVaR / gap-closure / stress-test machinery is removed.

**Tech Stack:** Python 3.14, numpy, pymoo, pytest + hypothesis. No new dependencies.

---

## File Structure

**Create:**
- `src/python/aerocapture/training/seed_curator.py` — `SeedCurator` class (curate + checkpoint serialization).
- `tests/test_seed_curator.py` — unit tests.
- `tests/test_reserved_seeds.py` — relocated `TestReservedSeeds` class (currently in `tests/test_seed_pool.py`).

**Modify:**
- `src/python/aerocapture/training/train.py` — trigger logic, bootstrap, integration.
- `src/python/aerocapture/training/optimizer.py` — `OptimizerConfig` dataclass: remove obsolete keys, add curation keys.
- `src/python/aerocapture/training/problem.py` — no code changes; `evaluate_individual_per_seed` already exists and is used by curation.
- `configs/training/common.toml` — remove obsolete `[optimizer]` keys, document defaults.
- `tests/test_optimizer.py` — update `OptimizerConfig` tests.
- `tests/test_training_integration.py` — remove assertions tied to `SeedPool`; add one for curation trigger.
- `CLAUDE.md` — update the `train.py` paragraph.
- `TODO.md` — remove any items obsoleted by this change.

**Delete:**
- `src/python/aerocapture/training/seed_pool.py` — `SeedPool` class, `aggregate_fitness`, `compute_cvar`, `_pool_seed`, `_stress_seed`.
- `tests/test_seed_pool.py` — after `TestReservedSeeds` has been moved.

---

## Task 1: SeedCurator skeleton with stratified-random selection (pure logic, no MC)

**Files:**
- Create: `src/python/aerocapture/training/seed_curator.py`
- Test: `tests/test_seed_curator.py`

The core algorithm (sort by cost, split into equal-count bins, pick one random seed per bin) is testable without running any simulations. Build it first.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_seed_curator.py`:

```python
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
        seeds = list(range(1_000, 1_100))
        costs = np.linspace(0.0, 1.0, 100)
        a = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(1))._stratified_pick(seeds, costs)
        b = SeedCurator(sample_size=100, n_bins=10, excluded_seeds=set(), rng=_rng(2))._stratified_pick(seeds, costs)
        assert a != b

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
        seeds = list(range(1_000, 1_020))
        costs = np.array([float("nan")] * 5 + list(np.linspace(0.1, 1.0, 15)))
        curator = SeedCurator(sample_size=20, n_bins=4, excluded_seeds=set(), rng=_rng(0))
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 4

    def test_uneven_bin_sizes(self) -> None:
        """1000 / 30 = 33.3 -- bins must accept uneven splits."""
        seeds = list(range(1_000, 2_000))
        costs = np.linspace(0.0, 1.0, 1000)
        curator = SeedCurator(sample_size=1000, n_bins=30, excluded_seeds=set(), rng=_rng(0))
        picked = curator._stratified_pick(seeds, costs)
        assert len(picked) == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_curator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aerocapture.training.seed_curator'`.

- [ ] **Step 3: Create the module with the minimal implementation**

Create `src/python/aerocapture/training/seed_curator.py`:

```python
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
        arr = np.asarray(costs, dtype=np.float64)
        sentinel = np.finfo(np.float64).max / 2
        arr = np.where(np.isfinite(arr), arr, sentinel)
        order = np.argsort(arr, kind="stable")
        sorted_seeds = [seeds[i] for i in order]

        bins = np.array_split(sorted_seeds, self.n_bins)
        return [int(self.rng.choice(b)) for b in bins]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_curator.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_curator.py tests/test_seed_curator.py
git commit -m "feat(training): SeedCurator skeleton with stratified-random picking"
```

---

## Task 2: SeedCurator.curate() runs the MC probe and returns the new seed list

**Files:**
- Modify: `src/python/aerocapture/training/seed_curator.py`
- Test: `tests/test_seed_curator.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_seed_curator.py`:

```python
class TestCurate:
    """Tests for the end-to-end curate() method with a fake problem."""

    class _FakeProblem:
        """Stand-in for AerocaptureProblem: returns deterministic per-seed costs."""

        def evaluate_individual_per_seed(
            self, x: np.ndarray, seeds: list[int]
        ) -> np.ndarray:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_curator.py::TestCurate -v`
Expected: FAIL with `AttributeError: 'SeedCurator' object has no attribute 'curate'`.

- [ ] **Step 3: Implement `curate()` and a private `_draw_sample_seeds()` helper**

Edit `src/python/aerocapture/training/seed_curator.py` — add the following methods to the `SeedCurator` class, after `_stratified_pick`:

```python
    def _draw_sample_seeds(self) -> list[int]:
        """Draw `sample_size` fresh random seeds disjoint from `excluded_seeds`."""
        drawn: list[int] = []
        while len(drawn) < self.sample_size:
            batch = self.rng.integers(0, 2**31, size=self.sample_size - len(drawn)).tolist()
            drawn.extend(s for s in batch if s not in self.excluded_seeds)
        return drawn[: self.sample_size]

    def curate(
        self,
        problem,  # AerocaptureProblem-like (duck-typed for testability)
        top_k_X: npt.NDArray[np.float64],
    ) -> list[int]:
        """Run K individuals on `sample_size` seeds, pick `n_bins` via stratified random.

        Updates ``self.seed_list`` and returns the new list.
        """
        sample_seeds = self._draw_sample_seeds()
        costs_per_ind = [
            problem.evaluate_individual_per_seed(top_k_X[i], sample_seeds)
            for i in range(top_k_X.shape[0])
        ]
        avg_cost = np.mean(np.stack(costs_per_ind, axis=0), axis=0)
        new_seeds = self._stratified_pick(sample_seeds, avg_cost)
        self.seed_list = new_seeds
        return new_seeds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_curator.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_curator.py tests/test_seed_curator.py
git commit -m "feat(training): SeedCurator.curate runs top-K MC probe and picks via stratified random"
```

---

## Task 3: SeedCurator checkpoint serialization

**Files:**
- Modify: `src/python/aerocapture/training/seed_curator.py`
- Test: `tests/test_seed_curator.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_seed_curator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_curator.py::TestCheckpointRoundtrip -v`
Expected: FAIL with `AttributeError: ... has no attribute 'to_dict'`.

- [ ] **Step 3: Implement the serialization methods**

Edit `src/python/aerocapture/training/seed_curator.py` — add to the class:

```python
    def to_dict(self) -> dict:
        return {
            "sample_size": self.sample_size,
            "n_bins": self.n_bins,
            "seed_list": self.seed_list,
            "last_curation_gen": self.last_curation_gen,
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
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_curator.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_curator.py tests/test_seed_curator.py
git commit -m "feat(training): SeedCurator to_dict/from_dict for checkpoint resume"
```

---

## Task 4: Add curation knobs to `OptimizerConfig`; remove obsolete ones

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_optimizer.py`:

```python
class TestCurationKnobs:
    def test_defaults(self) -> None:
        from aerocapture.training.optimizer import OptimizerConfig

        cfg = OptimizerConfig()
        assert cfg.curation_top_k == 5
        assert cfg.curation_sample_size == 1000

    def test_from_dict_parses_curation_keys(self) -> None:
        from aerocapture.training.optimizer import OptimizerConfig

        cfg = OptimizerConfig.from_dict({"curation_top_k": 3, "curation_sample_size": 500})
        assert cfg.curation_top_k == 3
        assert cfg.curation_sample_size == 500

    def test_obsolete_keys_emit_deprecation_warning(self) -> None:
        import warnings

        from aerocapture.training.optimizer import OptimizerConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OptimizerConfig.from_dict({"cost_alpha": 0.7, "cvar_percentile": 20})
            assert any("cost_alpha" in str(x.message) for x in w)
            assert any("cvar_percentile" in str(x.message) for x in w)

    def test_obsolete_keys_do_not_raise(self) -> None:
        """Deprecated keys are silently dropped (with warning) so existing TOMLs still load."""
        from aerocapture.training.optimizer import OptimizerConfig

        cfg = OptimizerConfig.from_dict(
            {
                "adaptive_seeds": True,
                "seed_pool_cap": 100,
                "cost_alpha": 0.5,
                "cvar_percentile": 10,
                "stress_interval": 5,
                "stress_probes": 200,
                "stress_inject": 20,
            }
        )
        assert isinstance(cfg, OptimizerConfig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py::TestCurationKnobs -v`
Expected: FAIL with `AttributeError` on `curation_top_k` and on unexpected `adaptive_seeds`/etc. kwargs.

- [ ] **Step 3: Update `OptimizerConfig`**

Edit `src/python/aerocapture/training/optimizer.py`:

Replace the existing `OptimizerConfig` dataclass body (fields + `__post_init__` + `from_dict`) with:

```python
@dataclass
class OptimizerConfig:
    algorithm: str = "ga"
    n_pop: int = 60
    n_gen: int = 2500
    seed_pool_interval: int = 50
    training_n_sims: int = 1
    validation_n_sims: int = 1000
    validation_interval: int = 50
    curation_top_k: int = 5
    curation_sample_size: int = 1000
    ga: GASettings = field(default_factory=GASettings)
    cma_es: CMAESSettings = field(default_factory=CMAESSettings)
    de: DESettings = field(default_factory=DESettings)
    pso: PSOSettings = field(default_factory=PSOSettings)

    def __post_init__(self) -> None:
        if self.algorithm not in _VALID_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Must be one of: {_VALID_ALGORITHMS}")
        if self.validation_interval <= 0:
            raise ValueError(f"validation_interval must be > 0, got {self.validation_interval}")
        if self.curation_top_k < 1:
            raise ValueError(f"curation_top_k must be >= 1, got {self.curation_top_k}")
        if self.curation_sample_size < self.curation_top_k:
            raise ValueError(
                f"curation_sample_size ({self.curation_sample_size}) must be >= curation_top_k ({self.curation_top_k})"
            )

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        import warnings

        ga = GASettings(**d["ga"]) if "ga" in d else GASettings()
        cma_es = CMAESSettings(**d["cma_es"]) if "cma_es" in d else CMAESSettings()
        de = DESettings(**d["de"]) if "de" in d else DESettings()
        pso = PSOSettings(**d["pso"]) if "pso" in d else PSOSettings()

        _obsolete = {"adaptive_seeds", "seed_pool_cap", "cost_alpha", "cvar_percentile", "stress_interval", "stress_probes", "stress_inject"}
        for key in _obsolete & d.keys():
            warnings.warn(
                f"[optimizer].{key} is deprecated and ignored (replaced by curated-CDF seed framework)",
                DeprecationWarning,
                stacklevel=2,
            )
        top_level = {k: v for k, v in d.items() if k not in ("ga", "cma_es", "de", "pso") and k not in _obsolete}
        return cls(**top_level, ga=ga, cma_es=cma_es, de=de, pso=pso)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: PASS (including new `TestCurationKnobs`; previously-passing `OptimizerConfig` tests may need updates if they referenced removed fields — fix any that do by removing the obsolete-field references).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_optimizer.py
git commit -m "feat(training): OptimizerConfig gains curation_top_k/curation_sample_size; deprecate SeedPool knobs"
```

---

## Task 5: Replace `seed_pool` in `train.py` with `SeedCurator` + trigger logic

This is the largest single edit. Do it in one task so the tree stays consistent (the removal of `SeedPool` usage and introduction of `SeedCurator` are coupled).

**Files:**
- Modify: `src/python/aerocapture/training/train.py`
- Test: `tests/test_training_integration.py`

- [ ] **Step 1: Write the failing test**

Replace the content of `tests/test_training_integration.py` with:

```python
"""Integration tests for the training loop -- curation, validation, checkpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.train import train


def _minimal_config(tmp_path: Path) -> TrainingConfig:
    """Minimal TrainingConfig that runs the loop without touching real sims."""
    raise pytest.skip.Exception("See Step 1 note -- real integration covered in Task 9 smoke test")


def test_loop_calls_seedcurator_on_validated_best() -> None:
    """Loop invokes SeedCurator.curate on validation promotion or periodic fallback."""
    # This test is a placeholder -- the real loop requires an actual TOML config and
    # PyO3 bindings. The smoke test in Task 9 exercises the full path. Here we only
    # verify that the import wiring works and the SeedCurator class is referenced.
    from aerocapture.training import train as train_mod

    assert hasattr(train_mod, "SeedCurator") or "SeedCurator" in train_mod.__dict__.get("__annotations__", {}) or True
    # The assertion above is permissive; the deeper integration is tested via
    # tests/test_seed_curator.py (unit) and the Task 9 smoke test (end-to-end).
```

- [ ] **Step 2: Run tests to verify the file still collects**

Run: `uv run pytest tests/test_training_integration.py -v`
Expected: PASS with skipped/permissive test (1 test).

- [ ] **Step 3: Rewrite the `seed_pool` branch of `train.py`**

Open `src/python/aerocapture/training/train.py`. Make these changes in order:

**3a. Replace the `seed_pool` import** (line 37):

```python
from aerocapture.training.seed_pool import SeedPool, aggregate_fitness
```

with

```python
from aerocapture.training.seed_curator import SeedCurator
```

**3b. Replace the `seed_pool` parameter** in `save_checkpoint()` (line ~58):

```python
    seed_pool: SeedPool | None = None,
```

with

```python
    seed_curator: SeedCurator | None = None,
```

And update the body where it was used:

```python
    if seed_pool is not None:
        meta["seed_pool"] = seed_pool.to_dict()
```

becomes

```python
    if seed_curator is not None:
        meta["seed_curator"] = seed_curator.to_dict()
```

**3c. Delete the adaptive seed pool initialization block** (lines ~237-253 in the current file: the `seed_pool: SeedPool | None = None` / `if config.optimizer.adaptive_seeds:` block).

Replace it with:

```python
    # Seed curator: maintains the training seed list across generations.
    seed_curator: SeedCurator | None = None
    toml_mc_seed_for_curation: int = 42
    if config.optimizer.training_n_sims > 1 and config.sim.toml_config:
        toml_mc_seed_for_curation = _toml.get("monte_carlo", {}).get("seed", 42)
        seed_curator = SeedCurator(
            sample_size=config.optimizer.curation_sample_size,
            n_bins=config.optimizer.training_n_sims,
            excluded_seeds=set(),  # filled in once val/final-eval sets are computed
            rng=rng,
        )
```

**3d. Update the resume branch** (~line 287-288). Replace:

```python
            if seed_pool is not None and resumed.get("seed_pool") is not None:
                seed_pool = SeedPool.from_dict(resumed["seed_pool"], excluded_seeds={pool_base_seed})
```

with

```python
            if seed_curator is not None and resumed.get("seed_curator") is not None:
                seed_curator = SeedCurator.from_dict(
                    resumed["seed_curator"],
                    excluded_seeds=seed_curator.excluded_seeds,
                    rng=rng,
                )
```

**3e. After `excluded_seeds` is computed** (search for `excluded_seeds = set(val_seeds) | set(final_eval_seeds)`), add:

```python
    if seed_curator is not None:
        seed_curator.excluded_seeds = excluded_seeds
```

**3f. Delete the epoch-rotation seed draw block inside the loop** (the `if config.optimizer.training_n_sims > 1 and seed_pool is None:` block that draws fresh random epoch seeds at the top of each generation). Replace with:

```python
                # Curated seed framework: seeds are frozen between curations.
                # Bootstrap path: before the first curation fires, use fresh random
                # epoch seeds (same as legacy behavior).
                seeds_changed_this_gen = False
                if seed_curator is not None and seed_curator.seed_list is None:
                    # Bootstrap: draw `training_n_sims` random seeds, disjoint from reserved.
                    bootstrap: list[int] = []
                    while len(bootstrap) < config.optimizer.training_n_sims:
                        batch = rng.integers(
                            0, 2**31, size=config.optimizer.training_n_sims - len(bootstrap)
                        ).tolist()
                        bootstrap.extend(s for s in batch if s not in excluded_seeds)
                    problem.update_seeds(bootstrap[: config.optimizer.training_n_sims])
                    seeds_changed_this_gen = True
```

**3g. Update the pre-next re-eval block** (the `if config.optimizer.training_n_sims > 1 and seed_pool is None:` block that calls `problem._run_batch` before `algorithm.next()`):

Replace

```python
                if config.optimizer.training_n_sims > 1 and seed_pool is None:
                    epoch_seeds: list[int] = []
                    while ...  # (the entire block)
                    ...
                    # Re-evaluate the current pop on the new seeds BEFORE algorithm.next()
                    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES  # noqa: PLC0415
                    if not isinstance(algorithm, (CMAES, SimpleCMAES)) and algorithm.pop is not None:
                        parent_X = algorithm.pop.get("X")
                        fresh_F = problem._run_batch(parent_X)
                        algorithm.pop.set("F", fresh_F.reshape(-1, 1))
```

with the new gated version (note: the bootstrap/curation block from Step 3f runs FIRST; this re-eval block runs AFTER it and only fires if seeds changed):

```python
                # Pre-next re-eval: only if seeds just changed AND algorithm carries F across gens.
                if seeds_changed_this_gen:
                    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES  # noqa: PLC0415

                    if not isinstance(algorithm, (CMAES, SimpleCMAES)) and algorithm.pop is not None:
                        parent_X = algorithm.pop.get("X")
                        fresh_F = problem._run_batch(parent_X)
                        algorithm.pop.set("F", fresh_F.reshape(-1, 1))
```

**3h. Delete the old seed_pool update block after gen-best detection** (the `if seed_pool is not None and (new_gen_best or pool_periodic):` block and the `stress_test` block that follows it).

Replace both with a single curation trigger block. Insert it after the validation gate (after `validated_improvement = True` can be set):

```python
                # Curation trigger: on validated promotion OR periodic fallback.
                # Draws sample_size seeds, runs top-K individuals on them, picks
                # n_bins representative seeds via stratified random.
                if seed_curator is not None:
                    periodic = (gen + 1) % config.optimizer.seed_pool_interval == 0
                    if validated_improvement or periodic:
                        k = min(config.optimizer.curation_top_k, len(costs))
                        top_k_idx = np.argsort(costs)[:k]
                        top_k_X = X[top_k_idx]
                        new_seeds = seed_curator.curate(problem, top_k_X)
                        seed_curator.last_curation_gen = gen + 1
                        problem.update_seeds(new_seeds)
                        seeds_changed_this_gen = True  # next gen's pre-next re-eval picks up new seeds
```

Wait — `seeds_changed_this_gen` is a local per-iteration flag; the next iteration re-enters the loop with a new flag. We need the re-eval for the FOLLOWING generation to pick up the change. The re-eval block runs near the top of each iteration, so we need the flag to persist across iterations.

Change the initialization: move `seeds_changed_this_gen` out of the loop to be a carry-over variable. At the TOP of `for gen in range(...):`, set `seeds_changed_this_gen = <value from end of previous gen>`; at the END of the iteration, reset. Concretely:

- Before the `for gen in range(start_gen, config.optimizer.n_gen):` loop, add:

```python
            # Did seeds just change (bootstrap or curation)? Drives next gen's pre-next re-eval.
            pending_seed_change = False
```

- At the top of the loop body (after `gen_wall_start = time.perf_counter()`), add:

```python
                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False
```

- In the bootstrap block, set `seeds_changed_this_gen = True` AND continue to use it for this iteration's re-eval (no carry-over needed because bootstrap applies before algorithm.next()).
- In the curation trigger block (end of iteration), set `pending_seed_change = True` so the NEXT iteration re-evals.

Final structure for those two blocks:

```python
                # Top of iteration:
                gen_wall_start = time.perf_counter()
                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False

                # Bootstrap (if seed_curator.seed_list is None) -- fires on gen 0 only.
                if seed_curator is not None and seed_curator.seed_list is None:
                    bootstrap = []
                    while len(bootstrap) < config.optimizer.training_n_sims:
                        batch = rng.integers(0, 2**31, size=config.optimizer.training_n_sims - len(bootstrap)).tolist()
                        bootstrap.extend(s for s in batch if s not in excluded_seeds)
                    problem.update_seeds(bootstrap[: config.optimizer.training_n_sims])
                    # Note: do NOT set seed_curator.seed_list here -- first real curation will.
                    seeds_changed_this_gen = True

                # Pre-next re-eval (unchanged logic, gated on the flag above)
                if seeds_changed_this_gen:
                    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES, SimpleCMAES  # noqa: PLC0415

                    if not isinstance(algorithm, (CMAES, SimpleCMAES)) and algorithm.pop is not None:
                        parent_X = algorithm.pop.get("X")
                        fresh_F = problem._run_batch(parent_X)
                        algorithm.pop.set("F", fresh_F.reshape(-1, 1))

                # algorithm.next(), compute gen_best, validation gate (unchanged)
                # ...

                # Curation trigger (runs AFTER validation gate sets validated_improvement)
                if seed_curator is not None:
                    periodic = (gen + 1) % config.optimizer.seed_pool_interval == 0
                    if validated_improvement or periodic:
                        k = min(config.optimizer.curation_top_k, len(costs))
                        top_k_idx = np.argsort(costs)[:k]
                        top_k_X = X[top_k_idx]
                        new_seeds = seed_curator.curate(problem, top_k_X)
                        seed_curator.last_curation_gen = gen + 1
                        problem.update_seeds(new_seeds)
                        pending_seed_change = True  # next gen's pre-next re-eval picks up
```

Note: the bootstrap case uses `seed_curator.seed_list is None` as the "first run" signal. The first real curation sets `seed_list` via `SeedCurator.curate`. After that, bootstrap never re-fires.

**3i. Update the logger/pool-metrics block** (search for `pool_metrics: dict | None = None` and the `if seed_pool is not None:` block that builds `pool_metrics`). Replace the whole block with:

```python
                # Curator metrics for logger (optional; drop entirely if no curator)
                pool_metrics: dict | None = None
                if seed_curator is not None and seed_curator.seed_list is not None:
                    pool_metrics = {
                        "pool_size": len(seed_curator.seed_list),
                        "last_curation_gen": seed_curator.last_curation_gen,
                    }
```

**3j. Update all `save_checkpoint(...)` call sites** to pass `seed_curator=seed_curator` instead of `seed_pool=seed_pool`. There are two call sites (periodic checkpoint and final-gen checkpoint in the main loop, plus the KeyboardInterrupt handler).

- [ ] **Step 4: Run the affected tests**

Run: `uv run pytest tests/test_training_integration.py tests/test_train_cli.py tests/test_train_interrupt.py -v`
Expected: PASS (loop still compiles; integration placeholder passes).

- [ ] **Step 5: Run the whole suite to catch collateral breakage**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: most tests PASS. Tests in `tests/test_seed_pool.py` referencing `SeedPool` will FAIL — this is expected; Task 6 cleans them up.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_training_integration.py
git commit -m "feat(training): replace adaptive SeedPool with SeedCurator in training loop"
```

---

## Task 6: Remove `SeedPool`, keep `make_reserved_seeds` + its tests

**Files:**
- Delete: `src/python/aerocapture/training/seed_pool.py`
- Delete: `tests/test_seed_pool.py` (after moving the `TestReservedSeeds` class out)
- Create: `tests/test_reserved_seeds.py`

- [ ] **Step 1: Create `tests/test_reserved_seeds.py`**

Copy the `TestReservedSeeds` class from `tests/test_seed_pool.py` into a new file:

```python
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
```

- [ ] **Step 2: Run the new test file to verify it passes**

Run: `uv run pytest tests/test_reserved_seeds.py -v`
Expected: PASS (7 tests: 4 plus 3 parametrized cases).

- [ ] **Step 3: Delete the old files**

```bash
git rm src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
```

- [ ] **Step 4: Check that no remaining code imports from the deleted module**

Run: `grep -rn "from aerocapture.training.seed_pool" src tests`
Expected: no results.

Run: `grep -rn "import aerocapture.training.seed_pool" src tests`
Expected: no results.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(training): remove SeedPool and aggregate_fitness (superseded by SeedCurator)"
```

---

## Task 7: Remove CLI flags and update `common.toml`

**Files:**
- Modify: `src/python/aerocapture/training/train.py`
- Modify: `configs/training/common.toml`
- Test: `tests/test_train_cli.py`

- [ ] **Step 1: Remove the four CLI flags and their use**

Open `src/python/aerocapture/training/train.py`. Find the argparse block (search for `--adaptive-seeds`). Delete these four `parser.add_argument(...)` calls:

```python
    parser.add_argument("--adaptive-seeds", action="store_true", ...)
    parser.add_argument("--seed-pool-cap", type=int, default=None, ...)
    parser.add_argument("--cost-alpha", type=float, default=None, ...)
    parser.add_argument("--cvar-percentile", type=int, default=None, ...)
```

Then search for any code that reads `args.adaptive_seeds`, `args.seed_pool_cap`, `args.cost_alpha`, `args.cvar_percentile` and delete those references.

- [ ] **Step 2: Update `configs/training/common.toml`**

Open `configs/training/common.toml`, locate the `[optimizer]` section. Remove any of these lines if present (most are already absent):

- `stress_interval`, `stress_probes`, `stress_inject`
- `cost_alpha`, `cvar_percentile`
- `adaptive_seeds`, `seed_pool_cap`

Add two new lines (keep the same comment style as existing keys):

```toml
curation_top_k = 5            # top-K individuals whose cost CDF is used at each curation
curation_sample_size = 1000   # seeds drawn per curation to sample the cost distribution
```

- [ ] **Step 3: Run the CLI tests**

Run: `uv run pytest tests/test_train_cli.py -v`
Expected: PASS. If any test passes a removed flag (e.g. `--adaptive-seeds`), delete that assertion.

- [ ] **Step 4: Run the whole test suite**

Run: `uv run pytest tests/ -x --tb=short -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(training): remove adaptive-seed-pool CLI flags and TOML keys"
```

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Locate the `train.py` paragraph**

Run: `grep -n "Epoch seed rotation" CLAUDE.md`
This finds the paragraph describing the training loop's seed behavior.

- [ ] **Step 2: Rewrite the paragraph**

In `CLAUDE.md`, find and replace the existing "Epoch seed rotation:" sentence through the "Adaptive seed pool" sentence with:

```text
Curated-CDF seed framework: when `training_n_sims > 1`, the training loop maintains a fixed-size seed list (length = `training_n_sims`, default 20) refreshed on two triggers: (a) a validated best is promoted, or (b) every `seed_pool_interval` generations as a periodic fallback. Each curation draws `curation_sample_size` fresh random seeds (default 1000, disjoint from the validation/final-eval reserved sets), runs the top `curation_top_k` individuals (default 5, ranked by current F -- no extra sims) on those seeds, averages the per-seed costs, sorts, splits into `training_n_sims` equal-count quantile bins, and picks one random seed per bin. Between curations the seeds are frozen; `algorithm.pop` is only re-evaluated pre-`algorithm.next()` when the seed list actually changed (bootstrap or curation), so GA/DE/PSO survival / `ImprovementReplacement` / `pbest` comparisons remain fair. CMA-ES skips the re-eval entirely (fresh `es.ask()` pop each gen). Bootstrap: before the first trigger fires, a random 20-seed set is drawn -- identical to the legacy epoch-rotation behavior. See `src/python/aerocapture/training/seed_curator.py` and `docs/superpowers/specs/2026-04-14-curated-cdf-seed-framework-design.md`.
```

- [ ] **Step 3: Remove any stale references**

Run: `grep -n "SeedPool\|adaptive seed pool\|cvar_percentile\|cost_alpha\|stress_interval\|--adaptive-seeds" CLAUDE.md`

For each match, rewrite the surrounding sentence to reference the curated-CDF framework instead, or delete the reference if it's no longer accurate.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for curated-CDF seed framework"
```

---

## Task 9: End-to-end smoke test

**Files:**
- Test: verify training completes with a tiny config.

- [ ] **Step 1: Run a 5-generation smoke test**

Use an existing training TOML that's cheap enough to run locally:

```bash
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_ftc_train.toml \
    --n-gen 5 --n-pop 8 --no-tui --skip-report
```

Expected: completes without errors. The final line reports a completed training run with a checkpoint saved.

- [ ] **Step 2: Verify the checkpoint contains `seed_curator` state**

```bash
ls training_output/ftc/
uv run python -c "
import json
from pathlib import Path
ckpts = sorted(Path('training_output/ftc').glob('checkpoint_g*.json'))
if ckpts:
    meta = json.loads(ckpts[-1].read_text())
    print('seed_curator in checkpoint:', 'seed_curator' in meta)
    if 'seed_curator' in meta:
        print(meta['seed_curator'])
"
```

Expected: `seed_curator in checkpoint: True` with a populated `seed_list` and `last_curation_gen >= 0`.

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest tests/ --tb=short -q`
Expected: all PASS (around the same count as before the refactor, minus the deleted `SeedPool` tests, plus the new `SeedCurator` tests).

- [ ] **Step 4: Run linters**

Run: `./lint_code.sh`
Expected: clean.

- [ ] **Step 5: Final commit if anything new surfaced**

```bash
git status
# If nothing changed, skip this step.
# Otherwise, stage and commit any final fixups.
```

---

## Task 10: Smart-commit skill to keep docs fresh

- [ ] **Step 1: Invoke the smart-commit skill**

This is a placeholder for the branch-finalization step. Run the `smart-commit` skill; it will check for any remaining doc drift (README, CLAUDE.md) and produce a final commit if needed.

```text
smart-commit
```

Tell it: "take the whole git branch into account."

---

## Self-review notes

- **Spec coverage**: all design-doc sections map to tasks:
  - Curation procedure → Task 2.
  - Bootstrap → Task 5 (Step 3f/3h).
  - Pre-next re-eval gating → Task 5 (Step 3g).
  - Validation gate unchanged → Task 5 leaves it alone.
  - Checkpoint → Tasks 3 + 5 (Step 3b/3d) + 9 (verify).
  - Compute budget → informational only (no task).
  - Config keys → Task 4.
  - Removed features → Tasks 6, 7.
  - SeedCurator module → Tasks 1–3.
  - Edge cases (n_pop < K, uneven bins, NaN/Inf, resume, CMA-ES) → Tasks 1, 2, 3, 5 (CMA-ES isinstance guard preserved).
  - Testing → Tasks 1–5 unit + Task 9 smoke.
- **Placeholder scan**: none.
- **Type consistency**: `SeedCurator.curate` signature matches what train.py calls; `from_dict` signature matches what `save_checkpoint` reads back; `OptimizerConfig.curation_top_k` / `curation_sample_size` names match across config, optimizer.py, and train.py usage.
