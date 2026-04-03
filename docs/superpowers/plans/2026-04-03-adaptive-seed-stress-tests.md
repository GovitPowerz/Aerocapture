# Adaptive Seed Pool: Stress Tests + Keep-Hardest Eviction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the adaptive seed pool a faithful proxy for the full MC distribution by adding hash-based seed generation, keep-hardest eviction, and periodic stress tests that inject hard seeds.

**Architecture:** Modify `SeedPool` in `seed_pool.py` with three changes: (1) hash-based `_pool_seed()` replaces consecutive integers, (2) keep-hardest eviction replaces gap-closure, (3) new `stress_test()` method probes fresh seeds and injects the worst. Wire stress tests into `train.py`'s adaptive seed loop with 3 new CLI args.

**Tech Stack:** Python (numpy, hashlib), pytest

**Spec:** `docs/superpowers/specs/2026-04-03-adaptive-seed-stress-tests-design.md`

---

### Task 1: Hash-based seed generation with exclusion

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py:60-86`
- Test: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing tests for hash-based seed generation**

Add to `tests/test_seed_pool.py`:

```python
from aerocapture.training.seed_pool import _pool_seed


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
        s_a = _pool_seed(42, 0)
        s_b = _pool_seed(99, 0)
        assert s_a != s_b

    def test_deterministic(self) -> None:
        assert _pool_seed(42, 7) == _pool_seed(42, 7)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py::TestPoolSeedHash -v`

Expected: FAIL with `ImportError: cannot import name '_pool_seed'`

- [ ] **Step 3: Implement `_pool_seed()` function**

Add to `src/python/aerocapture/training/seed_pool.py`, after the imports (before `compute_cvar`):

```python
import hashlib


def _pool_seed(base: int, index: int) -> int:
    """Generate a well-spread seed from (base, index) using SHA-256.

    Uses the "pool" namespace to ensure structural separation from
    the final MC eval's seed and stress test seeds.
    """
    h = hashlib.sha256(f"{base}:pool:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)
```

Note: `hashlib` is a stdlib module, already available.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py::TestPoolSeedHash -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Write failing tests for excluded seeds and updated `__init__`/`add_seeds`**

Add to `tests/test_seed_pool.py`:

```python
class TestSeedPoolExclusion:
    def test_excluded_seed_never_in_pool(self) -> None:
        """If a hash collision produces the excluded seed, it's skipped."""
        # Generate a pool seed and use it as the excluded seed to force the skip path
        excluded = _pool_seed(42, 0)
        pool = SeedPool(base_seed=42, max_size=50, excluded_seeds={excluded})
        pool.add_seeds(generation=0)
        assert excluded not in pool.seeds
        # Pool should still have 5 seeds (the collision was replaced)
        assert len(pool.seeds) == 5

    def test_pool_uses_hash_seeds_not_consecutive(self) -> None:
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)
        # Seeds should NOT be [42, 43, 44, 45, 46] anymore
        assert pool.seeds != [42, 43, 44, 45, 46]
        assert len(pool.seeds) == 5
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py::TestSeedPoolExclusion -v`

Expected: FAIL (SeedPool doesn't accept `excluded_seeds` yet).

- [ ] **Step 7: Update `SeedPool.__init__` and `add_seeds` to use hash-based generation with exclusion**

In `src/python/aerocapture/training/seed_pool.py`, replace `__init__` and `add_seeds`:

```python
    def __init__(
        self,
        base_seed: int,
        max_size: int = 100,
        alpha: float = 0.7,
        cvar_percentile: int = 20,
        excluded_seeds: set[int] | None = None,
    ) -> None:
        self.base_seed = base_seed
        self.max_size = max_size
        self.alpha = alpha
        self.cvar_percentile = cvar_percentile
        self.excluded_seeds: set[int] = excluded_seeds or set()
        self.seeds: list[int] = []
        self.difficulty: dict[int, float] = {}
        self.generation_added: dict[int, int] = {}
        self.n_evictions: int = 0
        self._next_index: int = 0  # tracks next hash index for seed generation

    def add_seeds(self, generation: int) -> None:
        """Add seeds to the pool for a given generation.

        Generation 0 bootstraps 5 seeds; subsequent generations add 1 per call.
        Seeds are generated via hash-based spread. Excluded seeds are skipped.
        """
        n_to_add = 5 if generation == 0 and self._next_index == 0 else 1
        added = 0
        while added < n_to_add:
            seed = _pool_seed(self.base_seed, self._next_index)
            self._next_index += 1
            if seed in self.excluded_seeds or seed in self.generation_added:
                continue
            self.seeds.append(seed)
            self.generation_added[seed] = generation
            added += 1
```

- [ ] **Step 8: Update `to_dict` and `from_dict` for new fields**

In `src/python/aerocapture/training/seed_pool.py`, update `to_dict`:

```python
    def to_dict(self) -> dict[str, Any]:
        """Serialize the pool to a JSON-compatible dict for checkpointing."""
        return {
            "base_seed": self.base_seed,
            "max_size": self.max_size,
            "alpha": self.alpha,
            "cvar_percentile": self.cvar_percentile,
            "seeds": self.seeds,
            "difficulty": {str(k): v for k, v in self.difficulty.items()},
            "generation_added": {str(k): v for k, v in self.generation_added.items()},
            "n_evictions": self.n_evictions,
            "next_index": self._next_index,
        }
```

Update `from_dict` to accept `excluded_seeds` and restore `_next_index`:

```python
    @classmethod
    def from_dict(cls, data: dict[str, Any], excluded_seeds: set[int] | None = None) -> SeedPool:
        """Restore a SeedPool from a checkpointed dict."""
        pool = cls(
            base_seed=int(data["base_seed"]),
            max_size=int(data["max_size"]),
            alpha=float(data.get("alpha", 0.7)),
            cvar_percentile=int(data.get("cvar_percentile", 20)),
            excluded_seeds=excluded_seeds,
        )
        pool.seeds = list(data["seeds"])
        pool.difficulty = {int(k): float(v) for k, v in dict(data["difficulty"]).items()}
        pool.generation_added = {int(k): int(v) for k, v in dict(data["generation_added"]).items()}
        pool.n_evictions = int(data.get("n_evictions", 0))
        pool._next_index = int(data.get("next_index", 0))
        return pool
```

- [ ] **Step 9: Fix existing tests that depend on consecutive-integer seeds**

Update `tests/test_seed_pool.py`:

In `TestSeedPoolGrowth`:

```python
class TestSeedPoolGrowth:
    def test_bootstrap_creates_5_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        # Seeds are hash-based now, not consecutive
        assert all(0 <= s < 2**31 for s in pool.seeds)
        assert len(set(pool.seeds)) == 5  # all unique

    def test_incremental_growth(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        pool.add_seeds(generation=1)
        assert len(pool.seeds) == 6

    def test_no_duplicate_seeds(self) -> None:
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
```

In `TestSeedPoolCheckpoint`, update both tests to not assert specific seed values (since they're hash-based now). The round-trip tests should construct the pool and populate seeds via `add_seeds` rather than setting `pool.seeds` directly, OR just keep them as-is since `from_dict` restores whatever seeds were saved:

```python
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
```

- [ ] **Step 10: Run all seed pool tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py -v`

Expected: All tests pass.

- [ ] **Step 11: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: hash-based seed generation with exclusion for adaptive seed pool"
```

---

### Task 2: Keep-hardest eviction

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py:102-122`
- Test: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing test for keep-hardest eviction**

Add to `tests/test_seed_pool.py`:

```python
class TestKeepHardestEviction:
    def test_easiest_seeds_evicted(self) -> None:
        """When over cap, the easiest seeds (lowest difficulty) are dropped."""
        pool = SeedPool(base_seed=0, max_size=3)
        pool.seeds = [10, 20, 30, 40, 50]
        pool.difficulty = {10: 100.0, 20: 1.0, 30: 50.0, 40: 2.0, 50: 75.0}
        pool.generation_added = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}
        pool.evict_redundant()
        assert len(pool.seeds) == 3
        # Hardest 3: seed 10 (100.0), seed 50 (75.0), seed 30 (50.0)
        assert set(pool.seeds) == {10, 50, 30}
        # Easiest 2 evicted: seed 20 (1.0), seed 40 (2.0)
        assert 20 not in pool.seeds
        assert 40 not in pool.seeds

    def test_hardest_seeds_always_survive(self) -> None:
        """Seeds with highest difficulty are never evicted."""
        pool = SeedPool(base_seed=0, max_size=2)
        pool.seeds = [1, 2, 3, 4, 5]
        pool.difficulty = {1: 1000.0, 2: 0.1, 3: 0.2, 4: 999.0, 5: 0.3}
        pool.generation_added = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        pool.evict_redundant()
        assert set(pool.seeds) == {1, 4}  # the two hardest

    def test_no_eviction_under_cap(self) -> None:
        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [1, 2, 3]
        pool.difficulty = {1: 1.0, 2: 2.0, 3: 3.0}
        pool.generation_added = {1: 0, 2: 1, 3: 2}
        pool.evict_redundant()
        assert len(pool.seeds) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py::TestKeepHardestEviction -v`

Expected: `test_easiest_seeds_evicted` and `test_hardest_seeds_always_survive` FAIL (old eviction logic gives different results).

- [ ] **Step 3: Replace `evict_redundant()` with keep-hardest strategy**

In `src/python/aerocapture/training/seed_pool.py`, replace the `evict_redundant` method:

```python
    def evict_redundant(self) -> None:
        """Evict seeds until pool size <= max_size.

        Keep-hardest strategy: drop the seed with the lowest difficulty
        score (easiest for the best individual). Hard seeds survive
        unconditionally.
        """
        while len(self.seeds) > self.max_size:
            easiest = min(self.seeds, key=lambda s: self.difficulty.get(s, 0.0))
            self.seeds.remove(easiest)
            del self.difficulty[easiest]
            del self.generation_added[easiest]
            self.n_evictions += 1
```

- [ ] **Step 4: Remove old eviction tests, run all tests**

In `tests/test_seed_pool.py`, remove the old `TestSeedPoolEviction` class entirely (the `TestKeepHardestEviction` class replaces it).

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: keep-hardest eviction replaces gap-closure in seed pool"
```

---

### Task 3: Stress test method on SeedPool

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py`
- Test: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing tests for `_stress_seed` and `stress_test`**

Add to `tests/test_seed_pool.py`:

```python
from aerocapture.training.seed_pool import _pool_seed, _stress_seed


class TestStressSeedHash:
    def test_stress_seeds_differ_from_pool_seeds(self) -> None:
        """Stress seeds use a different namespace, so no collisions with pool seeds."""
        pool_s = _pool_seed(42, 0)
        stress_s = _stress_seed(42, 0, 0)
        assert pool_s != stress_s

    def test_stress_seeds_vary_by_generation(self) -> None:
        s_gen0 = _stress_seed(42, 0, 0)
        s_gen1 = _stress_seed(42, 1, 0)
        assert s_gen0 != s_gen1

    def test_stress_seeds_within_range(self) -> None:
        for i in range(100):
            s = _stress_seed(42, 5, i)
            assert 0 <= s < 2**31


class TestStressTest:
    def test_injects_worst_seeds(self) -> None:
        """Stress test injects the N worst seeds into the pool."""
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)
        initial_size = len(pool.seeds)

        # Mock evaluator: cost = seed value (higher seed = harder)
        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([float(s) for s in seeds])

        metrics = pool.stress_test(generation=5, evaluator=evaluator, n_probes=20, n_inject=5)

        assert len(pool.seeds) == initial_size + 5
        assert metrics["n_probes"] == 20
        assert metrics["n_injected"] == 5
        assert "worst_cost" in metrics
        assert "median_cost" in metrics
        assert "capture_rate" in metrics

    def test_injected_seeds_survive_eviction(self) -> None:
        """Injected stress seeds are hard, so they survive keep-hardest eviction."""
        pool = SeedPool(base_seed=42, max_size=8)
        pool.add_seeds(generation=0)  # 5 seeds

        # Set low difficulty for existing seeds
        for s in pool.seeds:
            pool.difficulty[s] = 1.0

        # Evaluator returns very high cost for stress seeds
        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([10000.0] * len(seeds))

        pool.stress_test(generation=5, evaluator=evaluator, n_probes=10, n_inject=5)
        # Now pool has 10 seeds, max_size=8, so 2 should be evicted
        # The 2 easiest (from original 5 with difficulty=1.0) should go
        pool.evict_redundant()
        assert len(pool.seeds) == 8
        # All surviving seeds should have high difficulty (stress-injected ones)
        # or be the 3 remaining original seeds
        difficulties = [pool.difficulty[s] for s in pool.seeds]
        assert sum(d >= 10000.0 for d in difficulties) == 5  # all 5 injected survive

    def test_stress_test_metrics_capture_rate(self) -> None:
        """Capture rate is fraction of probes with cost below a threshold."""
        pool = SeedPool(base_seed=42, max_size=50)
        pool.add_seeds(generation=0)

        # Mock: half the probes "crash" (high cost), half succeed (low cost)
        def evaluator(seeds: list[int]) -> npt.NDArray[np.float64]:
            return np.array([1.0 if i % 2 == 0 else 50000.0 for i in range(len(seeds))])

        metrics = pool.stress_test(generation=5, evaluator=evaluator, n_probes=20, n_inject=5)
        # capture_rate should reflect the fraction with cost < crash threshold
        assert 0.0 <= metrics["capture_rate"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py::TestStressSeedHash tests/test_seed_pool.py::TestStressTest -v`

Expected: FAIL with `ImportError: cannot import name '_stress_seed'`

- [ ] **Step 3: Implement `_stress_seed()` and `stress_test()` method**

Add `_stress_seed` function after `_pool_seed` in `seed_pool.py`:

```python
def _stress_seed(base: int, generation: int, index: int) -> int:
    """Generate a stress-test seed from a separate hash namespace.

    Uses "stress" namespace to avoid collisions with pool seeds and
    the final MC eval seed.
    """
    h = hashlib.sha256(f"{base}:stress:{generation}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)
```

Add `stress_test` method to `SeedPool` class (after `evaluate_population`):

```python
    def stress_test(
        self,
        generation: int,
        evaluator: Callable[[list[int]], npt.NDArray[np.float64]],
        n_probes: int = 200,
        n_inject: int = 20,
    ) -> dict[str, Any]:
        """Probe fresh seeds and inject the hardest into the pool.

        Generates n_probes seeds from a separate hash namespace, evaluates
        the best individual on all of them, and injects the worst n_inject
        into the pool. Returns metrics including an estimated capture rate.

        Args:
            generation: Current generation number (used for seed namespace).
            evaluator: Callable(seeds) -> costs array for the best individual.
            n_probes: Number of fresh seeds to probe.
            n_inject: Number of worst seeds to inject into the pool.

        Returns:
            Dict with keys: n_probes, n_injected, worst_cost, median_cost, capture_rate.
        """
        # Generate probe seeds from stress namespace
        probe_seeds = []
        idx = 0
        while len(probe_seeds) < n_probes:
            s = _stress_seed(self.base_seed, generation, idx)
            idx += 1
            if s not in self.excluded_seeds and s not in self.generation_added:
                probe_seeds.append(s)

        # Evaluate best individual on all probe seeds
        costs = evaluator(probe_seeds)

        # Capture rate: fraction with DV < 10000 m/s (crash/hyperbolic threshold)
        capture_rate = float(np.mean(costs < 10000.0))

        # Sort by cost descending, take worst n_inject
        worst_indices = np.argsort(costs)[::-1][:n_inject]
        n_injected = 0
        for i in worst_indices:
            s = probe_seeds[i]
            if s not in self.generation_added:
                self.seeds.append(s)
                self.difficulty[s] = float(costs[i])
                self.generation_added[s] = generation
                n_injected += 1

        self.evict_redundant()

        return {
            "n_probes": n_probes,
            "n_injected": n_injected,
            "worst_cost": float(np.max(costs)),
            "median_cost": float(np.median(costs)),
            "capture_rate": capture_rate,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: add stress_test() method to adaptive seed pool"
```

---

### Task 4: Wire stress tests into train.py

**Files:**
- Modify: `src/python/aerocapture/training/train.py:300-315` (pool init)
- Modify: `src/python/aerocapture/training/train.py:505-528` (adaptive seed loop)
- Modify: `src/python/aerocapture/training/train.py:663-674` (pool metrics logging)
- Modify: `src/python/aerocapture/training/train.py:780-801` (CLI args)
- Modify: `src/python/aerocapture/training/train.py:336-337` (checkpoint restore)

- [ ] **Step 1: Add 3 new CLI arguments**

In `src/python/aerocapture/training/train.py`, after the `--cvar-percentile` argument (line 787), add:

```python
    parser.add_argument("--stress-interval", type=int, default=5, help="Run stress test every N generations (default: 5, only with --adaptive-seeds)")
    parser.add_argument("--stress-probes", type=int, default=200, help="Number of fresh seeds to probe per stress test (default: 200)")
    parser.add_argument("--stress-inject", type=int, default=20, help="Number of worst seeds to inject from each stress test (default: 20)")
```

After the existing `cfg.ga.cvar_percentile` assignment (around line 801), add:

```python
    cfg.ga.stress_interval = args.stress_interval
    cfg.ga.stress_probes = args.stress_probes
    cfg.ga.stress_inject = args.stress_inject
```

- [ ] **Step 2: Add fields to GAConfig dataclass**

In `src/python/aerocapture/training/config.py`, in the `GAConfig` dataclass (line 57), add after `cvar_percentile` (line 75):

```python
    stress_interval: int = 5
    stress_probes: int = 200
    stress_inject: int = 20
```

- [ ] **Step 3: Pass `excluded_seeds` to SeedPool init**

In `src/python/aerocapture/training/train.py`, at the pool initialization (around line 310), change:

```python
        seed_pool = SeedPool(
            base_seed=pool_base_seed,
            max_size=config.ga.seed_pool_cap,
            alpha=config.ga.cost_alpha,
            cvar_percentile=config.ga.cvar_percentile,
        )
```

to:

```python
        seed_pool = SeedPool(
            base_seed=pool_base_seed,
            max_size=config.ga.seed_pool_cap,
            alpha=config.ga.cost_alpha,
            cvar_percentile=config.ga.cvar_percentile,
            excluded_seeds={pool_base_seed},
        )
```

- [ ] **Step 4: Pass `excluded_seeds` on checkpoint restore**

At the checkpoint restore (around line 336-337), change:

```python
            if seed_pool is not None and resumed.get("seed_pool") is not None:
                seed_pool = SeedPool.from_dict(resumed["seed_pool"])
```

to:

```python
            if seed_pool is not None and resumed.get("seed_pool") is not None:
                seed_pool = SeedPool.from_dict(resumed["seed_pool"], excluded_seeds={pool_base_seed})
```

- [ ] **Step 5: Build stress test evaluator and wire into the adaptive seed loop**

In the adaptive seed loop section (after `seed_pool.evict_redundant()` around line 522, but before the migration block), add the stress test call. Find the block:

```python
                        seed_pool.evict_redundant()

                        n_pop = len(pop)
```

and insert between them:

```python
                    # Stress test: probe fresh seeds and inject hardest
                    stress_metrics: dict | None = None
                    if (
                        seed_pool is not None
                        and (gen + 1) % config.ga.stress_interval == 0
                        and _batch_evaluator is not None
                        and best_overall_chrom is not None
                    ):
                        def _stress_eval(seeds: list[int]) -> npt.NDArray[np.float64]:
                            return _batch_evaluator(best_overall_chrom, seeds)

                        stress_metrics = seed_pool.stress_test(
                            generation=gen,
                            evaluator=_stress_eval,
                            n_probes=config.ga.stress_probes,
                            n_inject=config.ga.stress_inject,
                        )

```

Note: This must go AFTER the per-subpop loop ends (after all subpops are processed) but BEFORE the common logging path. The `best_overall_chrom` is already updated at this point. Place it right after the migration block for adaptive mode (line 544) and before the common path comment at line 654.

- [ ] **Step 6: Add stress test metrics to pool logging**

In the pool metrics block (around line 664-674), change:

```python
                    pool_metrics: dict | None = None
                    if seed_pool is not None:
                        d_min, d_max = seed_pool.difficulty_range
                        difficulty_scores = sorted(seed_pool.difficulty.values())
                        pool_metrics = {
                            "pool_size": len(seed_pool.seeds),
                            "difficulty_min": d_min,
                            "difficulty_max": d_max,
                            "n_evictions": seed_pool.n_evictions,
                            "difficulty_scores": difficulty_scores,
                        }
```

to:

```python
                    pool_metrics: dict | None = None
                    if seed_pool is not None:
                        d_min, d_max = seed_pool.difficulty_range
                        difficulty_scores = sorted(seed_pool.difficulty.values())
                        pool_metrics = {
                            "pool_size": len(seed_pool.seeds),
                            "difficulty_min": d_min,
                            "difficulty_max": d_max,
                            "n_evictions": seed_pool.n_evictions,
                            "difficulty_scores": difficulty_scores,
                        }
                        if stress_metrics is not None:
                            pool_metrics["stress_test"] = stress_metrics
```

- [ ] **Step 7: Run lint**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff check src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/train.py`

Expected: No errors. Fix any issues.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/train.py src/python/aerocapture/training/config.py
git commit -m "feat: wire stress tests into adaptive seed training loop with CLI args"
```

---

### Task 5: Update integration test

**Files:**
- Modify: `tests/test_seed_pool.py`

- [ ] **Step 1: Update integration test for new behavior**

Replace `TestAdaptiveSeedIntegration` in `tests/test_seed_pool.py`:

```python
class TestAdaptiveSeedIntegration:
    """Integration test: adaptive seed pool with stress tests in a GA loop."""

    def test_pool_grows_evicts_and_stress_tests(self) -> None:
        """Verify pool grows, evicts hardest-first, and stress tests inject hard seeds."""
        pool = SeedPool(base_seed=0, max_size=8, alpha=0.7, cvar_percentile=20)

        rng = np.random.default_rng(42)
        pop = rng.integers(0, 2, size=(4, 10), dtype=np.int8)

        def evaluator(chrom: npt.NDArray[np.int8], seed: int) -> float:
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

        # Verify hardest seeds survived (keep-hardest eviction)
        difficulties = sorted(pool.difficulty.values())
        # The pool should have retained the harder end of the spectrum
        assert difficulties[-1] > difficulties[0]
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_seed_pool.py -v`

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_seed_pool.py
git commit -m "test: update integration test for stress tests and keep-hardest eviction"
```

---

### Task 6: Full lint and test verification

**Files:** None (verification only)

- [ ] **Step 1: Run ruff lint**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff check src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/train.py`

Expected: Clean.

- [ ] **Step 2: Run ruff format**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run ruff format --check src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/train.py`

Expected: Clean (run `ruff format` if not).

- [ ] **Step 3: Run mypy**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run mypy src/python/aerocapture/training/seed_pool.py`

Expected: No errors.

- [ ] **Step 4: Run full Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v 2>&1 | tail -40`

Expected: All tests pass.

- [ ] **Step 5: Run full lint script**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -20`

Expected: Clean.

---

### Task 7: Smart commit

Invoke the `smart-commit` skill, taking the whole git branch into account.
