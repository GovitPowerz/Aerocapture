# Adaptive Seed Pool: Stress Tests + Keep-Hardest Eviction

## Problem

The adaptive seed pool achieves 100% capture during training but only 69.5% on the independent 1000-sim MC test. Root cause: the pool grows by 1 consecutive-integer seed per generation (42, 43, 44, ...), capped at 100 seeds. The greedy gap-closure eviction strategy can remove genuinely hard seeds if their difficulty score is "close" to another seed's. The pool never actively probes for failure modes -- it passively hopes new seeds happen to be hard.

The pool's seeds and the final MC eval's seeds sample the dispersion space in fundamentally different ways (independent RNG streams vs. draws from a single stream), so the pool cannot discover failures it never generates.

The success criterion is not higher absolute capture, but that the training capture rate becomes a **reliable predictor** of the test rate (gap < 10 percentage points). Honest fitness signals let the GA do its job.

## Approach

Three changes, all in the Python training pipeline (no Rust changes):

1. **Hash-based seed generation** -- replace consecutive integers with well-spread hash-derived seeds, excluding the final MC eval's seed entirely (train/test firewall)
2. **Keep-hardest eviction** -- replace gap-closure with "drop the easiest seed" when over cap
3. **Periodic stress tests** -- every K generations, probe the best individual on N fresh seeds, inject the worst M into the pool

## Design

### 1. Hash-Based Seed Generation

Replace `base_seed + i` with a hash function:

```python
def _pool_seed(base: int, index: int) -> int:
    h = hashlib.sha256(f"{base}:pool:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)
```

- Generation 0 bootstraps indices 0-4 (5 seeds)
- Subsequent generations add index `5 + generation - 1`
- The `"pool"` namespace in the hash input ensures structural separation from the final MC eval's seed

**Exclusion rule:** `SeedPool.__init__` takes a new `excluded_seeds: set[int]` parameter (default empty). If a generated seed collides with an excluded seed, skip it and try `index + 1`. `train.py` passes `{toml_mc_seed}` to exclude the TOML's `[monte_carlo].seed` from training. Collision probability is ~1 in 2 billion per seed -- this is a safety net, not a hot path.

### 2. Keep-Hardest Eviction

Replace `evict_redundant()`:

```python
def evict_redundant(self) -> None:
    while len(self.seeds) > self.max_size:
        easiest = min(self.seeds, key=lambda s: self.difficulty.get(s, 0.0))
        self.seeds.remove(easiest)
        del self.difficulty[easiest]
        del self.generation_added[easiest]
        self.n_evictions += 1
```

When over cap, drop the seed with the lowest difficulty (easiest for the best individual). Hard seeds survive unconditionally. Seeds injected by stress tests (which are the hardest from fresh exploration) naturally float to the top and never get evicted.

### 3. Periodic Stress Tests

New method on `SeedPool`:

```python
def stress_test(
    self,
    generation: int,
    evaluator: Callable[[list[int]], npt.NDArray[np.float64]],
    n_probes: int = 200,
    n_inject: int = 20,
) -> dict[str, Any]:
```

**Mechanics:**

1. Generate `n_probes` fresh seeds from a separate hash namespace: `f"{base}:stress:{generation}:{i}"` (no collision with pool seeds or final MC seed)
2. Evaluate the **best individual only** on all `n_probes` seeds via a single `run_batch` call
3. Sort by cost descending, inject the worst `n_inject` seeds into the pool with their difficulty scores
4. Run `evict_redundant()` to stay within cap (the injected hard seeds survive, easy pool seeds get evicted)
5. Return metrics: `{"n_probes", "n_injected", "worst_cost", "median_cost", "capture_rate"}`

**Stress test evaluator callback:** Built in `train.py` alongside the existing `_batch_evaluator`. Takes the best individual's chromosome, encodes it as overrides, and calls `run_batch` with the stress test seeds. Same pattern as the pool batch evaluator but for a single individual.

**Schedule:** Every `stress_interval` generations (default 5). Called inside the adaptive seed loop after `evaluate_population` and before selection.

**Cost:** 200 sims every 5 generations = 40 sims/gen amortized, for 1 individual. Negligible vs. the ~4000 sims/gen for the full population.

### CLI Arguments

Three new arguments on `train.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--stress-interval` | 5 | Run stress test every N generations |
| `--stress-probes` | 200 | Number of fresh seeds to probe per stress test |
| `--stress-inject` | 20 | Number of worst seeds to inject from each stress test |

All three are only meaningful when `--adaptive-seeds` is also passed. Ignored otherwise.

### Logging

Stress test metrics are added to the per-generation JSONL log under `pool_metrics`:

```json
{
  "pool_metrics": {
    "pool_size": 100,
    "difficulty_min": 45.2,
    "difficulty_max": 15234.5,
    "n_evictions": 47,
    "stress_test": {
      "n_probes": 200,
      "n_injected": 18,
      "worst_cost": 15234.5,
      "median_cost": 312.7,
      "capture_rate": 0.73
    }
  }
}
```

The `stress_test` key is only present on generations where a stress test ran. The `capture_rate` field gives a running estimate of real-world performance -- if it reads 0.73, the training knows the true capture rate is around 73%, not the 100% the pool evaluation shows.

### Checkpoint

No new checkpoint fields. Injected stress-test seeds are stored in the pool's existing `seeds`, `difficulty`, and `generation_added` dicts, which are already serialized. On resume, the stress test schedule resumes naturally from `gen % stress_interval`.

`excluded_seeds` is NOT serialized -- it's a runtime parameter re-passed by `train.py` on resume. `from_dict()` gains an optional `excluded_seeds` parameter that `train.py` always provides.

## Files Changed

| File | Change |
|------|--------|
| `src/python/aerocapture/training/seed_pool.py` | Hash-based `_pool_seed()`, `excluded_seeds` param on `__init__`, keep-hardest `evict_redundant()`, new `stress_test()` method |
| `src/python/aerocapture/training/train.py` | Wire stress test into adaptive seed loop, 3 new CLI args (`--stress-interval`, `--stress-probes`, `--stress-inject`), pass `excluded_seeds` to pool, build stress test evaluator callback |
| `tests/` (seed pool tests) | Update existing tests for new eviction/init signature, add tests for hash spread, exclusion, keep-hardest, stress test injection, namespace separation, checkpoint roundtrip |

No changes to: Rust code, TOML configs, `param_spaces.py`, `evaluate.py`, `report.py`.

## Tests

### New tests

1. Hash-based seed spread: `_pool_seed(42, 0) != _pool_seed(42, 1)`, all seeds in `[0, 2^31)`
2. Excluded seed filtering: pool never contains the excluded seed
3. Keep-hardest eviction: build pool over cap, verify easiest seeds evicted, hardest survive
4. Stress test injection: mock evaluator returning known costs, verify worst N seeds injected and survive eviction
5. Stress test namespace separation: stress seeds don't collide with pool seeds
6. Checkpoint roundtrip: stress-injected seeds survive `to_dict()` / `from_dict()`

### Updated tests

Existing seed pool tests updated for:
- New `__init__` signature (`excluded_seeds` param)
- Keep-hardest eviction behavior (replaces gap-closure assertions)
