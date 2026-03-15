# Adaptive Seed Pool & Graceful Keyboard Interrupt — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace naive seed rotation with a difficulty-aware adaptive seed pool, and add graceful Ctrl+C handling that saves checkpoints and runs the final report.

**Architecture:** New `SeedPool` class manages a growing/evicting pool of MC seeds scored by population-relative difficulty. Fitness = `alpha * mean + (1-alpha) * CVaR`. The pool integrates into the existing GA loop as an alternative to `--rotate-seeds`. Keyboard interrupt is handled via `try/except KeyboardInterrupt` in `train()`.

**Tech Stack:** Python (numpy), existing PyO3 `aerocapture_rs.run_batch()` for batched evaluation.

**Spec:** `docs/superpowers/specs/2026-03-14-adaptive-seeds-graceful-interrupt-design.md`

---

## Chunk 1: SeedPool Core (add/evict/aggregate)

### Task 1: SeedPool — CVaR and aggregation helpers

**Files:**
- Create: `src/python/aerocapture/training/seed_pool.py`
- Create: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing tests for CVaR and fitness aggregation**

```python
# tests/test_seed_pool.py
"""Tests for the adaptive seed pool."""

from __future__ import annotations

import numpy as np
import pytest

from aerocapture.training.seed_pool import compute_cvar, aggregate_fitness


class TestComputeCvar:
    """Tests for CVaR (Conditional Value at Risk) computation."""

    def test_cvar_basic(self) -> None:
        """CVaR-20 of [1, 2, 3, 4, 5] = mean of worst 20% = 5.0."""
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_cvar(costs, percentile=20)
        assert result == pytest.approx(5.0)

    def test_cvar_50(self) -> None:
        """CVaR-50 of [1, 2, 3, 4] = mean of worst 50% = (3+4)/2 = 3.5."""
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_cvar(costs, percentile=50)
        assert result == pytest.approx(3.5)

    def test_cvar_floor_single_element(self) -> None:
        """When n_seeds * percentile / 100 < 1, floor to 1 sample (worst)."""
        costs = np.array([1.0, 2.0, 3.0])
        result = compute_cvar(costs, percentile=20)  # 3 * 0.2 = 0.6 -> floor to 1
        assert result == pytest.approx(3.0)

    def test_cvar_single_seed(self) -> None:
        """Single seed: CVaR = that cost."""
        costs = np.array([42.0])
        result = compute_cvar(costs, percentile=20)
        assert result == pytest.approx(42.0)


class TestAggregateFitness:
    """Tests for mean/CVaR blended fitness aggregation."""

    def test_alpha_1_is_pure_mean(self) -> None:
        """alpha=1.0 -> pure mean."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=1.0, cvar_percentile=20)
        assert result[0] == pytest.approx(3.0)

    def test_alpha_0_is_pure_cvar(self) -> None:
        """alpha=0.0 -> pure CVaR."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=0.0, cvar_percentile=20)
        assert result[0] == pytest.approx(5.0)

    def test_default_blend(self) -> None:
        """alpha=0.7 -> 0.7*mean + 0.3*CVaR."""
        cost_matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = aggregate_fitness(cost_matrix, alpha=0.7, cvar_percentile=20)
        expected = 0.7 * 3.0 + 0.3 * 5.0
        assert result[0] == pytest.approx(expected)

    def test_multiple_individuals(self) -> None:
        """Aggregation works row-wise for multiple individuals."""
        cost_matrix = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        result = aggregate_fitness(cost_matrix, alpha=1.0, cvar_percentile=20)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(30.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: FAIL with `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Implement CVaR and aggregation functions**

```python
# src/python/aerocapture/training/seed_pool.py
"""Adaptive seed pool for difficulty-aware GA training.

Manages a growing pool of MC seeds scored by population-relative
difficulty. Ensures coverage across the difficulty spectrum via
eviction of redundant seeds. Fitness is aggregated as a blend
of mean cost and CVaR (Conditional Value at Risk).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_cvar(costs: npt.NDArray[np.float64], percentile: int) -> float:
    """Compute CVaR (mean of the worst p% of costs).

    Args:
        costs: 1D array of costs for one individual across seeds.
        percentile: Tail fraction (e.g. 20 = worst 20%).

    Returns:
        Mean of the worst ceil(max(1, n * p/100)) costs.
    """
    n = len(costs)
    k = max(1, int(np.ceil(n * percentile / 100)))
    sorted_costs = np.sort(costs)
    return float(np.mean(sorted_costs[-k:]))


def aggregate_fitness(
    cost_matrix: npt.NDArray[np.float64],
    alpha: float,
    cvar_percentile: int,
) -> npt.NDArray[np.float64]:
    """Aggregate per-seed costs into scalar fitness per individual.

    fitness[i] = alpha * mean(costs[i]) + (1 - alpha) * CVaR_p(costs[i])

    Args:
        cost_matrix: Shape (n_individuals, n_seeds).
        alpha: Blend weight (1.0 = pure mean, 0.0 = pure CVaR).
        cvar_percentile: Tail fraction for CVaR.

    Returns:
        1D array of scalar fitness values (n_individuals,).
    """
    means = np.mean(cost_matrix, axis=1)
    cvars = np.array([compute_cvar(row, cvar_percentile) for row in cost_matrix])
    result: npt.NDArray[np.float64] = alpha * means + (1 - alpha) * cvars
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: add CVaR and fitness aggregation for adaptive seed pool"
```

---

### Task 2: SeedPool class — add, evict, difficulty scoring

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py`
- Modify: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing tests for SeedPool lifecycle**

Append to `tests/test_seed_pool.py`:

```python
from aerocapture.training.seed_pool import SeedPool


class TestSeedPoolGrowth:
    """Tests for seed pool growth and bootstrap."""

    def test_bootstrap_creates_5_seeds(self) -> None:
        """First update bootstraps with 5 seeds."""
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        assert pool.seeds == [100, 101, 102, 103, 104]

    def test_incremental_growth(self) -> None:
        """After bootstrap, each generation adds 1 seed."""
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5
        pool.add_seeds(generation=1)
        assert len(pool.seeds) == 6
        assert 105 in pool.seeds

    def test_no_duplicate_seeds(self) -> None:
        """Calling add_seeds for gen 0 twice doesn't duplicate."""
        pool = SeedPool(base_seed=100, max_size=50)
        pool.add_seeds(generation=0)
        pool.add_seeds(generation=0)
        assert len(pool.seeds) == 5


class TestSeedPoolEviction:
    """Tests for difficulty-based eviction."""

    def test_eviction_at_cap(self) -> None:
        """Pool evicts when over max_size."""
        pool = SeedPool(base_seed=0, max_size=7)
        pool.add_seeds(generation=0)  # 5 seeds
        pool.add_seeds(generation=1)  # 6 seeds
        pool.add_seeds(generation=2)  # 7 seeds
        pool.add_seeds(generation=3)  # 8 seeds -> should evict to 7

        # Need to set difficulties before eviction can work
        # Simulate: assign evenly spaced difficulties
        for i, seed in enumerate(pool.seeds):
            pool.difficulty[seed] = float(i * 10)

        pool.evict_redundant()
        assert len(pool.seeds) == 7

    def test_evict_closest_pair_older_one(self) -> None:
        """Eviction removes the older seed from the closest difficulty pair."""
        pool = SeedPool(base_seed=0, max_size=3)
        # Manually set up pool state
        pool.seeds = [10, 20, 30, 40]
        pool.difficulty = {10: 1.0, 20: 1.5, 30: 5.0, 40: 10.0}
        pool.generation_added = {10: 0, 20: 1, 30: 2, 40: 3}

        pool.evict_redundant()

        # Closest pair: 10 (diff=1.0) and 20 (diff=1.5), gap=0.5
        # Evict 10 (older, gen 0)
        assert len(pool.seeds) == 3
        assert 10 not in pool.seeds
        assert 20 in pool.seeds

    def test_no_eviction_under_cap(self) -> None:
        """No eviction when pool is at or under max_size."""
        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [1, 2, 3]
        pool.difficulty = {1: 1.0, 2: 2.0, 3: 3.0}
        pool.generation_added = {1: 0, 2: 1, 3: 2}

        pool.evict_redundant()
        assert len(pool.seeds) == 3


class TestSeedPoolScoring:
    """Tests for difficulty scoring from cost matrix."""

    def test_score_updates_difficulty(self) -> None:
        """score_difficulty extracts the best individual's costs."""
        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [10, 20, 30]

        # Cost matrix: 3 individuals x 3 seeds
        # Best individual = index 1 (lowest aggregated cost)
        cost_matrix = np.array([
            [100.0, 200.0, 300.0],
            [10.0, 20.0, 30.0],   # best
            [50.0, 60.0, 70.0],
        ])
        best_idx = 1
        pool.score_difficulty(cost_matrix, best_idx)

        assert pool.difficulty[10] == pytest.approx(10.0)
        assert pool.difficulty[20] == pytest.approx(20.0)
        assert pool.difficulty[30] == pytest.approx(30.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_pool.py::TestSeedPoolGrowth -v`
Expected: FAIL with `ImportError` (SeedPool not defined yet)

- [ ] **Step 3: Implement SeedPool class**

Add to `src/python/aerocapture/training/seed_pool.py`:

```python
class SeedPool:
    """Adaptive MC seed pool with difficulty-based eviction.

    Manages a growing pool of Monte Carlo seeds. Each seed's difficulty
    is scored by the best individual's cost on that seed (population-relative).
    When the pool exceeds max_size, the most redundant seed (closest
    difficulty pair, older one) is evicted.

    Args:
        base_seed: Base MC seed from TOML config.
        max_size: Maximum pool capacity (default 100).
        alpha: Blend weight for mean/CVaR aggregation (default 0.7).
        cvar_percentile: Tail fraction for CVaR (default 20).
    """

    def __init__(
        self,
        base_seed: int,
        max_size: int = 100,
        alpha: float = 0.7,
        cvar_percentile: int = 20,
    ) -> None:
        self.base_seed = base_seed
        self.max_size = max_size
        self.alpha = alpha
        self.cvar_percentile = cvar_percentile

        self.seeds: list[int] = []
        self.difficulty: dict[int, float] = {}
        self.generation_added: dict[int, int] = {}
        self.n_evictions: int = 0

    def add_seeds(self, generation: int) -> None:
        """Add new seed(s) for this generation.

        Generation 0 bootstraps with 5 seeds. Subsequent generations add 1.
        """
        if generation == 0:
            new_seeds = [self.base_seed + i for i in range(5)]
        else:
            new_seeds = [self.base_seed + generation + 4]  # offset by bootstrap count

        for seed in new_seeds:
            if seed not in self.generation_added:
                self.seeds.append(seed)
                self.generation_added[seed] = generation

    def score_difficulty(self, cost_matrix: npt.NDArray[np.float64], best_idx: int) -> None:
        """Update difficulty scores from the best individual's cost row.

        Args:
            cost_matrix: Shape (n_individuals, n_seeds).
            best_idx: Index of the best individual in cost_matrix.
        """
        best_costs = cost_matrix[best_idx]
        for i, seed in enumerate(self.seeds):
            self.difficulty[seed] = float(best_costs[i])

    def evict_redundant(self) -> None:
        """Evict the most redundant seed if pool exceeds max_size.

        Finds the pair of seeds with the smallest difficulty gap,
        then evicts the older one (lower generation_added).
        """
        while len(self.seeds) > self.max_size:
            # Sort seeds by difficulty
            scored = sorted(self.seeds, key=lambda s: self.difficulty.get(s, 0.0))
            # Find closest pair
            min_gap = float("inf")
            evict_candidate = scored[0]
            for i in range(len(scored) - 1):
                gap = abs(self.difficulty.get(scored[i + 1], 0.0) - self.difficulty.get(scored[i], 0.0))
                if gap < min_gap:
                    min_gap = gap
                    # Evict the older one
                    a, b = scored[i], scored[i + 1]
                    evict_candidate = a if self.generation_added.get(a, 0) <= self.generation_added.get(b, 0) else b

            self.seeds.remove(evict_candidate)
            del self.difficulty[evict_candidate]
            del self.generation_added[evict_candidate]
            self.n_evictions += 1

    @property
    def difficulty_range(self) -> tuple[float, float]:
        """Return (min, max) difficulty across the pool."""
        if not self.difficulty:
            return (0.0, 0.0)
        vals = list(self.difficulty.values())
        return (min(vals), max(vals))

    def to_dict(self) -> dict:
        """Serialize pool state for checkpoint."""
        return {
            "base_seed": self.base_seed,
            "max_size": self.max_size,
            "alpha": self.alpha,
            "cvar_percentile": self.cvar_percentile,
            "seeds": self.seeds,
            "difficulty": {str(k): v for k, v in self.difficulty.items()},
            "generation_added": {str(k): v for k, v in self.generation_added.items()},
            "n_evictions": self.n_evictions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SeedPool:
        """Restore pool from checkpoint data."""
        pool = cls(
            base_seed=data["base_seed"],
            max_size=data["max_size"],
            alpha=data.get("alpha", 0.7),
            cvar_percentile=data.get("cvar_percentile", 20),
        )
        pool.seeds = data["seeds"]
        pool.difficulty = {int(k): v for k, v in data["difficulty"].items()}
        pool.generation_added = {int(k): v for k, v in data["generation_added"].items()}
        pool.n_evictions = data.get("n_evictions", 0)
        return pool
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: add SeedPool class with growth, eviction, and difficulty scoring"
```

---

### Task 3: SeedPool checkpoint round-trip

**Files:**
- Modify: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing test for serialization round-trip**

Append to `tests/test_seed_pool.py`:

```python
class TestSeedPoolCheckpoint:
    """Tests for checkpoint serialization/deserialization."""

    def test_round_trip(self) -> None:
        """to_dict -> from_dict preserves all state."""
        pool = SeedPool(base_seed=42, max_size=50, alpha=0.8, cvar_percentile=25)
        pool.seeds = [42, 43, 44]
        pool.difficulty = {42: 1.0, 43: 5.0, 44: 10.0}
        pool.generation_added = {42: 0, 43: 0, 44: 1}
        pool.n_evictions = 2

        data = pool.to_dict()
        restored = SeedPool.from_dict(data)

        assert restored.base_seed == 42
        assert restored.max_size == 50
        assert restored.alpha == 0.8
        assert restored.cvar_percentile == 25
        assert restored.seeds == [42, 43, 44]
        assert restored.difficulty == {42: 1.0, 43: 5.0, 44: 10.0}
        assert restored.generation_added == {42: 0, 43: 0, 44: 1}
        assert restored.n_evictions == 2

    def test_round_trip_json_compatible(self) -> None:
        """Serialized dict is JSON-compatible (no numpy types)."""
        import json

        pool = SeedPool(base_seed=0, max_size=10)
        pool.seeds = [0, 1, 2]
        pool.difficulty = {0: 1.0, 1: 2.0, 2: 3.0}
        pool.generation_added = {0: 0, 1: 0, 2: 1}

        data = pool.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = SeedPool.from_dict(restored_data)
        assert restored.seeds == [0, 1, 2]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_pool.py::TestSeedPoolCheckpoint -v`
Expected: PASS (implementation already supports this)

- [ ] **Step 3: Commit**

```bash
git add tests/test_seed_pool.py
git commit -m "test: add SeedPool checkpoint round-trip tests"
```

---

## Chunk 2: Graceful Keyboard Interrupt

### Task 4: KeyboardInterrupt handling in train()

**Files:**
- Modify: `src/python/aerocapture/training/train.py:294-439` (wrap loop in try/except)
- Modify: `src/python/aerocapture/training/display.py` (add `stop()` method)
- Modify: `tests/test_seed_pool.py` (or create `tests/test_train_interrupt.py`)

- [ ] **Step 1: Add `stop()` method to display classes**

In `src/python/aerocapture/training/display.py`, add to `DisplayProtocol`:

```python
class DisplayProtocol(Protocol):
    def update(self, logger: TrainingLogger, current_run: int) -> None: ...
    def stop(self) -> None: ...  # ADD THIS
    def __enter__(self) -> DisplayProtocol: ...
    def __exit__(self, ...) -> None: ...
```

Add to `NoopDisplay`:

```python
def stop(self) -> None:
    pass
```

Add to `LiveDisplay`:

```python
def stop(self) -> None:
    """Stop the Live display (for clean interrupt output)."""
    if self._live is not None:
        self._live.stop()
```

- [ ] **Step 2: Wrap training loop with KeyboardInterrupt handling**

In `src/python/aerocapture/training/train.py`, wrap the `with display:` block (lines 294–439). The structure becomes:

```python
    interrupted = False

    with display:
        try:
            for run in range(start_run, config.ga.n_runs):
                # ... existing run loop (unchanged) ...

                for gen in range(gen_start, config.ga.n_gen):
                    # ... existing gen loop (unchanged) ...
                    pass

                cost_history.extend(gen_best_costs)
                logger.close()

        except KeyboardInterrupt:
            interrupted = True
            display.stop()
            print(f"\nInterrupted at run {run + 1}, gen {gen + 1}. Saving checkpoint...")
            save_checkpoint(
                save_dir, run, gen + 1, populations, all_costs,
                best_overall_cost, best_overall_chrom,
                cost_history + gen_best_costs, rng, config, cwd,
            )
            logger.close()

    return {
        "best_cost": best_overall_cost,
        "best_chromosome": best_overall_chrom,
        "cost_history": cost_history,
        "interrupted": interrupted,
    }
```

Key details:
- `display.stop()` tears down Rich TUI before printing to stdout
- Checkpoint is saved inside the `with display:` block (display still alive for cleanup)
- `logger.close()` ensures JSONL is flushed
- `interrupted: True` flag in return dict lets `__main__` know

- [ ] **Step 3: Write a test for interrupt handling**

Create `tests/test_train_interrupt.py`:

```python
"""Tests for graceful keyboard interrupt handling in train()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aerocapture.training.config import TrainingConfig
from aerocapture.training.train import train


class TestKeyboardInterrupt:
    """Tests that Ctrl+C saves checkpoint and returns cleanly."""

    def test_interrupt_returns_interrupted_flag(self, tmp_path: object) -> None:
        """train() returns interrupted=True on KeyboardInterrupt."""
        cfg = TrainingConfig()
        cfg.ga.n_gen = 100
        cfg.ga.n_pop = 2
        cfg.ga.n_runs = 1
        cfg.save_dir = str(tmp_path)

        gen_count = 0

        original_evaluate = None

        def mock_evaluate(*args, **kwargs):
            nonlocal gen_count
            gen_count += 1
            # Let initial population creation succeed, interrupt during GA loop
            if gen_count > 10:
                raise KeyboardInterrupt
            return 1e6 + gen_count, None

        # Mock create_initial_population to return a pre-built population
        mock_pop = np.zeros((2, cfg.chrom_length), dtype=np.int8)
        mock_costs = np.array([1e6, 1e6 + 1])

        with patch("aerocapture.training.train.create_initial_population", return_value=(mock_pop, mock_costs)), \
             patch("aerocapture.training.train.evaluate_chromosome", side_effect=mock_evaluate), \
             patch.object(type(cfg.sim), "executable", new_callable=lambda: property(lambda self: "dummy")), \
             patch("aerocapture.training.train.Path.exists", return_value=True):
            result = train(cfg, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        assert result["interrupted"] is True
        assert result["best_cost"] < float("inf")


class TestMutualExclusion:
    """Tests for --rotate-seeds / --adaptive-seeds mutual exclusion."""

    def test_both_flags_raises_error(self) -> None:
        """Passing both --rotate-seeds and --adaptive-seeds is an error."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "aerocapture.training.train",
             "--rotate-seeds", "--adaptive-seeds", "--toml", "dummy.toml"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr
```

- [ ] **Step 4: Run the interrupt test**

Run: `uv run pytest tests/test_train_interrupt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/display.py src/python/aerocapture/training/train.py tests/test_train_interrupt.py
git commit -m "feat: add graceful KeyboardInterrupt handling with checkpoint save"
```

---

## Chunk 3: Integrate SeedPool into Training Loop

### Task 5: CLI args and pool initialization in train()

**Files:**
- Modify: `src/python/aerocapture/training/train.py:448-580` (CLI args)
- Modify: `src/python/aerocapture/training/train.py:185-242` (train() signature + setup)
- Modify: `src/python/aerocapture/training/config.py` (add adaptive_seeds fields to GAConfig)

- [ ] **Step 1: Add adaptive seed config fields to GAConfig**

In `src/python/aerocapture/training/config.py`, add to `GAConfig` (after `rotate_seeds`):

```python
    adaptive_seeds: bool = False
    seed_pool_cap: int = 100
    cost_alpha: float = 0.7
    cvar_percentile: int = 20
```

- [ ] **Step 2: Add CLI args with mutual exclusion**

In `src/python/aerocapture/training/train.py`, replace the `--rotate-seeds` arg with a mutually exclusive group:

```python
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--rotate-seeds", action="store_true",
        help="Rotate MC dispersion seed each generation (prevents overfitting to fixed scenarios)")
    seed_group.add_argument("--adaptive-seeds", action="store_true",
        help="Use adaptive seed pool with difficulty-based eviction")
    parser.add_argument("--seed-pool-cap", type=int, default=100,
        help="Maximum adaptive seed pool size (default: 100)")
    parser.add_argument("--cost-alpha", type=float, default=0.7,
        help="Mean/CVaR blend weight: 1.0=pure mean, 0.0=pure CVaR (default: 0.7)")
    parser.add_argument("--cvar-percentile", type=int, default=20,
        help="CVaR tail fraction in percent (default: 20)")
```

Wire into config (in the `__main__` block after `cfg.ga.rotate_seeds = args.rotate_seeds`):

```python
    cfg.ga.rotate_seeds = args.rotate_seeds
    cfg.ga.adaptive_seeds = args.adaptive_seeds
    cfg.ga.seed_pool_cap = args.seed_pool_cap
    cfg.ga.cost_alpha = args.cost_alpha
    cfg.ga.cvar_percentile = args.cvar_percentile
```

- [ ] **Step 3: Initialize SeedPool in train()**

In `train()`, after the existing `base_mc_seed` setup (line ~242), add:

```python
    # Initialize adaptive seed pool
    seed_pool: SeedPool | None = None
    if config.ga.adaptive_seeds:
        if not config.sim.toml_config:
            msg = "adaptive_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        import tomllib

        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        with open(toml_path, "rb") as f:
            _toml = tomllib.load(f)
        pool_base_seed = _toml.get("monte_carlo", {}).get("seed")
        if pool_base_seed is None:
            msg = "adaptive_seeds requires [monte_carlo].seed in the TOML config"
            raise ValueError(msg)
        seed_pool = SeedPool(
            base_seed=pool_base_seed,
            max_size=config.ga.seed_pool_cap,
            alpha=config.ga.cost_alpha,
            cvar_percentile=config.ga.cvar_percentile,
        )
```

Add `from aerocapture.training.seed_pool import SeedPool` to imports at top.

- [ ] **Step 4: Restore seed pool from checkpoint**

In the resume block (around line 248), add after restoring other state:

```python
    if resumed is not None:
        # ... existing restore code ...
        if seed_pool is not None and "seed_pool" in resumed:
            seed_pool = SeedPool.from_dict(resumed["seed_pool"])
```

In `save_checkpoint()`, add pool state to `meta` dict:

```python
    # In save_checkpoint signature, add: seed_pool: SeedPool | None = None
    # In meta dict:
    if seed_pool is not None:
        meta["seed_pool"] = seed_pool.to_dict()
```

In `load_checkpoint()`, add to returned dict:

```python
    return {
        # ... existing fields ...
        "seed_pool": meta.get("seed_pool"),
    }
```

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_training_config.py tests/test_training_integration.py -v`
Expected: PASS (no existing behavior changed)

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py
git commit -m "feat: add adaptive-seeds CLI args, GAConfig fields, and pool initialization"
```

---

### Task 6: Replace per-individual evaluation with pool-based evaluation

**Files:**
- Modify: `src/python/aerocapture/training/seed_pool.py` (add `evaluate_population` method)
- Modify: `src/python/aerocapture/training/train.py:338-378` (generation inner loop)
- Modify: `tests/test_seed_pool.py`

- [ ] **Step 1: Write failing test for evaluate_population**

Append to `tests/test_seed_pool.py`:

```python
from unittest.mock import MagicMock

class TestSeedPoolEvaluation:
    """Tests for pool-based population evaluation."""

    def test_evaluate_population_calls_evaluator(self) -> None:
        """evaluate_population calls the evaluator for each (individual, seed) pair."""
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1, 2]
        pool.generation_added = {0: 0, 1: 0, 2: 0}

        population = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

        # Mock evaluator: returns seed value as cost
        def evaluator(chrom, seed):
            return float(seed) + float(chrom[0])

        fitness = pool.evaluate_population(population, evaluator)
        assert fitness.shape == (2,)
        # Individual 0 (chrom[0]=1): costs=[1, 2, 3], mean=2.0
        # Individual 1 (chrom[0]=0): costs=[0, 1, 2], mean=1.0
        assert fitness[0] == pytest.approx(2.0)  # alpha=1.0 -> pure mean
        assert fitness[1] == pytest.approx(1.0)

    def test_evaluate_population_updates_difficulty(self) -> None:
        """evaluate_population updates difficulty after evaluation."""
        pool = SeedPool(base_seed=0, max_size=10, alpha=1.0, cvar_percentile=20)
        pool.seeds = [0, 1]
        pool.generation_added = {0: 0, 1: 0}

        population = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)

        def evaluator(chrom, seed):
            return float(seed) * 10.0

        fitness = pool.evaluate_population(population, evaluator)
        # Best individual = index with lowest fitness
        # Both have same costs (seed-based, chrom ignored), so best_idx=0
        assert pool.difficulty[0] == pytest.approx(0.0)
        assert pool.difficulty[1] == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_pool.py::TestSeedPoolEvaluation -v`
Expected: FAIL with `AttributeError` (evaluate_population not defined)

- [ ] **Step 3: Implement evaluate_population**

Add to `SeedPool` class in `seed_pool.py`:

```python
    def evaluate_population(
        self,
        population: npt.NDArray[np.int8],
        evaluator: Callable[[npt.NDArray[np.int8], int], float],
        batch_evaluator: Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate all individuals on all pool seeds.

        If batch_evaluator is provided, uses it for per-individual batched
        evaluation (one call per individual across all seeds, leveraging
        run_batch() for Rayon parallelism). Falls back to the scalar
        evaluator otherwise.

        Args:
            population: Shape (n_pop, chrom_length).
            evaluator: Callable(chromosome, mc_seed) -> cost (scalar fallback).
            batch_evaluator: Callable(chromosome, seeds) -> costs array (n_seeds,).
                Evaluates one individual on all seeds in one batched call.

        Returns:
            1D fitness array (n_pop,) with aggregated fitness values.
        """
        n_pop = len(population)
        n_seeds = len(self.seeds)
        cost_matrix = np.full((n_pop, n_seeds), np.inf)

        if batch_evaluator is not None:
            # Fast path: one run_batch() call per individual
            for i in range(n_pop):
                cost_matrix[i] = batch_evaluator(population[i], self.seeds)
        else:
            # Fallback: scalar evaluator (tests, subprocess mode)
            for i in range(n_pop):
                for j, seed in enumerate(self.seeds):
                    cost_matrix[i, j] = evaluator(population[i], seed)

        # Aggregate fitness
        fitness = aggregate_fitness(cost_matrix, self.alpha, self.cvar_percentile)

        # Score difficulty from best individual
        best_idx = int(np.argmin(fitness))
        self.score_difficulty(cost_matrix, best_idx)

        return fitness
```

Add `from collections.abc import Callable` to the imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py
git commit -m "feat: add SeedPool.evaluate_population for batched evaluation"
```

---

### Task 7: Wire adaptive seed pool into the GA generation loop

**Files:**
- Modify: `src/python/aerocapture/training/train.py:338-436`

This is the core integration. When `seed_pool is not None`, the generation inner loop changes from per-individual `evaluate_chromosome()` calls to pool-based evaluation.

- [ ] **Step 1: Create evaluator callbacks (scalar + batch)**

In `train()`, before the generation loop (after `gen_best_costs: list[float] = []`, around line 336), define both callbacks:

```python
            # Build evaluator callbacks for adaptive seed pool
            def _pool_evaluator(chrom: npt.NDArray[np.int8], mc_seed: int) -> float:
                """Scalar fallback: one (chromosome, seed) pair."""
                cost, _ = evaluate_chromosome(chrom, base_network, config, cwd=cwd, mc_seed=mc_seed)
                return cost

            # Batch evaluator: leverages run_batch() for Rayon parallelism
            # Evaluates one individual on ALL seeds in a single batched call
            _batch_evaluator: Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]] | None = None
            if _HAS_PYO3 and config.sim.toml_config:
                def _make_batch_eval(
                    base_net: npt.NDArray[np.float64],
                    cfg: TrainingConfig,
                    working_dir: str | Path | None,
                ) -> Callable[[npt.NDArray[np.int8], list[int]], npt.NDArray[np.float64]]:
                    """Factory to avoid closure over mutable loop variables."""
                    def _batch_eval(chrom: npt.NDArray[np.int8], seeds: list[int]) -> npt.NDArray[np.float64]:
                        # Decode chromosome once (doesn't change across seeds)
                        if cfg.guidance_type == "neural_network":
                            weights = decode_direct(chrom, cfg) if cfg.ga.direct_encoding else perturb_network(chrom, base_net, cfg)
                            nn_path = Path(working_dir or cfg.sim.exec_dir) / cfg.sim.nn_param_file
                            write_nn_json(weights, cfg.network, nn_path)
                            # Build per-seed overrides
                            overrides_list = [{"monte_carlo.seed": s} for s in seeds]
                        else:
                            params = decode_params_from_chromosome(chrom, cfg)
                            # Build per-seed overrides with guidance params + seed
                            from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS
                            section = GUIDANCE_TOML_SECTIONS[cfg.guidance_type]
                            base_overrides = {f"guidance.{section}.{k}": v for k, v in params.items()}
                            base_overrides["guidance.type"] = cfg.guidance_type
                            overrides_list = [{**base_overrides, "monte_carlo.seed": s} for s in seeds]

                        toml_path = str((Path(working_dir or cfg.sim.exec_dir) / cfg.sim.toml_config).resolve())
                        results = _aero_rs.run_batch(
                            toml_path=toml_path,
                            overrides_list=overrides_list,
                        )
                        # Compute cost per seed from final records
                        costs = np.array([compute_cost(r.final_record.reshape(1, 52)) for r in results.results])
                        return costs
                    return _batch_eval

                _batch_evaluator = _make_batch_eval(base_network, config, cwd)
```

Note: add `from aerocapture.training.evaluate import _HAS_PYO3, _aero_rs, compute_cost, decode_direct, perturb_network, write_nn_json, decode_params_from_chromosome` to imports. Also add `_HAS_PYO3` and `_aero_rs` to the existing import line from evaluate.py.

- [ ] **Step 2: Add adaptive seed pool branch in generation loop**

Replace the generation inner loop (lines 341-378) with a branching structure. The existing code stays in an `else` branch; the new adaptive path goes in an `if seed_pool is not None` branch:

```python
                if seed_pool is not None:
                    # Adaptive seed pool path
                    seed_pool.add_seeds(gen)

                    for k in range(config.ga.n_subpop):
                        pop = populations[k]
                        pop_costs = all_costs[k]

                        # Create offspring
                        offspring = crossover_and_mutate(pop, pop_costs, config, rng)

                        # Evaluate combined population on all seeds
                        combined = np.vstack([pop, offspring])
                        combined_fitness = seed_pool.evaluate_population(
                            combined, _pool_evaluator, batch_evaluator=_batch_evaluator,
                        )

                        # Evict after scoring
                        seed_pool.evict_redundant()

                        # Tournament selection: keep best n_pop
                        n_pop = len(pop)
                        order = np.argsort(combined_fitness)
                        populations[k] = combined[order[:n_pop]]
                        all_costs[k] = combined_fitness[order[:n_pop]]

                        # Track best
                        gen_best = all_costs[k][0]
                        if gen_best < best_overall_cost:
                            best_overall_cost = gen_best
                            best_overall_chrom = populations[k][0].copy()

                    # Migration: skip local improvement in adaptive mode
                    # (local improvement uses single-seed eval, incompatible with pool)
                    if (gen + 1) % config.ga.migration_interval == 0 and config.ga.n_subpop > 1:
                        # Ring migration: exchange best individuals between subpopulations
                        # (no local improvement — re-evaluation happens next generation via pool)
                        for i in range(config.ga.n_subpop - 1):
                            best_idx = int(np.argmin(all_costs[i + 1]))
                            worst_idx = int(np.argmax(all_costs[i]))
                            populations[i][worst_idx] = populations[i + 1][best_idx].copy()
                            all_costs[i][worst_idx] = all_costs[i + 1][best_idx]
                        best_idx = int(np.argmin(all_costs[0]))
                        worst_idx = int(np.argmax(all_costs[-1]))
                        populations[-1][worst_idx] = populations[0][best_idx].copy()
                        all_costs[-1][worst_idx] = all_costs[0][best_idx]
                else:
                    # Original path (fixed seed or rotate-seeds)
                    mc_seed = (base_mc_seed + gen) if base_mc_seed is not None else None
                    # ... existing code lines 341-395 unchanged ...
```

- [ ] **Step 3: Pass seed_pool to save_checkpoint**

Update all `save_checkpoint()` calls to pass `seed_pool=seed_pool`. There are two call sites:
1. Regular checkpoint (line ~422)
2. Interrupt checkpoint (from Task 4)

Update `save_checkpoint()` signature to accept `seed_pool: SeedPool | None = None` and serialize it.

- [ ] **Step 4: Add pool metrics to logger**

In the `log_generation()` call (line ~406), add seed pool metrics when available:

```python
                # Log metrics
                pool_metrics = None
                if seed_pool is not None:
                    d_min, d_max = seed_pool.difficulty_range
                    pool_metrics = {
                        "pool_size": len(seed_pool.seeds),
                        "difficulty_min": d_min,
                        "difficulty_max": d_max,
                        "n_evictions": seed_pool.n_evictions,
                    }
```

Update `logger.log_generation()` to accept an optional `pool_metrics: dict | None = None` parameter and include it in the record.

- [ ] **Step 5: Update logger to accept pool metrics**

In `src/python/aerocapture/training/logger.py`, add to `log_generation()` signature:

```python
    def log_generation(
        self,
        generation: int,
        populations: list[npt.NDArray[np.int8]],
        costs: list[npt.NDArray[np.float64]],
        best_chromosome: npt.NDArray[np.int8],
        decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None,
        weight_stats: dict[str, dict[str, float]] | None = None,
        mc_seed: int | None = None,
        pool_metrics: dict | None = None,  # ADD THIS
    ) -> None:
```

And add at the end of the record dict construction (after the `mc_seed` block):

```python
        if pool_metrics is not None:
            record["pool_metrics"] = pool_metrics
```

- [ ] **Step 6: Run all training tests to verify no regressions**

Run: `uv run pytest tests/ -k "training" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/train.py src/python/aerocapture/training/logger.py
git commit -m "feat: integrate adaptive seed pool into GA generation loop"
```

---

### Task 7b: Integration test for adaptive seed pool in GA loop

**Files:**
- Modify: `tests/test_seed_pool.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_seed_pool.py`:

```python
class TestAdaptiveSeedIntegration:
    """Integration test: adaptive seed pool in the GA training loop."""

    def test_pool_grows_and_evicts_during_training(self) -> None:
        """Verify pool grows, evicts, and produces valid fitness across generations."""
        pool = SeedPool(base_seed=0, max_size=8, alpha=0.7, cvar_percentile=20)

        rng = np.random.default_rng(42)
        pop = rng.integers(0, 2, size=(4, 10), dtype=np.int8)

        # Simulate 10 generations of evaluation
        def evaluator(chrom: np.ndarray, seed: int) -> float:
            # Deterministic: cost depends on seed difficulty + chromosome quality
            quality = float(np.sum(chrom)) / len(chrom)
            return float(seed) * 10.0 + quality * 5.0

        for gen in range(10):
            pool.add_seeds(gen)
            fitness = pool.evaluate_population(pop, evaluator)

            assert fitness.shape == (4,)
            assert all(np.isfinite(fitness))

            pool.evict_redundant()

            # Pool should not exceed max_size
            assert len(pool.seeds) <= 8

        # After 10 gens: bootstrapped 5, added 9 more = 14 total, evicted to 8
        assert len(pool.seeds) == 8
        assert pool.n_evictions == 6  # 14 - 8 = 6 evictions

        # Difficulty should be populated for all active seeds
        assert len(pool.difficulty) == len(pool.seeds)

        # Difficulty range should be non-trivial (seeds have different costs)
        d_min, d_max = pool.difficulty_range
        assert d_max > d_min
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_seed_pool.py::TestAdaptiveSeedIntegration -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_seed_pool.py
git commit -m "test: add integration test for adaptive seed pool in GA loop"
```

---

## Chunk 4: Lint, Full Test Suite, Final Polish

### Task 8: Lint and type-check

**Files:**
- All modified files

- [ ] **Step 1: Run ruff**

Run: `uv run ruff check src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/train.py src/python/aerocapture/training/display.py src/python/aerocapture/training/logger.py src/python/aerocapture/training/config.py`
Expected: No errors. Fix any issues.

- [ ] **Step 2: Run ruff format**

Run: `uv run ruff format src/python/aerocapture/training/seed_pool.py tests/test_seed_pool.py`
Expected: Files formatted.

- [ ] **Step 3: Run mypy**

Run: `uv run mypy src/python/aerocapture/training/seed_pool.py src/python/aerocapture/training/train.py`
Expected: No errors. Fix any type issues.

- [ ] **Step 4: Commit fixes if any**

```bash
git add -A && git commit -m "style: fix lint and type issues"
```

---

### Task 9: Full test suite

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run lint_code.sh**

Run: `./lint_code.sh`
Expected: Clean

- [ ] **Step 3: Fix any failures and commit**

```bash
git add -A && git commit -m "fix: address test/lint failures from integration"
```

---

### Task 10: Smart commit with docs sync

- [ ] **Step 1: Use smart-commit skill**

Invoke the `smart-commit` skill. The skill should look at the entire git branch (all commits since diverging from `main`) to understand the full scope of changes when updating CLAUDE.md and README.md.

```
/smart-commit
```

This will sync CLAUDE.md and README.md with the new adaptive seed pool and graceful interrupt features, then commit everything.
