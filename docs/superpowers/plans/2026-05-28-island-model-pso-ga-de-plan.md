# Island-Model PSO/GA/DE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `algorithm = "islands"` optimizer mode that runs PSO, GA, and DE in parallel with periodic top-3 / worst-6 migration every K=25 generations, sharing one seed list / validation pool / problem instance — clean A/B against single-island PSO with the same per-island chromosome and budget.

**Architecture:** In-process trainer (one Python process). A new `IslandModel` class owns 3 `Island` instances, each wrapping a pymoo `Algorithm`. The outer loop in `train.py` calls `island_model.step(gen)` instead of `algorithm.next()`. Migration is in-place on `algorithm.pop`; PSO destinations get fresh velocity injection so the swarm can't immediately pull migrants into a collapsed `gbest`. Per-island identity-trigger validation, pooled top-K seed curation, per-island JSONL records, winning-island artifacts (`best_model.json` / `best_params.json`) drop into the existing Rust runtime untouched.

**Tech Stack:** Python 3.14, pymoo (PSO/GA/DE Algorithm objects), numpy, PyO3-bound Rust simulator (`aerocapture_rs`), pytest, Rich (TUI), `uv` for env management.

**Reference spec:** [`docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md`](../specs/2026-05-28-island-model-pso-ga-de-design.md)

---

## File Structure

### New files

- `src/python/aerocapture/training/island_model.py` (~280 LoC) — `Island`, `MigrationEvent`, `IslandModel`, `migrate()`, `inject_into_pso()`.
- `tests/test_island_model.py` — fast unit tests for `migrate()`, `inject_into_pso()`, config parsing, checkpoint round-trip, per-island validation gate, winner selection.
- `tests/test_island_model_smoke.py` — `@pytest.mark.slow` end-to-end 5-gen smoke + resume verification.
- `configs/training/msr_aller_islands_train.toml` — reference training config (Dense->Dense NN, k_period=25).

### Modified files

- `src/python/aerocapture/training/optimizer.py` — add `"islands"` to `_VALID_ALGORITHMS`, add `IslandSettings` dataclass and `OptimizerConfig.islands` field, extend `from_dict` parsing.
- `src/python/aerocapture/training/train.py` — `train()` gains an `algorithm == "islands"` branch that constructs an `IslandModel` and runs its outer loop instead of the single-algorithm one. Shared setup (problem, seed pools, warm-start, checkpoint load) is reused.
- `src/python/aerocapture/training/logger.py` — `TrainingLogger.log_generation` gains an optional `island_name: str | None = None` parameter; records carry the field for downstream charts/reports.
- `src/python/aerocapture/training/display.py` — `LiveDisplay` extended with a 3-column render mode when wired to an `IslandModel`.
- `src/python/aerocapture/training/charts.py` — two new chart functions: `chart_island_convergence_overlay`, `chart_migration_timeline`.
- `src/python/aerocapture/training/report.py` — Part 0 (island overlay + migration timeline) prepended; Parts 1–3 take an explicit `island_name` filter so they run on the winning island only.
- `src/python/aerocapture/training/warm_start.py` — the warm-start chromosome path returns one chromosome that gets fanned out to 3 islands' initial populations (see Task 13).
- `configs/training/common.toml` — add `[optimizer.islands]` sub-block with `enabled`, `k_period`, `k_top`, `pso_inject_velocity_scale`.

### Untouched (verify after implementation)

- All Rust code (`src/rust/`).
- `aerocapture-py` PyO3 bindings.
- `compare_guidance.py` (winning island's `best_model.json` plugs in identically).
- `seed_curator.py` (used as-is; islands pool top-K across the union of populations).
- `evaluate.py` constants (`VALIDATION_SEED_OFFSET`, `FINAL_EVAL_SEED_OFFSET`, `make_reserved_seeds`).

---

## Task 1: Optimizer config — `IslandSettings` + `"islands"` algorithm

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py` (lines 14, 19-103 region)
- Test: `tests/test_island_model.py` (new file)

- [ ] **Step 1: Write failing test for `OptimizerConfig.from_dict` with `algorithm = "islands"`**

Create `tests/test_island_model.py`:

```python
"""Unit tests for the 3-island PSO/GA/DE evolutionary trainer.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerocapture.training.optimizer import IslandSettings, OptimizerConfig


def test_optimizer_config_islands_parses() -> None:
    d = {
        "algorithm": "islands",
        "seed_strategy": "adaptive",
        "n_pop": 64,
        "training_n_sims": 20,
        "islands": {
            "enabled": True,
            "k_period": 25,
            "k_top": 3,
            "pso_inject_velocity_scale": 0.05,
        },
        "ga": {"crossover_eta": 15.0, "mutation_eta": 20.0},
        "pso": {"w": 0.7, "c1": 1.5, "c2": 1.5},
        "de": {"variant": "DE/rand/1/bin", "crossover_prob": 0.8, "scaling_factor": 0.6},
    }
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.algorithm == "islands"
    assert cfg.islands.enabled is True
    assert cfg.islands.k_period == 25
    assert cfg.islands.k_top == 3
    assert cfg.islands.pso_inject_velocity_scale == 0.05


def test_optimizer_config_islands_default_values() -> None:
    d = {"algorithm": "islands", "seed_strategy": "fixed"}
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.islands.enabled is True
    assert cfg.islands.k_period == 25
    assert cfg.islands.k_top == 3
    assert cfg.islands.pso_inject_velocity_scale == 0.05


def test_optimizer_config_islands_invalid_k_top_raises() -> None:
    with pytest.raises(ValueError, match="k_top"):
        IslandSettings(k_top=0)


def test_optimizer_config_islands_invalid_k_period_raises() -> None:
    with pytest.raises(ValueError, match="k_period"):
        IslandSettings(k_period=0)


def test_optimizer_config_islands_invalid_velocity_scale_raises() -> None:
    with pytest.raises(ValueError, match="pso_inject_velocity_scale"):
        IslandSettings(pso_inject_velocity_scale=-0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: 5 failures — `ImportError: cannot import name 'IslandSettings'` and "islands" not in `_VALID_ALGORITHMS`.

- [ ] **Step 3: Add `IslandSettings` dataclass and `islands` field to `OptimizerConfig`**

In `src/python/aerocapture/training/optimizer.py`:

```python
# Change line 14:
_VALID_ALGORITHMS = ("ga", "cma_es", "de", "pso", "islands")

# Add new dataclass after `PSOSettings` (around line 44):
@dataclass
class IslandSettings:
    enabled: bool = True
    k_period: int = 25
    k_top: int = 3
    pso_inject_velocity_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.k_period < 1:
            raise ValueError(f"k_period must be >= 1, got {self.k_period}")
        if self.k_top < 1:
            raise ValueError(f"k_top must be >= 1, got {self.k_top}")
        if self.pso_inject_velocity_scale < 0.0:
            raise ValueError(
                f"pso_inject_velocity_scale must be >= 0.0, got {self.pso_inject_velocity_scale}"
            )

# Add `islands` field to OptimizerConfig (after `pso`):
@dataclass
class OptimizerConfig:
    algorithm: str = "ga"
    seed_strategy: str = ""
    n_pop: int = 60
    n_gen: int = 2500
    seed_pool_interval: int = 50
    training_n_sims: int = 1
    validation_n_sims: int = 1000
    curation_top_k: int = 5
    curation_sample_size: int = 1000
    ga: GASettings = field(default_factory=GASettings)
    cma_es: CMAESSettings = field(default_factory=CMAESSettings)
    de: DESettings = field(default_factory=DESettings)
    pso: PSOSettings = field(default_factory=PSOSettings)
    islands: IslandSettings = field(default_factory=IslandSettings)
    # ... rest unchanged

# Extend from_dict to parse `islands` (around line 83-101):
        ga = GASettings(**d["ga"]) if "ga" in d else GASettings()
        cma_es = CMAESSettings(**d["cma_es"]) if "cma_es" in d else CMAESSettings()
        de = DESettings(**d["de"]) if "de" in d else DESettings()
        pso = PSOSettings(**d["pso"]) if "pso" in d else PSOSettings()
        islands = IslandSettings(**d["islands"]) if "islands" in d else IslandSettings()

        # ... obsolete-key warning block stays unchanged ...

        top_level = {
            k: v for k, v in d.items()
            if k not in ("ga", "cma_es", "de", "pso", "islands") and k not in _obsolete
        }
        return cls(
            **top_level, ga=ga, cma_es=cma_es, de=de, pso=pso, islands=islands,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: 5 passes.

- [ ] **Step 5: Run full existing optimizer tests to verify no regression**

```bash
uv run pytest tests/test_optimizer.py -v
```

Expected: all passes (existing single-algorithm paths untouched).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_island_model.py
git commit -m "feat(islands): add IslandSettings config dataclass and 'islands' algorithm

Adds [optimizer.islands] sub-block parsing (enabled, k_period, k_top,
pso_inject_velocity_scale) without wiring it into train() yet. The
'islands' algorithm value is now accepted by OptimizerConfig.from_dict."
```

---

## Task 2: Pure `migrate()` function + unit tests

**Files:**
- Create: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing tests for `migrate()` determinism, top-k selection, worst-slot replacement, no-self-migration**

Append to `tests/test_island_model.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.individual import Individual

from aerocapture.training.island_model import (
    Island,
    MigrationEvent,
    inject_into_pso,
    migrate,
)


class _FakeAlgo:
    """Minimal stand-in for pymoo Algorithm — only `.pop` is touched by migrate()."""

    def __init__(self, pop: Population) -> None:
        self.pop = pop


def _make_pop(X: np.ndarray, F: np.ndarray) -> Population:
    pop = Population.new("X", X)
    pop.set("F", F.reshape(-1, 1))
    return pop


def _make_island(name: str, X: np.ndarray, F: np.ndarray) -> Island:
    return Island(
        name=name,
        algorithm=_FakeAlgo(_make_pop(X, F)),
        last_validated_individual=None,
        best_overall_individual=None,
        best_overall_cost=float("inf"),
        best_val_cost=float("inf"),
        stagnation_counter=0,
    )


def test_migrate_top_k_selection() -> None:
    # Source island A: F = [10, 1, 5, 3, 20]; top-2 are indices 1 (F=1) and 3 (F=3)
    X_a = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    F_a = np.array([10.0, 1.0, 5.0, 3.0, 20.0])
    # Destination B: F = [100, 50, 200, 300, 75]; worst-2 are indices 3 (F=300) and 2 (F=200)
    X_b = np.array([[1.1], [1.2], [1.3], [1.4], [1.5]])
    F_b = np.array([100.0, 50.0, 200.0, 300.0, 75.0])
    islands = [_make_island("A", X_a, F_a), _make_island("B", X_b, F_b)]

    rng = np.random.default_rng(42)
    events = migrate(islands, k_top=2, current_gen=10, rng=rng)

    # B receives 2 from A (no self-migration), A receives 2 from B.
    assert len(events) == 4
    a_to_b = [e for e in events if e.src_island == "A" and e.dst_island == "B"]
    assert len(a_to_b) == 2
    # B's worst slots (sorted by ascending F) are 3 then 2; migrants overwrite them.
    new_F_b = islands[1].algorithm.pop.get("F").flatten()
    # B's slots 3 and 2 should now hold the values from A's top-2 (F=1.0 and F=3.0)
    migrant_F_values = sorted({float(new_F_b[3]), float(new_F_b[2])})
    assert migrant_F_values == [1.0, 3.0]


def test_migrate_no_self_migration() -> None:
    X = np.random.default_rng(0).random((10, 4))
    F = np.arange(10, dtype=float)
    islands = [
        _make_island("PSO", X.copy(), F.copy()),
        _make_island("GA", X.copy(), F.copy()),
        _make_island("DE", X.copy(), F.copy()),
    ]
    events = migrate(islands, k_top=3, current_gen=5, rng=np.random.default_rng(0))
    for e in events:
        assert e.src_island != e.dst_island


def test_migrate_determinism_with_fixed_rng() -> None:
    def run_once() -> list[MigrationEvent]:
        X = np.linspace(0.0, 1.0, 20).reshape(5, 4)
        F = np.array([2.0, 1.0, 3.0, 5.0, 4.0])
        islands = [
            _make_island("A", X.copy(), F.copy()),
            _make_island("B", X.copy() + 0.1, F.copy() + 10.0),
            _make_island("C", X.copy() + 0.2, F.copy() + 20.0),
        ]
        return migrate(islands, k_top=2, current_gen=7, rng=np.random.default_rng(123))

    e1, e2 = run_once(), run_once()
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2, strict=True):
        assert a.src_island == b.src_island
        assert a.dst_island == b.dst_island
        assert a.slot_idx == b.slot_idx
        assert a.F_migrant == b.F_migrant


def test_migrate_three_islands_total_event_count() -> None:
    # With 3 islands and k_top=3: each destination receives top-3 from each of 2
    # sources = 6 migrants. Total events = 3 * 6 = 18.
    X = np.random.default_rng(0).random((10, 4))
    F = np.arange(10, dtype=float)
    islands = [
        _make_island("PSO", X.copy(), F.copy()),
        _make_island("GA", X.copy(), F.copy() + 100.0),
        _make_island("DE", X.copy(), F.copy() + 200.0),
    ]
    events = migrate(islands, k_top=3, current_gen=1, rng=np.random.default_rng(0))
    assert len(events) == 18


def test_migrate_logs_f_displaced() -> None:
    X = np.zeros((5, 2))
    F_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    F_b = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    islands = [_make_island("A", X.copy(), F_a), _make_island("B", X.copy(), F_b)]
    events = migrate(islands, k_top=2, current_gen=0, rng=np.random.default_rng(0))
    # Migrants A->B replaced the 2 worst of B (originally F=500, F=400).
    a_to_b = [e for e in events if e.src_island == "A" and e.dst_island == "B"]
    f_displaced_set = sorted(e.F_displaced for e in a_to_b)
    assert f_displaced_set == [400.0, 500.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: 5 failures — `ImportError` from `aerocapture.training.island_model`.

- [ ] **Step 3: Create `island_model.py` with `Island`, `MigrationEvent`, and `migrate()`**

Create `src/python/aerocapture/training/island_model.py`:

```python
"""Three-island PSO/GA/DE evolutionary trainer with episodic migration.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm


@dataclass
class MigrationEvent:
    gen: int
    src_island: str
    dst_island: str
    slot_idx: int
    F_migrant: float
    F_displaced: float


@dataclass
class Island:
    """One algorithm-population pair in the 3-island model.

    `algorithm` is a pymoo Algorithm (PSO/GA/DE) whose `.pop` Population is
    mutated in-place by migration.
    """

    name: str
    algorithm: Any  # pymoo Algorithm or test stand-in (must expose .pop)
    last_validated_individual: npt.NDArray[np.float64] | None = None
    best_overall_individual: npt.NDArray[np.float64] | None = None
    best_overall_cost: float = float("inf")
    best_val_cost: float = float("inf")
    stagnation_counter: int = 0


def migrate(
    islands: list[Island],
    k_top: int,
    current_gen: int,
    rng: np.random.Generator,
    velocity_scale: float = 0.05,
) -> list[MigrationEvent]:
    """Apply full-mesh top-k -> worst-(k * (n_islands - 1)) migration in-place.

    Snapshots emigrants from every island BEFORE any in-place replacement
    (so a destination's incoming pool is not corrupted by its own outgoing).
    For PSO destinations, also resets V / pbest / pbest_F for the new slots
    via `inject_into_pso`.
    """
    # 1. Snapshot top-k emigrants from each island under current F.
    emigrants: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = {}
    for src in islands:
        F_src = src.algorithm.pop.get("F").flatten()
        top_idx = np.argsort(F_src, kind="stable")[:k_top]
        emigrants[src.name] = [
            (src.algorithm.pop[int(i)].X.copy(), float(F_src[int(i)]))
            for i in top_idx
        ]

    # 2. For each destination, apply replacements from all other islands.
    events: list[MigrationEvent] = []
    for dst in islands:
        incoming: list[tuple[npt.NDArray[np.float64], float, str]] = []
        for src in islands:
            if src.name == dst.name:
                continue
            for X_em, F_em in emigrants[src.name]:
                incoming.append((X_em, F_em, src.name))

        n_incoming = len(incoming)
        F_dst = dst.algorithm.pop.get("F").flatten()
        worst_slots = np.argsort(F_dst, kind="stable")[-n_incoming:]

        for slot_i, (X_new, F_new, src_name) in zip(worst_slots, incoming, strict=True):
            slot = int(slot_i)
            F_displaced = float(F_dst[slot])
            dst.algorithm.pop[slot].X = X_new
            dst.algorithm.pop[slot].F = np.array([F_new])

            if isinstance(dst.algorithm, PSO):
                inject_into_pso(
                    dst.algorithm,
                    slot=slot,
                    X=X_new,
                    F=F_new,
                    velocity_scale=velocity_scale,
                    rng=rng,
                )

            events.append(
                MigrationEvent(
                    gen=current_gen,
                    src_island=src_name,
                    dst_island=dst.name,
                    slot_idx=slot,
                    F_migrant=F_new,
                    F_displaced=F_displaced,
                )
            )

    return events


def inject_into_pso(
    algorithm: Algorithm,
    slot: int,
    X: npt.NDArray[np.float64],
    F: float,
    velocity_scale: float,
    rng: np.random.Generator,
) -> None:
    """Write a fresh velocity and reset pbest/pbest_F for a migrant PSO slot.

    Zero velocity is a trap: a collapsed swarm's gbest will pull the migrant
    in within 2-3 ticks. A small uniform velocity gives the migrant a few
    ticks to evaluate its neighborhood and (if better than gbest) become the
    new attractor itself.
    """
    n_params = X.shape[0]
    V = algorithm.pop.get("V")
    pbest = algorithm.pop.get("pbest")
    pbest_F = algorithm.pop.get("pbest_F")

    V[slot] = rng.uniform(-velocity_scale, velocity_scale, size=n_params)
    pbest[slot] = X.copy()
    pbest_F[slot] = np.array([F])

    algorithm.pop.set("V", V)
    algorithm.pop.set("pbest", pbest)
    algorithm.pop.set("pbest_F", pbest_F)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes (incl. the 5 from Task 1 + 5 new ones from Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): pure migrate() and inject_into_pso() functions

Adds Island/MigrationEvent dataclasses and the pure-function migration
primitive (full-mesh top-k -> worst-(k*(N-1)) replacement, PSO-aware
velocity injection). No outer-loop integration yet."
```

---

## Task 3: `inject_into_pso` unit tests with a real PSO instance

**Files:**
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing tests for `inject_into_pso` shape / value-range / persistence**

Append to `tests/test_island_model.py`:

```python
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem


class _UnitCubeProblem(Problem):
    """Trivial problem: f(x) = sum(x). 4 dims, [0,1] bounds, single objective."""

    def __init__(self) -> None:
        super().__init__(n_var=4, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        out["F"] = X.sum(axis=1).reshape(-1, 1)


def _make_real_pso() -> PSO:
    """Construct and run-once a small pymoo PSO so its pop has V/pbest/pbest_F populated."""
    problem = _UnitCubeProblem()
    pso = PSO(pop_size=10, w=0.7, c1=1.5, c2=1.5)
    pso.setup(problem, seed=0)
    pso.next()  # advance one gen so V/pbest fields exist on every individual
    return pso


def test_inject_into_pso_writes_velocity_in_range() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=3, X=X_new, F=0.42, velocity_scale=0.05, rng=rng)

    V = pso.pop.get("V")
    assert V[3].shape == (4,)
    assert np.all(np.abs(V[3]) <= 0.05)


def test_inject_into_pso_sets_pbest_to_current_position() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    X_new = np.array([0.11, 0.22, 0.33, 0.44])
    inject_into_pso(pso, slot=5, X=X_new, F=1.23, velocity_scale=0.05, rng=rng)

    pbest = pso.pop.get("pbest")
    pbest_F = pso.pop.get("pbest_F")
    np.testing.assert_array_equal(pbest[5], X_new)
    assert float(pbest_F[5][0]) == 1.23


def test_inject_into_pso_does_not_corrupt_other_slots() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    V_before = pso.pop.get("V").copy()
    pbest_before = pso.pop.get("pbest").copy()

    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=7, X=X_new, F=0.42, velocity_scale=0.05, rng=rng)

    V_after = pso.pop.get("V")
    pbest_after = pso.pop.get("pbest")
    for i in range(10):
        if i == 7:
            continue
        np.testing.assert_array_equal(V_before[i], V_after[i])
        np.testing.assert_array_equal(pbest_before[i], pbest_after[i])


def test_inject_into_pso_velocity_seeded_rng_deterministic() -> None:
    pso = _make_real_pso()
    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=0, X=X_new, F=0.0, velocity_scale=0.05,
                    rng=np.random.default_rng(42))
    V_first = pso.pop.get("V")[0].copy()

    pso2 = _make_real_pso()
    inject_into_pso(pso2, slot=0, X=X_new, F=0.0, velocity_scale=0.05,
                    rng=np.random.default_rng(42))
    V_second = pso2.pop.get("V")[0]
    np.testing.assert_array_equal(V_first, V_second)
```

- [ ] **Step 2: Run tests to verify they pass**

The implementation from Task 2 should already pass these. Run:

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes. (If `_UnitCubeProblem` has a pymoo `_evaluate` API mismatch, fix the signature here, not in `inject_into_pso`.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_island_model.py
git commit -m "test(islands): inject_into_pso with real pymoo PSO instance

Verifies velocity is in [-velocity_scale, velocity_scale] per dim,
pbest is reset to migrant position, other slots are untouched, and
RNG seeding is deterministic."
```

---

## Task 4: `IslandModel` constructor + `final_eval` + winner selection

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing tests for `IslandModel.__init__` and `final_eval`**

Append to `tests/test_island_model.py`:

```python
from aerocapture.training.island_model import IslandModel
from aerocapture.training.optimizer import (
    DESettings,
    GASettings,
    IslandSettings,
    OptimizerConfig,
    PSOSettings,
)


class _MockProblem:
    """Stand-in for AerocaptureProblem that does deterministic per-seed eval."""

    def __init__(self, n_var: int = 4) -> None:
        self.n_var = n_var

    def evaluate_individual_per_seed(
        self, X: np.ndarray, seeds: list[int]
    ) -> np.ndarray:
        # F = sum(X) + 0.01 * seed_idx (so different islands get different rms).
        base = float(np.sum(X))
        return np.array([base + 0.01 * s for s in seeds], dtype=np.float64)

    # AerocaptureProblem also exposes these — IslandModel.__init__ may call them.
    n_obj = 1
    n_ieq_constr = 0
    n_eq_constr = 0
    xl = None
    xu = None


def _make_islands_cfg() -> OptimizerConfig:
    return OptimizerConfig(
        algorithm="islands",
        seed_strategy="fixed",
        n_pop=8,
        n_gen=5,
        training_n_sims=2,
        validation_n_sims=4,
        ga=GASettings(),
        pso=PSOSettings(),
        de=DESettings(),
        islands=IslandSettings(k_period=2, k_top=2),
    )


def test_island_model_init_creates_three_named_islands() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg,
        problem=problem,
        n_params=4,
        validation_seeds=[100, 101, 102, 103],
        final_eval_seeds=[200, 201, 202, 203],
        base_mc_seed=42,
        rng=np.random.default_rng(0),
    )
    names = [i.name for i in model.islands]
    assert names == ["pso", "ga", "de"]


def test_island_model_final_eval_picks_lowest_rms_winner() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg,
        problem=problem,
        n_params=4,
        validation_seeds=[100, 101, 102, 103],
        final_eval_seeds=[200, 201, 202, 203],
        base_mc_seed=42,
        rng=np.random.default_rng(0),
    )
    # Hand-set each island's best_overall_individual so final_eval has work to do.
    model.islands[0].best_overall_individual = np.array([0.1, 0.1, 0.1, 0.1])  # sum=0.4
    model.islands[1].best_overall_individual = np.array([0.5, 0.5, 0.5, 0.5])  # sum=2.0
    model.islands[2].best_overall_individual = np.array([0.2, 0.2, 0.2, 0.2])  # sum=0.8

    results = model.final_eval()
    assert len(results) == 3
    # Winner is the island with lowest rms (which monotonically follows base sum).
    rms_by_island = {r["island"]: r["rms"] for r in results}
    assert rms_by_island["pso"] < rms_by_island["de"] < rms_by_island["ga"]


def test_island_model_final_eval_skips_islands_without_best() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    model.islands[0].best_overall_individual = np.array([0.1, 0.2, 0.3, 0.4])
    # ga and de have no best_overall — they should be skipped.
    results = model.final_eval()
    assert len(results) == 1
    assert results[0]["island"] == "pso"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py::test_island_model_init_creates_three_named_islands tests/test_island_model.py::test_island_model_final_eval_picks_lowest_rms_winner tests/test_island_model.py::test_island_model_final_eval_skips_islands_without_best -v
```

Expected: 3 failures — `ImportError: IslandModel`.

- [ ] **Step 3: Add `IslandModel` constructor and `final_eval` to `island_model.py`**

In `src/python/aerocapture/training/island_model.py`, add after `inject_into_pso`:

```python
from copy import deepcopy

from aerocapture.training.optimizer import OptimizerConfig, create_algorithm


_ISLAND_NAMES = ("pso", "ga", "de")


def _build_island(
    name: str,
    config: OptimizerConfig,
    n_params: int,
) -> Island:
    sub_config = deepcopy(config)
    sub_config.algorithm = name
    algorithm = create_algorithm(sub_config, n_params=n_params)
    return Island(name=name, algorithm=algorithm)


class IslandModel:
    """Owns 3 islands (PSO, GA, DE) and the migration / validation / final-eval flow."""

    def __init__(
        self,
        config: OptimizerConfig,
        problem: Any,
        n_params: int,
        validation_seeds: list[int],
        final_eval_seeds: list[int],
        base_mc_seed: int,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.problem = problem
        self.n_params = n_params
        self.validation_seeds = validation_seeds
        self.final_eval_seeds = final_eval_seeds
        self.base_mc_seed = base_mc_seed
        self.rng = rng
        self.islands: list[Island] = [
            _build_island(name, config, n_params) for name in _ISLAND_NAMES
        ]
        self.migration_log: list[MigrationEvent] = []

    def final_eval(self) -> list[dict[str, Any]]:
        """Re-evaluate each island's best_overall on the reserved final-eval seeds.

        Returns one record per island that had a validated best, sorted by rms ascending.
        The lowest-rms record is the winner. Caller writes best_model.json /
        best_params.json from `winner["X"]`.
        """
        results: list[dict[str, Any]] = []
        for island in self.islands:
            if island.best_overall_individual is None:
                continue
            costs = self.problem.evaluate_individual_per_seed(
                island.best_overall_individual,
                self.final_eval_seeds,
            )
            rms = float(np.sqrt(np.mean(costs ** 2)))
            results.append({
                "island": island.name,
                "X": island.best_overall_individual.copy(),
                "rms": rms,
                "mean": float(np.mean(costs)),
                "p95": float(np.percentile(costs, 95)),
                "capture_rate": float(np.mean(np.asarray(costs) < 3000.0)),
                "n_sims": len(self.final_eval_seeds),
            })
        results.sort(key=lambda r: r["rms"])
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): IslandModel constructor and final_eval winner selection

IslandModel owns 3 pymoo Algorithm instances (PSO/GA/DE) sharing a single
problem instance. final_eval() re-runs each island's best_overall on the
reserved final-eval seeds and returns records sorted by rms ascending."
```

---

## Task 5: `IslandModel.step` — advance + migrate + validate

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing test for `step()` advancing all islands and applying migration at gen K**

Append to `tests/test_island_model.py`:

```python
def test_island_model_step_advances_all_three_islands() -> None:
    """One step() call must invoke .next() on each island's algorithm."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()  # real pymoo problem so .next() works
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)

    # n_gen is set to 5; setup_initial_eval is needed before stepping.
    model.step(current_gen=0)
    # Each island should have a populated pop.
    for island in model.islands:
        assert island.algorithm.pop is not None
        assert len(island.algorithm.pop) == cfg.n_pop


def test_island_model_step_fires_migration_at_k_period() -> None:
    cfg = _make_islands_cfg()
    cfg.islands.k_period = 2  # migrate every 2 gens
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)

    # gens 0, 1: no migration. gen 2: migration fires.
    model.step(0); model.step(1)
    assert len(model.migration_log) == 0
    model.step(2)
    # k_top=2 with 3 islands -> 3 * 2 * 2 = 12 events.
    assert len(model.migration_log) == 12


def test_island_model_step_disabled_migration_never_fires() -> None:
    cfg = _make_islands_cfg()
    cfg.islands.enabled = False
    cfg.islands.k_period = 1  # would migrate every gen if enabled
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    for g in range(5):
        model.step(g)
    assert model.migration_log == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py::test_island_model_step_advances_all_three_islands tests/test_island_model.py::test_island_model_step_fires_migration_at_k_period tests/test_island_model.py::test_island_model_step_disabled_migration_never_fires -v
```

Expected: 3 failures — `AttributeError: 'IslandModel' has no attribute 'step'`.

- [ ] **Step 3: Add `step()` to `IslandModel`**

In `IslandModel`, add:

```python
    def step(self, current_gen: int) -> list[MigrationEvent]:
        """Advance every island one generation, then maybe migrate.

        Returns the migration events from this gen (empty list if no migration).
        Validation is intentionally separate — see `validate_each`.
        """
        # 1. Advance each island sequentially. Rayon parallelism inside each
        #    algorithm.next() saturates the CPU; three sequential 64-individual
        #    batches dominate the per-gen wall time.
        for island in self.islands:
            island.algorithm.next()

        # 2. Migration step: every k_period gens, never at gen 0.
        events: list[MigrationEvent] = []
        if (
            self.config.islands.enabled
            and current_gen > 0
            and current_gen % self.config.islands.k_period == 0
        ):
            events = migrate(
                self.islands,
                k_top=self.config.islands.k_top,
                current_gen=current_gen,
                rng=self.rng,
                velocity_scale=self.config.islands.pso_inject_velocity_scale,
            )
            self.migration_log.extend(events)
        return events
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): IslandModel.step() orchestrator

Advances each island via algorithm.next(), then applies migration every
k_period gens (skip at gen 0, skip when enabled=false). Migration events
are appended to self.migration_log."
```

---

## Task 6: Per-island validation gate (`validate_each`)

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing test for identity-trigger validation**

Append to `tests/test_island_model.py`:

```python
def test_validate_each_fires_only_when_argmin_changes() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    # Advance once so each island's pop has valid F.
    model.step(current_gen=0)

    # First call: every island's argmin differs from None -> all 3 validate.
    metrics = model.validate_each(current_gen=0)
    assert len(metrics) == 3
    for m in metrics:
        assert m["validated"] is True

    # Second call without changing pop: argmin unchanged -> no validation.
    metrics2 = model.validate_each(current_gen=1)
    assert all(m["validated"] is False for m in metrics2)


def test_validate_each_promotes_best_overall_on_rms_improvement() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[1, 2, 3], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    # Run first validation.
    model.validate_each(current_gen=0)
    initial_best = [
        island.best_overall_individual.copy() if island.best_overall_individual is not None else None
        for island in model.islands
    ]
    initial_costs = [island.best_val_cost for island in model.islands]
    assert all(c < float("inf") for c in initial_costs)
    assert all(b is not None for b in initial_best)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py::test_validate_each_fires_only_when_argmin_changes tests/test_island_model.py::test_validate_each_promotes_best_overall_on_rms_improvement -v
```

Expected: 2 failures — `AttributeError: 'IslandModel' has no attribute 'validate_each'`.

- [ ] **Step 3: Add `validate_each` to `IslandModel`**

In `IslandModel`:

```python
    def validate_each(self, current_gen: int) -> list[dict[str, Any]]:
        """Run identity-trigger validation per island.

        For each island, if its argmin differs from `last_validated_individual`,
        run validation on the reserved validation seeds; promote `best_overall_*`
        if val_rms < best_val_cost. Returns one summary dict per island for
        logging (includes a "validated" bool flag).
        """
        results: list[dict[str, Any]] = []
        for island in self.islands:
            pop = island.algorithm.pop
            X = pop.get("X")
            F = pop.get("F").flatten()
            argmin_idx = int(np.argmin(F))
            argmin_X = X[argmin_idx].copy()
            argmin_cost = float(F[argmin_idx])

            unchanged = (
                island.last_validated_individual is not None
                and np.array_equal(argmin_X, island.last_validated_individual)
            )
            if unchanged:
                island.stagnation_counter += 1
                results.append({
                    "island": island.name,
                    "validated": False,
                    "promoted": False,
                    "argmin_train_cost": argmin_cost,
                    "stagnation": island.stagnation_counter,
                })
                continue

            val_costs = self.problem.evaluate_individual_per_seed(
                argmin_X, self.validation_seeds,
            )
            val_rms = float(np.sqrt(np.mean(val_costs ** 2)))
            island.last_validated_individual = argmin_X

            promoted = val_rms < island.best_val_cost
            if promoted:
                island.best_val_cost = val_rms
                island.best_overall_individual = argmin_X.copy()
                island.best_overall_cost = argmin_cost
                island.stagnation_counter = 0
            else:
                island.stagnation_counter += 1

            results.append({
                "island": island.name,
                "validated": True,
                "promoted": promoted,
                "argmin_train_cost": argmin_cost,
                "val_rms": val_rms,
                "val_mean": float(np.mean(val_costs)),
                "val_p95": float(np.percentile(val_costs, 95)),
                "val_capture_rate": _capture_rate(np.asarray(val_costs)),
                "stagnation": island.stagnation_counter,
            })
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): per-island identity-trigger validation gate

validate_each runs validation only when an island's argmin differs from
its last_validated_individual. Promotes best_overall_* on val_rms
improvement; returns per-island summary records for logging."
```

---

## Task 7: Adaptive seed curator pooled-top-K integration

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing test for pooled top-K curation**

Append to `tests/test_island_model.py`:

```python
def test_pool_top_k_across_islands_unions_populations() -> None:
    """`pool_top_k_X` returns the K lowest-F individuals from the UNION of all
    island populations (search-space-wide signal, no per-island silo)."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    pooled = model.pool_top_k_X(k=5)
    assert pooled.shape == (5, 4)
    # The pooled cost values must be monotonically <= the worst individual in any
    # single island (by definition of pool-then-rank-then-take-top-K).
    pooled_costs = np.asarray([float(np.sum(x)) for x in pooled])
    assert pooled_costs.shape == (5,)
    assert np.all(pooled_costs == np.sort(pooled_costs))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_island_model.py::test_pool_top_k_across_islands_unions_populations -v
```

Expected: failure — `AttributeError: 'IslandModel' has no attribute 'pool_top_k_X'`.

- [ ] **Step 3: Add `pool_top_k_X` to `IslandModel`**

In `IslandModel`:

```python
    def pool_top_k_X(self, k: int) -> npt.NDArray[np.float64]:
        """Concatenate all island populations and return the K lowest-F rows.

        Used by the adaptive seed curator: the cost CDF is a search-space-wide
        signal, so the curator probes a representative slice across islands
        rather than per-island top-K (which would silo by algorithm).
        """
        all_X = []
        all_F = []
        for island in self.islands:
            all_X.append(island.algorithm.pop.get("X"))
            all_F.append(island.algorithm.pop.get("F").flatten())
        X = np.concatenate(all_X, axis=0)
        F = np.concatenate(all_F, axis=0)
        k = min(k, F.shape[0])
        top_idx = np.argsort(F, kind="stable")[:k]
        return X[top_idx]

    def re_evaluate_all_populations(self) -> None:
        """Re-evaluate every island's algorithm.pop under the current seed list.

        Called when the shared seed list changes (rotating strategy or adaptive
        curation). Mirrors the pre-`next()` re-eval block in the single-algorithm
        path in train.py.
        """
        for island in self.islands:
            if island.algorithm.pop is None:
                continue
            parent_X = island.algorithm.pop.get("X")
            fresh_F = self.problem._run_batch(parent_X)
            island.algorithm.pop.set("F", fresh_F.reshape(-1, 1))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): pool_top_k_X and re_evaluate_all_populations

pool_top_k_X feeds the adaptive seed curator with a search-space-wide
top-K slice from the UNION of all 3 populations. re_evaluate_all_populations
mirrors the pre-next() re-eval that the single-algorithm path runs when
seeds change."
```

---

## Task 8: Logger gains `island_name` field

**Files:**
- Modify: `src/python/aerocapture/training/logger.py`
- Test: extend `tests/test_logger.py` if it exists; otherwise inline in `tests/test_island_model.py`

- [ ] **Step 1: Inspect current `TrainingLogger.log_generation` signature**

```bash
uv run python -c "from aerocapture.training.logger import TrainingLogger; import inspect; print(inspect.signature(TrainingLogger.log_generation))"
```

Note the existing parameters (X, costs, best_individual, decode_fn, validation, improved, etc.).

- [ ] **Step 2: Write failing test for `island_name` field in JSONL record**

Append to `tests/test_island_model.py`:

```python
import json
import tempfile
from pathlib import Path

from aerocapture.training.logger import TrainingLogger


def test_logger_writes_island_name_field_when_provided() -> None:
    with tempfile.TemporaryDirectory() as td:
        logger = TrainingLogger(scheme="islands", run=0, output_dir=Path(td), config_hash="dummy")
        X = np.zeros((4, 2))
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        best = np.array([0.5, 0.5])
        logger.log_generation(
            generation=0,
            population=X,
            costs=costs,
            best_individual=best,
            decode_fn=None,
            island_name="pso",
        )
        jsonl_files = list(Path(td).glob("*.jsonl"))
        assert len(jsonl_files) == 1
        record = json.loads(jsonl_files[0].read_text().strip().splitlines()[-1])
        assert record["island_name"] == "pso"


def test_logger_omits_island_name_when_not_provided() -> None:
    with tempfile.TemporaryDirectory() as td:
        logger = TrainingLogger(scheme="ftc", run=0, output_dir=Path(td), config_hash="dummy")
        logger.log_generation(
            generation=0,
            population=np.zeros((4, 2)),
            costs=np.array([1.0, 2.0, 3.0, 4.0]),
            best_individual=np.array([0.5, 0.5]),
            decode_fn=None,
        )
        jsonl_files = list(Path(td).glob("*.jsonl"))
        record = json.loads(jsonl_files[0].read_text().strip().splitlines()[-1])
        # No island_name key when not provided (backwards compatibility).
        assert "island_name" not in record or record["island_name"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py::test_logger_writes_island_name_field_when_provided tests/test_island_model.py::test_logger_omits_island_name_when_not_provided -v
```

Expected: 1 failure (`island_name` not accepted by `log_generation`), 1 may pass depending on existing JSON dict construction.

- [ ] **Step 4: Add `island_name: str | None = None` parameter to `log_generation`**

In `src/python/aerocapture/training/logger.py`, edit `log_generation`:

```python
    def log_generation(
        self,
        generation: int,
        population: npt.NDArray[np.float64],
        costs: npt.NDArray[np.float64],
        best_individual: npt.NDArray[np.float64] | None,
        decode_fn: Any,
        validation: dict | None = None,
        improved: bool = False,
        weight_stats: list[dict] | None = None,
        island_name: str | None = None,   # NEW
        **extra: Any,
    ) -> None:
        # ... existing body up to the point where `record` dict is built ...
        record: dict[str, Any] = {
            # ... existing fields ...
        }
        if island_name is not None:
            record["island_name"] = island_name
        # ... rest of body (writing JSONL line) unchanged ...
```

Locate the existing `record = {...}` dict assembly and insert the `island_name` field conditionally. Do not rename any existing field.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
uv run pytest tests/ -k "logger" -v
```

Expected: all passes (incl. no regression in existing logger tests).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/logger.py tests/test_island_model.py
git commit -m "feat(islands): logger writes island_name field when provided

Backwards-compatible: when island_name is None (default), the field is
omitted from the JSONL record. When set, charts/report.py can filter
per-island records via this field."
```

---

## Task 9: Checkpoint v2 round-trip

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py`
- Modify: `tests/test_island_model.py`

- [ ] **Step 1: Write failing test for `IslandModel.checkpoint` and `from_checkpoint` round-trip**

Append to `tests/test_island_model.py`:

```python
def test_island_model_checkpoint_roundtrip() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)
    model.validate_each(current_gen=0)
    # Force a fake migration event into the log.
    model.migration_log.append(MigrationEvent(
        gen=1, src_island="ga", dst_island="pso", slot_idx=0,
        F_migrant=0.1, F_displaced=10.0,
    ))

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00005.npz"
        model.checkpoint(ckpt_path, generation=5)
        assert ckpt_path.exists()

        # Build a fresh model and restore from checkpoint.
        restored = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100, 101], final_eval_seeds=[200, 201],
            base_mc_seed=42, rng=np.random.default_rng(99),
        )
        for island in restored.islands:
            island.algorithm.setup(problem, seed=0)
        restored.from_checkpoint(ckpt_path)

    # Verify per-island state matches.
    for orig, rest in zip(model.islands, restored.islands, strict=True):
        assert orig.name == rest.name
        assert orig.best_val_cost == rest.best_val_cost
        assert orig.stagnation_counter == rest.stagnation_counter
        if orig.best_overall_individual is None:
            assert rest.best_overall_individual is None
        else:
            np.testing.assert_array_equal(
                orig.best_overall_individual, rest.best_overall_individual,
            )
    assert len(restored.migration_log) == 1
    assert restored.migration_log[0].src_island == "ga"


def test_island_model_resume_preserves_best_overall_per_island() -> None:
    """Regression guard: cross-gen training-cost incomparability rule must apply
    per-island. Restoring a checkpoint must NOT overwrite best_overall_* with
    the resumed population's gen-0 argmin (see project memory
    project_resume_cost_incomparability)."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    # Stamp each island's best_overall as a sentinel different from any pop member.
    sentinel = np.array([0.99, 0.99, 0.99, 0.99])
    for island in model.islands:
        island.best_overall_individual = sentinel.copy()
        island.best_val_cost = 0.123
        island.best_overall_cost = 0.456

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00010.npz"
        model.checkpoint(ckpt_path, generation=10)

        restored = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100], final_eval_seeds=[200],
            base_mc_seed=42, rng=np.random.default_rng(0),
        )
        for island in restored.islands:
            island.algorithm.setup(problem, seed=0)
        # Advance to a fresh pop with potentially-better argmin.
        restored.step(current_gen=0)
        restored.from_checkpoint(ckpt_path)

    # The sentinel must survive across the resume; the restored model must NOT
    # have replaced it with the gen-0 argmin.
    for island in restored.islands:
        np.testing.assert_array_equal(island.best_overall_individual, sentinel)
        assert island.best_val_cost == 0.123
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_island_model.py::test_island_model_checkpoint_roundtrip tests/test_island_model.py::test_island_model_resume_preserves_best_overall_per_island -v
```

Expected: 2 failures — `AttributeError: 'IslandModel' has no attribute 'checkpoint'`.

- [ ] **Step 3: Implement `checkpoint()` and `from_checkpoint()`**

In `IslandModel`:

```python
    def checkpoint(self, path: Path, generation: int) -> None:
        """Write a v2 atomic .npz checkpoint.

        Atomicity: writes to a tempfile in the same directory, then renames.
        Single file holds all 3 islands' state + migration log + RNG state.
        """
        import pickle  # noqa: PLC0415  -- local import keeps top of file slim

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.tmp")

        island_states = []
        for island in self.islands:
            pop = island.algorithm.pop
            island_states.append({
                "name": island.name,
                "pop_X": pop.get("X") if pop is not None else None,
                "pop_F": pop.get("F") if pop is not None else None,
                "pop_V": pop.get("V") if pop is not None and pop.get("V") is not None else None,
                "pop_pbest": pop.get("pbest") if pop is not None and pop.get("pbest") is not None else None,
                "pop_pbest_F": pop.get("pbest_F") if pop is not None and pop.get("pbest_F") is not None else None,
                "last_validated_individual": island.last_validated_individual,
                "best_overall_individual": island.best_overall_individual,
                "best_overall_cost": island.best_overall_cost,
                "best_val_cost": island.best_val_cost,
                "stagnation_counter": island.stagnation_counter,
            })

        np.savez(
            tmp,
            version=2,
            generation=generation,
            base_mc_seed=self.base_mc_seed,
            island_states=np.array(pickle.dumps(island_states), dtype=object),
            migration_log=np.array(pickle.dumps(self.migration_log), dtype=object),
            rng_state=np.array(pickle.dumps(self.rng.bit_generator.state), dtype=object),
        )
        tmp.rename(path)

    def from_checkpoint(self, path: Path) -> int:
        """Restore from a v2 checkpoint. Returns the generation at which it was saved.

        IMPORTANT: per-island best_overall_* are restored verbatim. The
        resumed population's gen-0 argmin must NOT be allowed to overwrite them
        (cross-gen training-cost incomparability under adaptive/rotating
        seeds — see project memory project_resume_cost_incomparability).
        The caller's outer loop is responsible for not re-running the gen-0
        init-best block when best_overall_individual is not None.
        """
        import pickle  # noqa: PLC0415

        with np.load(path, allow_pickle=True) as data:
            version = int(data["version"])
            if version != 2:
                raise ValueError(f"checkpoint version {version} unsupported; expected 2")
            generation = int(data["generation"])
            base_mc_seed = int(data["base_mc_seed"])
            island_states = pickle.loads(data["island_states"].item())
            migration_log = pickle.loads(data["migration_log"].item())
            rng_state = pickle.loads(data["rng_state"].item())

        assert base_mc_seed == self.base_mc_seed, (
            f"checkpoint base_mc_seed {base_mc_seed} != current {self.base_mc_seed}"
        )

        from pymoo.core.population import Population  # noqa: PLC0415

        for island, state in zip(self.islands, island_states, strict=True):
            assert island.name == state["name"], (
                f"checkpoint island order mismatch: {island.name} != {state['name']}"
            )
            island.last_validated_individual = state["last_validated_individual"]
            island.best_overall_individual = state["best_overall_individual"]
            island.best_overall_cost = float(state["best_overall_cost"])
            island.best_val_cost = float(state["best_val_cost"])
            island.stagnation_counter = int(state["stagnation_counter"])

            if state["pop_X"] is not None:
                pop = Population.new("X", state["pop_X"])
                pop.set("F", state["pop_F"])
                if state["pop_V"] is not None:
                    pop.set("V", state["pop_V"])
                if state["pop_pbest"] is not None:
                    pop.set("pbest", state["pop_pbest"])
                if state["pop_pbest_F"] is not None:
                    pop.set("pbest_F", state["pop_pbest_F"])
                island.algorithm.pop = pop

        self.migration_log = migration_log
        self.rng.bit_generator.state = rng_state
        return generation
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_island_model.py -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_island_model.py
git commit -m "feat(islands): checkpoint v2 atomic round-trip with resume safety

Single .npz holds all 3 islands' state + migration log + RNG state. Write
goes through tempfile + rename for atomicity. from_checkpoint restores
per-island best_overall_* VERBATIM — the resumed population's gen-0
argmin must not overwrite them (cross-gen cost-incomparability rule)."
```

---

## Task 10: `train.py` dispatch for `algorithm = "islands"`

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

- [ ] **Step 1: Locate the single-algorithm path in `train.py`**

Read `src/python/aerocapture/training/train.py` lines 900–1100 to understand the existing outer loop. The pattern is:

```python
algorithm = create_algorithm(config.optimizer, n_params=n_params)
# ... initial pop setup ...
algorithm.setup(problem, pop=initial_pop)
# ... gen-0 validation block ...
for gen in range(start_gen, config.optimizer.n_gen):
    # ... seed strategy handling ...
    algorithm.next()
    # ... validation gate, curation, logging, display, checkpoint ...
```

- [ ] **Step 2: Add `algorithm == "islands"` branch BEFORE the single-algorithm `create_algorithm` call**

In `train()` (around line 902 — adjust line number based on current state):

```python
    if config.optimizer.algorithm == "islands":
        return _train_islands(
            config=config,
            save_dir=save_dir,
            problem=problem,
            param_specs=param_specs,
            n_params=n_params,
            pop_array=pop_array,
            pop_costs=pop_costs,
            val_seeds=val_seeds,
            final_eval_seeds=final_eval_seeds,
            base_mc_seed=base_mc_seed,
            excluded_seeds=excluded_seeds,
            rng=rng,
            seed_curator=seed_curator,
            strategy=strategy,
            display=display,
            verbose=verbose,
            start_gen=start_gen,
            best_overall_individual=best_overall_individual,
            best_overall_cost=best_overall_cost,
            best_val_cost=best_val_cost,
            last_validated_individual=last_validated_individual,
            config_hash=config_hash,
            checkpoint_interval=checkpoint_interval,
            toml_abs_path=toml_abs_path,
        )
```

- [ ] **Step 3: Add the `_train_islands` helper function in `train.py`**

After `train()` (or just before — anywhere in the same module), add:

```python
def _train_islands(
    *,
    config: TrainingConfig,
    save_dir: Path,
    problem: AerocaptureProblem,
    param_specs: list[ParamSpec],
    n_params: int,
    pop_array: npt.NDArray[np.float64],
    pop_costs: npt.NDArray[np.float64] | None,
    val_seeds: list[int],
    final_eval_seeds: list[int],
    base_mc_seed: int,
    excluded_seeds: set[int],
    rng: np.random.Generator,
    seed_curator: SeedCurator | None,
    strategy: str,
    display: Any,
    verbose: bool,
    start_gen: int,
    best_overall_individual: npt.NDArray[np.float64] | None,
    best_overall_cost: float,
    best_val_cost: float,
    last_validated_individual: npt.NDArray[np.float64] | None,
    config_hash: str,
    checkpoint_interval: int,
    toml_abs_path: Path | None,
) -> dict[str, Any]:
    """Outer loop for the 3-island PSO/GA/DE trainer.

    Mirrors the single-algorithm path in train() but drives an IslandModel.
    Per-island JSONL records (3 per gen) are written via TrainingLogger with
    the `island_name` field set.
    """
    from aerocapture.training.island_model import IslandModel  # noqa: PLC0415

    island_model = IslandModel(
        config=config.optimizer,
        problem=problem,
        n_params=n_params,
        validation_seeds=val_seeds,
        final_eval_seeds=final_eval_seeds,
        base_mc_seed=base_mc_seed,
        rng=rng,
    )

    # Fan out the (possibly warm-started) initial population to all 3 islands.
    # Each island gets the same starting chromosome but its algorithm's own
    # internal state (e.g. PSO velocity init) is fresh.
    from pymoo.core.population import Population  # noqa: PLC0415
    from pymoo.core.evaluator import Evaluator  # noqa: PLC0415

    for island in island_model.islands:
        init_pop = Population.new("X", pop_array.copy())
        if pop_costs is not None:
            init_pop.set("F", pop_costs.reshape(-1, 1).copy())
        else:
            Evaluator().eval(problem, init_pop)
        island.algorithm.setup(problem, pop=init_pop)

    # Try resume.
    ckpt_files = sorted(save_dir.glob("checkpoint_g*.npz"))
    if ckpt_files:
        resumed_gen, resumed_curator_state = island_model.from_checkpoint(ckpt_files[-1])
        start_gen = resumed_gen + 1
        if resumed_curator_state is not None and seed_curator is not None:
            from aerocapture.training.seed_curator import SeedCurator  # noqa: PLC0415
            seed_curator = SeedCurator.from_dict(
                resumed_curator_state,
                excluded_seeds=seed_curator.excluded_seeds,
                rng=seed_curator.rng,
            )
        if verbose:
            print(f"  Resumed islands from gen {resumed_gen}, continuing from {start_gen}")

    # Logger and decode function (shared across islands).
    decode_fn = None
    if config.guidance_type != "neural_network":
        from aerocapture.training.encoding import decode_normalized  # noqa: PLC0415

        def _decode(x: npt.NDArray[np.float64]) -> dict[str, float]:
            return decode_normalized(x, param_specs)

        decode_fn = _decode

    logger = TrainingLogger(
        scheme=config.guidance_type,
        run=0,
        output_dir=save_dir,
        config_hash=config_hash,
    )

    pending_seed_change = False
    interrupted = False

    with display:
        try:
            for gen in range(start_gen, config.optimizer.n_gen):
                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False

                if strategy == "rotating":
                    fresh = _draw_disjoint_seeds(
                        rng, n=config.optimizer.training_n_sims, excluded=excluded_seeds,
                    )
                    problem.update_seeds(fresh)
                    seeds_changed_this_gen = True
                elif (
                    strategy == "adaptive"
                    and seed_curator is not None
                    and seed_curator.seed_list is None
                ):
                    bootstrap = _draw_disjoint_seeds(
                        rng, n=config.optimizer.training_n_sims, excluded=excluded_seeds,
                    )
                    problem.update_seeds(bootstrap)
                    seeds_changed_this_gen = True

                if seeds_changed_this_gen:
                    island_model.re_evaluate_all_populations()

                # Advance + (maybe) migrate.
                island_model.step(current_gen=gen)

                # Validate.
                val_records = island_model.validate_each(current_gen=gen)

                # Adaptive seed curation: pool top-K across islands.
                if seed_curator is not None:
                    elapsed = gen - seed_curator.last_curation_gen
                    periodic = elapsed >= config.optimizer.seed_pool_interval
                    any_promotion = any(r.get("promoted") for r in val_records)
                    if any_promotion or periodic:
                        k = config.optimizer.curation_top_k
                        top_k_X = island_model.pool_top_k_X(k)
                        new_seeds = seed_curator.curate(problem, top_k_X)
                        seed_curator.last_curation_gen = gen
                        problem.update_seeds(new_seeds)
                        pending_seed_change = True

                # Per-island JSONL records.
                for island, val_rec in zip(island_model.islands, val_records, strict=True):
                    X = island.algorithm.pop.get("X")
                    F = island.algorithm.pop.get("F").flatten()
                    validation_dict = None
                    if val_rec["validated"]:
                        validation_dict = {
                            "rms_cost": val_rec["val_rms"],
                            "mean_cost": val_rec["val_mean"],
                            "p95_cost": val_rec["val_p95"],
                            "capture_rate": val_rec["val_capture_rate"],
                            "n_sims": len(val_seeds),
                        }
                    logger.log_generation(
                        generation=gen,
                        population=X,
                        costs=F,
                        best_individual=island.best_overall_individual,
                        decode_fn=decode_fn,
                        validation=validation_dict,
                        improved=val_rec["promoted"],
                        island_name=island.name,
                    )

                display.update(logger, current_run=0)

                if gen % checkpoint_interval == 0 or gen == config.optimizer.n_gen - 1:
                    island_model.checkpoint(
                        save_dir / f"checkpoint_g{gen:05d}.npz",
                        generation=gen,
                        seed_curator_state=seed_curator.to_dict() if seed_curator is not None else None,
                    )

        except KeyboardInterrupt:
            interrupted = True
            island_model.checkpoint(
                save_dir / f"checkpoint_g{gen:05d}.npz",
                generation=gen,
                seed_curator_state=seed_curator.to_dict() if seed_curator is not None else None,
            )
            if verbose:
                print(f"\n  Interrupted at gen {gen}; checkpoint saved.")

    # Final eval -> winner -> write best_model.json / best_params.json.
    results = island_model.final_eval()
    if not results:
        if verbose:
            print("  No island had a validated best — skipping final-eval / artifact write.")
        return {
            "interrupted": interrupted, "winner": None,
            "results": [], "migration_log": island_model.migration_log,
        }

    winner = results[0]
    if verbose:
        print(
            f"  Winner: {winner['island']} rms={winner['rms']:.4e} "
            f"cap={winner['capture_rate']:.0%}"
        )

    _write_winner_artifacts(
        winner=winner, config=config, save_dir=save_dir,
        param_specs=param_specs, toml_abs_path=toml_abs_path,
    )

    return {
        "interrupted": interrupted,
        "winner": winner,
        "results": results,
        "migration_log": island_model.migration_log,
    }


def _write_winner_artifacts(
    *,
    winner: dict[str, Any],
    config: TrainingConfig,
    save_dir: Path,
    param_specs: list[ParamSpec],
    toml_abs_path: Path | None,
) -> None:
    """Write best_model.json / best_params.json from the winning island's chromosome.

    Mirrors the single-algorithm artifact-write block in train.py lines 370-393.
    """
    import json  # noqa: PLC0415
    from aerocapture.training.encoding import decode_normalized  # noqa: PLC0415

    best_individual = winner["X"]

    if config.guidance_type == "neural_network":
        n_scaff = 17 if config.network.optimize_scaffolding else 0
        n_weights = len(param_specs) - n_scaff
        weights = _decode_nn_weights(
            best_individual[:n_weights], param_specs[:n_weights],
        )
        write_nn_json(
            weights,
            config.network,
            save_dir / "best_model.json",
            input_mask=config.network.input_mask,
            output_param=config.network.output_parameterization,
        )
        # Also write to the deploy path the Rust TOML's `[data] neural_network` points at.
        if toml_abs_path is not None:
            cwd = toml_abs_path.parent
            nn_path = Path(cwd) / config.sim.nn_param_file
            write_nn_json(
                weights,
                config.network,
                nn_path,
                input_mask=config.network.input_mask,
                output_param=config.network.output_parameterization,
            )
        if n_scaff > 0:
            from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS  # noqa: PLC0415

            scaff_params = decode_normalized(
                best_individual[n_weights:], list(_NN_SCAFFOLDING_PARAMS),
            )
            for s in _NN_SCAFFOLDING_PARAMS:
                if s.is_integer and s.name in scaff_params:
                    scaff_params[s.name] = int(round(scaff_params[s.name]))
            with open(save_dir / "best_params.json", "w") as fp:
                json.dump(scaff_params, fp, indent=2)
    else:
        params = decode_normalized(best_individual, param_specs)
        with open(save_dir / "best_params.json", "w") as fp:
            json.dump(params, fp, indent=2)
```

**NOTE for the implementing engineer:** the `_write_winner_artifacts` body must mirror the existing artifact-write path that lives at the END of `train()` (after the main outer loop). Locate that block — search `write_nn_json(` and `best_model.json` in `train.py` — and copy its exact call signature here. Do not invent new arguments.

- [ ] **Step 4: Run existing single-algorithm tests to verify no regression**

```bash
uv run pytest tests/ -k "train and not slow" -v
```

Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(islands): wire algorithm='islands' into train()

Adds _train_islands helper that constructs an IslandModel, runs the
shared outer loop with per-island JSONL records, handles resume from
checkpoint_g*.npz, calls the adaptive seed curator with pooled top-K,
and writes best_model.json / best_params.json from the winning island."
```

---

## Task 11: TOML config — `[optimizer.islands]` defaults and reference training config

**Files:**
- Modify: `configs/training/common.toml`
- Create: `configs/training/msr_aller_islands_train.toml`

- [ ] **Step 1: Add `[optimizer.islands]` defaults to `common.toml`**

Read `configs/training/common.toml` first to find the `[optimizer]` section, then append:

```toml
[optimizer.islands]
enabled = true
k_period = 25
k_top = 3
pso_inject_velocity_scale = 0.05
```

- [ ] **Step 2: Create reference training config**

Create `configs/training/msr_aller_islands_train.toml`:

```toml
base = ["common.toml", "../missions/mars.toml"]

guidance_type = "neural_network"
results_suffix = "islands"

[network]
layer_sizes = [16, 32, 16, 2]
activations = ["swish", "swish", "linear"]

[optimizer]
algorithm = "islands"
seed_strategy = "adaptive"
n_pop = 64
n_gen = 2500
training_n_sims = 20
validation_n_sims = 1000
seed_pool_interval = 50
curation_top_k = 5
curation_sample_size = 1000

[optimizer.islands]
enabled = true
k_period = 25
k_top = 3
pso_inject_velocity_scale = 0.05

[optimizer.pso]
w = 0.7
c1 = 1.5
c2 = 1.5

[optimizer.ga]
crossover_eta = 15.0
mutation_eta = 20.0
mutation_prob = 0.05

[optimizer.de]
variant = "DE/rand/1/bin"
crossover_prob = 0.8
scaling_factor = 0.6
```

- [ ] **Step 3: Verify TOML loads via the existing loader**

```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
from aerocapture.training.optimizer import OptimizerConfig
d = load_toml_with_bases('configs/training/msr_aller_islands_train.toml')
cfg = OptimizerConfig.from_dict(d['optimizer'])
print(f'algorithm={cfg.algorithm}, islands.k_period={cfg.islands.k_period}')
"
```

Expected output: `algorithm=islands, islands.k_period=25`.

- [ ] **Step 4: Commit**

```bash
git add configs/training/common.toml configs/training/msr_aller_islands_train.toml
git commit -m "feat(islands): TOML config defaults and reference training file

Adds [optimizer.islands] defaults to common.toml and a reference
msr_aller_islands_train.toml that uses the same NN architecture as the
existing PSO baseline for a clean A/B comparison."
```

---

## Task 12: Integration smoke test (@slow)

**Files:**
- Create: `tests/test_island_model_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_island_model_smoke.py`:

```python
"""End-to-end smoke test for the 3-island trainer.

Runs 5 generations with k_period=1 (forces migration every gen) on a reduced
architecture, asserts that per-island JSONL records are produced, migration
events fire, the winner is selected, and the resulting best_model.json loads
via the Rust runtime.

Marked @slow because it runs ~50 MC sims per gen via the real simulator.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_islands_smoke_5_gens(tmp_path: Path) -> None:
    config_path = tmp_path / "islands_smoke.toml"
    config_path.write_text("""
base = ["../../../configs/training/common.toml", "../../../configs/missions/mars.toml"]

guidance_type = "neural_network"
results_suffix = "islands_smoke"

[network]
layer_sizes = [16, 8, 2]
activations = ["swish", "linear"]

[optimizer]
algorithm = "islands"
seed_strategy = "fixed"
n_pop = 16
n_gen = 5
training_n_sims = 2
validation_n_sims = 4

[optimizer.islands]
enabled = true
k_period = 1
k_top = 2
pso_inject_velocity_scale = 0.05

[monte_carlo]
seed = 12345
""")

    out_dir = tmp_path / "training_output" / "islands_smoke"
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "aerocapture.training.train",
            str(config_path), "--no-tui", "--skip-report",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"train.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify per-island JSONL records.
    jsonl_files = list(out_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    island_names = {r.get("island_name") for r in records if "island_name" in r}
    assert island_names == {"pso", "ga", "de"}

    # Verify ~3 records per gen (some gen-0 may differ).
    records_by_gen: dict[int, list] = {}
    for r in records:
        records_by_gen.setdefault(r["generation"], []).append(r)
    for g in range(1, 5):
        assert g in records_by_gen and len(records_by_gen[g]) == 3, (
            f"Expected 3 records at gen {g}, got {len(records_by_gen.get(g, []))}"
        )

    # Verify the winner artifact loads via the Rust runtime.
    best_model = out_dir / "best_model.json"
    assert best_model.exists()
    import aerocapture_rs  # noqa: PLC0415

    nn = aerocapture_rs.load_nn_json(str(best_model)) if hasattr(aerocapture_rs, "load_nn_json") else None
    # If load_nn_json isn't bound, fall back to nn_forward smoke.
    out = aerocapture_rs.nn_forward(str(best_model), [0.0] * 16, None)
    assert len(out) == 2
    assert all(isinstance(v, float) for v in out)


@pytest.mark.slow
def test_islands_resume_preserves_winner(tmp_path: Path) -> None:
    """Verify resume restores per-island best_overall_* verbatim."""
    # Run 3 gens, kill, resume for 2 more, check that the final winner's
    # rms_history is consistent (no spurious re-promotion).
    # Implementation: mirror the smoke test but split into two subprocess
    # invocations with the same out_dir; assert that the final best_overall
    # in the resumed run is at least as good as the pre-interrupt one.
    pytest.skip("Resume integration smoke deferred until checkpoint format stabilizes — covered by unit test test_island_model_resume_preserves_best_overall_per_island")
```

- [ ] **Step 2: Run smoke test**

```bash
uv run pytest tests/test_island_model_smoke.py -v -s --timeout 600
```

Expected: pass within ~3-5 minutes on a workstation (5 gens * 3 islands * 16 individuals * 2 seeds = 480 sims plus a few validations).

- [ ] **Step 3: Add CI entry**

The CI's `python-pyo3` job already runs `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py tests/test_gru_ppo_smoke.py`. Extend that list in `.github/workflows/ci.yml`:

```yaml
# search for "tests/test_pyo3.py" in the yml; add tests/test_island_model_smoke.py
```

The exact yml file location: `.github/workflows/ci.yml`. Find the line referencing the existing smoke tests and append `tests/test_island_model_smoke.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_island_model_smoke.py .github/workflows/ci.yml
git commit -m "test(islands): end-to-end smoke test with k_period=1

5-gen subprocess run on reduced 16->8->2 NN arch. Verifies per-island
JSONL records, 3 records per gen, winner artifact loads via aerocapture_rs.
Added to the python-pyo3 CI job."
```

---

## Task 13: Warm-start fan-out to 3 islands

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (the warm-start chromosome injection block)

The warm-start path in `train.py` builds a chromosome once (via `build_warm_start_chromosome`) and replicates it into `pop_array` with per-individual jitter. With `algorithm = "islands"`, `pop_array` is already used by `_train_islands` (see Task 10) as each island's `init_pop`. Each island's `algorithm.setup` clones the population, so no further changes are needed beyond what Task 10 already does.

- [ ] **Step 1: Verify warm-start interaction by reading the existing warm-start path**

```bash
uv run python -c "
import inspect
from aerocapture.training.warm_start import build_warm_start_chromosome
print(inspect.signature(build_warm_start_chromosome))
"
```

Read `src/python/aerocapture/training/train.py` for the call site of `build_warm_start_chromosome`. Confirm it produces a single chromosome that gets jittered across `n_pop`.

- [ ] **Step 2: Add a regression test verifying that warm-start replication works for islands**

Append to `tests/test_island_model.py`:

```python
def test_islands_use_warm_started_pop_array() -> None:
    """With warm-start active, the (jittered) starting population is fanned
    out to all 3 islands. Each island then sees the same X[0], same X[1], etc."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    n_pop = cfg.n_pop
    n_params = 4
    rng = np.random.default_rng(42)
    pop_array = rng.uniform(0.0, 1.0, size=(n_pop, n_params))  # simulates jittered warm-start output

    model = IslandModel(
        config=cfg, problem=problem, n_params=n_params,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )

    from pymoo.core.population import Population
    from pymoo.core.evaluator import Evaluator
    for island in model.islands:
        init_pop = Population.new("X", pop_array.copy())
        Evaluator().eval(problem, init_pop)
        island.algorithm.setup(problem, pop=init_pop)

    # All 3 islands' pre-`next()` X must equal pop_array.
    for island in model.islands:
        X = island.algorithm.pop.get("X")
        np.testing.assert_array_equal(X, pop_array)
```

- [ ] **Step 3: Run test to verify it passes (no production code change needed if Task 10 is correct)**

```bash
uv run pytest tests/test_island_model.py::test_islands_use_warm_started_pop_array -v
```

Expected: pass with no production-code change. If it fails, the fix is in `_train_islands` setup loop.

- [ ] **Step 4: Commit**

```bash
git add tests/test_island_model.py
git commit -m "test(islands): warm-started pop_array fans out to all 3 islands

Regression guard for the warm-start interaction: every island's init pop
matches the jittered warm-started chromosome 1:1."
```

---

## Task 14: Display 3-column TUI extension

**Files:**
- Modify: `src/python/aerocapture/training/display.py`

- [ ] **Step 1: Inspect `LiveDisplay.update` signature**

```bash
grep -n "def update\|class LiveDisplay" src/python/aerocapture/training/display.py
```

Note the current single-column rendering pattern.

- [ ] **Step 2: Add an `island_records: dict[str, dict] | None = None` mode to `LiveDisplay.update`**

In `src/python/aerocapture/training/display.py`, extend `LiveDisplay.update`:

```python
    def update(
        self,
        logger: TrainingLogger,
        current_run: int,
        island_records: dict[str, dict] | None = None,
    ) -> None:
        """Render either a single-column (legacy) or 3-column (islands) view.

        When `island_records` is provided (keyed by island name -> latest
        record dict), the layout switches to 3 columns showing each island's
        best/last_val/stagnation/diversity sparkline.
        """
        if island_records is None:
            # ... existing single-column rendering, unchanged ...
            return

        # 3-column layout
        from rich.columns import Columns  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415

        panels = []
        for name in ("pso", "ga", "de"):
            rec = island_records.get(name, {})
            content = (
                f"best: {_format_cost(rec.get('best_overall_cost', float('inf')))}\n"
                f"last_val: {_format_cost(rec.get('val_rms', float('inf')))}\n"
                f"stag: {rec.get('stagnation', 0)} gens\n"
                f"argmin: {_format_cost(rec.get('argmin_train_cost', float('inf')))}"
            )
            panels.append(Panel(content, title=name.upper(), border_style="cyan"))
        self._live.update(Columns(panels))
```

- [ ] **Step 3: Wire `_train_islands` to call the 3-column update**

In `train.py::_train_islands`, replace the existing `display.update(logger, current_run=0)` line with:

```python
                island_records = {
                    island.name: {
                        "best_overall_cost": island.best_overall_cost,
                        "val_rms": val_records[i].get("val_rms", float("inf")),
                        "stagnation": island.stagnation_counter,
                        "argmin_train_cost": val_records[i].get("argmin_train_cost", float("inf")),
                    }
                    for i, island in enumerate(island_model.islands)
                }
                display.update(logger, current_run=0, island_records=island_records)
```

- [ ] **Step 4: Smoke-test TUI**

```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_islands_train.toml --n-gen 3 --no-tui
```

Expected: runs to completion with `--no-tui` (NoopDisplay branch). Then re-run without `--no-tui` for visual inspection (manual step — not automated):

```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_islands_train.toml --n-gen 3
```

Expected: 3-column display visible during the run.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/display.py src/python/aerocapture/training/train.py
git commit -m "feat(islands): 3-column TUI render mode

LiveDisplay.update accepts an optional island_records dict and switches to
a Rich Columns layout with one Panel per island. Falls back to legacy
single-column when island_records is None. NoopDisplay needs no change."
```

---

## Task 15: Report Part 0 — island convergence overlay + migration timeline

**Files:**
- Modify: `src/python/aerocapture/training/charts.py` (add 2 new chart functions)
- Modify: `src/python/aerocapture/training/report.py` (insert Part 0; restrict Parts 1–3 to winning island)
- Modify: `src/typst/report.typ` (Part 0 section if templated)

- [ ] **Step 1: Add `chart_island_convergence_overlay` and `chart_migration_timeline` to `charts.py`**

Append to `src/python/aerocapture/training/charts.py`:

```python
def chart_island_convergence_overlay(
    records_by_island: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Overlay per-island best_overall_cost vs generation.

    Three colored lines (PSO=blue, GA=orange, DE=green) on a log-scale y-axis.
    Vertical dashed grey lines mark migration events. Used as Part 0 panel 1.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pso": "tab:blue", "ga": "tab:orange", "de": "tab:green"}
    for name, records in records_by_island.items():
        gens = [r["generation"] for r in records]
        costs = [r.get("best_overall_cost", float("nan")) for r in records]
        ax.plot(gens, costs, color=colors.get(name, "k"), label=name.upper(), linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validated best cost (RMS)")
    ax.set_title("Per-island convergence")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def chart_migration_timeline(
    migration_log: list[dict],
    n_gen: int,
    output_path: Path,
) -> None:
    """Scatter of migration events: x=generation, y=src->dst pair, color=F_migrant.

    Used as Part 0 panel 2. Shows which (src, dst) channels were most active
    and what fitness levels migrants carried.
    """
    if not migration_log:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No migration events", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, n_gen)
        fig.tight_layout()
        fig.savefig(output_path, format="svg")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    channels = sorted({f"{e['src_island']}->{e['dst_island']}" for e in migration_log})
    channel_y = {ch: i for i, ch in enumerate(channels)}
    gens = [e["gen"] for e in migration_log]
    ys = [channel_y[f"{e['src_island']}->{e['dst_island']}"] for e in migration_log]
    Fs = [e["F_migrant"] for e in migration_log]

    sc = ax.scatter(gens, ys, c=Fs, cmap="viridis", s=20)
    fig.colorbar(sc, ax=ax, label="F_migrant")
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Generation")
    ax.set_xlim(0, n_gen)
    ax.set_title(f"Migration events ({len(migration_log)} total)")
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
```

- [ ] **Step 2: In `report.py`, detect islands run + split records per island + call new charts**

Read `src/python/aerocapture/training/report.py` to locate the Parts 1-3 section. Add a Part 0 block at the top:

```python
def _render_island_part0(
    save_dir: Path,
    jsonl_path: Path,
    typst_dir: Path,
) -> tuple[str | None, list[str]]:
    """Detect an islands run and render Part 0 (overlay + migration timeline).

    Returns (winner_island_name, list_of_part0_svgs). When the JSONL contains
    no `island_name` field (single-algorithm run), returns (None, []).
    """
    import json  # noqa: PLC0415

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if not any("island_name" in r for r in records):
        return None, []

    records_by_island: dict[str, list[dict]] = {}
    for r in records:
        name = r.get("island_name")
        if name is None:
            continue
        records_by_island.setdefault(name, []).append(r)

    n_gen = max(r["generation"] for r in records) + 1

    # Load migration log from the latest checkpoint.
    import numpy as np  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    ckpt_files = sorted(save_dir.glob("checkpoint_g*.npz"))
    migration_log: list[dict] = []
    if ckpt_files:
        with np.load(ckpt_files[-1], allow_pickle=True) as data:
            raw = pickle.loads(data["migration_log"].item())
            migration_log = [
                {
                    "gen": e.gen, "src_island": e.src_island, "dst_island": e.dst_island,
                    "F_migrant": e.F_migrant,
                }
                for e in raw
            ]

    overlay_svg = typst_dir / "island_overlay.svg"
    timeline_svg = typst_dir / "migration_timeline.svg"
    chart_island_convergence_overlay(records_by_island, overlay_svg)
    chart_migration_timeline(migration_log, n_gen, timeline_svg)

    # Winner: lowest final best_overall_cost across islands.
    final_costs = {
        name: recs[-1].get("best_overall_cost", float("inf"))
        for name, recs in records_by_island.items()
    }
    winner = min(final_costs, key=final_costs.get)
    return winner, ["island_overlay.svg", "migration_timeline.svg"]
```

Then in the existing report generation flow, call `_render_island_part0` once at the start, and when a winner is returned, filter the records passed to Parts 1-3 to that island only:

```python
    winner_island, part0_svgs = _render_island_part0(save_dir, jsonl_path, typst_dir)
    if winner_island is not None:
        records = [r for r in records if r.get("island_name") == winner_island]
```

- [ ] **Step 3: Update Typst template `src/typst/report.typ` to include Part 0 SVGs when present**

Read `src/typst/report.typ` and add (near the top, after the cover page):

```typst
#if part0_present == true [
  = Part 0: Island Model Convergence

  #image("island_overlay.svg", width: 100%)
  #image("migration_timeline.svg", width: 100%)
]
```

Add `part0_present` to the JSON metadata `report.py` passes to Typst. Locate the metadata-dict-build block in `report.py` and add:

```python
    metadata["part0_present"] = winner_island is not None
    metadata["winner_island"] = winner_island or ""
```

- [ ] **Step 4: Smoke-test report**

```bash
# Run the smoke test (Task 12) to generate training_output then run report:
uv run python -m aerocapture.training.report training_output/<smoke-output-dir>/ \
    --toml configs/training/msr_aller_islands_train.toml
```

Expected: `report.pdf` produced; Part 0 section visible with overlay and timeline.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/charts.py src/python/aerocapture/training/report.py src/typst/report.typ
git commit -m "feat(islands): report Part 0 (overlay + migration timeline)

Two new charts: per-island convergence overlay (3 lines, log-scale) and
migration scatter (src->dst channel x gen, colored by F_migrant). report.py
detects islands runs via the island_name JSONL field, renders Part 0,
then filters Parts 1-3 to the winning island only."
```

---

## Task 16: Self-review pass

**Files:** all spec deliverables

- [ ] **Step 1: Confirm all spec sections are implemented**

Run through the spec (`docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md`) section by section:

- §3 Architecture: `island_model.py` exists, `train.py` dispatches, `optimizer.py` parses `IslandSettings`. ✓ (Tasks 1, 4, 10)
- §4 Per-gen sequencing: `_train_islands` runs the 1→7 step sequence exactly. ✓ (Task 10)
- §5 Migration mechanics: `migrate()` and `inject_into_pso()` exist with unit tests. ✓ (Tasks 2, 3)
- §6 Configuration: `[optimizer.islands]` block in `common.toml` + reference TOML. ✓ (Task 11)
- §7 Failure modes / resume: checkpoint v2 atomic write + cross-gen rule preserved per-island. ✓ (Task 9)
- §8 Testing pyramid: unit tests + smoke test + ablation (`enabled = false`) covered. ✓ (Tasks 2-9, 12)

- [ ] **Step 2: Run full Python test suite**

```bash
uv run pytest tests/ -v
```

Expected: all passes (incl. new islands tests and the 5 from Task 1).

- [ ] **Step 3: Run Rust regression**

```bash
cd src/rust && cargo test --release && cd ../..
```

Expected: all 474 existing Rust tests pass (no Rust changes were made).

- [ ] **Step 4: Run linting**

```bash
./lint_code.sh
```

Expected: clean.

- [ ] **Step 5: Manual A/B smoke (~30 min)**

```bash
# Run reference islands config for 50 gens.
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_islands_train.toml \
    --n-gen 50 --no-tui
# Verify report renders.
ls training_output/neural_network_islands/report.pdf
```

Expected: report renders without errors; Part 0 visible; a winning island is selected.

- [ ] **Step 6: Commit any cleanups discovered during self-review**

```bash
# If cleanups were made:
git add -p
git commit -m "chore(islands): post-review cleanups"
```

(If nothing changed, skip this step.)

---

## Task 17: smart-commit (final wrap-up)

**Per user planning preference:** the final task always invokes the `smart-commit` skill, with the whole branch in scope, so CLAUDE.md/README.md stay in sync.

- [ ] **Step 1: Verify branch state**

```bash
git status
git log --oneline main..HEAD
```

Expected: clean working tree, ~13–17 commits ahead of main.

- [ ] **Step 2: Invoke `smart-commit` skill**

Tell the agent / human operator: "Invoke the `smart-commit` skill, telling it to take the whole `feature/island-model-pso-ga-de` branch into account."

The skill will:
- Update CLAUDE.md with the new `algorithm = "islands"` section (under the GA Training & Comparison area).
- Update any relevant README sections (e.g., the training overview).
- Create a final docs-sync commit.

- [ ] **Step 3: Push and PR (user-triggered, NOT by Claude)**

Per user CLAUDE.md, Claude must NEVER push or ask to push. Hand back to the user.

---

## Notes for the Implementing Engineer

1. **The pymoo `algorithm.pop[i].X = X_new` assignment pattern works** because `Individual` objects expose `X` as a settable attribute, and `Population.set("F", ...)` propagates to underlying individuals. If this turns out not to work in your pymoo version (>=0.6), the alternative is to construct a new `Population` from the modified arrays and assign `island.algorithm.pop = new_pop` — but verify the pymoo PSO inner update reads from `.pop` correctly after such a swap.

2. **`AerocaptureProblem._run_batch` is the private method this codebase already uses to bypass `Evaluator().eval()` (which would skip individuals with existing F).** Reuse it in `IslandModel.re_evaluate_all_populations` — do NOT call `Evaluator().eval(problem, pop)`.

3. **Per-island `algorithm.next()` is called sequentially in Python, not in parallel.** The Rayon parallelism in the Rust simulator (called via `problem._run_batch` inside pymoo's evaluator) saturates the CPU per call. Three sequential 64-individual batches dominate per-gen wall time.

4. **The `_train_islands` function is intentionally a separate helper, not inlined into `train()`** — `train()` is already 600+ lines. Keeping islands in a separate function makes the diff reviewable and lets the engineer reason about the islands branch in isolation.

5. **Checkpoint format version=2 is encoded in the .npz itself** (`version` field). Resume is gated on `version == 2`; mismatches raise `ValueError` with a pointer to `--from-scratch`.
