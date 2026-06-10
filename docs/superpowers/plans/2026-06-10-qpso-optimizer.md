# QPSO Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algorithm = "qpso"` (canonical mbest Quantum-behaved PSO, Sun/Feng/Xu 2004) as a single-algorithm optimizer option, plus paper-study configs and a batch-4 runner script.

**Architecture:** New pymoo `Algorithm` subclass in `src/python/aerocapture/training/qpso.py` mirroring pymoo PSO's state conventions (`pop` = pbest, `particles` = current positions) so `warm_start_algorithm`, checkpoint/resume, and the manual `.next()` loop work untouched. Integration is a settings dataclass + factory branch in `optimizer.py` and three one-line touchpoints in `train.py`.

**Tech Stack:** Python 3.14, pymoo 0.6.1.6, numpy, pytest. Pure Python — no Rust changes.

**Spec:** `docs/superpowers/specs/2026-06-10-qpso-optimizer-design.md`

**Branch:** `feature/qpso-optimizer` (already created; spec committed).

**Conventions for every task:**
- Run all commands from the repo root.
- Before Task 1, run `uv sync --group dev` once so pytest/ruff/mypy are installed.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- mypy is strict (`disallow_untyped_defs`): every `def` in src AND tests needs annotations. pymoo is untyped (`ignore_missing_imports`), so pymoo types resolve to `Any` — annotate with the pymoo class names anyway for readability.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/python/aerocapture/training/qpso.py` | Create | The QPSO pymoo `Algorithm` subclass |
| `tests/test_qpso.py` | Create | Behavioral tests (pure Python, no Rust) |
| `src/python/aerocapture/training/optimizer.py` | Modify | `QPSOSettings`, `_VALID_ALGORITHMS`, `from_dict`, `create_algorithm` |
| `tests/test_optimizer.py` | Modify | Config + factory tests for qpso |
| `src/python/aerocapture/training/train.py` | Modify | Warm-start whitelist, settings printout, argparse help |
| `tests/test_warm_start_optimizer_seeding.py` | Modify | Extend jitter-algo loops with `"qpso"` |
| `configs/training/common.toml` | Modify | `[optimizer.qpso]` defaults |
| `configs/training/paper/opt_qpso.toml` | Create | Paper Study A, small net (dense_p515) |
| `configs/training/paper/optbig_qpso.toml` | Create | Paper Study A, big net (dense_p3998) |
| `run_paper_experiments4.sh` | Create | Batch-4 runner (QPSO @300 small; @60/@150/@300 big) |

Out of scope (spec section 6): `island_model.py`, `display.py`/`charts.py` islands panels, RL, Rust.

---

### Task 1: QPSO algorithm module

**Files:**
- Create: `tests/test_qpso.py`
- Create: `src/python/aerocapture/training/qpso.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_qpso.py` with exactly:

```python
"""Behavioral tests for the QPSO optimizer (pure Python, no Rust dependency).

QPSO mirrors pymoo PSO's state conventions (pop = pbest, particles = current
positions), so these tests drive it exactly like train.py does:
setup(problem, seed=...) then repeated .next().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from aerocapture.training.qpso import QPSO
from aerocapture.training.train import warm_start_algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class _Sphere(Problem):
    """Shifted sphere: f(x) = sum((x - 0.3)^2), bounds [0, 1]."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        out["F"] = ((X - 0.3) ** 2).sum(axis=1).reshape(-1, 1)


def _run(n_gen: int, seed: int, pop_size: int = 20, n_var: int = 10) -> QPSO:
    algo = QPSO(pop_size=pop_size, max_iter=n_gen)
    algo.setup(_Sphere(n_var=n_var), seed=seed)
    for _ in range(n_gen):
        algo.next()
    return algo


class TestAlphaSchedule:
    def test_first_iter_is_alpha_start(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 1
        assert algo._alpha() == pytest.approx(1.0)

    def test_last_iter_is_alpha_end(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 101
        assert algo._alpha() == pytest.approx(0.5)

    def test_midpoint(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 51
        assert algo._alpha() == pytest.approx(0.75)

    def test_past_max_iter_clamps_to_alpha_end(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 5000
        assert algo._alpha() == pytest.approx(0.5)

    def test_max_iter_1_no_division_by_zero(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=1)
        algo.n_iter = 1
        assert algo._alpha() == pytest.approx(1.0)


class TestSwarmBehavior:
    def test_deterministic_under_seed(self) -> None:
        a = _run(n_gen=10, seed=7)
        b = _run(n_gen=10, seed=7)
        assert np.array_equal(a.opt[0].X, b.opt[0].X)
        assert float(a.opt[0].F[0]) == float(b.opt[0].F[0])

    def test_different_seeds_differ(self) -> None:
        a = _run(n_gen=10, seed=7)
        b = _run(n_gen=10, seed=8)
        assert not np.array_equal(a.opt[0].X, b.opt[0].X)

    def test_positions_respect_bounds_every_generation(self) -> None:
        algo = QPSO(pop_size=20, max_iter=20)
        algo.setup(_Sphere(), seed=3)
        for _ in range(20):
            algo.next()
            X = algo.particles.get("X")
            assert (X >= 0.0).all() and (X <= 1.0).all()

    def test_pbest_monotonically_non_increasing(self) -> None:
        algo = QPSO(pop_size=20, max_iter=30)
        algo.setup(_Sphere(), seed=5)
        algo.next()
        prev_F = algo.pop.get("F").copy()
        for _ in range(29):
            algo.next()
            F = algo.pop.get("F")
            assert (F <= prev_F + 1e-15).all()
            prev_F = F.copy()

    def test_sphere_convergence(self) -> None:
        algo = QPSO(pop_size=20, max_iter=60)
        algo.setup(_Sphere(n_var=10), seed=42)
        algo.next()
        f_init = float(algo.opt[0].F[0])
        for _ in range(59):
            algo.next()
        f_final = float(algo.opt[0].F[0])
        assert f_final < f_init / 50.0


class TestWarmStartCompat:
    def test_seeded_chromosomes_survive_gen0(self) -> None:
        """The invariant warm_start_algorithm exists to protect: a seeded,
        pre-evaluated population must become the pbest baseline (not get
        wiped by pymoo's _initialize() + LHS resample on the first next())."""
        problem = _Sphere(n_var=4)
        rng = np.random.default_rng(0)
        X0 = rng.random((10, 4))
        pop = Population.new("X", X0)
        Evaluator().eval(problem, pop)
        F0 = pop.get("F").copy()

        algo = QPSO(pop_size=10, max_iter=50)
        warm_start_algorithm(algo, problem, pop, seed=1)

        # _initialize_advance hook contract: particles start at the seeded pop.
        assert np.array_equal(algo.particles.get("X"), X0)

        algo.next()
        # pbest baseline is the seeded pop: per-index F can only improve.
        assert (algo.pop.get("F") <= F0 + 1e-15).all()
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_qpso.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'aerocapture.training.qpso'`

- [ ] **Step 1.3: Implement the algorithm**

Create `src/python/aerocapture/training/qpso.py` with exactly:

```python
"""Quantum-behaved PSO (QPSO) -- canonical mbest form (Sun, Feng & Xu 2004).

Particles carry no velocity. Each generation, every position is resampled
from a delta-potential-well distribution centered on a per-particle local
attractor (a random convex mix of pbest and gbest), with characteristic
length alpha * |mbest - x| where mbest is the swarm's mean pbest. The
contraction-expansion coefficient alpha anneals linearly from alpha_start
to alpha_end over max_iter generations (theory bounds convergence at
alpha < e^gamma ~ 1.781).

State conventions mirror pymoo's PSO so train.py's warm_start_algorithm,
checkpointing (pop = pbest), and the manual .next() loop work untouched:
- self.pop is the personal-best population (checkpointed, read by _set_optimum)
- self.particles is the current swarm position population

Resume note: single-algo resume restarts n_iter at 1 while max_iter is the
bumped total (resumed + additional gens), so alpha restarts at alpha_start
on the stretched schedule -- same family of state reset as PSO's velocity
reinit on resume. Paper runs are single-shot --from-scratch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.replacement import ImprovementReplacement
from pymoo.operators.crossover.dex import repair_random_init
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput


class QPSO(Algorithm):
    def __init__(
        self,
        pop_size: int = 25,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        max_iter: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(output=SingleObjectiveOutput(), **kwargs)
        self.initialization = Initialization(LHS())
        self.pop_size = pop_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_iter = max_iter
        self.particles: Population | None = None

    def _alpha(self) -> float:
        progress = (self.n_iter - 1) / max(1, self.max_iter - 1)
        progress = min(1.0, max(0.0, progress))
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def _initialize_infill(self) -> Population:
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        self.particles = self.pop
        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self) -> Population:
        X = self.particles.get("X")
        P = self.pop.get("X")
        G = self.opt[0].X

        mbest = P.mean(axis=0)
        rs = self.random_state
        phi = rs.random(X.shape)
        attractor = phi * P + (1.0 - phi) * G[None, :]
        u = 1.0 - rs.random(X.shape)  # (0, 1]: keeps log(1/u) finite
        sign = np.where(rs.random(X.shape) < 0.5, 1.0, -1.0)
        Xp = attractor + sign * self._alpha() * np.abs(mbest[None, :] - X) * np.log(1.0 / u)

        if self.problem.has_bounds():
            Xp = repair_random_init(Xp, X, *self.problem.bounds(), random_state=rs)

        return Population.new(X=Xp)

    def _advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        assert infills is not None, "QPSO uses the ask-and-tell interface; 'infills' must be provided."
        self.particles = infills
        has_improved = ImprovementReplacement().do(self.problem, self.pop, infills, return_indices=True)
        self.pop[has_improved] = infills[has_improved]
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qpso.py -q`
Expected: 11 passed (5 alpha-schedule + 5 swarm-behavior + 1 warm-start)

- [ ] **Step 1.5: Commit**

```bash
git add src/python/aerocapture/training/qpso.py tests/test_qpso.py
git commit -m "feat(train): QPSO algorithm (canonical mbest, pymoo Algorithm subclass)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: optimizer.py integration

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 2.1: Write the failing tests**

In `tests/test_optimizer.py`:

(a) Extend the import block:

```python
from aerocapture.training.optimizer import (
    DESettings,
    GASettings,
    OptimizerConfig,
    PSOSettings,
    QPSOSettings,
    create_algorithm,
)
from aerocapture.training.qpso import QPSO
```

(b) In `test_all_algorithms_accepted`, change the tuple:

```python
        for algo in ("ga", "cma_es", "de", "pso", "qpso", "islands"):
```

(c) Add to `class TestOptimizerConfig` (after `test_from_toml_dict_cma_es`):

```python
    def test_from_toml_dict_qpso(self) -> None:
        d = {
            "algorithm": "qpso",
            "seed_strategy": "adaptive",
            "qpso": {"alpha_start": 0.9, "alpha_end": 0.4},
        }
        cfg = OptimizerConfig.from_dict(d)
        assert cfg.algorithm == "qpso"
        assert cfg.qpso.alpha_start == 0.9
        assert cfg.qpso.alpha_end == 0.4

    def test_qpso_defaults_when_subsection_missing(self) -> None:
        cfg = OptimizerConfig.from_dict({"algorithm": "qpso", "seed_strategy": "adaptive"})
        assert isinstance(cfg.qpso, QPSOSettings)
        assert cfg.qpso.alpha_start == 1.0
        assert cfg.qpso.alpha_end == 0.5

    def test_qpso_settings_rejects_zero_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha_start"):
            QPSOSettings(alpha_start=0.0)

    def test_qpso_settings_rejects_alpha_above_2(self) -> None:
        with pytest.raises(ValueError, match="alpha_end"):
            QPSOSettings(alpha_end=2.5)
```

(d) Add to `class TestCreateAlgorithm` (after `test_pso_returns_pso`):

```python
    def test_qpso_returns_qpso(self) -> None:
        cfg = OptimizerConfig(algorithm="qpso", seed_strategy="adaptive", n_pop=30, n_gen=500)
        algo = create_algorithm(cfg, n_params=10)
        assert isinstance(algo, QPSO)
        assert algo.pop_size == 30
        assert algo.max_iter == 500
        assert algo.alpha_start == 1.0
        assert algo.alpha_end == 0.5
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py -q`
Expected: collection error — `ImportError: cannot import name 'QPSOSettings'`

- [ ] **Step 2.3: Implement**

In `src/python/aerocapture/training/optimizer.py`:

(a) Add import after the pymoo imports (line 12, after `from pymoo.operators.mutation.pm import PM`):

```python
from aerocapture.training.qpso import QPSO
```

(b) Replace line 14:

```python
_VALID_ALGORITHMS = ("ga", "cma_es", "de", "pso", "qpso", "islands")
```

(c) Add after the `PSOSettings` dataclass (after line 43):

```python
@dataclass
class QPSOSettings:
    # Contraction-expansion coefficient, annealed linearly alpha_start -> alpha_end.
    # Theory (Sun et al. 2004): trajectories diverge above alpha ~ e^gamma ~ 1.781;
    # (0, 2] is a permissive cap.
    alpha_start: float = 1.0
    alpha_end: float = 0.5

    def __post_init__(self) -> None:
        for name in ("alpha_start", "alpha_end"):
            value = getattr(self, name)
            if not 0.0 < value <= 2.0:
                raise ValueError(f"{name} must be in (0, 2], got {value}")
```

(d) In `OptimizerConfig`, add a field after `pso`:

```python
    pso: PSOSettings = field(default_factory=PSOSettings)
    qpso: QPSOSettings = field(default_factory=QPSOSettings)
    islands: IslandSettings = field(default_factory=IslandSettings)
```

(e) In `from_dict`, add after the `pso = ...` line:

```python
        qpso = QPSOSettings(**d["qpso"]) if "qpso" in d else QPSOSettings()
```

and replace the `top_level` + `return` lines (the subsection-exclusion tuple MUST gain `"qpso"`, otherwise the dict leaks into `cls(**top_level)` as an unexpected kwarg):

```python
        top_level = {k: v for k, v in d.items() if k not in ("ga", "cma_es", "de", "pso", "qpso", "islands") and k not in _obsolete}
        return cls(**top_level, ga=ga, cma_es=cma_es, de=de, pso=pso, qpso=qpso, islands=islands)
```

(f) In `create_algorithm`, add after the `if algorithm == "pso":` block:

```python
    if algorithm == "qpso":
        qpso = config.qpso
        return QPSO(
            pop_size=config.n_pop,
            alpha_start=qpso.alpha_start,
            alpha_end=qpso.alpha_end,
            max_iter=config.n_gen,
        )
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py tests/test_qpso.py -q`
Expected: all pass (existing optimizer tests + 6 new + 12 from Task 1)

- [ ] **Step 2.5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_optimizer.py
git commit -m "feat(train): register qpso in OptimizerConfig + create_algorithm

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: train.py touchpoints

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (3 one-liners)
- Modify: `tests/test_warm_start_optimizer_seeding.py`

- [ ] **Step 3.1: Write the failing tests**

In `tests/test_warm_start_optimizer_seeding.py`:

(a) In `test_de_and_pso_match_ga_contract`, change the loop tuple:

```python
    for algo in ("de", "pso", "qpso"):
```

(b) Rename `test_row_0_is_exact_warm_start_chromosome_ga_de_pso` to `test_row_0_is_exact_warm_start_chromosome_jitter_algos` and change its loop tuple:

```python
    for algo in ("ga", "de", "pso", "qpso"):
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_warm_start_optimizer_seeding.py -q`
Expected: 2 failed — `ValueError: unknown algorithm 'qpso' for warm-start seeding`

- [ ] **Step 3.3: Implement the three edits**

In `src/python/aerocapture/training/train.py`:

(a) `_seed_initial_population` whitelist (line 277) — replace:

```python
    if algorithm_name not in ("ga", "de", "pso", "islands"):
```

with:

```python
    if algorithm_name not in ("ga", "de", "pso", "qpso", "islands"):
```

(b) Settings printout — after the existing block:

```python
        elif opt.algorithm == "pso":
            print(f"  PSO:       w={opt.pso.w}, c1={opt.pso.c1}, c2={opt.pso.c2}")
```

add:

```python
        elif opt.algorithm == "qpso":
            print(f"  QPSO:      alpha_start={opt.qpso.alpha_start}, alpha_end={opt.qpso.alpha_end}")
```

(c) argparse help (line 2118) — replace:

```python
    parser.add_argument("--algorithm", type=str, default=None, help="Optimization algorithm: ga, cma_es, de, pso (default: from TOML [optimizer])")
```

with:

```python
    parser.add_argument("--algorithm", type=str, default=None, help="Optimization algorithm: ga, cma_es, de, pso, qpso (default: from TOML [optimizer])")
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_warm_start_optimizer_seeding.py -q`
Expected: all pass

- [ ] **Step 3.5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_warm_start_optimizer_seeding.py
git commit -m "feat(train): qpso in warm-start seeding whitelist + settings printout + CLI help

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Configs + paper runner

**Files:**
- Modify: `configs/training/common.toml`
- Create: `configs/training/paper/opt_qpso.toml`
- Create: `configs/training/paper/optbig_qpso.toml`
- Create: `run_paper_experiments4.sh`

- [ ] **Step 4.1: Add `[optimizer.qpso]` defaults to common.toml**

In `configs/training/common.toml`, after the `[optimizer.pso]` block (lines 98-101), add:

```toml
# QPSO (quantum-behaved PSO, canonical mbest form): contraction-expansion
# coefficient annealed linearly alpha_start -> alpha_end over n_gen.
# Theory: divergence above alpha ~ 1.781 (e^gamma).
[optimizer.qpso]
alpha_start = 1.0
alpha_end = 0.5
```

- [ ] **Step 4.2: Create the paper configs**

Create `configs/training/paper/opt_qpso.toml`:

```toml
# Study A -- QPSO on the dense_p515 control architecture (compute-matched n_pop=300).
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_qpso/best_model.json"
results_suffix = ".paper_opt_qpso"

[optimizer]
algorithm = "qpso"
```

Create `configs/training/paper/optbig_qpso.toml`:

```toml
# Study A (big net) -- QPSO on dense_p3998 (~4000 params, compute-matched n_pop=300).
base = ["../sweep/dense_p3998.toml"]

[data]
neural_network = "training_output/paper_optbig_qpso/best_model.json"
results_suffix = ".paper_optbig_qpso"

[optimizer]
algorithm = "qpso"
```

- [ ] **Step 4.3: Create the batch-4 runner**

Create `run_paper_experiments4.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Batch 4 -- QPSO column for the optimizer comparison
# (spec: docs/superpowers/specs/2026-06-10-qpso-optimizer-design.md).
# Canonical mbest QPSO (Sun/Feng/Xu 2004), alpha annealed 1.0 -> 0.5.
# Mirrors the batch-2/3 grid: small net @300; big net @60/@150/@300.
# @60 uses the default output dir (matches batch 2); @150/@300 use
# --output-dir (matches batch 3).

# ── Study A (small net, dense_p515): compute-matched n_pop=300 ──
uv run python -m aerocapture.training.train configs/training/paper/opt_qpso.toml    --n-gen 2000 --n-pop 300 --from-scratch

# ── Study A (big net, dense_p3998): budget scaling @60/@150/@300 ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 60  --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_qpso150 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_qpso300 --from-scratch
```

Then: `chmod +x run_paper_experiments4.sh`

- [ ] **Step 4.4: Verify the config round-trip**

Run:

```bash
uv run python -c "
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.toml_utils import load_toml_with_bases
for p in ('configs/training/paper/opt_qpso.toml', 'configs/training/paper/optbig_qpso.toml'):
    cfg = OptimizerConfig.from_dict(load_toml_with_bases(p)['optimizer'])
    print(p, '->', cfg.algorithm, cfg.qpso)
"
```

Expected output (both lines):

```
configs/training/paper/opt_qpso.toml -> qpso QPSOSettings(alpha_start=1.0, alpha_end=0.5)
configs/training/paper/optbig_qpso.toml -> qpso QPSOSettings(alpha_start=1.0, alpha_end=0.5)
```

This exercises base inheritance (paper config -> sweep arch -> atan2 train config -> common.toml) AND the new `[optimizer.qpso]` block parsing.

- [ ] **Step 4.5: Commit**

```bash
git add configs/training/common.toml configs/training/paper/opt_qpso.toml configs/training/paper/optbig_qpso.toml run_paper_experiments4.sh
git commit -m "feat(paper): QPSO paper configs + batch-4 runner (@300 small; @60/@150/@300 big)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Verification (lint, targeted suite, E2E smoke)

**Files:** none (verification only; fix-ups if anything fails)

- [ ] **Step 5.1: Lint + type-check**

Run: `./lint_code.sh`
Expected: ruff (imports, format, lint) and mypy all clean. If ruff reformats `qpso.py`/`test_qpso.py` import order, accept its fixes and re-run.

- [ ] **Step 5.2: Targeted test suite**

Run: `uv run pytest tests/test_qpso.py tests/test_optimizer.py tests/test_warm_start_optimizer_seeding.py -q`
Expected: all pass, no warnings about unknown algorithms.

- [ ] **Step 5.3: E2E training smoke (requires Rust build)**

Build PyO3 bindings + binary, then run a 3-gen QPSO training on equilibrium glide (no ref-trajectory dependency, fast sims):

```bash
./build.sh
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml \
    --algorithm qpso --n-gen 3 --n-pop 6 --no-tui --skip-report \
    --output-dir /tmp/qpso_smoke --from-scratch
```

Expected: settings block prints `QPSO:      alpha_start=1.0, alpha_end=0.5`; 3 generations complete; `/tmp/qpso_smoke/` contains `checkpoint_g*.json` + `.npz` and `best_params.json`.

- [ ] **Step 5.4: Resume smoke (validates pop=pbest resume contract)**

```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml \
    --algorithm qpso --n-gen 2 --no-tui --skip-report \
    --output-dir /tmp/qpso_smoke
```

Expected: auto-resumes from the Task 5.3 checkpoint (resume message printed), runs 2 additional generations, exits cleanly. Then clean up: `rm -rf /tmp/qpso_smoke`.

- [ ] **Step 5.5: Commit any verification fix-ups**

Only if 5.1-5.4 required changes:

```bash
git add -u
git commit -m "fix(train): QPSO verification fix-ups

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Documentation sync + final commit (smart-commit)

- [ ] **Step 6.1: Invoke the `smart-commit` skill**

Invoke the `smart-commit` skill, telling it to take the **whole git branch** (`feature/qpso-optimizer`, all commits since `main`) into account. It syncs CLAUDE.md / README.md with the new optimizer (the `[optimizer]` algorithm list, `optimizer.py` description, `tests/` coverage notes) and commits everything remaining.
