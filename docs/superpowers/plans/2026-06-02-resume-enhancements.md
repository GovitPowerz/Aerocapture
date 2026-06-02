# Resume Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three resume-path capabilities to the GA/PSO training pipeline: re-validate the best individual on islands resume, resume with a larger/smaller population seeded from the checkpoint, and reset the validation baseline when `cost_transform` changes.

**Architecture:** A shared pure helper (`resize_population`) plus a config knob, wired into both the single-algorithm `train()` resume path and the `IslandModel` resume path. `cost_transform` is persisted into both checkpoint formats and compared on resume. Islands gains a `revalidate_each` method mirroring the single-algo resume re-validation, and a `resize_populations` method that handles PSO particle/velocity extension.

**Tech Stack:** Python 3.14, numpy, pymoo, pytest. Package: `src/python/aerocapture/training/`.

---

## Spec

`docs/superpowers/specs/2026-06-02-resume-enhancements-design.md`

## File Structure

- `src/python/aerocapture/training/population.py` — add `resize_population` pure helper (lives with `create_initial_population`).
- `src/python/aerocapture/training/optimizer.py` — add `grow_fresh_fraction` to `OptimizerConfig`.
- `src/python/aerocapture/training/train.py` — persist `cost_transform` in `save_checkpoint`/`load_checkpoint`; wire pop resize + transform-change reset into the single-algo resume path and `_train_islands`.
- `src/python/aerocapture/training/island_model.py` — persist `cost_transform`; add `revalidate_each` and `resize_populations`; extend `from_checkpoint` return.
- `tests/test_population_resize.py` — unit tests for `resize_population`.
- `tests/test_resume_enhancements.py` — config knob, checkpoint persistence, detection, and island method tests.

## Conventions (read before starting)

- Run Python from repo root with `uv run`. Tests: `uv run pytest <path> -v`.
- Populations are `float64` arrays of shape `(n_pop, n_params)` in the normalized `[0, 1]` hypercube. Fresh-random fill = `rng.random((k, n_params))`. Jitter = additive `rng.normal(0, sigma)` then `np.clip(_, 0, 1)`.
- `cost_transform` lives in `problem.cost_kwargs["cost_transform"]` (default `"linear"`), set at `train.py:764`.
- Lint after code changes: `./lint_code.sh` (ruff + mypy strict). Honor `feedback_ruff_fmt_skip` if multi-except parens appear (not expected here).
- Do NOT touch the cross-gen incomparability gate at `train.py:1239-1243` (guarded on `best_overall_individual is None`).

---

## Task 1: `resize_population` pure helper

**Files:**
- Modify: `src/python/aerocapture/training/population.py`
- Test: `tests/test_population_resize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_population_resize.py`:

```python
import numpy as np
import pytest

from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.population import resize_population


def _specs(n: int) -> list[ParamSpec]:
    return [ParamSpec(name=f"p{i}", p_min=0.0, p_max=1.0, default=0.5) for i in range(n)]


def test_equal_target_is_identity():
    rng = np.random.default_rng(0)
    pop = rng.random((5, 3))
    out = resize_population(pop, np.arange(5.0), 5, _specs(3), rng)
    assert np.array_equal(out, pop)


def test_grow_preserves_resumed_individuals_verbatim():
    rng = np.random.default_rng(1)
    pop = rng.random((4, 3))
    out = resize_population(pop, np.arange(4.0), 10, _specs(3), rng, fresh_fraction=0.2)
    assert out.shape == (10, 3)
    assert np.array_equal(out[:4], pop)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_grow_fresh_fraction_split():
    rng = np.random.default_rng(2)
    pop = np.full((10, 2), 0.5)
    # 10 new slots, fresh_fraction 0.2 -> round(0.2*10)=2 fresh, 8 clone+jitter.
    out = resize_population(pop, np.zeros(10), 20, _specs(2), rng, fresh_fraction=0.2, jitter_sigma=0.02)
    new = out[10:]
    # Clones sit within a few jitter sigmas of 0.5; fresh-random spread wider.
    near = np.abs(new - 0.5).max(axis=1) < 0.1
    assert near.sum() == 8
    assert (~near).sum() == 2


def test_shrink_keeps_best_by_cost():
    rng = np.random.default_rng(3)
    pop = rng.random((6, 2))
    costs = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.0])
    out = resize_population(pop, costs, 3, _specs(2), rng)
    assert out.shape == (3, 2)
    # Best 3 by cost are indices 5 (0.0), 1 (1.0), 3 (2.0).
    expected = pop[[5, 1, 3]]
    assert np.array_equal(out, expected)


def test_shrink_none_costs_keeps_first_n():
    rng = np.random.default_rng(4)
    pop = rng.random((6, 2))
    out = resize_population(pop, None, 3, _specs(2), rng)
    assert np.array_equal(out, pop[:3])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_population_resize.py -v`
Expected: FAIL with `ImportError: cannot import name 'resize_population'`.

- [ ] **Step 3: Implement `resize_population`**

Append to `src/python/aerocapture/training/population.py`:

```python
def resize_population(
    pop_X: npt.NDArray[np.float64],
    pop_F: npt.NDArray[np.float64] | None,
    target_n: int,
    specs: list[ParamSpec],
    rng: np.random.Generator,
    fresh_fraction: float = 0.2,
    jitter_sigma: float = 0.02,
) -> npt.NDArray[np.float64]:
    """Resize a normalized [0,1] population to ``target_n`` rows.

    Grow: keep all resumed rows verbatim, fill the rest with a
    ``fresh_fraction`` of fresh-random individuals and the remainder as
    clone+jitter of the resumed pool (round-robin). Shrink: keep the
    ``target_n`` lowest-``pop_F`` rows (first ``target_n`` rows when costs are
    unavailable). Equal: return ``pop_X`` unchanged.
    """
    n = pop_X.shape[0]
    n_params = pop_X.shape[1]
    if target_n == n:
        return pop_X
    if target_n < n:
        if pop_F is None:
            return pop_X[:target_n].copy()
        order = np.argsort(np.asarray(pop_F, dtype=np.float64))
        return pop_X[order[:target_n]].copy()

    n_new = target_n - n
    n_fresh = int(round(fresh_fraction * n_new))
    n_clone = n_new - n_fresh

    out = np.empty((target_n, n_params), dtype=np.float64)
    out[:n] = pop_X
    if n_clone > 0:
        src = pop_X[np.arange(n_clone) % n]
        jitter = rng.normal(0.0, jitter_sigma, size=(n_clone, n_params))
        out[n : n + n_clone] = np.clip(src + jitter, 0.0, 1.0)
    if n_fresh > 0:
        out[n + n_clone :] = rng.random((n_fresh, n_params))
    return out
```

Confirm the existing import block already has `from aerocapture.training.param_spaces import ParamSpec` (it does, line 10).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_population_resize.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint**

Run: `./lint_code.sh`
Expected: no errors on `population.py` / the new test.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/population.py tests/test_population_resize.py
git commit -m "feat(training): add resize_population helper for resume pop growth/shrink"
```

---

## Task 2: `grow_fresh_fraction` config knob

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py:63-91` (OptimizerConfig)
- Test: `tests/test_resume_enhancements.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resume_enhancements.py`:

```python
from aerocapture.training.optimizer import OptimizerConfig


def test_grow_fresh_fraction_default():
    cfg = OptimizerConfig(seed_strategy="fixed")
    assert cfg.grow_fresh_fraction == 0.2


def test_grow_fresh_fraction_from_dict():
    cfg = OptimizerConfig.from_dict({"seed_strategy": "fixed", "grow_fresh_fraction": 0.5})
    assert cfg.grow_fresh_fraction == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resume_enhancements.py -v`
Expected: FAIL with `AttributeError: 'OptimizerConfig' object has no attribute 'grow_fresh_fraction'`.

- [ ] **Step 3: Add the field**

In `src/python/aerocapture/training/optimizer.py`, add to the `OptimizerConfig` dataclass body (after `curation_sample_size: int = 1000`, before the sub-config fields):

```python
    grow_fresh_fraction: float = 0.2
```

In `__post_init__`, add a bounds check after the `curation_sample_size` check:

```python
        if not 0.0 <= self.grow_fresh_fraction <= 1.0:
            raise ValueError(f"grow_fresh_fraction must be in [0, 1], got {self.grow_fresh_fraction}")
```

`from_dict` already forwards unknown top-level keys via `top_level` splat, so `grow_fresh_fraction` flows through automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_resume_enhancements.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_resume_enhancements.py
git commit -m "feat(training): add [optimizer] grow_fresh_fraction knob"
```

---

## Task 3: Persist `cost_transform` in the single-algo checkpoint

**Files:**
- Modify: `src/python/aerocapture/training/train.py:496-638` (`save_checkpoint`, `load_checkpoint`)
- Test: `tests/test_resume_enhancements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume_enhancements.py`:

```python
import numpy as np

from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.train import load_checkpoint, save_checkpoint


def _make_config():
    from aerocapture.training.config import TrainingConfig

    return TrainingConfig()


def test_checkpoint_persists_cost_transform(tmp_path):
    rng = np.random.default_rng(0)
    specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5)]
    pop = rng.random((3, 1))
    save_checkpoint(
        tmp_path,
        generation=2,
        population=pop,
        costs=np.zeros(3),
        best_cost=1.0,
        best_individual=pop[0],
        cost_history=[1.0],
        rng=rng,
        config=_make_config(),
        cwd=None,
        param_specs=specs,
        cost_transform="log",
    )
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded["cost_transform"] == "log"


def test_load_checkpoint_legacy_cost_transform_defaults_none(tmp_path):
    import json

    # Hand-write a checkpoint pair with NO cost_transform key (legacy).
    (tmp_path / "checkpoint_g00000.json").write_text(
        json.dumps({"generation": 0, "best_cost": 1.0, "best_val_cost": 1.0, "cost_history": [], "rng_state": None})
    )
    np.savez(tmp_path / "checkpoint_g00000.npz", population=np.zeros((2, 1)), costs=np.zeros(2))
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded["cost_transform"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resume_enhancements.py::test_checkpoint_persists_cost_transform -v`
Expected: FAIL — `save_checkpoint() got an unexpected keyword argument 'cost_transform'`.

- [ ] **Step 3: Add the param to `save_checkpoint`**

In `src/python/aerocapture/training/train.py`, add a parameter to `save_checkpoint` (after `best_val_cost: float = np.inf,`):

```python
    cost_transform: str = "linear",
```

Add it to the `meta` dict (after `"best_val_cost": best_val_cost,`):

```python
        "cost_transform": cost_transform,
```

- [ ] **Step 4: Read it back in `load_checkpoint`**

In the `return` dict of `load_checkpoint` (after `"best_val_cost": meta.get("best_val_cost", float("inf")),`):

```python
        "cost_transform": meta.get("cost_transform", None),
```

(`None` signals a legacy checkpoint -> treated as "changed" by Task 4.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_resume_enhancements.py -v`
Expected: all passed (including the 2 legacy/persist tests).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_resume_enhancements.py
git commit -m "feat(training): persist cost_transform in single-algo checkpoint"
```

---

## Task 4: Wire single-algo resume — pop resize + cost_transform reset

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (resume restore ~969-975; save_checkpoint call sites 1433/1457/1482; resume block ~853-876)

This task has no new pure-function unit test (it is integration wiring); it is covered by the integration test in Task 8 and verified manually here. Keep edits minimal and exact.

- [ ] **Step 1: Pass `cost_transform` at all three `save_checkpoint` call sites**

At each of the three call sites (`train.py:1433`, `1457`, `1482`), add the keyword argument to the `save_checkpoint(...)` call (alongside the existing `best_val_cost=best_val_cost,`):

```python
                        cost_transform=problem.cost_kwargs.get("cost_transform", "linear"),
```

Match the indentation of each call site (the three sites differ in indentation). Verify with:

Run: `rg -n "save_checkpoint\(" src/python/aerocapture/training/train.py`
then read each call's argument block and confirm `cost_transform=` was added to all three.

- [ ] **Step 2: Detect a cost_transform change on resume**

In the single-algo resume block (inside `if resumed is not None:`, after `config.optimizer.n_gen += resumed["generation"]` at `train.py:876`), add:

```python
            saved_transform = resumed.get("cost_transform")
            current_transform = problem.cost_kwargs.get("cost_transform", "linear")
            cost_transform_changed = saved_transform is None or saved_transform != current_transform
            if cost_transform_changed and verbose:
                print(f"  cost_transform changed {saved_transform!r} -> {current_transform!r}; re-validating best under new metric")
```

NOTE: `problem` is constructed later (line 930), AFTER this resume block. So this detection must move to AFTER `problem` exists. Instead, place the detection right after the `problem = AerocaptureProblem(...)` block (after `train.py:938`):

```python
    cost_transform_changed = False
    if resumed is not None:
        saved_transform = resumed.get("cost_transform")
        current_transform = problem.cost_kwargs.get("cost_transform", "linear")
        cost_transform_changed = saved_transform is None or saved_transform != current_transform
        if cost_transform_changed and verbose:
            print(f"  cost_transform changed {saved_transform!r} -> {current_transform!r}; re-validating best under new metric")
```

Do NOT add the snippet inside the `if resumed is not None:` block at line 853 (problem does not exist yet there).

- [ ] **Step 3: Resize the resumed population to the configured n_pop**

In the resume restore branch (`train.py:969-975`), after `_check_resume_chromosome_shape(pop_array, expected_n_params=len(param_specs))`, add:

```python
        if config.optimizer.n_pop != pop_array.shape[0]:
            from aerocapture.training.population import resize_population  # noqa: PLC0415

            if verbose:
                print(f"  Resizing resumed population {pop_array.shape[0]} -> {config.optimizer.n_pop}")
            pop_array = resize_population(
                pop_array,
                pop_costs,
                config.optimizer.n_pop,
                param_specs,
                rng,
                fresh_fraction=config.optimizer.grow_fresh_fraction,
            )
            pop_costs = None  # force a single re-eval of the resized pop (train.py:1225)
```

Setting `pop_costs = None` makes the existing `Evaluator().eval(...)` path at `train.py:1222-1226` re-evaluate the resized population under the current seeds before `warm_start_algorithm`.

- [ ] **Step 4: Reset validation baseline on transform change (stagnation)**

The single-algo resume already re-validates `best_overall_individual` under the current config at `train.py:1275-1298`, which recomputes `best_val_cost` under the new transform automatically (the "re-validate, keep individual" semantics). The only additional reset is the stagnation counter, which the logger derives from validation promotions — there is no standalone counter variable to reset in the single-algo path (stagnation is computed in `metrics.py` from the JSONL `improvement` flags). So no extra code is needed here beyond the log line from Step 2.

Confirm this by checking there is no `stagnation_counter` local in the single-algo `train()` loop:

Run: `rg -n "stagnation" src/python/aerocapture/training/train.py`
Expected: matches are in display/logging only, not a resettable local in `train()`. (If a resettable local exists, set it to 0 when `cost_transform_changed`.)

- [ ] **Step 5: Smoke-check imports / syntax**

Run: `uv run python -c "import aerocapture.training.train"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(training): single-algo resume pop resize + cost_transform change notice"
```

---

## Task 5: Persist `cost_transform` in the islands checkpoint

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py:489-551` (`checkpoint`), `553-656` (`from_checkpoint`)
- Test: `tests/test_resume_enhancements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume_enhancements.py`:

```python
def test_islands_from_checkpoint_returns_cost_transform(tmp_path, monkeypatch):
    # Build a minimal IslandModel-like checkpoint and verify from_checkpoint
    # surfaces the saved cost_transform. We test the npz field directly to avoid
    # constructing a full pymoo island stack.
    import pickle

    import numpy as np

    npz = tmp_path / "checkpoint_g00000.npz"
    np.savez_compressed(
        npz.with_name(npz.stem + ".tmp.npz"),
        version=2,
        generation=0,
        base_mc_seed=42,
        cost_transform="log",
        island_states=np.array(pickle.dumps([]), dtype=object),
        migration_log=np.array(pickle.dumps([]), dtype=object),
        rng_state=np.array(pickle.dumps(np.random.default_rng(0).bit_generator.state), dtype=object),
        seed_curator_state=np.array(pickle.dumps(None), dtype=object),
    )
    (npz.with_name(npz.stem + ".tmp.npz")).rename(npz)

    with np.load(npz, allow_pickle=True) as data:
        assert "cost_transform" in data
        assert str(data["cost_transform"]) == "log"
```

- [ ] **Step 2: Run test to verify it fails**

This test exercises the npz schema we are about to write. Run it now to confirm it passes as a schema canary (it constructs the npz inline, so it will pass even before code changes — that is fine; it documents the on-disk shape). The real behavior change is verified by Step 5's manual check.

Run: `uv run pytest tests/test_resume_enhancements.py::test_islands_from_checkpoint_returns_cost_transform -v`
Expected: PASS (schema canary).

- [ ] **Step 3: Write `cost_transform` in `checkpoint`**

In `island_model.py`, inside `IslandModel.checkpoint`, add to the `np.savez_compressed(...)` call (after `base_mc_seed=self.base_mc_seed,`):

```python
            cost_transform=str(self.problem.cost_kwargs.get("cost_transform", "linear")),
```

- [ ] **Step 4: Read it in `from_checkpoint` and extend the return**

In `from_checkpoint`, inside the `with np.load(...)` block (after `base_mc_seed = int(data["base_mc_seed"])`):

```python
            saved_cost_transform = str(data["cost_transform"]) if "cost_transform" in data else None
```

Change the return statement at the end of `from_checkpoint` from:

```python
        return generation, seed_curator_state
```

to:

```python
        return generation, seed_curator_state, saved_cost_transform
```

Update the docstring's `Returns:` section to mention the third element `saved_cost_transform` (None for legacy checkpoints).

- [ ] **Step 5: Update the single caller in `train.py`**

In `_train_islands`, the call at `train.py:1620`:

```python
        resumed_gen, resumed_curator_state = island_model.from_checkpoint(resume_ckpt)
```

becomes:

```python
        resumed_gen, resumed_curator_state, resumed_cost_transform = island_model.from_checkpoint(resume_ckpt)
```

(`resumed_cost_transform` is consumed in Task 8.)

- [ ] **Step 6: Run tests + import check**

Run: `uv run pytest tests/test_resume_enhancements.py -v && uv run python -c "import aerocapture.training.train, aerocapture.training.island_model"`
Expected: all passed; no import error.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/island_model.py src/python/aerocapture/training/train.py tests/test_resume_enhancements.py
git commit -m "feat(training): persist cost_transform in islands checkpoint"
```

---

## Task 6: `IslandModel.revalidate_each`

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py` (add method near `validate_each`)
- Test: `tests/test_resume_enhancements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume_enhancements.py`. This uses a lightweight fake island + fake problem to isolate the re-validation logic:

```python
import types


class _FakeProblem:
    def __init__(self, rms):
        self._rms = rms
        self.cost_kwargs = {"cost_transform": "linear"}

    def evaluate_individual_records_per_seed(self, x, seeds):
        # costs whose RMS == self._rms regardless of x
        return np.full(len(seeds), self._rms, dtype=np.float64), [{} for _ in seeds]


def _fake_island(name, best_indiv, best_val_cost):
    return types.SimpleNamespace(
        name=name,
        best_overall_individual=best_indiv,
        best_val_cost=best_val_cost,
        last_validated_individual=None,
    )


def test_revalidate_each_recomputes_best_val_cost():
    from aerocapture.training.island_model import IslandModel

    model = IslandModel.__new__(IslandModel)  # bypass __init__
    model.islands = [
        _fake_island("pso", np.array([0.1, 0.2]), best_val_cost=999.0),
        _fake_island("ga", None, best_val_cost=999.0),  # no best -> skipped
    ]
    model.problem = _FakeProblem(rms=3.5)
    model.validation_seeds = [1, 2, 3]

    model.revalidate_each()

    assert model.islands[0].best_val_cost == 3.5
    assert np.array_equal(model.islands[0].last_validated_individual, np.array([0.1, 0.2]))
    # Island with no best_overall_individual is untouched.
    assert model.islands[1].best_val_cost == 999.0
    assert model.islands[1].last_validated_individual is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resume_enhancements.py::test_revalidate_each_recomputes_best_val_cost -v`
Expected: FAIL — `AttributeError: 'IslandModel' object has no attribute 'revalidate_each'`.

- [ ] **Step 3: Implement `revalidate_each`**

In `island_model.py`, add a method to `IslandModel` immediately after `validate_each` (after `island_model.py:451`):

```python
    def revalidate_each(self) -> None:
        """Re-validate each island's best_overall_individual under the current config.

        Called once after `from_checkpoint` on resume so each island's
        `best_val_cost` reflects the CURRENT cost_kwargs (notably a changed
        `cost_transform`) rather than the stale value baked into the checkpoint.
        Keeps the individual; only refreshes the metric baseline. Islands with no
        `best_overall_individual` (or with no validation seeds) are skipped.
        """
        if not self.validation_seeds:
            return
        for island in self.islands:
            if island.best_overall_individual is None:
                continue
            val_costs, _ = self.problem.evaluate_individual_records_per_seed(
                island.best_overall_individual,
                self.validation_seeds,
            )
            island.best_val_cost = float(np.sqrt(np.mean(val_costs**2)))
            island.last_validated_individual = island.best_overall_individual.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_resume_enhancements.py::test_revalidate_each_recomputes_best_val_cost -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_resume_enhancements.py
git commit -m "feat(training): IslandModel.revalidate_each for resume re-validation"
```

---

## Task 7: `IslandModel.resize_populations`

**Files:**
- Modify: `src/python/aerocapture/training/island_model.py` (add method; uses `resize_population`, `inject_into_pso`)
- Test: `tests/test_resume_enhancements.py`

This method resizes each island's restored population to `target_n`, re-evaluates under the problem's current seeds, re-stamps `rank` for GA/DE, and rebuilds PSO `particles` (positions + fresh velocity).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume_enhancements.py`. Uses a fake island whose `algorithm` is a `SimpleNamespace` with a real pymoo `Population`, and a problem stub returning deterministic costs:

```python
def test_resize_populations_grows_each_island():
    from pymoo.core.population import Population

    from aerocapture.training.island_model import IslandModel
    from aerocapture.training.param_spaces import ParamSpec

    class _P:
        def __init__(self):
            self.cost_kwargs = {"cost_transform": "linear"}

        def _run_batch(self, X):
            return np.arange(X.shape[0], dtype=np.float64)

    rng = np.random.default_rng(0)
    specs = [ParamSpec(name=f"p{i}", p_min=0.0, p_max=1.0, default=0.5) for i in range(2)]

    def _island(name):
        pop = Population.new("X", rng.random((4, 2)))
        pop.set("F", np.arange(4.0).reshape(-1, 1))
        algo = types.SimpleNamespace(pop=pop)
        return types.SimpleNamespace(name=name, algorithm=algo)

    model = IslandModel.__new__(IslandModel)
    model.islands = [_island("ga"), _island("de")]
    model.problem = _P()
    model.n_params = 2

    changed = model.resize_populations(
        target_n=10, specs=specs, rng=rng, fresh_fraction=0.2, velocity_scale=0.05
    )
    assert changed is True
    for isl in model.islands:
        assert isl.algorithm.pop.get("X").shape == (10, 2)
        assert isl.algorithm.pop.get("F").shape[0] == 10


def test_resize_populations_noop_when_size_matches():
    from pymoo.core.population import Population

    from aerocapture.training.island_model import IslandModel
    from aerocapture.training.param_spaces import ParamSpec

    class _P:
        cost_kwargs = {"cost_transform": "linear"}

        def _run_batch(self, X):
            return np.zeros(X.shape[0])

    rng = np.random.default_rng(0)
    specs = [ParamSpec(name="p0", p_min=0.0, p_max=1.0, default=0.5)]
    pop = Population.new("X", rng.random((5, 1)))
    pop.set("F", np.zeros((5, 1)))
    island = types.SimpleNamespace(name="ga", algorithm=types.SimpleNamespace(pop=pop))
    model = IslandModel.__new__(IslandModel)
    model.islands = [island]
    model.problem = _P()
    model.n_params = 1

    changed = model.resize_populations(target_n=5, specs=specs, rng=rng, fresh_fraction=0.2, velocity_scale=0.05)
    assert changed is False
```

NOTE: these fake islands use non-PSO `algorithm` namespaces (no `isinstance(algorithm, PSO)`), so the PSO particle branch is not exercised by unit tests. The GA/DE `FitnessSurvival` re-stamp and the PSO particle rebuild are verified together by the integration test in Task 8 (real island stack). Keep the PSO branch guarded by `isinstance(island.algorithm, PSO)` so the fakes skip it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resume_enhancements.py -k resize_populations -v`
Expected: FAIL — `AttributeError: 'IslandModel' object has no attribute 'resize_populations'`.

- [ ] **Step 3: Implement `resize_populations`**

In `island_model.py`, add to `IslandModel` after `revalidate_each`. Confirm `Population`, `PSO`, `FitnessSurvival`, `inject_into_pso`, and `resize_population` are importable (PSO + Population are already imported at module top; import the rest locally):

```python
    def resize_populations(
        self,
        target_n: int,
        specs: list[ParamSpec],
        rng: np.random.Generator,
        fresh_fraction: float,
        velocity_scale: float,
    ) -> bool:
        """Resize every island's restored population to ``target_n``.

        Grows (clone+jitter + fresh-random) or shrinks (best-N by F) each
        island's pop, re-evaluates the resized pop under the problem's CURRENT
        seeds, re-stamps GA/DE ``rank`` via FitnessSurvival, and rebuilds PSO
        ``particles`` (positions = new pop, fresh velocity). Returns True if any
        island changed size.
        """
        from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival  # noqa: PLC0415
        from pymoo.core.population import Population  # noqa: PLC0415

        from aerocapture.training.population import resize_population  # noqa: PLC0415

        any_changed = False
        for island in self.islands:
            pop = island.algorithm.pop
            if pop is None:
                continue
            cur_X = pop.get("X")
            if cur_X.shape[0] == target_n:
                continue
            any_changed = True
            cur_F = pop.get("F").flatten()
            new_X = resize_population(cur_X, cur_F, target_n, specs, rng, fresh_fraction=fresh_fraction)
            new_F = self.problem._run_batch(new_X)
            new_pop = Population.new("X", new_X)
            new_pop.set("F", new_F.reshape(-1, 1))
            if not isinstance(island.algorithm, PSO):
                new_pop = FitnessSurvival().do(self.problem, new_pop, n_survive=len(new_pop))
            island.algorithm.pop = new_pop
            if isinstance(island.algorithm, PSO):
                particles = Population.new("X", new_X.copy())
                particles.set("F", new_F.reshape(-1, 1).copy())
                particles.set("V", rng.uniform(-velocity_scale, velocity_scale, size=new_X.shape))
                island.algorithm.particles = particles
            if hasattr(island.algorithm, "_set_optimum"):
                island.algorithm._set_optimum()
        return any_changed
```

Add `from aerocapture.training.param_spaces import ParamSpec` to the module-level imports if not already present (check first: `rg -n "import ParamSpec|param_spaces import" src/python/aerocapture/training/island_model.py`). If absent, add it to avoid a NameError in the method signature annotation under `from __future__ import annotations` (annotations are strings, so a missing import only fails if evaluated; still import it for clarity and mypy).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_resume_enhancements.py -k resize_populations -v`
Expected: 2 passed.

- [ ] **Step 5: Lint**

Run: `./lint_code.sh`
Expected: no errors in `island_model.py`.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/island_model.py tests/test_resume_enhancements.py
git commit -m "feat(training): IslandModel.resize_populations for resume pop growth/shrink"
```

---

## Task 8: Wire `_train_islands` resume — revalidate + resize + transform reset

**Files:**
- Modify: `src/python/aerocapture/training/train.py:1619-1644` (islands resume block)
- Test: integration test in `tests/test_resume_enhancements.py`

- [ ] **Step 1: Add the wiring in `_train_islands`**

In `train.py`, inside `if resume_ckpt is not None:` (the block starting `train.py:1619`), AFTER the existing curator-seed push (after `train.py:1642`, the `problem.update_seeds(seed_curator.seed_list)` line) and BEFORE the closing `if verbose:` print at line 1643, insert:

```python
        # Reconcile population size to the configured n_pop (supports resuming a
        # small-pop run with a bigger pop). Done AFTER the curated seeds are
        # pushed so the re-eval of new individuals uses the right seeds.
        if config.optimizer.n_pop != island_model.islands[0].algorithm.pop.get("X").shape[0]:
            if verbose:
                old_n = island_model.islands[0].algorithm.pop.get("X").shape[0]
                print(f"  Resizing islands populations {old_n} -> {config.optimizer.n_pop}")
            island_model.resize_populations(
                target_n=config.optimizer.n_pop,
                specs=param_specs,
                rng=rng,
                fresh_fraction=config.optimizer.grow_fresh_fraction,
                velocity_scale=config.optimizer.islands.pso_inject_velocity_scale,
            )

        # Re-validate each island's best under the current config (refreshes
        # best_val_cost; auto-handles a changed cost_transform).
        if val_seeds:
            island_model.revalidate_each()

        # cost_transform change notice + stagnation reset.
        current_transform = problem.cost_kwargs.get("cost_transform", "linear")
        if resumed_cost_transform is None or resumed_cost_transform != current_transform:
            if verbose:
                print(f"  cost_transform changed {resumed_cost_transform!r} -> {current_transform!r}; re-validated best under new metric")
            for island in island_model.islands:
                island.stagnation_counter = 0
```

`resumed_cost_transform` comes from the Task 5 return-tuple change. `param_specs` is a parameter of `_train_islands` (confirm: `rg -n "def _train_islands" -A25 src/python/aerocapture/training/train.py` shows `param_specs: list[ParamSpec]`).

- [ ] **Step 2: Integration test — islands resume grows the population and re-validates**

Append to `tests/test_resume_enhancements.py`. This builds a real `IslandModel`, checkpoints it, then resumes with a larger n_pop and a different cost_transform via a fresh model + the wiring methods directly (exercises `from_checkpoint` -> `resize_populations` -> `revalidate_each` end-to-end against a stub problem):

```python
def test_islands_resume_grow_and_revalidate(tmp_path):
    from pymoo.core.population import Population

    from aerocapture.training.island_model import IslandModel
    from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
    from aerocapture.training.param_spaces import ParamSpec

    class _P:
        def __init__(self):
            self.cost_kwargs = {"cost_transform": "linear"}

        def _run_batch(self, X):
            return np.linspace(1.0, 2.0, X.shape[0])

        def evaluate_individual_records_per_seed(self, x, seeds):
            return np.full(len(seeds), 1.23, dtype=np.float64), [{} for _ in seeds]

    specs = [ParamSpec(name=f"p{i}", p_min=0.0, p_max=1.0, default=0.5) for i in range(2)]
    rng = np.random.default_rng(0)

    # k_top=1 so k_top*(n_islands-1)=2 <= n_pop=4 (IslandModel.__init__ guard).
    cfg = OptimizerConfig(seed_strategy="fixed", algorithm="islands", n_pop=4, validation_n_sims=3, islands=IslandSettings(k_top=1))
    model = IslandModel(
        config=cfg, problem=_P(), n_params=2, validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12], base_mc_seed=42, rng=rng,
    )
    # Seed each island with a real pop so checkpoint has something to write.
    for isl in model.islands:
        pop = Population.new("X", rng.random((4, 2)))
        pop.set("F", np.arange(4.0).reshape(-1, 1))
        isl.algorithm.pop = pop
        isl.algorithm.is_initialized = True
        isl.algorithm.n_iter = 1
        isl.best_overall_individual = pop.get("X")[0].copy()
        isl.best_val_cost = 999.0
    ckpt = tmp_path / "checkpoint_g00000.npz"
    model.checkpoint(ckpt, generation=0)

    # Resume into a BIGGER model.
    cfg2 = OptimizerConfig(seed_strategy="fixed", algorithm="islands", n_pop=12, validation_n_sims=3, islands=IslandSettings(k_top=1))
    p2 = _P()
    p2.cost_kwargs = {"cost_transform": "log"}  # changed transform
    model2 = IslandModel(
        config=cfg2, problem=p2, n_params=2, validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12], base_mc_seed=42, rng=np.random.default_rng(1),
    )
    gen, _curator, saved_transform = model2.from_checkpoint(ckpt)
    assert gen == 0
    assert saved_transform == "linear"

    model2.resize_populations(target_n=12, specs=specs, rng=np.random.default_rng(2), fresh_fraction=0.2, velocity_scale=0.05)
    model2.revalidate_each()

    for isl in model2.islands:
        assert isl.algorithm.pop.get("X").shape == (12, 2)
        assert isl.best_val_cost == 1.23  # re-validated under new metric, not the stale 999.0
```

- [ ] **Step 3: Run the integration test**

Run: `uv run pytest tests/test_resume_enhancements.py::test_islands_resume_grow_and_revalidate -v`
Expected: PASS. If `IslandModel.__init__` requires extra args, inspect with `rg -n "def __init__" -A20 src/python/aerocapture/training/island_model.py` and adjust the constructor call to match (do not change production code to fit the test).

- [ ] **Step 4: Import + full new-test run**

Run: `uv run python -c "import aerocapture.training.train" && uv run pytest tests/test_resume_enhancements.py tests/test_population_resize.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_resume_enhancements.py
git commit -m "feat(training): islands resume pop resize + revalidate + cost_transform reset"
```

---

## Task 9: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Lint the whole Python tree**

Run: `./lint_code.sh`
Expected: ruff clean, mypy strict clean.

- [ ] **Step 2: Run the training test suite (the areas this plan touches)**

Run: `uv run pytest tests/test_population_resize.py tests/test_resume_enhancements.py tests/test_train_interrupt.py tests/test_optimizer_config.py -v`
Expected: all passed. (`test_train_interrupt.py` guards the resume/checkpointed-best invariant; `test_optimizer_config.py` guards the config knobs.)

- [ ] **Step 3: Broader regression sweep**

Run: `uv run pytest tests -q -k "resume or checkpoint or island or optimizer or population"`
Expected: all passed. Investigate any failure before proceeding — do not skip.

- [ ] **Step 4: Confirm no placeholder leaks**

Run: `rg -n "TODO|FIXME|XXX" src/python/aerocapture/training/population.py src/python/aerocapture/training/island_model.py`
Expected: no new markers introduced by this work.

---

## Task 10: Documentation + smart-commit

**Files:** docs as needed via the skill.

- [ ] **Step 1: Run smart-commit over the whole branch**

Invoke the `smart-commit` skill, instructing it to take the WHOLE git branch (`feature/resume-enhancements`) into account so CLAUDE.md / README stay in sync with the new resume capabilities (`grow_fresh_fraction` knob, cost_transform persistence, islands re-validation/resize on resume).

---

## Self-Review Notes

- **Spec coverage:** Feature #1 islands re-validation -> Task 6 + Task 8. Feature #2 pop growth/shrink -> Task 1 (helper), Task 4 (single-algo), Task 7 + Task 8 (islands). Feature #3 cost_transform persist+detect+reset -> Task 3 + Task 5 (persist), Task 4 (single-algo detect/log), Task 8 (islands detect/reset). `grow_fresh_fraction` TOML knob -> Task 2. Shrink-to-best-N -> Task 1. PSO velocity extension -> Task 7. All spec sections mapped.
- **Type consistency:** `resize_population(pop_X, pop_F, target_n, specs, rng, fresh_fraction=0.2, jitter_sigma=0.02)` used identically in Tasks 1, 4, 7. `from_checkpoint` returns a 3-tuple everywhere after Task 5 (defined Task 5, consumed Task 8). `resize_populations(target_n, specs, rng, fresh_fraction, velocity_scale)` consistent Tasks 7/8. `revalidate_each()` no-arg consistent Tasks 6/8.
- **Single-algo cost_transform:** existing resume re-validation (`train.py:1275-1298`) already recomputes `best_val_cost` under the current transform, so feature #3 for single-algo is the persist + log notice only (no functional reset beyond that), as documented in Task 4 Step 4.
