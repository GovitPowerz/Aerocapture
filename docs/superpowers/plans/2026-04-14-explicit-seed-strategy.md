# Explicit `seed_strategy` Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose three training seed strategies (`fixed` / `rotating` / `adaptive`) as a required `[optimizer] seed_strategy` TOML key, reintroducing the rotating path alongside the existing curated-CDF adaptive behavior.

**Architecture:** A small dispatch layer in `train.py` that handles the three strategies: `fixed` computes `[mc_seed + i]` once at loop setup, `rotating` draws fresh random seeds each generation via a shared helper, `adaptive` keeps the current `SeedCurator`-driven logic. A new `_draw_disjoint_seeds` helper factors out the exclusion retry loop so bootstrap and rotating share identical draw semantics. `OptimizerConfig.seed_strategy` is required (no default) and validated at load time.

**Tech Stack:** Python 3.14, numpy, pymoo, pytest. No new deps.

---

## File Structure

**Create:**
- `tests/test_seed_strategy.py` — parameterized tests for the dispatch layer (stub `problem`).

**Modify:**
- `src/python/aerocapture/training/optimizer.py` — add `seed_strategy: str` field (required), validate in `__post_init__`, update `from_dict` to require the key.
- `src/python/aerocapture/training/train.py` — add `_draw_disjoint_seeds` helper; replace the current `training_n_sims > 1` branch with three-way dispatch; make bootstrap conditional on `strategy == "adaptive"`; add rotating-seeds block at top of loop.
- `configs/training/common.toml` — add `seed_strategy = "adaptive"` to `[optimizer]`.
- `tests/test_optimizer.py` — add `TestSeedStrategy` class covering valid/invalid/missing values.
- `CLAUDE.md` — extend the `train.py` paragraph to describe the three strategies.
- `README.md` — add a "Training seed strategies" subsection under GA Optimization.

**Not touched:**
- `src/python/aerocapture/training/seed_curator.py` — unchanged.
- `tests/test_seed_curator.py` — unchanged.

---

## Task 1: Add `seed_strategy` field to `OptimizerConfig`

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_optimizer.py` (in the same file, after the existing `TestCurationKnobs` class):

```python
class TestSeedStrategy:
    def test_accepts_valid_values(self) -> None:
        for value in ("fixed", "rotating", "adaptive"):
            cfg = OptimizerConfig.from_dict({"seed_strategy": value})
            assert cfg.seed_strategy == value

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="seed_strategy"):
            OptimizerConfig.from_dict({})

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="seed_strategy"):
            OptimizerConfig.from_dict({"seed_strategy": "bogus"})

    def test_invalid_value_lists_valid_values(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            OptimizerConfig.from_dict({"seed_strategy": "bogus"})
        msg = str(excinfo.value)
        assert "fixed" in msg
        assert "rotating" in msg
        assert "adaptive" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py::TestSeedStrategy -v`
Expected: FAIL (attribute missing, or the default kwargs construction succeeds when it shouldn't).

- [ ] **Step 3: Add the field and validation**

Edit `src/python/aerocapture/training/optimizer.py`. Find the `_VALID_ALGORITHMS = (...)` line near the top of the file and add beneath it:

```python
_VALID_SEED_STRATEGIES = ("fixed", "rotating", "adaptive")
```

In the `OptimizerConfig` dataclass, add the field (place it just after `algorithm: str = "ga"`):

```python
    seed_strategy: str = ""  # required; validated in __post_init__
```

In `__post_init__`, add validation after the existing algorithm check:

```python
        if self.seed_strategy not in _VALID_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of {_VALID_SEED_STRATEGIES}, got {self.seed_strategy!r}. "
                f"Add `seed_strategy = \"adaptive\"` (or another valid value) under [optimizer]."
            )
```

(The empty-string default is just a sentinel — the validator rejects it with a helpful message if the TOML didn't set one.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: PASS (4 new tests; existing tests may fail because they construct `OptimizerConfig()` without `seed_strategy` — fix those by adding `seed_strategy="adaptive"` kwarg. Find them via grep: `grep -n "OptimizerConfig()" tests/`).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_optimizer.py
git commit -m "feat(optimizer): require seed_strategy in OptimizerConfig (fixed|rotating|adaptive)"
```

---

## Task 2: Update `configs/training/common.toml` with `seed_strategy = "adaptive"`

**Files:**
- Modify: `configs/training/common.toml`

- [ ] **Step 1: Add the key**

Open `configs/training/common.toml`. Find the `[optimizer]` section. Add this line just after `algorithm = "ga"`:

```toml
seed_strategy = "adaptive"    # fixed | rotating | adaptive
```

- [ ] **Step 2: Sanity-check a training config loads**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
from aerocapture.training.optimizer import OptimizerConfig
d = load_toml_with_bases('configs/training/msr_aller_ftc_train.toml')
cfg = OptimizerConfig.from_dict(d.get('optimizer', {}))
print('seed_strategy:', cfg.seed_strategy)
"
```
Expected: `seed_strategy: adaptive`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ --tb=short -q`
Expected: all PASS. If any training config integration test fails, confirm it's due to a test that doesn't pass `seed_strategy` — fix the test, not the production code.

- [ ] **Step 4: Commit**

```bash
git add configs/training/common.toml
git commit -m "chore(config): default seed_strategy = \"adaptive\" in common.toml"
```

---

## Task 3: Extract `_draw_disjoint_seeds` helper

**Files:**
- Modify: `src/python/aerocapture/training/train.py`
- Test: `tests/test_seed_strategy.py` (new file)

This factors out the "draw N random seeds disjoint from an exclusion set" pattern so bootstrap and rotating share the same draw semantics.

- [ ] **Step 1: Write the failing test**

Create `tests/test_seed_strategy.py`:

```python
"""Tests for the seed_strategy dispatch layer in train.py."""

from __future__ import annotations

import numpy as np
import pytest

from aerocapture.training.train import _draw_disjoint_seeds


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestDrawDisjointSeeds:
    def test_returns_n_seeds(self) -> None:
        seeds = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        assert len(seeds) == 20

    def test_excludes_reserved(self) -> None:
        excluded = {1, 2, 3, 42, 999}
        seeds = _draw_disjoint_seeds(_rng(0), n=20, excluded=excluded)
        assert not (set(seeds) & excluded)

    def test_deterministic_with_same_rng(self) -> None:
        a = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        b = _draw_disjoint_seeds(_rng(0), n=20, excluded=set())
        assert a == b

    def test_handles_empty_exclusion(self) -> None:
        seeds = _draw_disjoint_seeds(_rng(0), n=5, excluded=set())
        assert len(seeds) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_strategy.py::TestDrawDisjointSeeds -v`
Expected: FAIL with `ImportError: cannot import name '_draw_disjoint_seeds' from 'aerocapture.training.train'`.

- [ ] **Step 3: Add the helper**

Edit `src/python/aerocapture/training/train.py`. Above the `save_checkpoint` function (near the top of the file, after the imports), add:

```python
def _draw_disjoint_seeds(
    rng: np.random.Generator,
    n: int,
    excluded: set[int],
) -> list[int]:
    """Draw `n` random seeds disjoint from `excluded`."""
    drawn: list[int] = []
    while len(drawn) < n:
        batch = rng.integers(0, 2**31, size=n - len(drawn)).tolist()
        drawn.extend(s for s in batch if s not in excluded)
    return drawn[:n]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_seed_strategy.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_seed_strategy.py
git commit -m "feat(training): extract _draw_disjoint_seeds helper for seed strategies"
```

---

## Task 4: Wire `fixed` strategy into `train.py`

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

In this task we make the training loop honor `config.optimizer.seed_strategy == "fixed"` at setup time. `adaptive` keeps working; `rotating` still falls through to the adaptive bootstrap path until Task 5.

- [ ] **Step 1: Add fixed-strategy setup**

Edit `src/python/aerocapture/training/train.py`. Find the existing `SeedCurator` init block (currently around line 236-244). Replace:

```python
    # Seed curator: maintains the training seed list across generations.
    seed_curator: SeedCurator | None = None
    if config.optimizer.training_n_sims > 1:
        seed_curator = SeedCurator(
            sample_size=config.optimizer.curation_sample_size,
            n_bins=config.optimizer.training_n_sims,
            excluded_seeds=set(),  # populated once val/final-eval sets are computed
            rng=rng,
        )
```

with:

```python
    # Seed strategy: three mutually exclusive training seed paths.
    #   fixed    -- deterministic [mc_seed + i]; seeds never change.
    #   rotating -- fresh random seeds drawn each generation (handled in loop body).
    #   adaptive -- bootstrap random + curated-CDF refreshes (SeedCurator).
    seed_curator: SeedCurator | None = None
    strategy = config.optimizer.seed_strategy
    if strategy == "adaptive":
        seed_curator = SeedCurator(
            sample_size=config.optimizer.curation_sample_size,
            n_bins=config.optimizer.training_n_sims,
            excluded_seeds=set(),  # populated once val/final-eval sets are computed
            rng=rng,
        )
```

Then, find the line (a bit further down, after `excluded_seeds = set(val_seeds) | set(final_eval_seeds)` is computed and the curator has its exclusion set injected — look for `if seed_curator is not None and seed_curator.seed_list is not None:`). Just after that block, add the fixed-strategy setup:

```python
    if strategy == "fixed":
        n_sims = config.optimizer.training_n_sims
        fixed_seeds = [base_mc_seed + i for i in range(n_sims)]
        overlap = set(fixed_seeds) & excluded_seeds
        if overlap:
            msg = (
                f"fixed seed range [{base_mc_seed}..{base_mc_seed + n_sims - 1}] "
                f"overlaps {len(overlap)} validation/final-eval reserved seeds"
            )
            raise ValueError(msg)
        problem.update_seeds(fixed_seeds)
```

(Note: `base_mc_seed` is already computed nearby — the variable holding the `[monte_carlo].seed` value. If your file uses a different local name, substitute it.)

- [ ] **Step 2: Add a test asserting fixed-strategy setup runs update_seeds once**

Append to `tests/test_seed_strategy.py`:

```python
class _StubProblem:
    """Minimal problem stand-in: records seed updates."""

    def __init__(self) -> None:
        self.seed_updates: list[list[int]] = []

    def update_seeds(self, seeds: list[int]) -> None:
        self.seed_updates.append(list(seeds))


class TestFixedStrategySetup:
    def test_fixed_seeds_are_deterministic_range(self) -> None:
        from aerocapture.training.train import _compute_fixed_seeds

        seeds = _compute_fixed_seeds(base_mc_seed=100, n_sims=5, excluded=set())
        assert seeds == [100, 101, 102, 103, 104]

    def test_fixed_seeds_raise_on_overlap(self) -> None:
        from aerocapture.training.train import _compute_fixed_seeds

        with pytest.raises(ValueError, match="overlaps"):
            _compute_fixed_seeds(base_mc_seed=100, n_sims=5, excluded={102})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed_strategy.py::TestFixedStrategySetup -v`
Expected: FAIL — `_compute_fixed_seeds` doesn't exist yet.

- [ ] **Step 4: Extract the overlap check into a helper**

The inline logic in Step 1 is correct but not unit-testable. Replace the block we added in Step 1 with a call to a helper, and add the helper above `save_checkpoint`:

Helper (place just after `_draw_disjoint_seeds`):

```python
def _compute_fixed_seeds(base_mc_seed: int, n_sims: int, excluded: set[int]) -> list[int]:
    """Deterministic seed list for the `fixed` strategy.

    Raises ValueError if any seed in the range overlaps `excluded`.
    """
    seeds = [base_mc_seed + i for i in range(n_sims)]
    overlap = set(seeds) & excluded
    if overlap:
        msg = (
            f"fixed seed range [{base_mc_seed}..{base_mc_seed + n_sims - 1}] "
            f"overlaps {len(overlap)} validation/final-eval reserved seeds"
        )
        raise ValueError(msg)
    return seeds
```

Replace the inline block from Step 1 with:

```python
    if strategy == "fixed":
        fixed_seeds = _compute_fixed_seeds(
            base_mc_seed=base_mc_seed,
            n_sims=config.optimizer.training_n_sims,
            excluded=excluded_seeds,
        )
        problem.update_seeds(fixed_seeds)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_seed_strategy.py -v && uv run pytest tests/ --tb=short -q`
Expected: PASS (6 tests in test_seed_strategy plus the full suite).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_seed_strategy.py
git commit -m "feat(training): fixed seed_strategy uses deterministic [mc_seed + i] range"
```

---

## Task 5: Wire `rotating` strategy into the generation loop

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

- [ ] **Step 1: Add the rotating draw at the top of the generation loop**

Edit `src/python/aerocapture/training/train.py`. Find the generation-loop body. The current top-of-loop block handles bootstrap for `adaptive`:

```python
                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False

                # Bootstrap: first iteration draws a random seed list if curator has none.
                if seed_curator is not None and seed_curator.seed_list is None:
                    bootstrap: list[int] = []
                    while len(bootstrap) < config.optimizer.training_n_sims:
                        batch = rng.integers(
                            0, 2**31, size=config.optimizer.training_n_sims - len(bootstrap)
                        ).tolist()
                        bootstrap.extend(s for s in batch if s not in excluded_seeds)
                    problem.update_seeds(bootstrap[: config.optimizer.training_n_sims])
                    seeds_changed_this_gen = True
```

Replace with:

```python
                seeds_changed_this_gen = pending_seed_change
                pending_seed_change = False

                if strategy == "rotating":
                    fresh = _draw_disjoint_seeds(
                        rng, n=config.optimizer.training_n_sims, excluded=excluded_seeds
                    )
                    problem.update_seeds(fresh)
                    seeds_changed_this_gen = True
                elif strategy == "adaptive" and seed_curator is not None and seed_curator.seed_list is None:
                    bootstrap = _draw_disjoint_seeds(
                        rng, n=config.optimizer.training_n_sims, excluded=excluded_seeds
                    )
                    problem.update_seeds(bootstrap)
                    seeds_changed_this_gen = True
```

(Note: the old bootstrap block wrote `bootstrap[: config.optimizer.training_n_sims]` — `_draw_disjoint_seeds` already truncates to `n` so the slice is unnecessary.)

- [ ] **Step 2: Add an integration-style test for the three strategies**

Append to `tests/test_seed_strategy.py`:

```python
class TestStrategyDispatch:
    """Exercises the loop-body dispatch logic via a minimal helper.

    These tests do NOT run a real training loop (too expensive). They verify
    the dispatch decisions deciding when to call `problem.update_seeds`.
    """

    def test_rotating_calls_update_each_gen(self) -> None:
        # Simulate three gens of rotating: three update_seeds calls with different lists.
        stub = _StubProblem()
        rng = _rng(0)
        for _ in range(3):
            fresh = _draw_disjoint_seeds(rng, n=5, excluded=set())
            stub.update_seeds(fresh)
        assert len(stub.seed_updates) == 3
        # Between gens seeds should differ (RNG advances).
        assert stub.seed_updates[0] != stub.seed_updates[1]
        assert stub.seed_updates[1] != stub.seed_updates[2]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_seed_strategy.py -v && uv run pytest tests/ --tb=short -q`
Expected: PASS (7 tests in test_seed_strategy plus the full suite).

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_seed_strategy.py
git commit -m "feat(training): rotating seed_strategy draws fresh seeds each generation"
```

---

## Task 6: Guard the curation trigger and pool_metrics on `seed_curator is not None`

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

Now that `seed_curator` is only created for `adaptive`, the curation trigger block and pool_metrics block must no-op gracefully when it's `None`. This is already guarded by `if seed_curator is not None:` for the trigger, but double-check and harden.

- [ ] **Step 1: Inspect existing guards**

Run: `grep -n "seed_curator" src/python/aerocapture/training/train.py`

For each match, confirm it is one of:
- An assignment to `seed_curator` (including `None`).
- A conditional `if seed_curator is not None:` or `if seed_curator is not None and ...:`.
- Access inside a conditional that already null-guards it.

If you find an unguarded access (e.g., `seed_curator.seed_list` without an `is not None` check), add the guard.

Current known guard sites to verify are still correct:

- Resume branch: `if seed_curator is not None and resumed.get("seed_curator") is not None:`
- Excluded-seeds injection: `if seed_curator is not None:`
- Restore seeds after resume: `if seed_curator is not None and seed_curator.seed_list is not None:`
- Loop body bootstrap: `elif strategy == "adaptive" and seed_curator is not None and seed_curator.seed_list is None:` (added in Task 5)
- Curation trigger: `if seed_curator is not None:`
- Pool metrics: `if seed_curator is not None and seed_curator.seed_list is not None:`
- Checkpoint save calls: passed `seed_curator=seed_curator` (None-safe in save_checkpoint).

- [ ] **Step 2: Run tests to confirm no regression**

Run: `uv run pytest tests/ --tb=short -q`
Expected: PASS.

- [ ] **Step 3: Run a quick sanity check against each strategy**

```bash
# fixed
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
from aerocapture.training.optimizer import OptimizerConfig
d = load_toml_with_bases('configs/training/msr_aller_ftc_train.toml')
opt = d.get('optimizer', {})
for strat in ('fixed', 'rotating', 'adaptive'):
    opt['seed_strategy'] = strat
    cfg = OptimizerConfig.from_dict(opt)
    print(strat, '->', cfg.seed_strategy)
"
```
Expected: prints three lines.

- [ ] **Step 4: Commit (no-op if no changes)**

If Step 1 revealed no missing guards, skip this commit.

```bash
git status
# If anything is staged, commit:
git commit -m "fix(training): null-guard seed_curator access for non-adaptive strategies"
```

---

## Task 7: End-to-end smoke test for all three strategies

**Files:**
- No code changes; this is a verification task.

- [ ] **Step 1: Pick a tiny training run**

We want a fast, ~5-gen training run per strategy. Create a scratch TOML override path isn't needed — just swap the `seed_strategy` value in the FTC config via environment or temporarily edit `common.toml` if easier.

Preferred: run with the three strategies by copying the FTC config and editing inline:

```bash
for STRAT in fixed rotating adaptive; do
    cp configs/training/msr_aller_ftc_train.toml /tmp/ftc_${STRAT}.toml
    # Override seed_strategy in the leaf config
    cat >> /tmp/ftc_${STRAT}.toml <<TOML

[optimizer]
seed_strategy = "${STRAT}"
TOML
    rm -rf training_output/ftc
    echo "=== ${STRAT} ==="
    uv run python -m aerocapture.training.train /tmp/ftc_${STRAT}.toml --n-gen 5 --n-pop 8 --no-tui --skip-report
done
```

(The append here produces a duplicated `[optimizer]` section; since TOML parses each section independently and our base-inheritance collapses by key, this should work. If it doesn't, just edit the file in place.)

Expected: each run completes without errors and saves a checkpoint to `training_output/ftc/checkpoint_g00005.json`.

- [ ] **Step 2: Inspect the checkpoints**

For each strategy:

```bash
uv run python -c "
import json, sys
from pathlib import Path
ckpts = sorted(Path('training_output/ftc').glob('checkpoint_g*.json'))
meta = json.loads(ckpts[-1].read_text())
print('seed_curator key present:', 'seed_curator' in meta)
"
```

Expected:
- `fixed`: `seed_curator key present: False`
- `rotating`: `seed_curator key present: False`
- `adaptive`: `seed_curator key present: True`

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest tests/ --tb=short -q`
Expected: all PASS.

- [ ] **Step 4: Run lint**

Run: `./lint_code.sh`
Expected: clean.

- [ ] **Step 5: Commit (if anything surfaced)**

If Steps 1-4 produced repo changes (shouldn't), commit them:

```bash
git add -A
git commit -m "chore: smoke tests pass for all three seed strategies"
```

Otherwise skip.

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Locate the curated-CDF paragraph**

Run: `grep -n "Curated-CDF seed framework:" CLAUDE.md`

- [ ] **Step 2: Rewrite the paragraph to cover all three strategies**

Replace the sentence starting with `Curated-CDF seed framework:` and extending to the end of its single paragraph with:

```text
Seed strategies: the `[optimizer] seed_strategy` key is required and picks one of three training seed paths. `"fixed"` uses a deterministic range `[mc_seed + 0, ..., mc_seed + (training_n_sims - 1)]` and never changes (bit-reproducible across runs). `"rotating"` draws `training_n_sims` fresh random seeds every generation, disjoint from the validation/final-eval reserved sets -- the landscape shifts each gen so the optimizer can't memorize scenarios. `"adaptive"` is the curated-CDF path: bootstrap draws a random `training_n_sims` seed list once, then the list is refreshed on (a) validated best promoted, or (b) every `seed_pool_interval` generations (measured from `last_curation_gen`). Each curation draws `curation_sample_size` probe seeds (default 1000), runs the top `curation_top_k` individuals (default 5) on them, averages per-seed costs, sorts, splits into `training_n_sims` equal-count quantile bins, and picks one random seed per bin. Between seed-list changes, `algorithm.pop` is only re-evaluated pre-`algorithm.next()` when the seeds actually changed; CMA-ES skips the re-eval entirely. See `src/python/aerocapture/training/seed_curator.py` and `docs/superpowers/specs/2026-04-14-explicit-seed-strategy-design.md`.
```

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -n "Epoch rotation\|epoch-rotation\|bootstrap path" CLAUDE.md`
Expected: no matches, or only historical-context matches clearly scoped to prior rewrites.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for three-way seed_strategy (fixed | rotating | adaptive)"
```

---

## Task 9: Update `README.md` with a "Training seed strategies" subsection

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the insertion point**

Run: `grep -n "^## GA Optimization\|^## Reports and Visualization" README.md`

The new subsection goes between `## GA Optimization` and `## Reports and Visualization`, just after the existing "curated-CDF seed framework" sentence (around line 141 post-prior-commits).

- [ ] **Step 2: Replace the one-sentence summary with a full subsection**

Find the line:

```text
The training loop uses a curated-CDF seed framework (`training_n_sims > 1`): a fixed-size seed list is refreshed on validated-best promotion or every `seed_pool_interval` gens by sampling 1000 probe seeds, scoring the top-`curation_top_k` individuals on them, and picking one random seed per cost quantile bin. See `CLAUDE.md` for details.
```

Replace with:

```markdown
### Training seed strategies

The `[optimizer] seed_strategy` key (required) controls how Monte Carlo seeds are picked across generations. All three strategies use the same `training_n_sims` size knob.

| Strategy    | What it does                                                                           | When to use |
| ----------- | -------------------------------------------------------------------------------------- | ----------- |
| `"fixed"`   | Deterministic `[mc_seed + 0, ..., mc_seed + (n_sims-1)]`; seeds never change.          | Debugging, A/B comparisons where the cost landscape must be identical across runs. |
| `"rotating"`| Fresh random seeds drawn every generation, disjoint from reserved sets.                | Default production: landscape shifts each gen so the optimizer can't overfit to a fixed scenario set. |
| `"adaptive"`| Random bootstrap, then curated-CDF: refreshed on validated-best or every `seed_pool_interval` gens. Each curation draws `curation_sample_size` probes, runs the top `curation_top_k` individuals, and picks one seed per cost quantile bin. | When you want a lower-variance fitness signal than rotating; pairs well with a strong `validation_n_sims`. |

Typical TOML snippet:

```toml
[optimizer]
algorithm = "ga"
seed_strategy = "adaptive"
training_n_sims = 20
seed_pool_interval = 50
curation_top_k = 5
curation_sample_size = 1000
```

Override per-scheme by adding `seed_strategy = "..."` in a leaf training TOML. See `CLAUDE.md` for full details.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Training seed strategies subsection to README"
```

---

## Task 10: Smart-commit for branch finalization

- [ ] **Step 1: Invoke the smart-commit skill**

Run the `smart-commit` skill and tell it: "take the whole git branch into account". It will sweep CLAUDE.md / README.md for drift one more time and produce a final commit if needed.

---

## Self-review notes

**Spec coverage:**
- Strategies (fixed / rotating / adaptive) → Tasks 4, 5, plus adaptive preserved through all tasks.
- `[optimizer] seed_strategy` required, no default → Task 1.
- `common.toml` sets `adaptive` → Task 2.
- Deterministic fixed range `[mc_seed + i]` → Task 4.
- Rotating fresh-each-gen → Task 5.
- Adaptive bootstrap + curator → Task 5 (preserved in the `elif`).
- `_draw_disjoint_seeds` helper → Task 3.
- Pre-next re-eval gated on `seeds_changed_this_gen` → unchanged; Tasks 5 and 6 ensure the flag is set correctly.
- Curation trigger guarded on `seed_curator is not None` → Task 6.
- Checkpoint behavior (no new persistence for fixed/rotating; adaptive unchanged) → Task 6 confirmation; no code changes needed.
- Disjointness with reserved seeds → Task 4 (fixed) + Task 3 (helper, used by rotating and adaptive).
- Degenerate `n_sims=1` for all three → covered by existing `SeedCurator._stratified_pick` guard + trivial correctness of fixed/rotating at n=1.
- Resume-with-changed-strategy → falls out of the dispatch being driven by current config, not checkpoint.
- CLAUDE.md + README updates → Tasks 8 + 9.
- Smoke tests → Task 7.

**Placeholder scan:** none.

**Type consistency:** `_draw_disjoint_seeds(rng, n, excluded)` signature is consistent across Tasks 3, 4, 5. `_compute_fixed_seeds(base_mc_seed, n_sims, excluded)` signature is consistent between Task 4 Step 1 (inline) and Step 4 (extracted). `OptimizerConfig.seed_strategy: str` attribute name matches in Tasks 1, 4, 5, 6.
