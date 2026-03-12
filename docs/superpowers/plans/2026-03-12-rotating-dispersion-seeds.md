# Rotating Dispersion Seeds Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in per-generation Monte Carlo seed rotation to GA training, with full parent re-evaluation, to prevent overfitting to fixed dispersion scenarios.

**Architecture:** Python-side only. A new `patch_toml_mc_seed()` helper creates temp TOMLs with overridden `[monte_carlo].seed`. The GA loop in `train.py` computes `mc_seed = base_mc_seed + gen` each generation and threads it through all evaluations. All parents are re-evaluated on the new seed before tournament selection.

**Tech Stack:** Python 3.14, tomllib (stdlib), existing TOML writer in evaluate.py

**Spec:** `docs/superpowers/specs/2026-03-12-rotating-dispersion-seeds-design.md`

---

## Chunk 1: Configuration + Logger

### Task 1: Add `rotate_seeds` to GAConfig

**Files:**
- Modify: `src/python/aerocapture/training/config.py:57-71`
- Create: `tests/test_training_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_config.py`:

```python
"""Tests for training configuration dataclasses."""

from __future__ import annotations

from aerocapture.training.config import GAConfig


def test_ga_config_rotate_seeds_default_false() -> None:
    ga = GAConfig()
    assert ga.rotate_seeds is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_config.py::test_ga_config_rotate_seeds_default_false -v`
Expected: FAIL with `AttributeError: 'GAConfig' object has no attribute 'rotate_seeds'`

- [ ] **Step 3: Add the field to GAConfig**

In `src/python/aerocapture/training/config.py`, add to the `GAConfig` dataclass (after line 70, `n_runs`):

```python
    rotate_seeds: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training_config.py::test_ga_config_rotate_seeds_default_false -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_training_config.py
git commit -m "feat(training): add rotate_seeds flag to GAConfig"
```

### Task 2: Add `mc_seed` to TrainingLogger

**Files:**
- Modify: `src/python/aerocapture/training/logger.py:40-86`
- Modify: `tests/test_training_logger.py`

- [ ] **Step 1: Write the failing test — mc_seed recorded when provided**

In `tests/test_training_logger.py`, add to `TestTrainingLogger`:

```python
    def test_mc_seed_recorded(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn, mc_seed=99)
        assert logger.buffer[0]["mc_seed"] == 99
        logger.close()
```

- [ ] **Step 2: Write the failing test — mc_seed absent when not provided**

```python
    def test_mc_seed_absent_by_default(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        assert "mc_seed" not in logger.buffer[0]
        logger.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_logger.py::TestTrainingLogger::test_mc_seed_recorded tests/test_training_logger.py::TestTrainingLogger::test_mc_seed_absent_by_default -v`
Expected: FAIL with `TypeError: log_generation() got an unexpected keyword argument 'mc_seed'`

- [ ] **Step 4: Add `mc_seed` parameter to `log_generation()`**

In `src/python/aerocapture/training/logger.py`, modify the `log_generation` signature (line 40-48) to add `mc_seed: int | None = None` after the existing `weight_stats` parameter:

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
    ) -> None:
```

Then, after the `weight_stats` block (after line 82), add:

```python
        if mc_seed is not None:
            record["mc_seed"] = mc_seed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_logger.py -v`
Expected: ALL PASS (including existing tests — they don't pass `mc_seed`, so behavior is unchanged)

- [ ] **Step 6: Write the failing test — mc_seed written to JSONL file**

```python
    def test_mc_seed_written_to_jsonl(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn, mc_seed=77)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        record = json.loads(jsonl_file.read_text().strip())
        assert record["mc_seed"] == 77
```

- [ ] **Step 7: Run and verify it passes (should already pass)**

Run: `uv run pytest tests/test_training_logger.py::TestTrainingLogger::test_mc_seed_written_to_jsonl -v`
Expected: PASS (the record is already written to file in the existing code path)

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/logger.py tests/test_training_logger.py
git commit -m "feat(training): add mc_seed to TrainingLogger JSONL output"
```

---

## Chunk 2: TOML Seed Patching

### Task 3: Add `patch_toml_mc_seed()` helper

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`
- Test: `tests/test_toml_patching.py` (existing — add to it)

- [ ] **Step 1: Check existing TOML patching tests**

Run: `uv run pytest tests/test_toml_patching.py -v --co` to list existing test names. Understand the patterns used.

- [ ] **Step 2: Write the failing test — patch_toml_mc_seed creates temp file with overridden seed**

In `tests/test_toml_patching.py`, add:

```python
import tomllib
from aerocapture.training.evaluate import patch_toml_mc_seed

class TestPatchTomlMcSeed:
    def test_overrides_seed(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text('[monte_carlo]\nseed = 42\n\n[mission]\ntype = "aerocapture"\n')
        patched = patch_toml_mc_seed(base, 99)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 99
            assert data["mission"]["type"] == "aerocapture"
        finally:
            patched.unlink(missing_ok=True)

    def test_adds_seed_when_missing(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text('[mission]\ntype = "aerocapture"\n')
        patched = patch_toml_mc_seed(base, 55)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 55
        finally:
            patched.unlink(missing_ok=True)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_toml_patching.py::TestPatchTomlMcSeed -v`
Expected: FAIL with `ImportError: cannot import name 'patch_toml_mc_seed'`

- [ ] **Step 4: Implement `patch_toml_mc_seed()`**

In `src/python/aerocapture/training/evaluate.py`, add just before `evaluate_chromosome()` (before line 432). This keeps it near the other TOML helpers but doesn't split the private `_write_toml`/`_write_toml_section`/`_toml_value` cluster:

```python
def patch_toml_mc_seed(base_toml_path: str | Path, mc_seed: int) -> Path:
    """Create a temp TOML with [monte_carlo].seed overridden.

    Args:
        base_toml_path: Path to the base TOML config.
        mc_seed: The Monte Carlo seed to set.

    Returns:
        Path to the temp TOML file (caller must clean up).
    """
    import os
    import tomllib

    base_toml_path = Path(base_toml_path)
    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed

    fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="mc_seed_")
    output_path = Path(path_str)
    os.close(fd)
    _write_toml(toml_data, output_path)
    return output_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_toml_patching.py::TestPatchTomlMcSeed -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py tests/test_toml_patching.py
git commit -m "feat(training): add patch_toml_mc_seed() helper"
```

### Task 4: Add `mc_seed` to `evaluate_chromosome()`

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:432-483`
- Test: `tests/test_toml_patching.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_toml_patching.py`, add:

```python
import inspect
from aerocapture.training.evaluate import evaluate_chromosome

class TestEvaluateChromosomeMcSeed:
    def test_mc_seed_param_exists(self) -> None:
        sig = inspect.signature(evaluate_chromosome)
        assert "mc_seed" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_toml_patching.py::TestEvaluateChromosomeMcSeed::test_mc_seed_param_exists -v`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Implement `mc_seed` in `evaluate_chromosome()`**

Modify `evaluate_chromosome()` signature in `src/python/aerocapture/training/evaluate.py` (line 432-436):

```python
def evaluate_chromosome(
    xbit: npt.NDArray[np.int8],
    base_network: npt.NDArray[np.float64],
    config: TrainingConfig,
    cwd: str | Path | None = None,
    mc_seed: int | None = None,
) -> tuple[float, npt.NDArray[np.float64] | None]:
```

**NN path** (lines 455-460) — when `mc_seed` is set, create a patched temp TOML:

```python
    if config.guidance_type == "neural_network":
        weights = decode_direct(xbit, config) if config.ga.direct_encoding else perturb_network(xbit, base_network, config)
        nn_path = Path(cwd) / config.sim.nn_param_file
        write_nn_json(weights, config.network, nn_path)
        if mc_seed is not None:
            assert config.sim.toml_config is not None
            patched_toml = patch_toml_mc_seed(Path(cwd) / config.sim.toml_config, mc_seed)
            try:
                orig_toml = config.sim.toml_config
                config.sim.toml_config = str(patched_toml)
                final = run_simulation(config, cwd=cwd)
            finally:
                config.sim.toml_config = orig_toml
                patched_toml.unlink(missing_ok=True)
        else:
            final = run_simulation(config, cwd=cwd)
```

**Non-NN path** (lines 462-477) — compose the seed patch into the existing `write_guidance_toml` flow. Add `mc_seed` parameter to `write_guidance_toml()`:

First, modify `write_guidance_toml()` signature (line 295-300) to accept an optional seed:

```python
def write_guidance_toml(
    base_toml_path: str | Path,
    guidance_type: str,
    params: dict[str, float],
    output_path: str | Path | None = None,
    mc_seed: int | None = None,
) -> Path:
```

Then, after the guidance params are set (after line 322 `toml_data["guidance"][section_name] = params`), add:

```python
    if mc_seed is not None:
        toml_data.setdefault("monte_carlo", {})["seed"] = mc_seed
```

Now the non-NN path in `evaluate_chromosome` simply passes `mc_seed` through:

```python
    else:
        params = decode_params_from_chromosome(xbit, config)
        if config.sim.toml_config is None:
            msg = f"toml_config must be set for guidance_type={config.guidance_type}"
            raise ValueError(msg)
        base_toml = Path(cwd) / config.sim.toml_config
        patched_toml = write_guidance_toml(base_toml, config.guidance_type, params, mc_seed=mc_seed)
        try:
            orig_toml = config.sim.toml_config
            config.sim.toml_config = str(patched_toml)
            final = run_simulation(config, cwd=cwd)
        finally:
            config.sim.toml_config = orig_toml
            patched_toml.unlink(missing_ok=True)
```

Note: `config.sim.toml_config` restoration is now in `finally` for both paths, fixing the existing fragility.

- [ ] **Step 4: Add a test for seed composition in write_guidance_toml**

```python
class TestWriteGuidanceTomlMcSeed:
    def test_mc_seed_composed_into_patched_toml(self, tmp_path: Path) -> None:
        from aerocapture.training.evaluate import write_guidance_toml

        base = tmp_path / "base.toml"
        base.write_text(
            '[mission]\ntype = "aerocapture"\n\n[monte_carlo]\nseed = 42\n\n'
            '[guidance]\ntype = "equilibrium_glide"\n\n[guidance.equilibrium_glide]\nk_hdot = 1.0\n'
        )
        patched = write_guidance_toml(base, "equilibrium_glide", {"k_hdot": 2.0}, mc_seed=99)
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 99
            assert data["guidance"]["equilibrium_glide"]["k_hdot"] == 2.0
        finally:
            patched.unlink(missing_ok=True)

    def test_no_mc_seed_preserves_original(self, tmp_path: Path) -> None:
        from aerocapture.training.evaluate import write_guidance_toml

        base = tmp_path / "base.toml"
        base.write_text(
            '[mission]\ntype = "aerocapture"\n\n[monte_carlo]\nseed = 42\n\n'
            '[guidance]\ntype = "equilibrium_glide"\n\n[guidance.equilibrium_glide]\nk_hdot = 1.0\n'
        )
        patched = write_guidance_toml(base, "equilibrium_glide", {"k_hdot": 2.0})
        try:
            with open(patched, "rb") as f:
                data = tomllib.load(f)
            assert data["monte_carlo"]["seed"] == 42
        finally:
            patched.unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_toml_patching.py::TestEvaluateChromosomeMcSeed tests/test_toml_patching.py::TestWriteGuidanceTomlMcSeed -v`
Expected: PASS

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py tests/test_toml_patching.py
git commit -m "feat(training): add mc_seed parameter to evaluate_chromosome()"
```

---

## Chunk 3: GA Loop Integration + CLI

### Task 5: Wire seed rotation into the GA loop

**Files:**
- Modify: `src/python/aerocapture/training/train.py:185-407`
- Test: `tests/test_training_integration.py`

- [ ] **Step 1: Write the failing test — evaluate_chromosome called with mc_seed when rotate_seeds enabled**

In `tests/test_training_integration.py`, add the import `from unittest.mock import patch` (already present) and add:

```python
class TestRotateSeedsIntegration:
    def test_evaluate_called_with_mc_seed_when_rotate_enabled(self, tmp_path: Path) -> None:
        """When rotate_seeds=True, evaluate_chromosome receives mc_seed arg."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.save_dir = str(tmp_path)
        config.sim.toml_config = "dummy.toml"

        # Create a dummy TOML with [monte_carlo].seed
        dummy_toml = tmp_path / "dummy.toml"
        dummy_toml.write_text('[monte_carlo]\nseed = 10\n')

        mock_eval_calls: list[dict] = []

        def tracking_eval(*args, **kwargs):
            mock_eval_calls.append(kwargs.copy())
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=tracking_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # All evaluate calls should have mc_seed set
        assert len(mock_eval_calls) > 0
        for call in mock_eval_calls:
            assert "mc_seed" in call
            assert call["mc_seed"] == 10 + 0  # base_mc_seed(10) + gen(0)
```

- [ ] **Step 2: Write the failing test — evaluate NOT called with mc_seed when rotate_seeds disabled**

```python
    def test_evaluate_called_without_mc_seed_when_rotate_disabled(self, tmp_path: Path) -> None:
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = False
        config.save_dir = str(tmp_path)

        mock_eval_calls: list[dict] = []

        def tracking_eval(*args, **kwargs):
            mock_eval_calls.append(kwargs.copy())
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=tracking_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # No evaluate call should have mc_seed
        for call in mock_eval_calls:
            assert call.get("mc_seed") is None
```

- [ ] **Step 3: Write the failing test — parents re-evaluated when rotate_seeds enabled**

```python
    def test_parents_reevaluated_when_rotate_enabled(self, tmp_path: Path) -> None:
        """With rotate_seeds, total evals per gen = 2*n_pop (offspring + parents)."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.save_dir = str(tmp_path)
        config.sim.toml_config = "dummy.toml"

        dummy_toml = tmp_path / "dummy.toml"
        dummy_toml.write_text('[monte_carlo]\nseed = 10\n')

        eval_count = 0

        def counting_eval(*args, **kwargs):
            nonlocal eval_count
            eval_count += 1
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=counting_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # 1 gen, 1 subpop: 4 offspring + 4 parents = 8 evals
        assert eval_count == 8
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_integration.py::TestRotateSeedsIntegration -v`
Expected: FAIL

- [ ] **Step 5: Implement seed rotation in `train()`**

In `src/python/aerocapture/training/train.py`:

**5a.** Add `base_mc_seed` reading at the top of `train()`, after the `save_dir` setup (after line 219). Raise an error if `rotate_seeds` is enabled but `[monte_carlo].seed` is missing from the TOML:

```python
    # Read base MC seed from TOML for seed rotation
    base_mc_seed: int | None = None
    if config.ga.rotate_seeds:
        if not config.sim.toml_config:
            msg = "rotate_seeds requires a TOML config with [monte_carlo].seed"
            raise ValueError(msg)
        import tomllib

        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        with open(toml_path, "rb") as f:
            _toml = tomllib.load(f)
        base_mc_seed = _toml.get("monte_carlo", {}).get("seed")
        if base_mc_seed is None:
            msg = "rotate_seeds requires [monte_carlo].seed in the TOML config"
            raise ValueError(msg)
```

**5b.** Inside the generation loop (line 316), compute `mc_seed` at the top of each generation. Note this is inside the `for gen` loop but outside the `for k in range(config.ga.n_subpop)` loop:

```python
            for gen in range(gen_start, config.ga.n_gen):
                mc_seed = (base_mc_seed + gen) if base_mc_seed is not None else None
```

**5c.** Thread `mc_seed` into offspring evaluation (inside the `for k` subpop loop, lines 326-333):

```python
                    for i in range(len(offspring)):
                        cost, _ = evaluate_chromosome(
                            offspring[i],
                            base_network,
                            config,
                            cwd=cwd,
                            mc_seed=mc_seed,
                        )
                        offspring_costs[i] = cost
```

**5d.** Add parent re-evaluation before tournament (after offspring eval, before the combine step at line 336). This is inside the `for k` subpop loop. Note: `pop_costs` aliases `all_costs[k]` (numpy array), so in-place mutation updates the stored costs:

```python
                    # Re-evaluate parents on current seed when rotating
                    if mc_seed is not None:
                        for i in range(len(pop)):
                            cost, _ = evaluate_chromosome(
                                pop[i],
                                base_network,
                                config,
                                cwd=cwd,
                                mc_seed=mc_seed,
                            )
                            pop_costs[i] = cost
```

**5e.** Thread `mc_seed` into `logger.log_generation()` call (line 369):

```python
                logger.log_generation(
                    gen + 1,
                    populations,
                    all_costs,
                    best_overall_chrom if best_overall_chrom is not None else populations[0][0],
                    decode_fn,
                    weight_stats=ws,
                    mc_seed=mc_seed,
                )
```

- [ ] **Step 6: Run the integration tests**

Run: `uv run pytest tests/test_training_integration.py::TestRotateSeedsIntegration -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: ALL PASS (existing tests unaffected — they don't set `rotate_seeds`)

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_training_integration.py
git commit -m "feat(training): wire seed rotation into GA loop with parent re-evaluation"
```

### Task 6: Add `--rotate-seeds` CLI flag

**Files:**
- Modify: `src/python/aerocapture/training/train.py:410-474` (the `__main__` block)

- [ ] **Step 1: Add argparse argument**

In `src/python/aerocapture/training/train.py`, after the `--no-tui` argument (line 429), add:

```python
    parser.add_argument("--rotate-seeds", action="store_true", help="Rotate MC dispersion seed each generation (prevents overfitting to fixed scenarios)")
```

- [ ] **Step 2: Wire it to config**

After `cfg.guidance_type = args.guidance` (line 436), add:

```python
    cfg.ga.rotate_seeds = args.rotate_seeds
```

- [ ] **Step 3: Run lint**

Run: `uv run ruff check src/python/aerocapture/training/train.py && uv run ruff format --check src/python/aerocapture/training/train.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(training): add --rotate-seeds CLI flag"
```

### Task 7: Run full validation

- [ ] **Step 1: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run linting and type checking**

Run: `./lint_code.sh`
Expected: PASS

- [ ] **Step 3: Verify CLI help shows new flag**

Run: `uv run python -m aerocapture.training.train --help`
Expected: Output includes `--rotate-seeds` with description

- [ ] **Step 4: Final commit (if any lint fixes needed)**

```bash
git add -u
git commit -m "style: fix lint issues from seed rotation feature"
```
