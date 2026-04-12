# Epoch Seed Rotation + Validation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static seed evaluation with epoch-based seed rotation so the optimizer can't memorize scenarios, and add a validation gate for honest generalization monitoring.

**Architecture:** New TOML-configurable `training_n_sims` controls how many fresh random seeds each generation draws. A separate validation gate (fixed seeds, large N) fires periodically and on new-best to log true generalization performance. Both features integrate into the existing pymoo training loop without touching problem.py or the Rust side.

**Tech Stack:** Python (pymoo, numpy), TOML config, Rich TUI

---

### Task 1: Add training_n_sims / validation fields to OptimizerConfig

**Files:**
- Modify: `src/python/aerocapture/training/optimizer.py:46-75`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for new config fields**

In `tests/test_optimizer.py`, add to `TestOptimizerConfig`:

```python
def test_default_training_n_sims(self):
    cfg = OptimizerConfig()
    assert cfg.training_n_sims == 1

def test_default_validation_n_sims(self):
    cfg = OptimizerConfig()
    assert cfg.validation_n_sims == 1000

def test_default_validation_interval(self):
    cfg = OptimizerConfig()
    assert cfg.validation_interval == 50

def test_from_dict_training_n_sims(self):
    d = {"algorithm": "ga", "training_n_sims": 20}
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.training_n_sims == 20

def test_from_dict_validation_fields(self):
    d = {"algorithm": "ga", "validation_n_sims": 500, "validation_interval": 25}
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.validation_n_sims == 500
    assert cfg.validation_interval == 25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer.py::TestOptimizerConfig::test_default_training_n_sims tests/test_optimizer.py::TestOptimizerConfig::test_default_validation_n_sims tests/test_optimizer.py::TestOptimizerConfig::test_default_validation_interval tests/test_optimizer.py::TestOptimizerConfig::test_from_dict_training_n_sims tests/test_optimizer.py::TestOptimizerConfig::test_from_dict_validation_fields -v`
Expected: FAIL (attributes don't exist)

- [ ] **Step 3: Add fields to OptimizerConfig**

In `src/python/aerocapture/training/optimizer.py`, add 3 fields to the `OptimizerConfig` dataclass after `stress_inject`:

```python
    training_n_sims: int = 1
    validation_n_sims: int = 1000
    validation_interval: int = 50
```

No changes to `from_dict()` needed -- the top-level keys are already passed through via `top_level = {k: v for k, v in d.items() if k not in ("ga", "cma_es", "de", "pso")}` and `cls(**top_level, ...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_optimizer.py -v`
Expected: All pass (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/optimizer.py tests/test_optimizer.py
git commit -m "feat: add training_n_sims and validation config to OptimizerConfig"
```

---

### Task 2: Add validation dict support to TrainingLogger

**Files:**
- Modify: `src/python/aerocapture/training/logger.py:40-104`
- Test: `tests/test_training_logger.py`

- [ ] **Step 1: Write failing tests for validation logging**

In `tests/test_training_logger.py`, add:

```python
def test_validation_recorded(self, logger: TrainingLogger) -> None:
    val = {
        "mean_cost": 42.3,
        "median_cost": 38.1,
        "std_cost": 15.2,
        "p95_cost": 112.5,
        "worst_cost": 8432.0,
        "capture_rate": 0.97,
        "n_sims": 1000,
    }
    logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, validation=val)
    assert logger.buffer[0]["validation"] == val
    logger.close()

def test_validation_absent_by_default(self, logger: TrainingLogger) -> None:
    logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
    assert "validation" not in logger.buffer[0]
    logger.close()

def test_validation_written_to_jsonl(self, logger: TrainingLogger, tmp_path: Path) -> None:
    val = {"mean_cost": 42.3, "capture_rate": 0.97, "n_sims": 1000}
    logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, validation=val)
    logger.close()
    jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
    record = json.loads(jsonl_file.read_text().strip())
    assert record["validation"]["mean_cost"] == 42.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training_logger.py::TestTrainingLogger::test_validation_recorded tests/test_training_logger.py::TestTrainingLogger::test_validation_absent_by_default tests/test_training_logger.py::TestTrainingLogger::test_validation_written_to_jsonl -v`
Expected: FAIL (unexpected keyword argument 'validation')

- [ ] **Step 3: Add validation parameter to log_generation**

In `src/python/aerocapture/training/logger.py`, add `validation: dict | None = None` parameter to `log_generation()`:

```python
    def log_generation(
        self,
        generation: int,
        population: npt.NDArray[np.float64],
        costs: npt.NDArray[np.float64],
        best_individual: npt.NDArray[np.float64],
        decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None,
        weight_stats: dict[str, dict[str, float]] | None = None,
        mc_seed: int | None = None,
        pool_metrics: dict | None = None,
        gen_elapsed_s: float | None = None,
        gen_best_individual: npt.NDArray[np.float64] | None = None,
        validation: dict | None = None,
    ) -> None:
```

Then, after the `gen_elapsed_s` block (around line 100), add:

```python
        if validation is not None:
            record["validation"] = validation
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training_logger.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/logger.py tests/test_training_logger.py
git commit -m "feat: add validation dict support to TrainingLogger"
```

---

### Task 3: Add validation line to TUI display

**Files:**
- Modify: `src/python/aerocapture/training/display.py:74-134`

- [ ] **Step 1: Add validation metrics to _build_panel**

In `src/python/aerocapture/training/display.py`, in the `_build_panel` method, after the stagnation block (after line 125) and before the best params block, add:

```python
        # Validation metrics (shown when present)
        val = latest.get("validation")
        if val is not None:
            val_line = f"Val: mean={_format_cost(val['mean_cost'])} p95={_format_cost(val['p95_cost'])} cap={val['capture_rate']:.0%}"
            lines.append(val_line)
```

- [ ] **Step 2: Verify display still works (manual check)**

No automated test needed -- the display is a TUI component. The NoopDisplay path is unaffected. The validation line only appears when `latest.get("validation")` is non-None.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/display.py
git commit -m "feat: show validation metrics in TUI display"
```

---

### Task 4: Implement epoch seed rotation in the training loop

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

This is the core feature. Changes are in the generation loop of `train()`.

- [ ] **Step 1: Add epoch seed rotation before algorithm.next()**

In `src/python/aerocapture/training/train.py`, inside the generation loop (after `gen_wall_start = time.perf_counter()` at line 408, before `algorithm.next()` at line 411), add:

```python
                # Epoch seed rotation: fresh random seeds each generation
                if config.optimizer.training_n_sims > 1 and seed_pool is None:
                    epoch_seeds = rng.integers(0, 2**31, size=config.optimizer.training_n_sims).tolist()
                    problem.update_seeds(epoch_seeds)
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run pytest tests/test_problem.py tests/test_training_integration.py -v`
Expected: All pass (default `training_n_sims=1` means no rotation, backward compatible)

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: epoch seed rotation in training loop"
```

---

### Task 5: Implement validation gate in the training loop

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

- [ ] **Step 1: Initialize validation seeds at startup**

In `src/python/aerocapture/training/train.py`, after the `problem_seeds` setup (around line 320), add validation seed initialization:

```python
    # Validation gate: fixed seeds for honest generalization monitoring
    val_seeds: list[int] | None = None
    if config.optimizer.validation_n_sims > 0:
        val_rng = np.random.default_rng((mc_seed_val or 42) + 999)
        val_seeds = val_rng.integers(0, 2**31, size=config.optimizer.validation_n_sims).tolist()
```

- [ ] **Step 2: Add validation gate logic after best-tracking**

In the generation loop, after the best-tracking block (after `best_overall_individual = X[gen_best_idx].copy()` around line 422, before the seed pool update block at line 424), add:

We need to detect "new best" before updating `best_overall_cost`. Replace the best-tracking block (lines 418-422) with:

```python
                gen_best_idx = int(np.argmin(costs))
                gen_best_cost = float(costs[gen_best_idx])
                new_best_this_gen = gen_best_cost < best_overall_cost
                if new_best_this_gen:
                    best_overall_cost = gen_best_cost
                    best_overall_individual = X[gen_best_idx].copy()
```

Then after that block, add the validation gate:

```python
                # Validation gate: periodic + on-new-best
                validation_metrics: dict | None = None
                if val_seeds is not None and best_overall_individual is not None:
                    should_validate = new_best_this_gen or (gen + 1) % config.optimizer.validation_interval == 0
                    if should_validate:
                        val_costs = problem.evaluate_individual_per_seed(best_overall_individual, val_seeds)
                        val_captured = val_costs < 10000.0
                        validation_metrics = {
                            "mean_cost": float(np.mean(val_costs)),
                            "median_cost": float(np.median(val_costs)),
                            "std_cost": float(np.std(val_costs)),
                            "p95_cost": float(np.percentile(val_costs, 95)),
                            "worst_cost": float(np.max(val_costs)),
                            "capture_rate": float(np.mean(val_captured)),
                            "n_sims": len(val_seeds),
                        }
```

- [ ] **Step 3: Thread validation_metrics to the logger**

In the `logger.log_generation()` call (around line 503), add the `validation` keyword argument:

```python
                logger.log_generation(
                    gen + 1,
                    X,
                    costs,
                    best_overall_individual if best_overall_individual is not None else X[0],
                    decode_fn,
                    weight_stats=ws,
                    pool_metrics=pool_metrics,
                    gen_elapsed_s=gen_elapsed_s,
                    gen_best_individual=gen_best_individual,
                    validation=validation_metrics,
                )
```

- [ ] **Step 4: Also handle the seed pool re-evaluation best-tracking**

The seed pool block (lines 424-446) also updates `best_overall_cost`. The `new_best_this_gen` flag must also cover this case. After the seed pool re-tracking block:

```python
                    if gen_best_cost < best_overall_cost:
                        best_overall_cost = gen_best_cost
                        best_overall_individual = X[gen_best_idx].copy()
```

Change to:

```python
                    if gen_best_cost < best_overall_cost:
                        best_overall_cost = gen_best_cost
                        best_overall_individual = X[gen_best_idx].copy()
                        new_best_this_gen = True
```

And move the validation gate block to AFTER all the seed pool / stress test / corridor blocks, right before the logging section. This ensures validation fires after all best-tracking is finalized for the generation.

- [ ] **Step 5: Verify existing tests still pass**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat: validation gate with periodic + on-new-best evaluation"
```

---

### Task 6: Update common.toml with new settings

**Files:**
- Modify: `configs/training/common.toml`

- [ ] **Step 1: Add new keys to [optimizer] section**

In `configs/training/common.toml`, add after the `seed_pool_interval = 50` line:

```toml
training_n_sims = 20          # fresh random seeds per individual per generation (epoch rotation)
validation_n_sims = 1000      # sims for validation gate (fixed seeds, honest generalization metric)
validation_interval = 50      # periodic validation every N gens (also triggers on new best)
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/common.toml
git commit -m "config: add training_n_sims=20 and validation gate settings to common.toml"
```

---

### Task 7: Run full test suite and lint

**Files:** None (verification only)

- [ ] **Step 1: Run linter**

Run: `./lint_code.sh`
Expected: No errors

- [ ] **Step 2: Run full Python test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All pass

- [ ] **Step 3: Run Rust checks**

Run: `./check_all.sh`
Expected: All pass (no Rust changes, but verify nothing broke)

- [ ] **Step 4: Fix any issues found**

Address lint or test failures if any.

- [ ] **Step 5: Commit fixes if needed**

---

### Task 8: Smart commit (final)

- [ ] **Step 1: Invoke `smart-commit` skill**

Use the `smart-commit` skill, telling it to take the whole git branch into account. This will sync CLAUDE.md and README.md with the changes and create a final commit.
