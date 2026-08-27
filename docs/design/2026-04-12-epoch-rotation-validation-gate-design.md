# Design: Epoch Seed Rotation + Validation Gate

## Problem

The training pipeline evaluates each individual on a fixed set of MC seeds (typically 1). The optimizer finds a good solution for those specific scenarios within ~20 generations, then stalls because the fitness landscape is static. This happens across all guidance schemes and all optimizers (GA, CMA-ES, DE, PSO). The adaptive seed pool (`--adaptive-seeds`) was designed to address this but converges to a stable "hardest-N" set, producing the same stall with extra complexity.

## Solution

Two layered features:

1. **Epoch seed rotation:** Each generation draws fresh random seeds for evaluation. The landscape shifts every generation, so the optimizer can never memorize specific scenarios. Individuals must generalize to survive.
2. **Validation gate:** Periodic + on-new-best large-N evaluation on fixed seeds. Provides an honest, stable metric of true generalization performance without influencing training.

## Design

### TOML Configuration

New keys in `[optimizer]` section (added to `configs/training/common.toml`):

```toml
[optimizer]
training_n_sims = 20         # seeds per individual per generation (epoch rotation)
validation_n_sims = 1000     # sims for validation gate
validation_interval = 50     # periodic validation every N gens
```

- `training_n_sims` defaults to 1 (backward compatible -- single fixed seed as today).
- When `training_n_sims > 1`, epoch rotation activates automatically. No flag needed.
- `--adaptive-seeds` remains independent. If both `training_n_sims > 1` and `--adaptive-seeds` are set, adaptive seeds take precedence (the pool manages its own seed set).
- `validation_n_sims` defaults to 1000. Set to 0 to disable validation entirely.
- `validation_interval` defaults to 50. Validation also fires on every new global best (no cooldown).

### Epoch Seed Rotation

**Where:** `train.py`, generation loop, before `algorithm.next()`.

**Mechanism:** Each generation, draw `training_n_sims` seeds from the training RNG and update the problem's seed list:

```python
if config.optimizer.training_n_sims > 1 and seed_pool is None:
    epoch_seeds = rng.integers(0, 2**31, size=config.optimizer.training_n_sims).tolist()
    problem.update_seeds(epoch_seeds)
```

**Aggregation:** Already handled -- `_run_batch_pyo3` in `problem.py` loops over `self.seeds` and aggregates by RMS. No change needed.

**Best tracking:** Compare raw RMS costs across generations as-is (option a). The validation gate provides the honest cross-generation metric. Overcomplicating best-tracking in the training loop adds noise for no gain.

**Checkpoint compatibility:** Seeds are ephemeral (regenerated from RNG state on resume). No serialization needed. The RNG state is already checkpointed, so the seed sequence is deterministic on resume.

### Validation Gate

**Where:** `train.py`, after best-tracking logic in the generation loop.

**Trigger:** Fires when either:
1. `(gen + 1) % validation_interval == 0` (periodic)
2. A new global best was found this generation

No cooldown on the new-best trigger.

**Mechanism:** Uses a separate RNG stream so training reproducibility is not disturbed:

```python
# Initialized once at startup:
validation_rng = np.random.default_rng(base_seed + 999)
val_seeds = validation_rng.integers(0, 2**31, size=config.optimizer.validation_n_sims).tolist()

# On trigger:
val_costs = problem.evaluate_individual_per_seed(best_overall_individual, val_seeds)
```

The validation seeds are fixed for the entire training run (drawn once at startup). This gives a stable benchmark so validation cost is comparable across generations.

**Logging:** Add optional `validation` dict to the JSONL record:

```json
{
  "validation": {
    "mean_cost": 42.3,
    "median_cost": 38.1,
    "std_cost": 15.2,
    "p95_cost": 112.5,
    "worst_cost": 8432.0,
    "capture_rate": 0.97,
    "n_sims": 1000
  }
}
```

Only present on gens where validation fired. Absent otherwise.

**TUI display:** Show a validation line when it fires, e.g.:
```
Val: mean=42.3 p95=112.5 cap=97.0%
```

### OptimizerConfig Changes

Add 3 fields to `OptimizerConfig` dataclass in `optimizer.py`:

```python
training_n_sims: int = 1
validation_n_sims: int = 1000
validation_interval: int = 50
```

Wire through `from_dict()` as top-level keys.

### Logger Changes

`log_generation()` gets an optional `validation: dict | None` parameter. When non-None, it's included in the JSONL record.

### Display Changes

`LiveDisplay` shows validation metrics when present in the latest log record.

## Files Modified

| File | Change |
|------|--------|
| `configs/training/common.toml` | Add `training_n_sims`, `validation_n_sims`, `validation_interval` to `[optimizer]` |
| `src/python/aerocapture/training/optimizer.py` | Add 3 fields to `OptimizerConfig` + `from_dict` |
| `src/python/aerocapture/training/train.py` | Epoch seed rotation before `algorithm.next()`, validation gate after best-tracking |
| `src/python/aerocapture/training/logger.py` | Accept + write `validation` dict to JSONL |
| `src/python/aerocapture/training/display.py` | Show validation metrics in TUI |
| `tests/` | Tests for new config fields, epoch rotation behavior, validation gate |

## What Does NOT Change

- `problem.py` -- already handles multiple seeds via `update_seeds()` + RMS aggregation
- `seed_pool.py` -- untouched, still works with `--adaptive-seeds`
- `evaluate.py`, `param_spaces.py`, `encoding.py` -- no changes
- Rust simulator -- nothing needed
- `report.py` / `charts.py` -- validation data is in JSONL but no report charts yet (future improvement)
- Checkpoint format -- no new arrays; RNG state already saved

## Success Criterion

A scheme trained with `training_n_sims = 20` should:
1. Show continued improvement past generation 20 (the current stall point)
2. Show validation cost that tracks or improves over training (no overfitting)
3. Perform comparably or better on `compare_guidance` (unseen scenarios) vs. the old single-seed training
