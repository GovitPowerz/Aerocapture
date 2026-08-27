# Adaptive Seed Pool & Graceful Keyboard Interrupt

**Date**: 2026-03-14
**Status**: Draft
**Scope**: `src/python/aerocapture/training/`

## Problem

The current `--rotate-seeds` strategy (`mc_seed = base_seed + generation`) has three issues:

1. **Variance noise**: Some seeds produce trivially easy scenarios, others brutally hard. The GA wastes generations on uninformative extremes and the best individual's fitness oscillates wildly.
2. **Coverage gaps**: Sequential seeds don't guarantee coverage of the difficulty spectrum — they may cluster in difficulty.
3. **Robustness signal**: Selection pressure doesn't explicitly reward consistency across difficulty levels. An individual that aces easy seeds but fails hard ones can still win tournaments.

Additionally, there is no keyboard interrupt handling — Ctrl+C during training loses all progress since the last checkpoint with no final report.

## Design

### Feature 1: Adaptive Seed Pool

#### New class: `SeedPool` (`src/python/aerocapture/training/seed_pool.py`)

**Core state:**
- `seeds: list[int]` — active MC seeds in the pool
- `difficulty: dict[int, float]` — best individual's cost on each seed (population-relative)
- `generation_added: dict[int, int]` — generation when each seed entered the pool
- `max_size: int` — pool capacity cap (default 100)

**Difficulty metric:** Population-relative — the cost of the current best individual on a given seed. This naturally adapts: a seed that was "hard" at gen 5 may become "easy" at gen 50 and get evicted for redundancy.

**Lifecycle per generation:**

1. **Add**: Introduce a new seed (`base_mc_seed + generation`). On the first generation, bootstrap with 5 seeds (`base_mc_seed` through `base_mc_seed + 4`) to avoid single-scenario optimization in early generations.
2. **Evaluate**: Run ALL individuals on ALL pool seeds via `run_batch()` → cost matrix `(n_pop, n_seeds)`.
3. **Score**: Extract difficulty from the best individual's row in the cost matrix (no extra simulations).
4. **Evict** (if pool > max_size): Sort seeds by difficulty. Find the pair with the smallest difficulty gap. Evict the older one (lower `generation_added`) — prefer fresher seeds, since they better reflect the current difficulty landscape. If the older seed has survived many generations, it likely occupies a difficulty niche that the newer seed now covers equally well.
5. **Aggregate**: `fitness[i] = alpha * mean(costs[i]) + (1 - alpha) * CVaR_p(costs[i])` where CVaR_p is the mean of the worst p% of costs.

**Fitness aggregation:**
- `alpha` (default 0.7): blend weight between mean and CVaR.
- `cvar_percentile` (default 20): tail fraction for CVaR computation.
- **CVaR floor**: When `n_seeds * p / 100 < 1` (early pool growth), CVaR uses at least 1 sample (the worst cost).
- This blend rewards consistent performance (mean) while penalizing catastrophic failures on hard seeds (CVaR).

**Pool growth:** Bootstraps with 5 seeds, then grows by 1 per generation until hitting the cap. After cap, each new seed triggers an eviction. At steady state with `max_size=100`, each individual is evaluated on 100 sims per generation. With `n_pop=20`, that's 2,000 sims/generation — this is acceptable given PyO3 `run_batch()` with Rayon parallelism (confirmed by user).

**Batching:** Evaluation leverages `aerocapture_rs.run_batch()` for Rayon-parallelized execution. All (individual x seed) combinations are batched efficiently. Each individual's seed pool evaluation is a single `run_batch()` call with per-seed TOML overrides.

**Checkpoint integration:** Pool state (seeds, difficulty, generation_added) serialized as a new top-level `"seed_pool"` key in the checkpoint `.json` file. On resume, the restored population is re-evaluated on the restored seed pool in the first generation (difficulty scores are stale since they reference the old best individual).

#### Config & CLI

New CLI args (mutually exclusive group with `--rotate-seeds`):
- `--adaptive-seeds` — enables the adaptive seed pool
- `--seed-pool-cap 100` — maximum pool size
- `--cost-alpha 0.7` — mean/CVaR blend weight
- `--cvar-percentile 20` — tail fraction for CVaR

If both `--adaptive-seeds` and `--rotate-seeds` are passed, `argparse` raises an error.

#### Integration with training loop

**Subpopulation handling:** The seed pool is shared across all subpopulations. Each subpopulation's individuals are evaluated on the same pool. Migration exchanges individuals with their aggregated fitness scores. The `migrate()` function receives aggregated fitness values (scalars), not per-seed cost matrices.

When `--adaptive-seeds` is active:
- The per-generation evaluation flow changes from individual-level `evaluate_chromosome()` calls to a bulk `pool.evaluate_population(population, config)` returning a fitness array.
- Tournament selection uses the aggregated fitness.
- `pool.update(generation, best_individual)` handles add/score/evict each generation.

When `--adaptive-seeds` is NOT active:
- Behavior is identical to current code. `evaluate_chromosome()` and `--rotate-seeds` work as before.

#### Logging/display

Pool diagnostics added to per-generation logger output:
- `pool_size`: current number of seeds in the pool
- `difficulty_range`: `[min_difficulty, max_difficulty]` across pool seeds
- `n_evictions`: cumulative eviction count

### Feature 2: Graceful Keyboard Interrupt

**Mechanism:** `try/except KeyboardInterrupt` wrapping the main training loop in `train()`.

**On interrupt:**
1. Call `display.stop()` to cleanly tear down the Rich TUI before printing to stdout
2. Print: `"Interrupted at run R, generation G. Saving checkpoint..."`
3. Save checkpoint immediately (reuse `save_checkpoint()`)
4. Save best model/params to disk (reuse existing logic)
5. Return results dict with `interrupted: True` flag

**Post-interrupt flow:** The `__main__` block's existing post-training pipeline runs normally:
- Save best model/params
- Run `run_final_evaluation()` + `generate_final_report()` (unless `--skip-final-report`)

The final report runs on the best individual found so far, even if training was cut short.

**Double-interrupt protection:** First Ctrl+C = graceful shutdown. Second Ctrl+C during checkpoint saving or final report = hard exit (natural Python behavior). No complex signal handler state machine.

**Changes:** All in `train.py` — wrap the loop in try/except, add `interrupted` to return dict. No changes to other files.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/python/aerocapture/training/seed_pool.py` | Create | `SeedPool` class: add/score/evict/evaluate/aggregate |
| `src/python/aerocapture/training/train.py` | Modify | Integrate `SeedPool` in training loop, add CLI args, add KeyboardInterrupt handling |
| `tests/test_seed_pool.py` | Create | Unit tests for SeedPool: growth, eviction, difficulty scoring, aggregation, checkpoint round-trip |
| `tests/test_train.py` | Modify | Test keyboard interrupt handling, adaptive-seeds CLI args |

## What Stays the Same

- `evaluate_chromosome()` — still used for non-adaptive mode and final report
- `evaluate.py` — no changes needed (existing `mc_seed` parameter suffices)
- `--rotate-seeds` — old behavior preserved, mutually exclusive with `--adaptive-seeds`
- Final report pipeline — unchanged, runs on best individual regardless of training mode
- All other training infrastructure (migration, population, param_spaces, etc.)
