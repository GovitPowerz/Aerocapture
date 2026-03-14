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

1. **Add**: Introduce a new seed (`base_mc_seed + generation`).
2. **Score**: Evaluate the best individual on ALL pool seeds to update difficulty scores.
3. **Evict** (if pool > max_size): Sort seeds by difficulty. Find the pair with the smallest difficulty gap. Evict the older one (lower `generation_added`) — prefer fresher seeds.
4. **Evaluate**: Run ALL individuals on ALL pool seeds → cost matrix `(n_pop, n_seeds)`.
5. **Aggregate**: `fitness[i] = alpha * mean(costs[i]) + (1 - alpha) * CVaR_p(costs[i])` where CVaR_p is the mean of the worst p% of costs.

**Fitness aggregation:**
- `alpha` (default 0.7): blend weight between mean and CVaR.
- `cvar_percentile` (default 20): tail fraction for CVaR computation.
- This blend rewards consistent performance (mean) while penalizing catastrophic failures on hard seeds (CVaR).

**Pool growth:** Starts at 1 seed, grows by 1 per generation until hitting the cap. After cap, each new seed triggers an eviction. This means early generations are cheap (few sims) and cost grows linearly until stabilizing at `max_size` sims per individual.

**Batching:** Evaluation leverages `aerocapture_rs.run_batch()` for Rayon-parallelized execution. All (individual x seed) combinations can be batched efficiently.

**Checkpoint integration:** Pool state (seeds, difficulty, generation_added) serialized alongside existing checkpoint data in the `.json` file.

#### Config & CLI

New CLI args:
- `--adaptive-seeds` — enables the adaptive seed pool (mutually exclusive with `--rotate-seeds`)
- `--seed-pool-cap 100` — maximum pool size
- `--cost-alpha 0.7` — mean/CVaR blend weight
- `--cvar-percentile 20` — tail fraction for CVaR

#### Integration with training loop

When `--adaptive-seeds` is active:
- The per-generation evaluation flow changes from individual-level `evaluate_chromosome()` calls to a bulk `pool.evaluate_population(population, config)` returning a fitness array.
- Tournament selection uses the aggregated fitness.
- `pool.update(generation, best_individual)` handles add/score/evict each generation.

When `--adaptive-seeds` is NOT active:
- Behavior is identical to current code. `evaluate_chromosome()` and `--rotate-seeds` work as before.

### Feature 2: Graceful Keyboard Interrupt

**Mechanism:** `try/except KeyboardInterrupt` wrapping the main training loop in `train()`.

**On interrupt:**
1. Print: `"Interrupted at run R, generation G. Saving checkpoint..."`
2. Save checkpoint immediately (reuse `save_checkpoint()`)
3. Save best model/params to disk (reuse existing logic)
4. Return results dict with `interrupted: True` flag

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
| `src/python/aerocapture/training/evaluate.py` | Modify | Add single-seed evaluation helper for pool scoring (thin wrapper around existing code) |
| `tests/test_seed_pool.py` | Create | Unit tests for SeedPool: growth, eviction, difficulty scoring, aggregation, checkpoint round-trip |
| `tests/test_train.py` | Modify | Test keyboard interrupt handling, adaptive-seeds CLI args |

## What Stays the Same

- `evaluate_chromosome()` — still used for non-adaptive mode and final report
- `--rotate-seeds` — old behavior preserved, mutually exclusive with `--adaptive-seeds`
- Display/logging — fed aggregated fitness values, no structural changes
- Final report pipeline — unchanged, runs on best individual regardless of training mode
- All other training infrastructure (migration, population, param_spaces, etc.)
