# Resume Enhancements: Re-validation, Population Growth, Cost-Transform Reset

Date: 2026-06-02
Status: Design approved, pending implementation plan

## Overview

Three related enhancements to the training resume path (`train.py` single-algorithm
loop and `island_model.py` islands path). All three live in the checkpoint/restore
machinery and share two cross-cutting changes (checkpoint format + restore logic),
so they are landed as one cohesive spec with three independent, testable pieces.

1. On resume, re-evaluate the best validation individual of every optimizer.
2. Resume an optimization with a larger (or smaller) population, seeding the new
   population from the checkpointed one.
3. Resume with a different `cost_transform`, which re-validates the best under the
   new metric and resets the stagnation counter.

## Current State (grounding)

- **Single-algo already re-validates the checkpointed best on resume**
  (`train.py:1275-1298`): it calls `problem.evaluate_individual_records_per_seed`
  on `best_overall_individual` under the current config and recomputes
  `best_val_cost`. No change needed there for feature #1.
- **Islands does NOT re-validate on resume**: `IslandModel.from_checkpoint`
  (`island_model.py:553-657`) restores each island's `best_val_cost` verbatim from
  the checkpoint (`island_model.py:530` saves it, restore reads it back). This is
  feature #1's real gap.
- **`cost_transform` is never persisted**: read fresh from TOML each run
  (`train.py:764`), threaded into `AerocaptureProblem.cost_kwargs`
  (`problem.py:48`), applied in `evaluate.py:compute_cost`
  (`evaluate.py:299-311`). Not in any checkpoint or cache key.
- **`n_pop` is pinned to the resumed population's shape**: single-algo restore at
  `train.py:969-975` (`pop_array = resumed["population"]`); islands restore inside
  `from_checkpoint`. No resize mechanism exists. Changing `[optimizer] n_pop`
  between runs is silently ignored on resume.
- The "cross-gen training-cost incomparability" gate (`train.py:1239-1243`) is
  guarded on `best_overall_individual is None` (fresh start only) and must remain
  so. None of these features touch that invariant.
- `_check_resume_chromosome_shape` (`train.py:220-238`) and the islands width guard
  in `from_checkpoint` validate chromosome **width** (`n_params`). Population
  **count** (`n_pop`) is orthogonal, so growing/shrinking the population does not
  trip these guards.

## Architecture

Shared pure helpers called from both the single-algo and islands restore paths,
mirroring the existing `_apply_seed_strategy` / `_maybe_curate` shared-helper
pattern. Inlining the resize/eval logic in each path would duplicate tricky code
across `train.py` and `island_model.py` and be harder to unit-test.

## Feature 1: Re-validate best on resume (islands)

Single-algo is already correct (`train.py:1275-1298`); no change.

Add a method to `IslandModel`, e.g. `revalidate_each(val_seeds, problem)`:
for each island where `best_overall_individual is not None` and `val_seeds` is
non-empty, recompute `best_val_cost` via
`problem.evaluate_individual_records_per_seed(best_overall_individual, val_seeds)`
(RMS of per-seed costs, matching `validate_each` and the single-algo path) under
the **current** config, and set
`last_validated_individual = best_overall_individual.copy()`.

Call it from `_train_islands` immediately after
`island_model.from_checkpoint(...)` (`train.py:1620`), only when `val_seeds` is
set. This makes islands honest after resume and (combined with feature #3)
auto-handles a transform change.

## Feature 2: Resume with a larger (or smaller) population

New pure helper in a shared module (e.g. `training/population_resize.py`):

```
resize_population(pop_X, pop_F, target_n, param_specs, rng, fresh_fraction=0.2) -> new_pop_X
```

Behavior:
- **Grow** (`target_n > N`): keep all `N` resumed individuals verbatim in their
  original slots; fill the remaining `target_n - N` slots as
  `round(fresh_fraction * (target_n - N))` **fresh-random** individuals (sampled
  from `param_specs`, i.e. normalized [0,1] hypercube) and the rest **clone+jitter**
  (round-robin over the resumed pool, additive Gaussian jitter in normalized space,
  clipped to [0,1]).
- **Shrink** (`target_n < N`): keep the `target_n` lowest-`pop_F` individuals
  (truncate to best N by training cost). When `pop_F` is None/unavailable, fall
  back to keeping the first `target_n` rows.
- **Equal**: identity (return `pop_X` unchanged).

`fresh_fraction` defaults to 0.2, TOML-overridable via `[optimizer]
grow_fresh_fraction` (added to `OptimizerConfig` + `from_dict` in `optimizer.py`,
default 0.2).

Jitter magnitude: a small fixed normalized-space sigma (e.g. 0.02, matching the
warm-start jitter convention) so clones stay in the resumed basin.

### Wiring (single-algo)

After the resume restore (`train.py:969-975`), if
`config.optimizer.n_pop != pop_array.shape[0]`, call `resize_population`. New and
jittered individuals have no valid cost, so the grown population must be
re-evaluated once before `warm_start_algorithm`: set `pop_costs = None` so the
existing initial-eval path runs a single batch. (Shrink keeps the surviving
individuals' costs, but setting `pop_costs = None` and re-evaluating is the simple
uniform choice; the cost is one MC batch on resume.)

### Wiring (islands)

Resize each island's restored `pop_X` to the new `n_pop` inside `from_checkpoint`
(or a helper it calls), using `resize_population` with that island's `pop_F` for
the shrink case. For **PSO** islands additionally:
- extend `particles_X` to the new positions (= the resized `pop_X`),
- extend `particles_V` with fresh velocities `U(-s, +s)` where
  `s = pso_inject_velocity_scale` (the existing migration velocity knob),
- recompute `particles_F` (re-evaluate, or let the post-resume re-eval handle it).

The migration constraint `k_top * (n_islands - 1) <= n_pop` is re-validated at
`IslandModel.__init__` with the new `n_pop`; growing only relaxes it. Chromosome
width is unchanged, so the width guard still passes.

## Feature 3: Cost-transform change resets the validation best

- **Persist** `cost_transform` in both checkpoint formats:
  - single-algo: add to `save_checkpoint` JSON meta and read in `load_checkpoint`.
  - islands: add to the `.npz` top-level alongside `version` / `base_mc_seed`.
- **Auto-detect** on resume: compare the persisted value to the current
  `problem.cost_kwargs["cost_transform"]`. Checkpoints without a persisted value
  (legacy) are treated as "unknown" -> assume changed.
- **Reset semantics = "re-validate, keep individual"**: feature #1 (islands) and
  the existing single-algo resume both unconditionally re-validate the
  checkpointed best under the current config, so `best_val_cost` is recomputed
  under the new transform automatically -- the individual is kept, the metric
  baseline is refreshed. On a detected change additionally:
  - emit a clear log line:
    `cost_transform changed 'X' -> 'Y'; re-validating best under new metric`.
  - reset the stagnation counter to 0 (single-algo) / per island (islands), since
    "improvement" is now measured under a different metric and the old count is
    meaningless.

This keeps `best_overall_individual` (cost transforms in scope are monotonic, so
the per-sim argmin individual is unchanged; only the RMS-across-seeds ranking can
shift, which the re-validation handles fairly going forward).

## Testing

Unit:
- `resize_population`: grow (output count == target_n; first N rows == resumed
  verbatim; fresh-vs-clone split matches `fresh_fraction`; jittered clones within
  [0,1] and within jitter sigma of a source row); shrink (selects best-N by
  `pop_F`); equal (identity).
- `cost_transform` round-trips in both checkpoint formats.
- transform-change detection (changed / unchanged / legacy-missing).

Integration:
- islands resume re-validates: `best_val_cost` recomputed under current config
  (differs from a deliberately-stale checkpointed value).
- single-algo resume with grown `n_pop`: population is the new size, the resumed
  individuals are preserved, training proceeds.
- resume with changed `cost_transform`: stagnation reset, re-validation fires,
  log line emitted.

## Out of Scope

- Changing chromosome width on resume (still guarded -> `--from-scratch`).
- Non-monotonic cost transforms.
- Per-island heterogeneous `n_pop`.

## Final Step

Run the `smart-commit` skill taking the whole git branch into account.
