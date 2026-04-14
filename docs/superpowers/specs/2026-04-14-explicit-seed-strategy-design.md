# Explicit `seed_strategy` Configuration

**Status:** design
**Date:** 2026-04-14
**Context:** Builds on the curated-CDF seed framework
(`docs/superpowers/specs/2026-04-14-curated-cdf-seed-framework-design.md`).
Reintroduces the rotating-seed path as a first-class option and exposes three
explicit strategies via a single TOML knob.

## Motivation

The curated-CDF framework removed the old epoch-rotation path on the assumption
that curation would subsume it. Empirical A/B testing on this project finds
rotating seeds competitive with curated (on par or slightly better on several
schemes). Rather than pick one default, expose all three as a required
`[optimizer] seed_strategy` key so users can choose per-scheme, per-run, without
CLI flags.

## Strategies

| Strategy    | Seeds used each generation                                                               | Refresh trigger        | Checkpoint state |
| ----------- | ---------------------------------------------------------------------------------------- | ---------------------- | ---------------- |
| `"fixed"`   | `[mc_seed + 0, mc_seed + 1, ..., mc_seed + (n_sims-1)]`                                 | Never                  | None             |
| `"rotating"`| `n_sims` fresh random seeds drawn each generation (disjoint from reserved sets)          | Every generation       | None             |
| `"adaptive"`| Bootstrap = random `n_sims` draw once; then curated list via `SeedCurator`               | Validated best OR every `seed_pool_interval` gens | `SeedCurator` state |

All three exclude validation and final-evaluation reserved seed sets from their
draw pools.

All three use the same `training_n_sims` knob for the seed-list length.

## Configuration

New `[optimizer]` key, **required** (no default):

```toml
[optimizer]
seed_strategy = "fixed" | "rotating" | "adaptive"
training_n_sims = 20
```

`common.toml` sets `seed_strategy = "adaptive"` so existing training configs
inheriting from it keep current behavior. Leaf configs may override with
`"rotating"` or `"fixed"` without touching common.

Missing key or invalid value → `ValueError` at config load with a message
listing valid values.

## Integration with the training loop

Dispatch lives in `train.py` at loop setup and loop body. The existing
`SeedCurator` integration remains unchanged for the `"adaptive"` branch.

### Loop setup (before the generation loop)

```python
strategy = config.optimizer.seed_strategy
n_sims = config.optimizer.training_n_sims

seed_curator: SeedCurator | None = None

if strategy == "fixed":
    fixed_seeds = [base_mc_seed + i for i in range(n_sims)]
    # Enforce disjointness with reserved pools; abort if overlap is non-empty.
    if set(fixed_seeds) & excluded_seeds:
        raise ValueError("fixed seed range overlaps validation/final-eval reserved sets")
    problem.update_seeds(fixed_seeds)
elif strategy == "rotating":
    pass  # per-gen draw in the loop body
elif strategy == "adaptive":
    seed_curator = SeedCurator(
        sample_size=config.optimizer.curation_sample_size,
        n_bins=n_sims,
        excluded_seeds=excluded_seeds,
        rng=rng,
    )
```

### Loop body (top of each generation)

```python
seeds_changed_this_gen = pending_seed_change
pending_seed_change = False

if strategy == "rotating":
    fresh = _draw_disjoint_seeds(rng, n_sims, excluded_seeds)
    problem.update_seeds(fresh)
    seeds_changed_this_gen = True

elif strategy == "adaptive" and seed_curator.seed_list is None:
    # Existing bootstrap block (first iteration only).
    bootstrap = _draw_disjoint_seeds(rng, n_sims, excluded_seeds)
    problem.update_seeds(bootstrap)
    seeds_changed_this_gen = True

# strategy == "fixed" → no action; seeds set at loop setup.
```

`_draw_disjoint_seeds(rng, n, excluded)` is a small helper that factors the
bootstrap/rotating logic out of both paths. It draws `n` random seeds and
retries until `n` disjoint values are collected.

### Pre-`algorithm.next()` re-evaluation

Unchanged logic, gated on `seeds_changed_this_gen`:
- `fixed`: True only on the first gen (seeds set at loop entry).
- `rotating`: True every gen.
- `adaptive`: True at bootstrap + after each curation.
- CMA-ES skip applies to all strategies.

### Curation trigger

Unchanged. Fires only when `strategy == "adaptive"`, wrapped in
`if seed_curator is not None:`. For `"fixed"` and `"rotating"` the trigger
block is skipped entirely.

## Degenerate cases

Allowed (no validation error) for all three strategies at `n_sims = 1`:

- `fixed` + `n_sims=1`: single seed = `mc_seed`.
- `rotating` + `n_sims=1`: one fresh random seed each generation.
- `adaptive` + `n_sims=1`: 1000-seed probe, one quantile bin, one random pick
  per curation. The `SeedCurator._stratified_pick` `n_bins == len(seeds)` guard
  continues to protect against `n_bins > len(seeds)`, which is the only real
  failure mode here (never triggered by this combination).

## Checkpoint compatibility

- `fixed`, `rotating`: nothing new persisted. Existing checkpoint fields
  untouched.
- `adaptive`: existing `seed_curator` state persists as today.

On resume, the strategy is re-read from the current config (not from the
checkpoint). Switching `seed_strategy` between runs is legal and takes effect
immediately on resume. Any stale `seed_curator` state in the checkpoint is
ignored unless the active strategy is `"adaptive"`.

## Removed features

None. The feature is additive. The `SeedCurator` class, curation trigger, and
related tests remain in place for the `"adaptive"` path.

## Tests

**`OptimizerConfig`:**
- `from_dict` accepts each of `"fixed"`, `"rotating"`, `"adaptive"`.
- Missing `seed_strategy` raises `ValueError` with a helpful message.
- Invalid `seed_strategy` raises `ValueError` listing valid values.

**Training loop dispatch (`tests/test_seed_strategy.py`):**
- A parameterized test per strategy verifies `problem.update_seeds` is called
  with the correct seeds at the correct cadence. Uses a lightweight stub for
  `problem` to avoid running real sims.
- `fixed`: called once with `[mc_seed, mc_seed+1, ...]`; never called again.
- `rotating`: called every generation with a fresh random list; no two gens
  produce identical lists (low-probability collision ignored, as with existing
  `test_different_rng_differs`).
- `adaptive`: called at bootstrap and again after each curation trigger.

**Disjointness:**
- `fixed`: unit test asserts `_validate_fixed_seeds_disjoint` raises when the
  generated range overlaps a reserved set.
- `rotating`: unit test asserts every drawn seed is disjoint from an injected
  excluded set.

**Checkpoint round-trip:**
- `fixed` + `rotating`: resume from a checkpoint written under a different
  strategy — the strategy in the current config wins.
- `adaptive`: existing `SeedCurator.from_dict` tests stay as-is.

**Regression:**
- Extend the Task 9 smoke test style: run 5 gens for each strategy on the FTC
  config and confirm training completes without errors and writes a checkpoint.

## Edge cases

- **Resume with changed strategy**: If the checkpoint was written under
  `"adaptive"` but the TOML now says `"fixed"`, the `seed_curator` key in the
  checkpoint is ignored; training proceeds with the fixed seed list. No data is
  lost; the curator state is merely unused.
- **`fixed` range overlap with reserved**: `make_reserved_seeds` uses offsets
  `VALIDATION_SEED_OFFSET = 1_000_000` and `FINAL_EVAL_SEED_OFFSET = 2_000_000`
  relative to `base_mc_seed`. A `fixed` range of length 20 starting at
  `mc_seed` will not overlap unless `mc_seed + n_sims` pushes past the first
  reserved offset — at `mc_seed = 42`, `n_sims` would need to exceed ~1M. The
  check is therefore near-trivial in practice but included for safety.
- **CMA-ES + any strategy**: CMA-ES samples fresh populations via `es.ask()`,
  so the pre-next re-eval is always skipped regardless of strategy. The three
  strategies still change which seeds get used when the offspring are
  evaluated; the `es.ask()` logic is orthogonal.

## Docs

- `CLAUDE.md` `train.py` bullet: expand the curated-CDF paragraph to describe
  all three strategies.
- `README.md`: add a "Training seed strategies" subsection to the GA
  Optimization block with the three-row table above and a short example per
  strategy.

## Open questions

None. All structural decisions resolved during brainstorming:

- Strategy menu: `fixed` / `rotating` / `adaptive`.
- `seed_strategy` is required (no default).
- `fixed` means deterministic `[mc_seed + i for i in range(n_sims)]`.
- Degenerate `n_sims=1` cases are allowed for all three.
- Checkpoint strategy is read from the current config on resume, not from the
  checkpoint itself.
