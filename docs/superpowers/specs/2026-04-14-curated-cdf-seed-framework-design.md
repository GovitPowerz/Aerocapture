# Curated-CDF Adaptive Seed Framework

**Status:** design
**Date:** 2026-04-14
**Supersedes:** adaptive seed pool (`SeedPool`, CVaR aggregation, gap-closure eviction, stress testing)

## Motivation

Training fitness under rotating Monte Carlo seeds is noisy: 20 random seeds per generation form a high-variance estimator of the 1000-sim "true" cost. The optimizer spends compute chasing apparent improvements that are actually seed-luck. Conversely, a frozen 20-seed set causes the optimizer to memorize those scenarios, failing on the full MC distribution.

The goal is a 20-seed training subset that approximates the full 1000-sim cost CDF of the current best candidates, refreshed periodically so the landscape stays honest without being pure noise.

## Overview

A single persistent list of `training_n_sims` seeds (default 20) drives all training fitness evaluations. The list is refreshed on two triggers:

1. A validated best is promoted (`val_rms < best_val_cost` in the validation gate).
2. Every `seed_pool_interval` generations (default 50), as a periodic fallback.

Between triggers the seed list is frozen. Every generation reuses the same 20 seeds; no per-generation random draws.

### Curation procedure

When a trigger fires:

1. Take the top-`curation_top_k` individuals from the current population, ranked by their current `F` (no extra evaluations for ranking; `K = min(curation_top_k, n_pop)`).
2. Draw `curation_sample_size` fresh random seeds (default 1000), disjoint from validation and final-eval reserved sets.
3. Run each of the K individuals on those 1000 seeds → `cost_matrix` of shape `(K, 1000)`.
4. Average across individuals → `avg_cost` of shape `(1000,)`. Non-finite values are replaced with a large sentinel so they sort to the tail.
5. Sort the 1000 seeds by `avg_cost`, split into `training_n_sims` equal-count quantile bins (50 seeds each when 1000 / 20).
6. Pick one random seed from each bin via the curation RNG → new training seed list.
7. `problem.update_seeds(new_seeds)`.

### Bootstrap

Before the first trigger fires, training uses pure random epoch-rotation seeds — identical to the current behavior when `training_n_sims > 1`. The first curation fires at the earliest of (first validated best) or (generation `seed_pool_interval`).

## Integration with the existing loop

### Pre-`algorithm.next()` re-evaluation

The fairness re-eval (introduced to make pymoo's internal cross-generation F comparisons consistent) still runs, but only when the seed list actually changed this generation. Between curations the seeds are unchanged, `algorithm.pop`'s F is already on them, and re-eval would be pure waste. CMA-ES continues to skip the re-eval regardless.

### Validation gate

Unchanged. Triggers on gen-best parameter identity (`np.array_equal` against `last_validated_individual`), runs on reserved `val_seeds`, gates promotion on `val_rms < best_val_cost`. A successful promotion is now also a curation trigger.

### Event ordering per generation

After `algorithm.next()`:

1. Compute `gen_best_individual` and `new_gen_best`.
2. Run validation gate if `new_gen_best`; set `validated_improvement = True` if `val_rms < best_val_cost`.
3. Curation check: if `validated_improvement or (gen + 1) % seed_pool_interval == 0`, run curation and update the seed list.
4. Logger + TUI update (Last val / Best val / Stagnant as already implemented).
5. Next generation's pre-next re-eval picks up the new seeds if they changed.

### Checkpoint

`seed_list` and `last_curation_gen` are added to the checkpoint state. On resume, training continues on the same curated seeds; periodic-fallback countdown resumes from `last_curation_gen`.

### Compute budget

With `n_pop = 60`, `training_n_sims = 20`, `seed_pool_interval = 50`:

- Per generation: 60 × 20 = 1200 training sims (unchanged vs. epoch rotation).
- Per curation: `K × curation_sample_size = 5 × 1000 = 5000` sims. Amortized: ~100 sims/generation.
- Validation: 1000 sims per validated-best candidate (unchanged).

Net cost: roughly +8% over epoch rotation in exchange for a much lower-variance fitness signal and reduced risk of the optimizer memorizing a fixed 20-scenario landscape.

## Configuration

New `[optimizer]` keys, all with defaults:

- `curation_top_k = 5` — number of top individuals averaged for the CDF.
- `curation_sample_size = 1000` — size of the probe MC drawn at each curation.

Existing keys retain their meaning:

- `training_n_sims = 20` — curated seed list size.
- `seed_pool_interval = 50` — periodic re-curation fallback.

## Removed features

The old adaptive seed pool is fully superseded.

**Code removed:**

- `SeedPool` class in `seed_pool.py`, including CVaR aggregation, difficulty scoring, gap-closure eviction, stress testing. All these were heuristics for a growing, ranked seed pool; the new scheme curates explicitly from a cost CDF and none of them have a purpose.
- Related tests in `test_seed_pool.py`.

**CLI flags removed:**

- `--adaptive-seeds`
- `--seed-pool-cap`
- `--cost-alpha`
- `--cvar-percentile`

**Config keys removed from `common.toml`** (emit a deprecation warning if present in a user TOML):

- `[optimizer] stress_interval`, `stress_probes`, `stress_inject`
- `[optimizer] cost_alpha`, `cvar_percentile`

## What's kept

- Reserved seed streams (`VALIDATION_SEED_OFFSET`, `FINAL_EVAL_SEED_OFFSET`, `make_reserved_seeds`) — the curation probe draws from the same non-reserved RNG space as epoch rotation.
- Validation gate with identity-based trigger.
- Pre-`next()` re-evaluation for GA/DE/PSO; skip for CMA-ES.

## New module: `SeedCurator`

Replaces `SeedPool` with a smaller, stateless-by-design class.

```python
class SeedCurator:
    def __init__(
        self,
        sample_size: int,
        n_bins: int,
        excluded_seeds: set[int],
        rng: np.random.Generator,
    ) -> None: ...

    def curate(
        self,
        problem: AerocaptureProblem,
        top_k_X: npt.NDArray[np.float64],  # (K, n_params), normalized
    ) -> list[int]:
        """Run K * sample_size MC, return n_bins representative seeds."""

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict, excluded_seeds: set[int], rng: np.random.Generator) -> SeedCurator: ...
```

Responsibilities:

- Draw `sample_size` fresh random seeds (excluding reserved).
- Run the K individuals on the 1000 seeds via `problem.evaluate_individual_per_seed` (once per top-K individual, returning a per-seed cost vector).
- Average cost across individuals, sort, stratified-random pick.
- Return the new seed list.

State stored in dict: `seed_list`, `last_curation_gen`. RNG and exclusion set are injected, not serialized.

The training loop in `train.py` owns trigger logic; `SeedCurator` only knows how to perform one curation given its inputs.

## Edge cases

- **`n_pop < curation_top_k`** → use `K = min(curation_top_k, n_pop)`.
- **`curation_sample_size` not divisible by `training_n_sims`** → use `np.array_split`-style uneven bins (e.g., 1000/20 = 50 exact; 1000/30 = 33 or 34 per bin).
- **NaN/Inf in the `cost_matrix`** → replaced with a large sentinel before sorting; the corresponding seeds land in the high-cost bin.
- **Resume from checkpoint** → `seed_list` and `last_curation_gen` are restored; periodic fallback resumes its countdown from `last_curation_gen`, not from gen 0.
- **CMA-ES** → curation works identically; only the pre-next re-eval stays skipped.

## Testing

**`SeedCurator` unit tests:**

- Stratified selection produces exactly `n_bins` seeds, one per bin.
- Same RNG seed + same cost matrix → deterministic output.
- Different RNG seeds → different picks with identical bin structure.
- Excluded seeds never appear in output.
- NaN/Inf handling: non-finite costs don't crash; they land in the top bin.

**Integration tests:**

- Training loop with curation triggers correctly on validated-best and periodic fallback.
- `seed_list` changes only at trigger generations; stable between.
- Checkpoint roundtrip preserves `seed_list` and `last_curation_gen`.

**Regression:**

- Training completes with the new scheme on at least one small config (e.g., 5-gen smoke test with `n_pop = 8`), producing a final checkpoint.

**Not tested (cost-prohibitive, follow-up benchmark):**

- Empirical demonstration that curated-CDF reduces fitness noise vs. pure random rotation.

## Open questions

None. All structural decisions resolved during brainstorming:

- CDF source: top-K population average with K=5 default.
- Selection: stratified random within equal-count quantile bins.
- Trigger: validated best + periodic fallback every `seed_pool_interval` gens.
- Top-K ranking: current gen F (no extra sims).
- Bootstrap: pure random epoch rotation until first trigger.
