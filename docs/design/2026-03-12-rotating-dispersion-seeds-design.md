# Design: Rotating Dispersion Seeds for GA Training

## Problem

The GA training pipeline evaluates every chromosome against the same fixed set of Monte Carlo dispersion scenarios (determined by `[monte_carlo].seed` in the TOML config). With typically 20 sims per evaluation, the GA can overfit to quirks of those specific 20 draws. This has been observed in practice: schemes trained with `seed=42` degrade on unseen scenarios in `compare_guidance`.

## Solution

Rotate the Monte Carlo seed each generation so the GA is exposed to a wider variety of dispersion scenarios over the course of training. Re-evaluate all parent individuals on the new seed before tournament selection to ensure fair within-generation comparison.

## Design

### Seed Computation

Each generation deterministically computes its MC seed:

```
mc_seed = base_mc_seed + generation_number
```

where `base_mc_seed` is the `[monte_carlo].seed` value read from the TOML config (not the `--seed` GA RNG seed, which controls crossover/mutation randomness — these are independent concerns). The base MC seed is read once at startup and stored in `TrainingConfig`. This keeps training fully reproducible.

### evaluate.py Changes

`evaluate_chromosome()` gets a new optional parameter `mc_seed: int | None = None`. When provided, the TOML's `[monte_carlo].seed` is overridden before running the simulator.

**New helper: `patch_toml_mc_seed()`** — a lightweight function that reads a TOML file, sets `[monte_carlo].seed`, and writes to a temp file. This is distinct from `write_guidance_toml()` (which patches guidance params) because the two serve different purposes and the NN path doesn't use `write_guidance_toml()`.

**NN path:** Currently writes weights to JSON and runs the base TOML directly. When `mc_seed` is set, it calls `patch_toml_mc_seed()` to create a temp TOML with the patched seed, temporarily overrides `config.sim.toml_config`, runs the sim, then restores and cleans up (same pattern as the non-NN guidance param patching at lines 470-477).

**Non-NN path:** Already creates a patched temp TOML via `write_guidance_toml()`. When `mc_seed` is set, the seed is applied to the same TOML dict before `_write_toml()` writes it out. This composes naturally — the temp TOML contains both the guidance params and the patched seed.

### train.py GA Loop Changes

When `config.ga.rotate_seeds` is enabled:

1. At startup, read `base_mc_seed` from the TOML config's `[monte_carlo].seed`
2. At the top of each generation: `mc_seed = base_mc_seed + gen`
3. Offspring are evaluated with `mc_seed` passed to `evaluate_chromosome()`
4. **All parents are re-evaluated** on the same `mc_seed` before tournament selection
5. Tournament proceeds as before: combine parents + offspring, sort by cost, keep top N

This means `2 * n_pop * n_subpop` evaluations per generation (parents + offspring, across all subpopulations) instead of `n_pop * n_subpop`. With evaluations taking seconds, this is acceptable.

When `rotate_seeds` is disabled (default), behavior is identical to today — no `mc_seed` is passed, and the TOML's baked-in seed is used unchanged.

### Configuration

- `GAConfig` (in `config.py`) gets `rotate_seeds: bool = False`
- CLI gets `--rotate-seeds` flag (default off), wired as `cfg.ga.rotate_seeds = args.rotate_seeds` in the `__main__` block
- No TOML training config file changes — seed rotation is a GA strategy choice. The TOML's `[monte_carlo].seed` is still read as the base seed, but at runtime, `evaluate_chromosome()` patches it per-generation when rotation is active.

### Checkpoint Compatibility

No checkpoint format change. The generation number is already saved, which implicitly determines the seed sequence on resume. On the first generation after resume, all parents are re-evaluated on the new seed, so stale costs from the previous generation are harmless.

### Logging

`logger.log_generation()` accepts an optional `mc_seed: int | None` parameter and records it in the JSONL output. This enables post-hoc analysis of per-seed difficulty. Existing tests for the logger will be updated to cover the new parameter.

### What Does NOT Change

- Rust simulator (no code changes)
- Dispersion draw logic in `dispersions.rs`
- Cost function in `evaluate.py`
- Display/TUI
- `compare_guidance.py`
- Checkpoint format
- TOML training config files (the `[monte_carlo].seed` field is read but not modified on disk; runtime patching uses temp files)

## Success Criterion

A scheme trained with `--rotate-seeds` should show comparable or better performance on `compare_guidance` (unseen scenarios) vs. the same scheme trained without the flag.

## Files Modified

| File | Change |
|---|---|
| `src/python/aerocapture/training/evaluate.py` | Add `mc_seed` param to `evaluate_chromosome()`, new `patch_toml_mc_seed()` helper, seed patching for both NN and non-NN paths |
| `src/python/aerocapture/training/train.py` | Read `base_mc_seed` from TOML, compute per-generation `mc_seed`, re-evaluate all parents (across subpopulations), thread `mc_seed` through evaluations, add `--rotate-seeds` CLI flag with wiring to config |
| `src/python/aerocapture/training/config.py` | Add `rotate_seeds: bool = False` to `GAConfig` |
| `src/python/aerocapture/training/logger.py` | Accept and record `mc_seed` in JSONL output |
| `tests/` | Update logger tests to cover `mc_seed` parameter |
