# Remove `--guidance` CLI Flag

**Date:** 2026-03-20
**Status:** Approved

## Problem

The `--guidance` CLI flag in `train.py` is redundant — every TOML training config already specifies `[guidance] type = "..."`. This creates a divergence risk (CLI says `ftc`, TOML says `energy_controller`) and forces users to type the same information twice. The non-TOML code path (legacy NN-only mode) is unused in practice.

## Design

### CLI changes (`train.py` `__main__` block)

1. **Replace `--toml` (optional named) and `--guidance` (named with default)** with a single **positional `toml` argument** (required).
2. **Read `guidance.type`** from the resolved TOML (after base inheritance). Error with a clear message if the key is missing (e.g. `"TOML config must contain [guidance] type = '...'"`)
3. **Validate the guidance type** is a recognized scheme — replaces the `argparse choices=` validation that's being removed.
4. **Remove the non-TOML code path** entirely — the `else` branch and the `if cfg.guidance_type != "neural_network" and not args.toml` guard.
5. **Remove `--cwd`** — unconditionally set `cwd = "."` (TOML mode always runs from repo root).
6. **Ordering constraint:** guidance type must be extracted from the TOML *before* `save_dir` computation, auto-resume checkpoint detection, and NN-specific config reading — all of which depend on `cfg.guidance_type`.
7. **Update hardcoded error messages** — the ref-trajectory error message (line ~847) tells users the old `--guidance --toml` syntax; update to match new positional syntax.

**Before:**
```bash
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20
```

**After:**
```bash
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20
```

### Downstream consumers — no changes

- `TrainingConfig.guidance_type` field stays (set from TOML instead of CLI).
- `train()` function signature unchanged.
- `compare_guidance.py`, `final_report.py` already derive scheme from TOML/directory.
- Tests construct `TrainingConfig` directly — unaffected.

### Documentation updates

- CLAUDE.md: update training command examples *and* the `train.py` description line (currently documents `--guidance <scheme> --toml <config>` signature).
- README.md: update training command examples.
- Historical spec/plan docs: leave as-is.

## Files changed

| File | Change |
|------|--------|
| `src/python/aerocapture/training/train.py` | Remove `--guidance`, `--toml`, `--cwd`; add positional `toml`; read guidance type from TOML; remove non-TOML path |
| `CLAUDE.md` | Update CLI examples |
| `README.md` | Update CLI examples |
