# Re-validate best individual on training resume

## Problem

When resuming a pymoo training run, the TUI displays stale validation context:

```
Best val  g372: RMS=4.10e+02 mean=3.82e+02 p95=6.80e+02 cap=100%
No improvement yet
```

- `Best val g372` is whatever validation record happened to land in the resumed-session buffer (or is absent entirely if no new best has been promoted yet).
- `No improvement yet` fires because the fresh in-memory `TrainingLogger` buffer has no `improvement: True` records from this session.

Root cause: `src/python/aerocapture/training/train.py:533` gates the initial-validation block on `start_gen == 0`, so resume skips it entirely.

## Change

Drop the `start_gen == 0` guard. On every training start (fresh or resume):

1. Re-validate `best_overall_individual` on the reserved `val_seeds`.
2. Update `best_val_cost` + `last_validated_individual` with the RMS result.
3. Log the record at `generation=start_gen` (not hardcoded `0`) with `improved=True`.

## Why it's safe

- `val_seeds` are deterministic: `make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, validation_n_sims)`. Re-running the same individual on the same seeds yields the same RMS unless the Rust physics changed between sessions -- in which case the refreshed value is the honest one.
- `improved=True` on resume treats the checkpointed best as this session's baseline improvement. Stagnation counter and "Best val" row populate immediately; no misleading "No improvement yet".
- Extra cost: one MC batch (`validation_n_sims`, typically 1000) per resume. Negligible vs. a multi-hour resumed session.

## Out of scope

- Re-validating multiple candidates (only the checkpointed best).
- Changes to fixed/rotating/adaptive seed strategy logic.
- Changes to checkpoint format or RL training path.

## Files touched

- `src/python/aerocapture/training/train.py` -- remove `start_gen == 0` from the guard, change hardcoded `0` to `start_gen` in the `logger.log_generation(...)` call, update the verbose print to say "Resume" vs "Gen 0" as appropriate.

## Verification

- `pytest tests/` -- full suite passes (no test should depend on validation being skipped on resume).
- Manual smoke: start a short training, Ctrl+C, resume with `--n-gen 5`, confirm the TUI populates `Best val` / stagnation counter within the first gen and the JSONL log has a validation record at `generation=start_gen`.
