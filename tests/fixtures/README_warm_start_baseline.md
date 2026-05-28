# Warm-Start Magnitude-Only Equivalence Baseline

The test `tests/test_warm_start_equivalence_gate.py` requires a baseline JSON
recorded from the **pre-refactor pipeline** (the code on `main` before the
2026-05-22 warm-start-all-archs branch is merged).

To record:

1. Check out `main` at the commit just before the warm-start-all-archs branch.
2. Run:
   ```bash
   uv run python -m aerocapture.training.train \
       configs/training/msr_aller_nn_train_consolidated.toml \
       --n-gen 20 --n-pop 32 \
       --no-tui --skip-report \
       --output-dir /tmp/baseline_run
   ```
3. Extract the best validation RMS from the JSONL log in `/tmp/baseline_run/`.
4. Write `tests/fixtures/warm_start_magonly_baseline.json`:
   ```json
   {
     "val_rms_after_20_gens": <measured_value>,
     "recorded_at": "YYYY-MM-DD",
     "config": "msr_aller_nn_train_consolidated.toml",
     "n_gen": 20,
     "n_pop": 32
   }
   ```
5. Commit the fixture file.

The test is skipped when the fixture is absent.
