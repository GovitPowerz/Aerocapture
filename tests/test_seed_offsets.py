"""Reserved-seed-pool offsets are a single registry, distinct, at documented values (H6)."""

from __future__ import annotations

from aerocapture.training.evaluate import (
    CALIBRATION_SEED_OFFSET,
    FINAL_EVAL_SEED_OFFSET,
    NN_INPUT_REPORT_SEED_OFFSET,
    RL_TRAINING_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    WARM_START_SEED_OFFSET,
)


def test_offsets_distinct_and_documented() -> None:
    offsets = {
        "validation": VALIDATION_SEED_OFFSET,
        "final_eval": FINAL_EVAL_SEED_OFFSET,
        "rl_training": RL_TRAINING_SEED_OFFSET,
        "warm_start": WARM_START_SEED_OFFSET,
        "nn_input_report": NN_INPUT_REPORT_SEED_OFFSET,
        "calibration": CALIBRATION_SEED_OFFSET,
    }
    assert list(offsets.values()) == [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000]
    assert len(set(offsets.values())) == 6  # all disjoint


def test_modules_reference_registry() -> None:
    import aerocapture.training.calibrate_inputs as ci
    import aerocapture.training.nn_input_report as nir

    assert ci.CALIBRATION_SEED_OFFSET == CALIBRATION_SEED_OFFSET
    assert nir.NN_INPUT_REPORT_SEED_OFFSET == NN_INPUT_REPORT_SEED_OFFSET
