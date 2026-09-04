"""Reserved-seed-pool offsets are a single registry, distinct, at documented values (H6)."""

from __future__ import annotations

from aerocapture.training.seeds import (
    CALIBRATION_SEED_OFFSET,
    CONFIRM_EVAL_SEED_OFFSET,
    FINAL_EVAL_SEED_OFFSET,
    HEADLINE_REQUOTE_SEED_OFFSET,
    MAMBA3_EVAL_SEED_OFFSET,
    NN_INPUT_REPORT_SEED_OFFSET,
    PROBE_EVAL_SEED_OFFSET,
    RL_TRAINING_SEED_OFFSET,
    STRESS_EVAL_SEED_OFFSET,
    SWEEP_EVAL_SEED_OFFSET,
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
        "sweep_eval": SWEEP_EVAL_SEED_OFFSET,
        "headline_requote": HEADLINE_REQUOTE_SEED_OFFSET,
        "stress_eval": STRESS_EVAL_SEED_OFFSET,
        "probe_eval": PROBE_EVAL_SEED_OFFSET,
        "confirm_eval": CONFIRM_EVAL_SEED_OFFSET,
    }
    assert list(offsets.values()) == [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000, 20_000_000]
    assert len(set(offsets.values())) == len(offsets)  # all disjoint
    assert MAMBA3_EVAL_SEED_OFFSET == PROBE_EVAL_SEED_OFFSET  # legacy alias


def test_modules_reference_registry() -> None:
    import aerocapture.training.calibrate_inputs as ci
    import aerocapture.training.nn_input_report as nir
    import aerocapture.training.param_sweep as ps

    assert ci.CALIBRATION_SEED_OFFSET == CALIBRATION_SEED_OFFSET
    assert nir.NN_INPUT_REPORT_SEED_OFFSET == NN_INPUT_REPORT_SEED_OFFSET
    assert ps.SWEEP_EVAL_SEED_OFFSET == SWEEP_EVAL_SEED_OFFSET


def test_no_seed_offset_defined_outside_the_registry() -> None:
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src/python/aerocapture"
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py") if p.name != "seeds.py" and re.search(r"^[A-Z_]+_SEED_OFFSET\s*=", p.read_text(), re.M)]
    assert offenders == [], offenders
