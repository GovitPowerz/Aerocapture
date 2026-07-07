"""Unit tests for the CfC probe driver + shared probe machinery."""

from __future__ import annotations

import numpy as np
from aerocapture.training.experiments.probe_common import aggregate, arch_toml, cvar95


def test_cvar95_is_worst_5pct_mean() -> None:
    x = np.arange(100.0)
    cv = cvar95(x)
    assert cv > float(np.percentile(x, 95))
    assert cv == float(np.mean(x[x >= np.percentile(x, 95)]))


def test_cvar95_empty_is_nan() -> None:
    assert np.isnan(cvar95(np.array([])))


def test_aggregate_mean_std() -> None:
    per_rep = [
        {"rms_cost": 10.0, "capture_rate": 1.0, "dv_p50": 100.0, "dv_p95": 200.0, "cvar95": 250.0},
        {"rms_cost": 12.0, "capture_rate": 0.9, "dv_p50": 110.0, "dv_p95": 220.0, "cvar95": 270.0},
    ]
    agg = aggregate(per_rep)
    assert agg["n_repeats"] == 2
    assert agg["dv_p95"]["mean"] == 210.0
    assert agg["dv_p95"]["std"] == 10.0


def test_arch_toml_renders_blocks() -> None:
    arch = [
        {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"},
        {"type": "cfc", "input_size": 32, "hidden_size": 32, "backbone_units": 32},
    ]
    s = arch_toml(arch)
    assert s.count("[[network.architecture]]") == 2
    assert 'type = "cfc"' in s
    assert "backbone_units = 32" in s
    assert 'activation = "swish"' in s


def test_probe_offset_alias() -> None:
    from aerocapture.training.evaluate import MAMBA3_EVAL_SEED_OFFSET, PROBE_EVAL_SEED_OFFSET

    assert PROBE_EVAL_SEED_OFFSET == 10_000_000
    assert MAMBA3_EVAL_SEED_OFFSET == PROBE_EVAL_SEED_OFFSET
