"""Unit tests for the pure pieces of the Mamba-3 ablation driver."""

from __future__ import annotations

import numpy as np
from aerocapture.training.experiments.mamba3_ablation import ARMS, _aggregate, _cvar95, _leaf_toml


def test_arms_cover_the_2x2() -> None:
    assert set(ARMS.values()) == {("euler", "real"), ("trapezoidal", "real"), ("euler", "complex"), ("trapezoidal", "complex")}


def test_cvar95_is_worst_5pct_mean() -> None:
    x = np.arange(100.0)  # p95 = 94.05
    cv = _cvar95(x)
    assert cv > float(np.percentile(x, 95))
    assert cv == float(np.mean(x[x >= np.percentile(x, 95)]))


def test_cvar95_empty_is_nan() -> None:
    assert np.isnan(_cvar95(np.array([])))


def test_aggregate_mean_std() -> None:
    per_rep = [
        {"rms_cost": 10.0, "capture_rate": 1.0, "dv_p50": 100.0, "dv_p95": 200.0, "cvar95": 250.0},
        {"rms_cost": 12.0, "capture_rate": 0.9, "dv_p50": 110.0, "dv_p95": 220.0, "cvar95": 270.0},
    ]
    agg = _aggregate(per_rep)
    assert agg["n_repeats"] == 2
    assert agg["dv_p95"]["mean"] == 210.0
    assert agg["dv_p95"]["std"] == 10.0


def test_leaf_toml_carries_flags_and_seed() -> None:
    toml = _leaf_toml("both", "trapezoidal", "complex", 20260709, __import__("pathlib").Path("training_output/mamba3/both_s2"), 500, 10)
    assert 'discretization = "trapezoidal"' in toml
    assert 'state_mode = "complex"' in toml
    assert "seed = 20260709" in toml
    assert 'type = "mamba3"' in toml
