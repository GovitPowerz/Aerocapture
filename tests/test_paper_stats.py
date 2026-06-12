"""Tests for the paper's statistical helpers (locked 2026-06-12 reporting rules)."""

import numpy as np
import pytest
from aerocapture.training.paper_stats import (
    actual_sims,
    bootstrap_ci,
    cvar,
    paired_comparison,
    run_stats,
)


def test_cvar_is_mean_of_worst_tail() -> None:
    x = np.arange(1.0, 101.0)  # 1..100
    assert cvar(x, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_cvar_small_sample_uses_at_least_one() -> None:
    assert cvar(np.array([3.0, 1.0]), 0.95) == 3.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng_x = np.random.default_rng(0).normal(100.0, 5.0, size=1000)
    lo, hi = bootstrap_ci(rng_x, np.mean, n_boot=2000, seed=1)
    assert lo < float(np.mean(rng_x)) < hi
    assert hi - lo < 2.0  # ~2*1.96*5/sqrt(1000)


def test_run_stats_capture_conditional() -> None:
    ifinal = np.array([3.0, 3.0, 3.0, 2.0])
    ecc = np.array([0.5, 0.5, 1.2, 0.5])  # third sim: ifinal 3 but hyperbolic
    dv = np.array([100.0, 200.0, 999.0, 999.0])
    s = run_stats(ifinal, ecc, dv, n_boot=200, seed=0)
    assert s["n"] == 4
    assert s["capture_pct"] == 50.0
    assert s["dv_mean"] == pytest.approx(150.0)
    assert "dv_p99" in s and "dv_cvar95" in s and "dv_mean_ci" in s


def test_paired_comparison_sign_and_win_rate() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(120.0, 10.0, size=500)
    b = a + 2.0  # b uniformly worse by 2 m/s
    cap = np.ones(500, dtype=bool)
    p = paired_comparison(a, cap, b, cap, n_boot=500, seed=0)
    assert p["n_pairs"] == 500
    assert p["delta_mean"] == pytest.approx(-2.0)  # a - b
    assert p["win_rate_a"] == 1.0
    assert p["wilcoxon_p"] < 1e-6
    assert p["delta_mean_ci"][1] < 0  # CI excludes zero


def test_paired_comparison_drops_either_failed() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    p = paired_comparison(a, np.array([True, False, True]), b, np.array([True, True, False]), n_boot=100, seed=0)
    assert p["n_pairs"] == 1


def test_actual_sims_formula() -> None:
    # 3 gens, n_pop=2, n_sims=10; validation fired twice; one curation event;
    # seeds changed once after the curation -> one parent re-eval.
    records = [
        {"all_costs": [1, 2], "pool_metrics": {"last_curation_gen": 0}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 5.0}, "pool_metrics": {"last_curation_gen": 1}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 4.0}, "pool_metrics": {"last_curation_gen": 1}},
    ]
    s = actual_sims(records, training_n_sims=10, validation_n_sims=1000, curation_sample_size=1000, curation_top_k=1)
    assert s["training"] == 3 * 2 * 10
    assert s["validation"] == 2 * 1000
    assert s["curation"] == 2 * 1 * 1000  # two distinct last_curation_gen values
    assert s["reeval"] == 2 * 2 * 10  # parent re-eval on each curation-change gen
    assert s["total"] == s["training"] + s["validation"] + s["curation"] + s["reeval"]
