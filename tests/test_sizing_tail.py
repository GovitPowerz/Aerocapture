"""The one sizing-tail scorer (design 2026-10-30, #220): estimator, record, pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aerocapture.training import sizing_tail as st
from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_ECC, FR_IFINAL, FR_INTEGRATED_FLUX, FR_MAX_G_LOAD, FR_MAX_HEAT_FLUX, CellResult
from aerocapture.training.cost import build_cost_kwargs

COST_KWARGS = build_cost_kwargs({"flight": {"constraints": {"max_heat_flux": 200.0, "max_load_factor": 15.0, "max_heat_load": 25000.0}}})


def _records(n: int = 100, *, n_captured: int | None = None) -> np.ndarray:
    """n rows, the first `n_captured` captured with DV 100..., no violation."""
    k = n if n_captured is None else n_captured
    fr = np.zeros((n, 52))
    fr[:k, FR_IFINAL] = 3.0
    fr[:k, FR_ECC] = 0.5
    fr[k:, FR_IFINAL] = 1.0
    fr[k:, FR_ECC] = 1.2
    fr[:k, FR_DV_TOTAL] = np.arange(1, k + 1, dtype=float) + 99.0  # 100..99+k
    fr[k:, FR_DV_TOTAL] = 5000.0
    fr[:, FR_MAX_HEAT_FLUX] = 100.0
    fr[:, FR_MAX_G_LOAD] = 5.0
    fr[:, FR_INTEGRATED_FLUX] = 20.0  # MJ/m2 -> 20000 kJ/m2
    return fr


# --- the estimator -----------------------------------------------------------------------


def test_cvar_is_mean_of_worst_tail() -> None:
    x = np.arange(1.0, 101.0)
    assert st.cvar(x, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_cvar_small_sample_uses_at_least_one() -> None:
    assert st.cvar(np.array([3.0, 1.0]), 0.95) == 3.0


def test_cvar_on_no_sample_is_nan_without_a_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(st.cvar(np.array([]), 0.95))


@pytest.mark.parametrize(("n", "k"), [(1000, 50), (970, 49), (997, 50), (10, 1), (5, 1)])
def test_cvar_tail_size_is_the_paper_stats_arithmetic(n: int, k: int) -> None:
    """`round(n * (1 - level))`, never `round(n * 0.05)`: at n = 970 the two differ (48.50000000000004
    rounds to 49, 48.5 rounds to 48), which is where the inline copies and `paper_stats.cvar` disagreed."""
    assert st.cvar_tail_size(n, 0.95) == k


def test_cvar_tail_size_970_differs_from_the_retired_inline_copy() -> None:
    assert st.cvar_tail_size(970, 0.95) == 49 != round(970 * 0.05)


def test_bootstrap_ci_brackets_the_mean() -> None:
    x = np.random.default_rng(0).normal(100.0, 5.0, size=1000)
    lo, hi = st.bootstrap_ci(x, np.mean, n_boot=2000, seed=1)
    assert lo < float(np.mean(x)) < hi
    assert hi - lo < 2.0


# --- the DV block ----------------------------------------------------------------------------


def test_dv_tail_keys_rounding_and_optional_cis() -> None:
    dvc = np.arange(1.0, 1001.0) / 3.0
    plain = st.dv_tail(dvc)
    assert set(plain) == {"dv_mean", "dv_p50", "dv_p95", "dv_p99", "dv_p999", "dv_p9987", "dv_cvar95", "dv_cvar99", "dv_cvar999", "dv_max_descriptive"}
    assert all(v == round(v, 2) for v in plain.values())
    assert plain["dv_cvar95"] == pytest.approx(round(float(np.sort(dvc)[-50:].mean()), 2))
    with_ci = st.dv_tail(dvc, n_boot=200)
    assert set(with_ci) - set(plain) == {"dv_mean_ci", "dv_p95_ci", "dv_cvar95_ci", "dv_p99_ci", "dv_cvar99_ci"}
    assert with_ci["dv_cvar95_ci"][0] <= with_ci["dv_cvar95"] <= with_ci["dv_cvar95_ci"][1]
    assert st.dv_tail(np.array([])) == {}


def test_run_stats_is_capture_conditional() -> None:
    ifinal = np.array([3.0, 3.0, 3.0, 2.0])
    ecc = np.array([0.5, 0.5, 1.2, 0.5])  # third sim: ifinal 3 but hyperbolic
    dv = np.array([100.0, 200.0, 999.0, 999.0])
    s = st.run_stats(ifinal, ecc, dv, n_boot=200)
    assert (s["n"], s["n_captured"], s["capture_pct"]) == (4, 2, 50.0)
    assert s["dv_mean"] == pytest.approx(150.0)
    assert "dv_cvar95_ci" in s and "capture_pct_ci" in s


# --- the record ------------------------------------------------------------------------------


def test_tail_record_capture_tail_cost_and_estimator() -> None:
    rec = st.tail_record(_records(100, n_captured=90), COST_KWARGS)
    assert (rec["n"], rec["n_captured"], rec["capture_pct"]) == (100, 90, 90.0)
    assert rec["dv_max_descriptive"] == 189.0
    assert rec["dv_cvar95"] == pytest.approx(np.mean([185, 186, 187, 188, 189]))  # 90 * (1 - 0.95) = 4.5000...04 -> 5 samples
    assert rec["estimator"] == st.CVAR_ESTIMATOR
    assert rec["rms_cost"] > 0 and np.isfinite(rec["rms_cost"])
    assert (rec["viol_pct"], rec["viol_n"]) == (0.0, 0)


def test_tail_record_viol_pct_is_the_union_of_the_constraints() -> None:
    fr = _records(100)
    fr[[0, 1], FR_MAX_HEAT_FLUX] = 250.0  # flux on rows 0, 1
    fr[[1, 2], FR_INTEGRATED_FLUX] = 30.0  # heat load on rows 1, 2 (row 1 twice)
    rec = st.tail_record(fr, COST_KWARGS)
    assert (rec["heat_flux_viol_n"], rec["g_load_viol_n"], rec["heat_load_viol_n"]) == (2, 0, 2)
    assert (rec["viol_n"], rec["viol_pct"]) == (3, 3.0)  # not max (2), not sum (4)
    assert (rec["heat_flux_viol_pct"], rec["heat_load_viol_pct"]) == (2.0, 2.0)


def test_tail_record_scores_a_missing_limit_at_the_cost_functions_default() -> None:
    """A config that omits a limit is scored at `build_cost_kwargs`' default (200 / 15 / 25000), as the
    gate scores it: no None-becomes-inf, no 'never violated'."""
    fr = _records(10)
    fr[0, FR_MAX_G_LOAD] = 20.0
    rec = st.tail_record(fr, build_cost_kwargs({}))
    assert rec["g_load_viol_n"] == 1 and rec["viol_n"] == 1


def test_tail_record_omits_an_unassessed_constraint() -> None:
    rec = st.tail_record(_records(10), {k: v for k, v in COST_KWARGS.items() if k != "g_load_limit"})
    assert "g_load_viol_pct" not in rec and "heat_flux_viol_pct" in rec


def _cell(fr: np.ndarray, toml: Path, overrides: dict[str, object]) -> CellResult:
    return CellResult(final_records=fr, dispersions=np.zeros((len(fr), 26)), trajectories=None, seeds=list(range(len(fr))), toml_path=toml, overrides=overrides)


def test_score_cell_names_the_regime_flown_and_the_pool(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[monte_carlo]\nnoise_seeding = "legacy"\n')
    bare = tmp_path / "bare.toml"
    bare.write_text("[monte_carlo]\nseed = 7\n")
    fr = _records(20)
    assert st.score_cell(_cell(fr, bare, {"monte_carlo.noise_seeding": "legacy"}), COST_KWARGS, pool="p")["noise_seeding"] == "legacy"
    assert st.score_cell(_cell(fr, legacy, {}), COST_KWARGS, pool="p")["noise_seeding"] == "legacy"
    rec = st.score_cell(_cell(fr, bare, {}), COST_KWARGS, pool="FINAL_EVAL 2M")
    assert (rec["noise_seeding"], rec["pool"]) == ("per_draw", "FINAL_EVAL 2M")
    assert rec["estimator"] == st.CVAR_ESTIMATOR


def test_score_cell_scores_at_the_limit_the_flight_overrode(tmp_path: Path) -> None:
    """heat_load_slope re-flies a cell under `flight.constraints.max_heat_load`; the record's violation
    rule and `rms_cost` follow the override (the limit the simulator flew at), not the TOML's limit."""
    from aerocapture.training.cost import compute_cost

    fr = _records(10)
    fr[0, FR_INTEGRATED_FLUX] = 27.0  # 27000 kJ/m2: over the 25000 of COST_KWARGS, under a 30000 ceiling
    legacy: dict[str, object] = {"monte_carlo.noise_seeding": "legacy"}
    assert st.score_cell(_cell(fr, tmp_path / "unused.toml", legacy), COST_KWARGS)["heat_load_viol_n"] == 1
    rec = st.score_cell(_cell(fr, tmp_path / "unused.toml", {**legacy, "flight.constraints.max_heat_load": 30000.0}), COST_KWARGS)
    assert (rec["heat_load_viol_n"], rec["viol_n"]) == (0, 0)
    assert rec["rms_cost"] == round(compute_cost(fr, **{**COST_KWARGS, "heat_load_limit": 30000.0}), 2)


def test_cost_rms_matches_compute_cost() -> None:
    from aerocapture.training.cost import compute_cost

    fr = _records(50, n_captured=45)
    assert st.tail_record(fr, COST_KWARGS)["rms_cost"] == round(compute_cost(fr, **COST_KWARGS), 2)


# --- pairing ---------------------------------------------------------------------------------


def test_paired_comparison_sign_and_win_rate() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(120.0, 10.0, size=500)
    b = a + 2.0
    cap = np.ones(500, dtype=bool)
    p = st.paired_comparison(a, cap, b, cap, n_boot=500, seed=0)
    assert p["n_pairs"] == 500
    assert p["delta_mean"] == pytest.approx(-2.0)
    assert p["win_rate_a"] == 1.0
    assert p["wilcoxon_p"] < 1e-6
    assert p["delta_mean_ci"][1] < 0


def test_paired_comparison_drops_either_failed() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    p = st.paired_comparison(a, np.array([True, False, True]), b, np.array([True, True, False]), n_boot=100, seed=0)
    assert p["n_pairs"] == 1


def test_paired_tail_deltas_and_cis() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(120.0, 10.0, size=400)
    b = a + 5.0
    cap_a = np.ones(400, dtype=bool)
    cap_b = cap_a.copy()
    cap_b[:4] = False  # b loses 1% of the scenarios
    p = st.paired_tail(a, cap_a, b, cap_b, n_boot=300, seed=0)
    assert p["n_pairs"] == 396
    assert p["delta_cvar95"] < 0 and p["delta_cvar95_ci"][0] <= p["delta_cvar95"] <= p["delta_cvar95_ci"][1]
    assert p["delta_p95"] < 0 and p["delta_p95_ci"][1] < 0
    assert p["delta_capture_pts"] == 1.0 and p["delta_capture_pts_ci"][0] <= 1.0 <= p["delta_capture_pts_ci"][1]


def test_paired_tail_with_a_side_that_never_captures_is_nan_not_a_crash() -> None:
    """The depth scripts pair after every cell is flown: an IndexError there would lose the flights."""
    dv = np.full(50, 100.0)
    p = st.paired_tail(dv, np.ones(50, dtype=bool), dv, np.zeros(50, dtype=bool), n_boot=20)
    assert np.isnan(p["delta_p95"]) and np.isnan(p["delta_cvar95"]) and p["delta_capture_pts"] == 100.0


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "articles/paper/data/runs"


def test_paired_tail_reproduces_a_committed_results_json_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """results.json's paired block is the frozen shape of `paired_tail`: the committed numbers
    (point deltas and the scenario-resampled CIs at seed 0, 10 000 resamples) must come back, in the
    committed key order (`aggregate_results` writes without sort_keys)."""
    import json

    import pandas as pd

    monkeypatch.syspath_prepend(str(REPO / "articles/paper/scripts"))
    from aggregate_results import _PAIRED_STAT_KEYS  # type: ignore[import-not-found]

    results = json.loads((REPO / "articles/paper/data/results.json").read_text())
    pair = results["paired"]["v4_nn_vs_jointftc"]
    cols = ["ifinal", "eccentricity", "dv_total_m_s"]
    da, db = (pd.read_parquet(RUNS / pair[side] / "final_eval.parquet", columns=cols) for side in ("a", "b"))
    n = min(len(da), len(db))
    da, db = da.head(n), db.head(n)
    ca = st.capture_mask(da["ifinal"].to_numpy(), da["eccentricity"].to_numpy())
    cb = st.capture_mask(db["ifinal"].to_numpy(), db["eccentricity"].to_numpy())
    got = st.paired_tail(da["dv_total_m_s"].to_numpy(), ca, db["dv_total_m_s"].to_numpy(), cb)
    assert {"a": pair["a"], "b": pair["b"], **{k: got[k] for k in _PAIRED_STAT_KEYS if k in got}} == pair
    assert ["a", "b", *(k for k in _PAIRED_STAT_KEYS if k in got)] == list(pair)


# --- two scripts that used to disagree ---------------------------------------------------------


def test_quote_marginal_and_fresh_pool_requote_report_the_same_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before #220 quote_marginal scored CVaR95 as the mean at or above the 95th percentile and
    fresh_pool_requote as the worst n // 20: one flight, two numbers. Both now read the record."""
    import json

    from aerocapture.training import cell_eval

    n = 970  # the captured count where the retired estimators and the record all differ
    fr = _records(n)
    fr[:, FR_DV_TOTAL] = np.sqrt(np.arange(1, n + 1, dtype=float)) * 20.0
    toml = REPO / "configs/training/sweep/mamba_p962.toml"
    fake = lambda _d, _t, seeds=None, *, pool=None, **kw: _cell(fr, toml, dict(kw.get("extra_overrides") or {}))  # noqa: E731
    monkeypatch.syspath_prepend(str(REPO / "experiments/ou_marginal"))
    monkeypatch.syspath_prepend(str(REPO / "articles/paper/scripts"))
    import fresh_pool_requote  # type: ignore[import-not-found]
    import quote_marginal  # type: ignore[import-not-found]

    # quote_marginal binds evaluate_cell at import (another test module may have imported it first); fresh_pool_requote imports it inside main.
    monkeypatch.setattr(quote_marginal, "evaluate_cell", fake)
    monkeypatch.setattr(cell_eval, "evaluate_cell", fake)
    quoted = quote_marginal.score(str(toml.relative_to(REPO)), None, np.arange(n), "frozen")
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "best_model.json").write_text("{}")
    fresh_pool_requote.main([str(run_dir), "--toml", str(toml), "--n-sims", str(n), "--noise-seeding", "legacy"])
    requoted = json.loads((run_dir / "fresh_pool_requote.json").read_text())

    for k in ("capture_pct", "dv_p50", "dv_p95", "dv_p99", "dv_cvar95"):
        assert quoted[k] == requoted[k], k
    x = np.sort(fr[:, FR_DV_TOTAL])
    assert quoted["dv_cvar95"] == round(float(x[-49:].mean()), 2)  # round(970 * 0.05000...04) = 49 samples
    assert requoted["estimator"] == st.CVAR_ESTIMATOR and requoted["noise_seeding"] == "legacy" and requoted["pool"] == "fresh (offset 8M)"
