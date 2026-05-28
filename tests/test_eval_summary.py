"""compute_eval_summary + format_eval_summary cover the same stats block
print_eval_summary writes to stdout, but as structured data so consumers
(warm-start report PDF, RL report, etc.) can embed it."""

from __future__ import annotations

import numpy as np
from aerocapture.training.report import compute_eval_summary, format_eval_summary


def _synthetic_final_records(n: int = 1000, *, capture_rate: float = 0.995) -> np.ndarray:
    """Build a (n, 52) final_records array with deterministic stats so the
    formatted block is predictable. The column indices match charts._FR_* but
    only the few fields the summary reads need realistic values; the rest are 0."""
    from aerocapture.training import charts

    rng = np.random.default_rng(0)
    records = np.zeros((n, 52))
    n_captured = int(round(capture_rate * n))
    # Mark first n_captured rows as captures (ifinal=3, ecc<1).
    records[:n_captured, charts._FR_IFINAL] = 3
    records[:n_captured, charts._FR_ECC] = 0.5
    # Non-captures: ifinal=1, ecc>=1 so the summary excludes them.
    records[n_captured:, charts._FR_IFINAL] = 1
    records[n_captured:, charts._FR_ECC] = 1.2
    # Captured-only stats: small DV, small apoapsis err, small inclination err.
    records[:n_captured, charts._FR_DV_TOTAL] = rng.uniform(400, 800, n_captured)
    records[:n_captured, charts._FR_APO_ERR] = rng.uniform(-3000, 7000, n_captured)
    records[:n_captured, charts._FR_PERI_ERR] = rng.uniform(-100, 50, n_captured)
    records[:n_captured, charts._FR_INCL_ERR] = rng.uniform(-1.0, 0.5, n_captured)
    # Constraint columns (over ALL sims).
    records[:, charts._FR_MAX_HEAT_FLUX] = rng.uniform(150, 200, n)
    records[:, charts._FR_MAX_G_LOAD] = rng.uniform(2.0, 3.5, n)
    records[:, charts._FR_INTEGRATED_FLUX] = rng.uniform(18, 20, n)  # MJ/m2
    return records


def test_compute_eval_summary_returns_expected_keys() -> None:
    records = _synthetic_final_records(n=200)
    summary = compute_eval_summary(records, n_sims=200)
    assert summary["n_sims"] == 200
    assert summary["n_captured"] > 0
    assert 0.0 <= summary["capture_rate"] <= 1.0
    for k in ("p50", "p95", "rms"):
        assert k in summary["cost"]
    assert summary["captured"] is not None
    for axis in ("dv", "apoapsis", "periapsis", "inclination"):
        assert {"p50", "p95", "mean"} == set(summary["captured"][axis])
    for axis in ("heat_flux", "g_load", "heat_load"):
        assert {"p50", "p95", "max", "limit", "viol_pct"} == set(summary["constraints"][axis])


def test_compute_eval_summary_handles_zero_captures() -> None:
    records = _synthetic_final_records(n=50, capture_rate=0.0)
    summary = compute_eval_summary(records, n_sims=50)
    assert summary["n_captured"] == 0
    assert summary["captured"] is None
    # Cost / constraints still computed over all sims.
    assert summary["cost"]["rms"] > 0
    assert summary["constraints"]["heat_flux"]["p50"] > 0


def test_compute_eval_summary_with_constraint_limits() -> None:
    records = _synthetic_final_records(n=100)
    cost_kwargs = {"heat_flux_limit": 195.0, "g_load_limit": 3.0, "heat_load_limit": 19.5}
    summary = compute_eval_summary(records, n_sims=100, cost_kwargs=cost_kwargs)
    # viol_pct populated; record sample has values up to 200 → some violations.
    assert summary["constraints"]["heat_flux"]["limit"] == 195.0
    assert summary["constraints"]["heat_flux"]["viol_pct"] is not None
    assert 0.0 <= summary["constraints"]["heat_flux"]["viol_pct"] <= 100.0


def test_format_eval_summary_lines_match_user_spec() -> None:
    """Format must match the exact pattern the user requested:
    Final evaluation (N sims):
      Objective cost:     p50=...  p95=...  RMS=...
      Capture rate:       k/N (rate%)
      Delta-V (m/s):      p50=...  p95=...  mean=...
      ...
    """
    records = _synthetic_final_records(n=1000)
    summary = compute_eval_summary(records, n_sims=1000, cost_kwargs={"heat_flux_limit": 200.0})
    lines = format_eval_summary(summary, indent="    ")
    # Header
    assert lines[0] == "Final evaluation (1000 sims):"
    # Cost line
    assert lines[1].startswith("    Objective cost:") and "p50=" in lines[1] and "RMS=" in lines[1]
    # Capture rate
    assert lines[2].startswith("    Capture rate:") and "/1000" in lines[2]
    # The captured-only rows: Delta-V, Apoapsis, Periapsis, Inclin.
    bodies = [line.lstrip() for line in lines[3:7]]
    assert bodies[0].startswith("Delta-V (m/s):")
    assert bodies[1].startswith("Apoapsis err (km):")
    assert bodies[2].startswith("Periapsis err (km):")
    assert bodies[3].startswith("Inclin. err (deg):")
    # Constraint block
    constraint_bodies = [line.lstrip() for line in lines[7:10]]
    assert constraint_bodies[0].startswith("Heat flux (kW/m2):")
    assert constraint_bodies[1].startswith("G-load (g):")
    assert constraint_bodies[2].startswith("Heat load (kJ/m2):")
    # Heat flux violation present because we passed a limit
    assert "> 200" in constraint_bodies[0]


def test_format_eval_summary_no_violations_section_when_limit_absent() -> None:
    """When cost_kwargs omits a limit, the line still prints stats but no "% > X" suffix."""
    records = _synthetic_final_records(n=20)
    summary = compute_eval_summary(records, n_sims=20)  # no cost_kwargs
    lines = format_eval_summary(summary)
    flux_line = next(line for line in lines if "Heat flux" in line)
    assert "%" not in flux_line.split("max=")[1]


def test_format_eval_summary_omits_captured_block_when_no_captures() -> None:
    records = _synthetic_final_records(n=20, capture_rate=0.0)
    summary = compute_eval_summary(records, n_sims=20)
    lines = format_eval_summary(summary)
    # No Delta-V / Apoapsis / Inclin lines
    assert not any("Delta-V" in line for line in lines)
    assert not any("Apoapsis" in line for line in lines)
    # But cost + capture rate + constraint blocks are still there.
    assert any("Capture rate:" in line for line in lines)
    assert any("Heat flux" in line for line in lines)
