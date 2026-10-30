"""Tests for aerocapture.training.report."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_read_constraint_limits_includes_heat_load(tmp_path: Path) -> None:
    """_read_constraint_limits is `build_cost_kwargs`' limits, so the PDF colors a violator as
    constrained exactly where the stats block, the cost and the gate count it (D3)."""
    from aerocapture.training.report import _read_constraint_limits

    toml = tmp_path / "m.toml"
    toml.write_text("[flight.constraints]\nmax_heat_flux = 200.0\nmax_load_factor = 15.0\nmax_heat_load = 25000.0\n")
    heat_flux, g_load, heat_load = _read_constraint_limits(toml)
    assert heat_flux == 200.0
    assert g_load == 15.0
    assert heat_load == 25000.0


def test_heat_load_violator_constrained_when_limit_absent(tmp_path: Path) -> None:
    """A captured heat-load-only violator must classify as CONSTRAINED even when the
    config omits max_heat_load (Earth/ESR-style). Otherwise classification (which reads
    _read_constraint_limits) colors it OK while the stats block + cost — which default
    the heat-load limit to 25000.0 — count it as a violation (residual D3 mismatch)."""
    from aerocapture.training import charts
    from aerocapture.training.report import _read_constraint_limits

    toml = tmp_path / "m.toml"
    # Heat-flux and g-load limits present, but NO max_heat_load (cf. configs/missions/earth.toml).
    toml.write_text("[flight.constraints]\nmax_heat_flux = 200.0\nmax_load_factor = 15.0\n")

    heat_flux, g_load, heat_load = _read_constraint_limits(toml)
    assert heat_load == 25000.0  # build_cost_kwargs' default, the one every scorer uses

    # One captured trajectory: heat-flux + g-load within limits, heat load over 25000 kJ/m².
    rec = np.zeros((1, 52), dtype=np.float64)
    rec[0, charts._FR_IFINAL] = 3  # exited atmosphere
    rec[0, charts._FR_ECC] = 0.5  # bound orbit -> captured
    rec[0, charts._FR_MAX_HEAT_FLUX] = 100.0  # below 200
    rec[0, charts._FR_MAX_G_LOAD] = 3.0  # below 15
    rec[0, charts._FR_INTEGRATED_FLUX] = 30.0  # MJ/m² -> 30000 kJ/m² > 25000

    cls = charts.classify_trajectories(rec, heat_flux_limit=heat_flux, g_load_limit=g_load, heat_load_limit=heat_load)
    assert cls[0] == charts.TRAJ_CONSTRAINED


def test_read_constraint_limits_defaults_every_missing_limit_like_the_cost_function(tmp_path: Path) -> None:
    """Until #220 a missing flux or g limit read None (never violated) while the cost penalized at 200 / 15."""
    from aerocapture.training.cost import build_cost_kwargs
    from aerocapture.training.report import _read_constraint_limits

    toml = tmp_path / "m.toml"
    toml.write_text("[flight.constraints]\nmax_heat_load = 30000.0\n")
    kw = build_cost_kwargs({"flight": {"constraints": {"max_heat_load": 30000.0}}})
    assert _read_constraint_limits(toml) == (kw["heat_flux_limit"], kw["g_load_limit"], 30000.0) == (200.0, 15.0, 30000.0)
