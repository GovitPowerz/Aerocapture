"""Tests for aerocapture.training.report."""

from __future__ import annotations

from pathlib import Path


def test_read_constraint_limits_includes_heat_load(tmp_path: Path) -> None:
    """_read_constraint_limits must return the heat-load limit so the PDF colors
    heat-load-only violators as constrained, matching the stats block (D3)."""
    from aerocapture.training.report import _read_constraint_limits

    toml = tmp_path / "m.toml"
    toml.write_text("[flight.constraints]\nmax_heat_flux = 200.0\nmax_load_factor = 15.0\nmax_heat_load = 25000.0\n")
    heat_flux, g_load, heat_load = _read_constraint_limits(toml)
    assert heat_flux == 200.0
    assert g_load == 15.0
    assert heat_load == 25000.0
