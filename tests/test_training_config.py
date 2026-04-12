"""Tests for training configuration dataclasses."""

from __future__ import annotations

import tomllib
from pathlib import Path

from aerocapture.training.optimizer import OptimizerConfig


def test_optimizer_config_defaults() -> None:
    opt = OptimizerConfig()
    assert opt.algorithm == "ga"
    assert opt.n_pop == 60
    assert opt.n_gen == 2500
    assert opt.adaptive_seeds is False


def test_dv_threshold_parsed_from_toml(tmp_path: Path) -> None:
    """Verify dv_threshold is correctly extracted from TOML cost_function section."""
    toml_content = """\
[cost_function]
dv_threshold = 500.0
g_load_limit = 15.0
heat_flux_limit = 200.0
g_load_weight = 1000.0
heat_flux_weight = 1000.0
"""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(toml_content)

    with open(toml_file, "rb") as f:
        _toml = tomllib.load(f)

    cost_cfg = _toml.get("cost_function", {})
    cost_kwargs = {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
        "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
    }
    assert cost_kwargs["dv_threshold"] == 500.0


def test_dv_threshold_default_when_missing(tmp_path: Path) -> None:
    """When dv_threshold is absent from TOML, default to 1000.0."""
    toml_content = """\
[cost_function]
g_load_limit = 15.0
"""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(toml_content)

    with open(toml_file, "rb") as f:
        _toml = tomllib.load(f)

    cost_cfg = _toml.get("cost_function", {})
    dv_threshold = float(cost_cfg.get("dv_threshold", 1000.0))
    assert dv_threshold == 1000.0
