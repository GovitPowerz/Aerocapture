"""Table-driven tests for the single scaffolding deploy rule (deploy_overrides)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aerocapture.training.deploy_overrides import INTEGER_PARAM_NAMES, load_scaffolding_overrides, overrides_from_params, resolve_eval_toml
from aerocapture.training.param_spaces import PARAM_SPACES
from aerocapture.training.toml_utils import set_dot_path


@pytest.mark.parametrize(
    "key, scheme, expected_path",
    [
        ("lateral.tau", "ftc", "guidance.lateral.tau"),
        ("exit.exit_velocity_threshold", "equilibrium_glide", "guidance.ftc.exit_velocity_threshold"),
        ("nav.density_filter_gain", "ftc", "navigation.density_filter_gain"),
        ("thermal.heat_flux_activation", "fnpag", "guidance.thermal_limiter.heat_flux_activation"),
        ("shaping.max_bank_acceleration", "ftc", "guidance.command_shaping.max_bank_acceleration"),
        ("k_hdot_scale", "equilibrium_glide", "guidance.equilibrium_glide.k_hdot_scale"),
    ],
)
def test_every_prefix_routes(key: str, scheme: str, expected_path: str) -> None:
    assert overrides_from_params({key: 1.5}, scheme)[expected_path] == 1.5


def test_shaping_key_sets_enabled_flag() -> None:
    assert overrides_from_params({"shaping.max_bank_acceleration": 7.0}, "ftc")["guidance.command_shaping.enabled"] is True


def test_no_shaping_key_no_enabled_flag() -> None:
    assert "guidance.command_shaping.enabled" not in overrides_from_params({"lateral.tau": 3.0}, "ftc")


def test_integer_coercion_reads_param_specs() -> None:
    assert "lateral.max_reversals" in INTEGER_PARAM_NAMES
    assert {s.name for specs in PARAM_SPACES.values() for s in specs if s.is_integer} <= INTEGER_PARAM_NAMES
    v = overrides_from_params({"lateral.max_reversals": 4.2}, "ftc")["guidance.lateral.max_reversals"]
    assert isinstance(v, int) and v == 4


def test_ref_bank_gene_is_never_routed() -> None:
    assert overrides_from_params({"ref_bank": 65.0, "k": 1.0}, "ftc") == {"guidance.ftc.k": 1.0}


def test_scaffolding_only_drops_unprefixed_keys() -> None:
    assert overrides_from_params({"lateral.tau": 1.0, "w0": 0.3}, "", scaffolding_only=True) == {"guidance.lateral.tau": 1.0}


def test_load_scaffolding_overrides_absent_file(tmp_path: Path) -> None:
    assert load_scaffolding_overrides(tmp_path) == {}


def test_load_scaffolding_overrides_reads_best_params(tmp_path: Path) -> None:
    (tmp_path / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9, "shaping.max_bank_acceleration": 5.0, "w0": 1.0}))
    assert load_scaffolding_overrides(tmp_path) == {
        "navigation.density_filter_gain": 0.9,
        "guidance.command_shaping.max_bank_acceleration": 5.0,
        "guidance.command_shaping.enabled": True,
    }


def test_resolve_eval_toml_prefers_optimized_toml(tmp_path: Path) -> None:
    cell = tmp_path / "ftc"
    cell.mkdir()
    (cell / "optimized_ftc.toml").write_text("")
    (cell / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9}))
    toml, overrides = resolve_eval_toml(tmp_path / "base.toml", cell)
    assert toml == cell / "optimized_ftc.toml"
    assert overrides == {}


def test_resolve_eval_toml_glob_fallback_for_custom_dir_name(tmp_path: Path) -> None:
    cell = tmp_path / "ftc_joint_ref"
    cell.mkdir()
    (cell / "optimized_ftc.toml").write_text("")
    toml, _ = resolve_eval_toml(tmp_path / "base.toml", cell)
    assert toml == cell / "optimized_ftc.toml"


def test_resolve_eval_toml_nn_cell_uses_base_plus_scaffolding(tmp_path: Path) -> None:
    cell = tmp_path / "mamba"
    cell.mkdir()
    (cell / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9}))
    toml, overrides = resolve_eval_toml(tmp_path / "base.toml", cell)
    assert toml == tmp_path / "base.toml"
    assert overrides == {"navigation.density_filter_gain": 0.9}


def test_set_dot_path_creates_intermediate_tables() -> None:
    data: dict = {"guidance": {"type": "ftc"}}
    set_dot_path(data, "guidance.command_shaping.enabled", True)
    set_dot_path(data, "navigation.density_filter_gain", 0.8)
    assert data == {"guidance": {"type": "ftc", "command_shaping": {"enabled": True}}, "navigation": {"density_filter_gain": 0.8}}
