"""Unit tests for compare_guidance._apply_optimized_params_to_toml."""

from aerocapture.training.compare_guidance import _apply_optimized_params_to_toml
from aerocapture.training.param_spaces import route_param_path


def _make_toml_data() -> dict:
    return {
        "guidance": {
            "type": "ftc",
            "command_shaping": {},
        },
        "navigation": {},
    }


def test_apply_optimized_params_routes_correctly() -> None:
    toml_data = _make_toml_data()
    params = {
        "lateral.max_reversals": 3.7,
        "exit.kp": 1.5,
        "shaping.max_bank_acceleration": 10.0,
        "nav.density_filter_gain": 0.8,
    }
    _apply_optimized_params_to_toml(toml_data, params, "ftc")

    # max_reversals must be coerced to int and rounded (3.7 -> 4)
    max_rev_path = route_param_path("lateral.max_reversals", "ftc")
    assert max_rev_path == "guidance.lateral.max_reversals"
    assert toml_data["guidance"]["lateral"]["max_reversals"] == 4
    assert isinstance(toml_data["guidance"]["lateral"]["max_reversals"], int)

    # float param lands at its routed path unchanged
    kp_path = route_param_path("exit.kp", "ftc")
    assert kp_path == "guidance.ftc.kp"
    assert toml_data["guidance"]["ftc"]["kp"] == 1.5

    # shaping param lands at its routed path and flips enabled=True
    shaping_path = route_param_path("shaping.max_bank_acceleration", "ftc")
    assert shaping_path == "guidance.command_shaping.max_bank_acceleration"
    assert toml_data["guidance"]["command_shaping"]["max_bank_acceleration"] == 10.0
    assert toml_data["guidance"]["command_shaping"]["enabled"] is True

    # nav param lands at its routed path
    nav_path = route_param_path("nav.density_filter_gain", "ftc")
    assert nav_path == "navigation.density_filter_gain"
    assert toml_data["navigation"]["density_filter_gain"] == 0.8


def test_no_shaping_key_does_not_set_enabled() -> None:
    toml_data = _make_toml_data()
    params = {"lateral.tau": 30.0}
    _apply_optimized_params_to_toml(toml_data, params, "ftc")
    assert "enabled" not in toml_data["guidance"]["command_shaping"]


def test_unprefixed_key_routes_to_scheme_block() -> None:
    toml_data = _make_toml_data()
    params = {"gain": 2.5}
    _apply_optimized_params_to_toml(toml_data, params, "equilibrium_glide")
    assert toml_data["guidance"]["equilibrium_glide"]["gain"] == 2.5
