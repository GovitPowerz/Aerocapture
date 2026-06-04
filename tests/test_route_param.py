"""Tests for route_param_path: all 6 canonical routing cases."""

from aerocapture.training.param_spaces import route_param_path


def test_lateral_prefix() -> None:
    assert route_param_path("lateral.tau", "ftc") == "guidance.lateral.tau"


def test_exit_prefix() -> None:
    assert route_param_path("exit.exit_velocity_threshold", "ftc") == "guidance.ftc.exit_velocity_threshold"


def test_nav_prefix() -> None:
    assert route_param_path("nav.density_filter_gain", "ftc") == "navigation.density_filter_gain"


def test_thermal_prefix() -> None:
    assert route_param_path("thermal.heat_flux_activation", "ftc") == "guidance.thermal_limiter.heat_flux_activation"


def test_shaping_prefix() -> None:
    assert route_param_path("shaping.max_bank_acceleration", "ftc") == "guidance.command_shaping.max_bank_acceleration"


def test_unprefixed_routes_to_scheme() -> None:
    assert route_param_path("k", "eqglide") == "guidance.eqglide.k"


def test_unprefixed_uses_scheme_name() -> None:
    assert route_param_path("k_hdot_scale", "equilibrium_glide") == "guidance.equilibrium_glide.k_hdot_scale"
