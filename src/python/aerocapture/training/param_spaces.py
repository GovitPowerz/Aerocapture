"""Parameter space definitions for each guidance scheme.

Each guidance scheme has a list of ParamSpec entries defining the
tunable parameters, their bounds, defaults, and whether to use
log-scale encoding (useful for gains spanning orders of magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    """Single tunable parameter specification."""

    name: str
    p_min: float
    p_max: float
    default: float
    log_scale: bool = False
    is_integer: bool = False


# Exit phase params shared by all unsigned-magnitude schemes.
# Prefixed with "exit." so evaluate.py routes them to [guidance.ftc] in TOML.
_EXIT_PARAMS: list[ParamSpec] = [
    ParamSpec("exit.exit_velocity_threshold", 3000.0, 5500.0, 4400.0),
    ParamSpec("exit.exit_pdyn_margin", 0.5, 4.0, 1.75),
    ParamSpec("exit.exit_radial_vel_gain", 1.0, 30.0, 10.0),
    ParamSpec("exit.exit_altitude_threshold", 30.0, 90.0, 60.0),  # km
]

# Lateral guidance params shared by all unsigned-magnitude schemes.
# Prefixed with "lateral." so evaluate.py routes them to [guidance.lateral] in TOML.
_LATERAL_PARAMS: list[ParamSpec] = [
    ParamSpec("lateral.tau", 2.0, 60.0, 15.0),  # seconds
    ParamSpec("lateral.threshold", 0.01, 2.0, 0.5),  # degrees (TOML units)
    ParamSpec("lateral.min_reversal_interval", 1.0, 30.0, 5.0),  # seconds
    ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),  # MJ/kg
    ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),  # MJ/kg
    ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0, is_integer=True),
]

# Thermal safety limiter params shared by all unsigned-magnitude schemes.
# Prefixed with "thermal." so evaluate.py routes them to [guidance.thermal_limiter] in TOML.
_THERMAL_LIMITER_PARAMS: list[ParamSpec] = [
    ParamSpec("thermal.heat_flux_activation", 0.6, 1.0, 1.0),
    ParamSpec("thermal.heat_load_activation", 0.6, 1.0, 1.0),
    ParamSpec("thermal.heat_flux_ramp_exponent", 0.5, 3.0, 1.0),
    ParamSpec("thermal.heat_load_ramp_exponent", 0.5, 3.0, 1.0),
]

# Command shaping params shared by all schemes.
# Prefixed with "shaping." so evaluate.py routes them to [guidance.command_shaping] in TOML.
_SHAPING_PARAMS: list[ParamSpec] = [
    ParamSpec("shaping.max_bank_acceleration", 2.0, 15.0, 5.0),  # deg/s^2
]

# Navigation density filter params shared by all unsigned-magnitude schemes.
# Prefixed with "nav." so routing sends them to [navigation] in TOML
# (density filter config lives in [navigation], not [guidance], and affects all schemes).
_NAV_PARAMS: list[ParamSpec] = [
    ParamSpec("nav.density_filter_gain", 0.3, 1.0, 0.8),
    ParamSpec("nav.density_gain_max_delta", 0.01, 0.5, 0.1),
]

# Combined scaffolding pack used when training a neural-network scheme with
# `scaffolding = "full"`. Same specs FTC trains, same order. The
# routing in `problem.py::_build_overrides` already handles every prefix.
_NN_SCAFFOLDING_PARAMS: list[ParamSpec] = [
    *_NAV_PARAMS,
    *_LATERAL_PARAMS,
    *_EXIT_PARAMS,
    *_THERMAL_LIMITER_PARAMS,
    *_SHAPING_PARAMS,
]

# Live-in-full_neural scaffolding params: nav density filter feeds the NN's
# observation vector, command shaping shapes its output. These 3 have standalone
# defaults, so they can be optimized without seeding from FTC. Used for
# `scaffolding = "live"` (full_neural schemes that want nav/shaping tuned but
# don't need the FTC-only lateral/exit/thermal pack).
_NN_LIVE_PARAMS: list[ParamSpec] = [
    *_NAV_PARAMS,
    *_SHAPING_PARAMS,
]


def active_scaffolding_specs(scaffolding: str) -> list[ParamSpec]:
    """Resolve the active scaffolding ParamSpec pack for a `scaffolding` value.

    "off" -> [], "live" -> nav+shaping (3), "full" -> the 17-param FTC pack.
    Raises KeyError on any other value (caught at config load with a clearer
    message).
    """
    return {
        "off": [],
        "live": _NN_LIVE_PARAMS,
        "full": _NN_SCAFFOLDING_PARAMS,
    }[scaffolding]


# TOML section key matches the guidance type name used in [guidance] type field
PARAM_SPACES: dict[str, list[ParamSpec]] = {
    "equilibrium_glide": [
        ParamSpec("k_hdot_scale", 0.05, 1.0, 0.3),
        ParamSpec("v_ratio_threshold", 0.9, 1.5, 1.1),
        ParamSpec("velocity_bias_high", 0.0, 0.5, 0.15),
        ParamSpec("velocity_bias_low", 0.0, 1.0, 0.3),
        ParamSpec("alt_bias_threshold", 20.0, 80.0, 40.0),
        ParamSpec("cos_bank_min", -1.0, 0.0, -0.5),
        ParamSpec("cos_bank_max", 0.5, 1.0, 0.95),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
        *_SHAPING_PARAMS,
    ],
    "energy_controller": [
        ParamSpec("gain", 1e-8, 1e-5, 5e-7, log_scale=True),
        ParamSpec("kp", 0.1, 5.0, 1.0),
        ParamSpec("kd", 0.0, 3.0, 0.5),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
        *_SHAPING_PARAMS,
    ],
    "pred_guid": [
        ParamSpec("k_drag_high", 0.1, 3.0, 0.8),
        ParamSpec("k_drag_low", 0.05, 2.0, 0.3),
        ParamSpec("pdyn_threshold", 10.0, 500.0, 100.0, log_scale=True),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
        *_SHAPING_PARAMS,
    ],
    "fnpag": [
        ParamSpec("energy_tol", 1e2, 1e5, 1e4, log_scale=True),
        ParamSpec("prediction_dt", 0.5, 5.0, 2.0),
        ParamSpec("bank_min_deg", 10.0, 40.0, 20.0),
        ParamSpec("bank_max_high_deg", 100.0, 170.0, 140.0),
        ParamSpec("bank_max_low_deg", 70.0, 130.0, 100.0),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
        *_SHAPING_PARAMS,
    ],
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("altitude_damping", 0.3, 1.5, 0.7),
        ParamSpec("altitude_frequency", 0.01, 0.2, 0.08),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        ParamSpec("pressure_coeff_base", -500.0, -10.0, -134.4),
        ParamSpec("pressure_coeff_scale_height", 4.0, 15.0, 6.9),
        ParamSpec("gain_fade_start_km", 60.0, 90.0, 80.0),
        ParamSpec("gain_fade_end_km", 85.0, 120.0, 100.0),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
        *_SHAPING_PARAMS,
    ],
    "piecewise_constant": [
        *(ParamSpec(f"bank_angle_{i}", -180.0, 180.0, 65.0) for i in range(10)),
        *_SHAPING_PARAMS,
    ],
}


def make_piecewise_constant_specs(n_segments: int) -> list[ParamSpec]:
    """Build PARAM_SPACES['piecewise_constant'] for arbitrary segment count.

    Mirrors the static entry above (which keeps the 10-segment default for
    backward compat) but produces N bank_angle_* specs + the shared shaping
    params. Used by train.py when [guidance.piecewise_constant] sets a
    non-default n_segments / bank_angles length.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    return [
        *(ParamSpec(f"bank_angle_{i}", -180.0, 180.0, 65.0) for i in range(n_segments)),
        *_SHAPING_PARAMS,
    ]


# TOML section name for each guidance type (used in [guidance.<section>])
GUIDANCE_TOML_SECTIONS: dict[str, str] = {
    "equilibrium_glide": "equilibrium_glide",
    "energy_controller": "energy_controller",
    "pred_guid": "pred_guid",
    "fnpag": "fnpag",
    "ftc": "ftc",
    "piecewise_constant": "piecewise_constant",
}

# Schemes that require a pre-computed reference trajectory
REQUIRES_REF_TRAJECTORY: set[str] = {"energy_controller", "pred_guid", "fnpag", "ftc"}

# Mapping from GA param prefix to canonical TOML section prefix.
# Keys include the trailing dot; values include the trailing dot.
_GUIDANCE_PREFIX_SECTIONS: dict[str, str] = {
    "lateral.": "guidance.lateral.",
    "exit.": "guidance.ftc.",
    "nav.": "navigation.",
    "thermal.": "guidance.thermal_limiter.",
    "shaping.": "guidance.command_shaping.",
}

# Tuple of scaffolding prefixes for callers that need an `str.startswith` guard.
SCAFFOLDING_PREFIXES: tuple[str, ...] = tuple(_GUIDANCE_PREFIX_SECTIONS)


def route_param_path(key: str, scheme: str) -> str:
    """Map a (possibly prefixed) GA param name to its TOML dot-path.

    lateral./exit./nav./thermal./shaping. route to their fixed sections;
    an unprefixed key routes to the scheme's own guidance.<scheme>.* block.
    """
    for prefix, section in _GUIDANCE_PREFIX_SECTIONS.items():
        if key.startswith(prefix):
            return section + key.removeprefix(prefix)
    return f"guidance.{scheme}.{key}"
