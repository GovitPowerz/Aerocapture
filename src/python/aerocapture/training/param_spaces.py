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
    ParamSpec("lateral.corridor_slope", 5000.0, 20000.0, 13080.458),
    ParamSpec("lateral.corridor_intercept", 0.0, 0.1, 0.0),
    ParamSpec("lateral.lateral_activation", -5.0, -0.5, -2.5),
    ParamSpec("lateral.lateral_inhibition", -10.0, -2.0, -8.0),
    ParamSpec("lateral.max_reversals", 1.0, 10.0, 5.0),
]

# Thermal safety limiter params shared by all unsigned-magnitude schemes.
# Prefixed with "thermal." so evaluate.py routes them to [guidance.thermal_limiter] in TOML.
_THERMAL_LIMITER_PARAMS: list[ParamSpec] = [
    ParamSpec("thermal.heat_flux_activation", 0.6, 1.0, 1.0),
    ParamSpec("thermal.heat_load_activation", 0.6, 1.0, 1.0),
    ParamSpec("thermal.heat_flux_ramp_exponent", 0.5, 3.0, 1.0),
    ParamSpec("thermal.heat_load_ramp_exponent", 0.5, 3.0, 1.0),
]

# Navigation density filter params shared by all unsigned-magnitude schemes.
# Prefixed with "nav." so evaluate.py routes them to [guidance.ftc] in TOML
# (density filter config lives in the FTC TOML section but affects all schemes).
_NAV_PARAMS: list[ParamSpec] = [
    ParamSpec("nav.density_filter_gain", 0.3, 1.0, 0.8),
    ParamSpec("nav.density_gain_max_delta", 0.01, 0.5, 0.1),
]

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
    ],
    "energy_controller": [
        ParamSpec("gain", 1e-8, 1e-5, 5e-7, log_scale=True),
        ParamSpec("kp", 0.1, 5.0, 1.0),
        ParamSpec("kd", 0.0, 3.0, 0.5),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
    ],
    "pred_guid": [
        ParamSpec("k_drag_high", 0.1, 3.0, 0.8),
        ParamSpec("k_drag_low", 0.05, 2.0, 0.3),
        ParamSpec("pdyn_threshold", 10.0, 500.0, 100.0, log_scale=True),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
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
    ],
    "ftc": [
        ParamSpec("capture_damping", 0.3, 1.5, 0.7),
        ParamSpec("capture_frequency", 0.01, 0.2, 0.072),
        ParamSpec("altitude_damping", 0.3, 1.5, 0.7),
        ParamSpec("altitude_frequency", 0.01, 0.2, 0.08),
        ParamSpec("capture_pdyn_margin", 1.0, 3.0, 1.75),
        ParamSpec("pressure_coeff_base", -0.01, -0.0001, -0.001),
        ParamSpec("pressure_coeff_scale_height", 5.0, 20.0, 10.0),
        ParamSpec("gain_fade_start_km", 60.0, 90.0, 80.0),
        ParamSpec("gain_fade_end_km", 85.0, 120.0, 100.0),
        *_NAV_PARAMS,
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
    ],
    "piecewise_constant": [
        ParamSpec("bank_angle_0", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_1", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_2", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_3", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_4", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_5", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_6", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_7", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_8", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_9", -180.0, 180.0, 65.0),
    ],
}

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
