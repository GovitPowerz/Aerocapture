"""Tests for aerocapture.training.sensitivity module."""
from __future__ import annotations

import pytest


class TestBuildProblem:
    def test_build_problem_returns_salib_dict(self) -> None:
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS, build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "medium"},
            "atmosphere": {"level": "medium"},
            "aerodynamics": {"level": "medium"},
            "navigation": {"level": "medium"},
            "mass": {"level": "medium"},
            "vehicle": {"level": "medium"},
            "pilot": {"level": "medium"},
            "nav_filter": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        assert problem["num_vars"] == 26
        assert problem["names"] == DISPERSION_COLUMNS
        assert len(problem["bounds"]) == 26
        assert len(problem["dists"]) == 26

    def test_build_problem_distribution_types(self) -> None:
        from aerocapture.training.sensitivity import build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "medium"},
            "atmosphere": {"level": "medium"},
            "aerodynamics": {"level": "medium"},
            "navigation": {"level": "medium"},
            "mass": {"level": "medium"},
            "vehicle": {"level": "medium"},
            "pilot": {"level": "medium"},
            "nav_filter": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        dists = problem["dists"]
        assert isinstance(dists, list)
        # Initial state (0-5) = Gaussian
        assert dists[0] == "norm"
        assert dists[5] == "norm"
        # Atmosphere (6) = Uniform
        assert dists[6] == "unif"
        # Aerodynamics (7-9) = Uniform
        assert dists[7] == "unif"
        assert dists[9] == "unif"
        # Navigation (10-16) = Gaussian
        assert dists[10] == "norm"
        assert dists[16] == "norm"
        # Mass (17) = Uniform
        assert dists[17] == "unif"
        # Vehicle (18-19) = Uniform
        assert dists[18] == "unif"
        assert dists[19] == "unif"
        # Pilot (20-22) = Uniform
        assert dists[20] == "unif"
        assert dists[22] == "unif"
        # Nav filter (23) = Gaussian
        assert dists[23] == "norm"
        # Wind (24-25) = Uniform
        assert dists[24] == "unif"
        assert dists[25] == "unif"

    def test_dispersion_columns_length(self) -> None:
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS

        assert len(DISPERSION_COLUMNS) == 26

    def test_build_problem_off_level_has_zero_bounds(self) -> None:
        from aerocapture.training.sensitivity import build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "off"},
            "atmosphere": {"level": "off"},
            "aerodynamics": {"level": "off"},
            "navigation": {"level": "off"},
            "mass": {"level": "off"},
            "vehicle": {"level": "off"},
            "pilot": {"level": "off"},
            "nav_filter": {"level": "off"},
        }
        problem = build_problem(mc_config)
        bounds = problem["bounds"]
        assert isinstance(bounds, list)
        # All Gaussian bounds should have sigma=0
        assert bounds[0] == [0.0, 0.0]  # altitude
        assert bounds[3] == [0.0, 0.0]  # velocity
        assert bounds[10] == [0.0, 0.0]  # nav_altitude
        assert bounds[23] == [0.0, 0.0]  # filter_gain
        # All Uniform bounds should be [0, 0]
        assert bounds[6] == [0.0, 0.0]   # density
        assert bounds[7] == [0.0, 0.0]   # drag_coeff
        assert bounds[17] == [0.0, 0.0]  # mass

    def test_build_problem_medium_level_nonzero_bounds(self) -> None:
        import math
        from aerocapture.training.sensitivity import build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "medium"},
            "atmosphere": {"level": "medium"},
            "aerodynamics": {"level": "medium"},
            "navigation": {"level": "medium"},
            "mass": {"level": "medium"},
            "vehicle": {"level": "medium"},
            "pilot": {"level": "medium"},
            "nav_filter": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        bounds = problem["bounds"]
        assert isinstance(bounds, list)
        # altitude sigma = 0.1 km = 100 m
        assert bounds[0] == [0.0, 100.0]
        # velocity sigma = 1.0 m/s
        assert bounds[3] == [0.0, 1.0]
        # atmosphere density hw = 50% -> 0.5
        assert bounds[6] == [-0.5, 0.5]
        # drag hw = 5% -> 0.05
        assert bounds[7] == pytest.approx([-0.05, 0.05])
        # nav_altitude sigma = 0.667 km = 667.0 m
        assert bounds[10] == pytest.approx([0.0, 667.0])
        # nav_drag_accel sigma = 0.1 m/s²
        assert bounds[16] == [0.0, 0.1]
        # mass hw = 1% -> 0.01
        assert bounds[17] == [-0.01, 0.01]
        # filter_gain sigma = 0.10
        assert bounds[23] == [0.0, 0.10]

    def test_build_problem_wind_absent_zero_width(self) -> None:
        from aerocapture.training.sensitivity import build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "off"},
            "atmosphere": {"level": "off"},
            "aerodynamics": {"level": "off"},
            "navigation": {"level": "off"},
            "mass": {"level": "off"},
            "vehicle": {"level": "off"},
            "pilot": {"level": "off"},
            "nav_filter": {"level": "off"},
            # no "wind" key
        }
        problem = build_problem(mc_config)
        bounds = problem["bounds"]
        assert isinstance(bounds, list)
        # wind absent -> both dimensions zero-width
        assert bounds[24] == [0.0, 0.0]
        assert bounds[25] == [0.0, 0.0]

    def test_build_problem_wind_level_medium(self) -> None:
        import math
        from aerocapture.training.sensitivity import build_problem

        mc_config: dict[str, object] = {
            "seed": 42,
            "initial_state": {"level": "off"},
            "atmosphere": {"level": "off"},
            "aerodynamics": {"level": "off"},
            "navigation": {"level": "off"},
            "mass": {"level": "off"},
            "vehicle": {"level": "off"},
            "pilot": {"level": "off"},
            "nav_filter": {"level": "off"},
            "wind": {"level": "medium"},
        }
        problem = build_problem(mc_config)
        bounds = problem["bounds"]
        assert isinstance(bounds, list)
        # medium: scale [0.5, 1.5], direction_bias 10 deg -> rad
        assert bounds[24] == pytest.approx([0.5, 1.5])
        expected_dir_hw = 10.0 * math.pi / 180.0
        assert bounds[25] == pytest.approx([-expected_dir_hw, expected_dir_hw])

    def test_build_problem_missing_domain_defaults_to_off(self) -> None:
        from aerocapture.training.sensitivity import build_problem

        # Minimal config -- missing all optional domains
        mc_config: dict[str, object] = {"seed": 42}
        problem = build_problem(mc_config)
        # Should not raise, all dims should be zero-width
        assert problem["num_vars"] == 26
        bounds = problem["bounds"]
        assert isinstance(bounds, list)
        assert bounds[0] == [0.0, 0.0]   # altitude
        assert bounds[6] == [0.0, 0.0]   # density
        assert bounds[23] == [0.0, 0.0]  # filter_gain
