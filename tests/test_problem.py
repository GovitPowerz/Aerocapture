"""Tests for AerocaptureProblem pymoo integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.problem import AerocaptureProblem


def _make_specs() -> list[ParamSpec]:
    return [
        ParamSpec("tau", 2.0, 60.0, 30.0),
        ParamSpec("threshold", 0.5, 5.0, 2.0),
        ParamSpec("gain", 1e-8, 1e-5, 1e-6, log_scale=True),
    ]


class TestProblemShape:
    def test_n_var(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.n_var == 3

    def test_bounds_zero_one(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert np.all(p.xl == 0.0)
        assert np.all(p.xu == 1.0)

    def test_single_objective(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.n_obj == 1


class TestProblemEvaluation:
    def test_evaluate_returns_correct_shape(self) -> None:
        specs = _make_specs()
        p = AerocaptureProblem(
            param_specs=specs,
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={
                "dv_threshold": 1000.0,
                "g_load_limit": 15.0,
                "heat_flux_limit": 200.0,
                "heat_load_limit": 25000.0,
                "g_load_weight": 1000.0,
                "heat_flux_weight": 1000.0,
                "heat_load_weight": 1000.0,
            },
            scheme="equilibrium_glide",
        )
        X = np.random.default_rng(0).random((5, 3))
        out: dict[str, Any] = {}
        with patch.object(p, "_run_batch", return_value=np.array([100.0, 200.0, 150.0, 300.0, 50.0])):
            p._evaluate(X, out)
        assert out["F"].shape == (5, 1)

    def test_evaluate_values_are_finite(self) -> None:
        specs = _make_specs()
        p = AerocaptureProblem(
            param_specs=specs,
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={
                "dv_threshold": 1000.0,
                "g_load_limit": 15.0,
                "heat_flux_limit": 200.0,
                "heat_load_limit": 25000.0,
                "g_load_weight": 1000.0,
                "heat_flux_weight": 1000.0,
                "heat_load_weight": 1000.0,
            },
            scheme="equilibrium_glide",
        )
        X = np.random.default_rng(1).random((3, 3))
        out: dict[str, Any] = {}
        with patch.object(p, "_run_batch", return_value=np.array([100.0, 200.0, 150.0])):
            p._evaluate(X, out)
        assert np.all(np.isfinite(out["F"]))


class TestSeedUpdate:
    def test_update_seeds(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        assert p.seeds == [42]
        p.update_seeds([1, 2, 3])
        assert p.seeds == [1, 2, 3]


class TestBuildOverrides:
    def test_lateral_prefix(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"lateral.tau": 30.0}, mc_seed=42)
        assert overrides["guidance.lateral.tau"] == 30.0
        assert overrides["monte_carlo.seed"] == 42

    def test_nav_prefix(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"nav.density_filter_gain": 0.5})
        assert overrides["navigation.density_filter_gain"] == 0.5

    def test_exit_prefix(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"exit.exit_velocity_threshold": 4000.0})
        assert overrides["guidance.ftc.exit_velocity_threshold"] == 4000.0

    def test_thermal_prefix(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"thermal.heat_flux_activation": 0.8})
        assert overrides["guidance.thermal_limiter.heat_flux_activation"] == 0.8

    def test_shaping_prefix(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"shaping.max_bank_acceleration": 5.0})
        assert overrides["guidance.command_shaping.max_bank_acceleration"] == 5.0

    def test_unprefixed_goes_to_scheme(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"hdot_gain": 1.5})
        assert overrides["guidance.equilibrium_glide.hdot_gain"] == 1.5

    def test_n_sims_always_one(self) -> None:
        p = AerocaptureProblem(
            param_specs=_make_specs(),
            toml_path="dummy.toml",
            seeds=[42],
            cost_kwargs={},
            scheme="equilibrium_glide",
        )
        overrides = p._build_overrides({"hdot_gain": 1.5}, mc_seed=99)
        assert overrides["simulation.n_sims"] == 1
        assert overrides["monte_carlo.seed"] == 99
