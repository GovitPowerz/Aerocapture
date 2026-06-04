"""Tests for AerocaptureProblem pymoo integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.problem import AerocaptureProblem


def _make_minimal_problem() -> AerocaptureProblem:
    return AerocaptureProblem(
        param_specs=[ParamSpec("tau", 2.0, 60.0, 30.0)],
        toml_path="dummy.toml",
        seeds=[42],
        cost_kwargs={},
        scheme="ftc",
    )


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


def test_problem_n_nn_weight_specs_live_pack() -> None:
    """A live-scaffolding NN problem caps weights at len(param_specs) - 3."""
    from aerocapture.training.config import NetworkConfig

    arch = [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}]
    net = NetworkConfig(architecture=arch, scaffolding="live")
    # 3 placeholder weight specs + 3 live scaffolding specs
    specs = [ParamSpec(f"w{i}", -1.0, 1.0, 0.0) for i in range(3)] + [
        ParamSpec("nav.density_filter_gain", 0.3, 1.0, 0.8),
        ParamSpec("nav.density_gain_max_delta", 0.01, 0.5, 0.1),
        ParamSpec("shaping.max_bank_acceleration", 2.0, 15.0, 5.0),
    ]
    prob = AerocaptureProblem(
        param_specs=specs,
        toml_path="x.toml",
        seeds=[0],
        cost_kwargs={},
        scheme="neural_network",
        nn_config=net,
    )
    assert prob._n_nn_weight_specs == 3


def test_problem_n_nn_weight_specs_off_and_full() -> None:
    """off subtracts 0; full subtracts 17."""
    from aerocapture.training.config import NetworkConfig
    from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS

    arch = [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}]
    base_specs = [ParamSpec(f"w{i}", -1.0, 1.0, 0.0) for i in range(5)]

    net_off = NetworkConfig(architecture=arch, scaffolding="off")
    prob_off = AerocaptureProblem(param_specs=base_specs, toml_path="x.toml", seeds=[0], cost_kwargs={}, scheme="neural_network", nn_config=net_off)
    assert prob_off._n_nn_weight_specs == 5

    net_full = NetworkConfig(architecture=arch, scaffolding="full")
    full_specs = [*base_specs, *_NN_SCAFFOLDING_PARAMS]
    prob_full = AerocaptureProblem(param_specs=full_specs, toml_path="x.toml", seeds=[0], cost_kwargs={}, scheme="neural_network", nn_config=net_full)
    assert prob_full._n_nn_weight_specs == len(base_specs)


def test_evaluate_aborts_after_consecutive_failures(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A persistent batch-eval failure must raise after threshold, not silently return 1e9 forever (D6)."""
    import pytest
    from aerocapture.training import problem as problem_mod

    prob = _make_minimal_problem()

    def always_fail(self: AerocaptureProblem, X: object) -> object:
        raise RuntimeError("simulated systemic break")

    monkeypatch.setattr(problem_mod.AerocaptureProblem, "_run_batch", always_fail)

    X = np.zeros((4, prob.n_var), dtype=np.float64)
    out: dict = {}
    for _ in range(problem_mod._MAX_CONSECUTIVE_EVAL_FAILURES - 1):
        prob._evaluate(X, out)
        assert np.all(out["F"] == 1e9)
    with pytest.raises(RuntimeError, match="consecutive"):
        prob._evaluate(X, out)


def test_evaluate_resets_failure_counter_on_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Counter resets to zero after a successful batch; transient failure is tolerated."""
    from aerocapture.training import problem as problem_mod

    prob = _make_minimal_problem()
    calls: dict[str, int] = {"n": 0}

    def flaky(self: AerocaptureProblem, X: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return np.full(4, 5.0)

    monkeypatch.setattr(problem_mod.AerocaptureProblem, "_run_batch", flaky)
    X = np.zeros((4, prob.n_var), dtype=np.float64)
    out: dict = {}
    prob._evaluate(X, out)  # transient failure -> 1e9, counter = 1
    prob._evaluate(X, out)  # success -> 5.0, counter reset to 0
    assert np.all(out["F"] == 5.0)
    assert prob._consecutive_eval_failures == 0
