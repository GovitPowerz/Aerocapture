"""Behavioral tests for the QPSO optimizer (pure Python, no Rust dependency).

QPSO mirrors pymoo PSO's state conventions (pop = pbest, particles = current
positions), so these tests drive it exactly like train.py does:
setup(problem, seed=...) then repeated .next().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from aerocapture.training.qpso import QPSO
from aerocapture.training.train import warm_start_algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class _Sphere(Problem):
    """Shifted sphere: f(x) = sum((x - 0.3)^2), bounds [0, 1]."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        out["F"] = ((X - 0.3) ** 2).sum(axis=1).reshape(-1, 1)


def _run(n_gen: int, seed: int, pop_size: int = 20, n_var: int = 10) -> QPSO:
    algo = QPSO(pop_size=pop_size, max_iter=n_gen)
    algo.setup(_Sphere(n_var=n_var), seed=seed)
    for _ in range(n_gen):
        algo.next()
    return algo


class TestAlphaSchedule:
    def test_first_iter_is_alpha_start(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 1
        assert algo._alpha() == pytest.approx(1.0)

    def test_last_iter_is_alpha_end(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 101
        assert algo._alpha() == pytest.approx(0.5)

    def test_midpoint(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 51
        assert algo._alpha() == pytest.approx(0.75)

    def test_past_max_iter_clamps_to_alpha_end(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=101)
        algo.n_iter = 5000
        assert algo._alpha() == pytest.approx(0.5)

    def test_max_iter_1_no_division_by_zero(self) -> None:
        algo = QPSO(alpha_start=1.0, alpha_end=0.5, max_iter=1)
        algo.n_iter = 1
        assert algo._alpha() == pytest.approx(1.0)


class TestSwarmBehavior:
    def test_deterministic_under_seed(self) -> None:
        a = _run(n_gen=10, seed=7)
        b = _run(n_gen=10, seed=7)
        assert np.array_equal(a.opt[0].X, b.opt[0].X)
        assert float(a.opt[0].F[0]) == float(b.opt[0].F[0])

    def test_different_seeds_differ(self) -> None:
        a = _run(n_gen=10, seed=7)
        b = _run(n_gen=10, seed=8)
        assert not np.array_equal(a.opt[0].X, b.opt[0].X)

    def test_positions_respect_bounds_every_generation(self) -> None:
        algo = QPSO(pop_size=20, max_iter=20)
        algo.setup(_Sphere(), seed=3)
        for _ in range(20):
            algo.next()
            assert algo.particles is not None
            X = algo.particles.get("X")
            assert (X >= 0.0).all() and (X <= 1.0).all()

    def test_pbest_monotonically_non_increasing(self) -> None:
        algo = QPSO(pop_size=20, max_iter=30)
        algo.setup(_Sphere(), seed=5)
        algo.next()
        prev_F = algo.pop.get("F").copy()
        for _ in range(29):
            algo.next()
            F = algo.pop.get("F")
            assert (prev_F + 1e-15 >= F).all()
            prev_F = F.copy()

    def test_sphere_convergence(self) -> None:
        algo = QPSO(pop_size=20, max_iter=60)
        algo.setup(_Sphere(n_var=10), seed=42)
        algo.next()
        f_init = float(algo.opt[0].F[0])
        for _ in range(59):
            algo.next()
        f_final = float(algo.opt[0].F[0])
        assert f_final < f_init / 50.0


class TestWarmStartCompat:
    def test_seeded_chromosomes_survive_gen0(self) -> None:
        """The invariant warm_start_algorithm exists to protect: a seeded,
        pre-evaluated population must become the pbest baseline (not get
        wiped by pymoo's _initialize() + LHS resample on the first next())."""
        problem = _Sphere(n_var=4)
        rng = np.random.default_rng(0)
        X0 = rng.random((10, 4))
        pop = Population.new("X", X0)
        Evaluator().eval(problem, pop)
        F0 = pop.get("F").copy()

        algo = QPSO(pop_size=10, max_iter=50)
        warm_start_algorithm(algo, problem, pop, seed=1)

        # _initialize_advance hook contract: particles start at the seeded pop.
        assert algo.particles is not None
        assert np.array_equal(algo.particles.get("X"), X0)

        algo.next()
        # pbest baseline is the seeded pop: per-index F can only improve.
        assert (algo.pop.get("F") <= F0 + 1e-15).all()
