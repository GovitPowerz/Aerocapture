"""Quantum-behaved PSO (QPSO) -- canonical mbest form (Sun, Feng & Xu 2004).

Particles carry no velocity. Each generation, every position is resampled
from a delta-potential-well distribution centered on a per-particle local
attractor (a random convex mix of pbest and gbest), with characteristic
length alpha * |mbest - x| where mbest is the swarm's mean pbest. The
contraction-expansion coefficient alpha anneals linearly from alpha_start
to alpha_end over max_iter generations (theory bounds convergence at
alpha < e^gamma ~ 1.781).

State conventions mirror pymoo's PSO so train.py's warm_start_algorithm,
checkpointing (pop = pbest), and the manual .next() loop work untouched:
- self.pop is the personal-best population (checkpointed, read by _set_optimum)
- self.particles is the current swarm position population

Resume note: single-algo resume restarts n_iter at 1 while max_iter is the
bumped total (resumed + additional gens), so alpha restarts at alpha_start
on the stretched schedule -- same family of state reset as PSO's velocity
reinit on resume. Paper runs are single-shot --from-scratch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.replacement import ImprovementReplacement
from pymoo.operators.crossover.dex import repair_random_init
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput


class QPSO(Algorithm):
    def __init__(
        self,
        pop_size: int = 25,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        max_iter: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(output=SingleObjectiveOutput(), **kwargs)
        self.initialization = Initialization(LHS())
        self.pop_size = pop_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_iter = max_iter
        self.particles: Population | None = None

    def _alpha(self) -> float:
        progress: float = float(self.n_iter - 1) / max(1, int(self.max_iter) - 1)
        progress = min(1.0, max(0.0, progress))
        return float(self.alpha_start) + (float(self.alpha_end) - float(self.alpha_start)) * progress

    def _initialize_infill(self) -> Population:
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        self.particles = self.pop
        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self) -> Population:
        assert self.particles is not None, "particles not initialized"
        X = self.particles.get("X")
        P = self.pop.get("X")
        G = self.opt[0].X

        mbest = P.mean(axis=0)
        rs = self.random_state
        phi = rs.random(X.shape)
        attractor = phi * P + (1.0 - phi) * G[None, :]
        u = 1.0 - rs.random(X.shape)  # (0, 1]: keeps log(1/u) finite
        sign = np.where(rs.random(X.shape) < 0.5, 1.0, -1.0)
        Xp = attractor + sign * self._alpha() * np.abs(mbest[None, :] - X) * np.log(1.0 / u)

        if self.problem.has_bounds():
            Xp = repair_random_init(Xp, X, *self.problem.bounds(), random_state=rs)

        return Population.new(X=Xp)

    def _advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        assert infills is not None, "QPSO uses the ask-and-tell interface; 'infills' must be provided."
        self.particles = infills
        has_improved = ImprovementReplacement().do(self.problem, self.pop, infills, return_indices=True)
        self.pop[has_improved] = infills[has_improved]
