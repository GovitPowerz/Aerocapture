"""Regression: single-algo training with validation_n_sims=0 must promote a
later generation's best, not freeze the gen-0 argmin (defect D1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.param_spaces import PARAM_SPACES

from tests.fixtures.fake_problem import FakeProblem, run_trainer

GEN0_MIN = 1000.0


class _ImprovingProblem(FakeProblem):
    """Every batch evaluation is 100 cheaper than the previous one."""

    def evaluate_population_per_seed(self, X: npt.NDArray[np.float64], seeds: list[int]) -> npt.NDArray[np.float64]:
        self.n_evals += 1
        base = max(GEN0_MIN - 100.0 * (self.n_evals - 1), 10.0)
        return (base + np.arange(X.shape[0], dtype=np.float64))[:, None] * np.ones((1, len(seeds)))


def test_no_validation_promotes_later_generation(tmp_path: Path) -> None:
    cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"), guidance_type="equilibrium_glide")
    cfg.optimizer.n_gen = 4
    cfg.optimizer.n_pop = 4
    cfg.optimizer.validation_n_sims = 0  # validation gate OFF -> exercises the D1 path
    cfg.save_dir = str(tmp_path / "training_output")

    problem = _ImprovingProblem(list(PARAM_SPACES[cfg.guidance_type]), seeds=[42])
    _, result = run_trainer(cfg, problem, cwd=str(tmp_path))

    assert result["interrupted"] is False
    assert result["best_cost"] < GEN0_MIN - 100.0
