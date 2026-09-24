"""A pymoo Problem implementing `problem.PerSeedEvaluator` without the simulator.

Drives both trainer adapters through the loop contract in the fast suite: the
per-seed cost is an analytic function of the chromosome and the seed, the
records are captured rows (ifinal = 3) so the validation payload, the
feasibility check and the eval summary all run unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from aerocapture.training.config import TrainingConfig
from aerocapture.training.display import NoopDisplay
from aerocapture.training.logger import TrainingLogger
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.parquet_output import FINAL_RECORD_LEN
from aerocapture.training.problem import training_rms
from aerocapture.training.trainer import IslandsTrainer, SingleAlgoTrainer, Trainer, run_loop
from pymoo.core.problem import Problem


class FakeProblem(Problem):
    """Per-seed cost = sum((x - 0.3)^2) * (1 + seed / 1e4), or `constant_cost`
    everywhere. `interrupt_after` raises KeyboardInterrupt on the first per-seed
    evaluation past that count (an injected Ctrl+C)."""

    def __init__(
        self,
        param_specs: list[ParamSpec],
        *,
        seeds: list[int],
        toml_path: str = "fake.toml",
        interrupt_after: int | None = None,
        constant_cost: float | None = None,
    ) -> None:
        super().__init__(n_var=len(param_specs), n_obj=1, xl=0.0, xu=1.0)
        self.param_specs = param_specs
        self.toml_path = toml_path
        self.seeds = list(seeds)
        self.cost_kwargs: dict[str, Any] = {}
        self.n_evals = 0
        self.interrupt_after = interrupt_after
        self.constant_cost = constant_cost

    def update_seeds(self, seeds: list[int]) -> None:
        self.seeds = list(seeds)

    def _evaluate(self, X: npt.NDArray[np.float64], out: dict, *args: Any, **kwargs: Any) -> None:
        out["F"] = training_rms(self, X).reshape(-1, 1)

    def evaluate_population_per_seed(self, X: npt.NDArray[np.float64], seeds: list[int]) -> npt.NDArray[np.float64]:
        self.n_evals += 1
        if self.interrupt_after is not None and self.n_evals > self.interrupt_after:
            raise KeyboardInterrupt
        if self.constant_cost is not None:
            flat: npt.NDArray[np.float64] = np.full((X.shape[0], len(seeds)), self.constant_cost, dtype=np.float64)
            return flat
        base = np.sum((np.asarray(X, dtype=np.float64) - 0.3) ** 2, axis=1)
        scale = 1.0 + np.asarray(seeds, dtype=np.float64) / 1e4
        costs: npt.NDArray[np.float64] = base[:, None] * scale[None, :]
        return costs

    def evaluate_population_records_per_seed(self, X: npt.NDArray[np.float64], seeds: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        costs = self.evaluate_population_per_seed(X, seeds)
        records = np.zeros((X.shape[0], len(seeds), FINAL_RECORD_LEN), dtype=np.float64)
        records[..., 31] = 3.0  # ifinal: captured
        records[..., 9] = 0.5  # eccentricity
        records[..., 41] = 100.0  # DV total
        return costs, records

    def evaluate_individual_per_seed(self, x: npt.NDArray[np.float64], seeds: list[int]) -> npt.NDArray[np.float64]:
        row: npt.NDArray[np.float64] = self.evaluate_population_per_seed(x.reshape(1, -1), seeds)[0]
        return row

    def evaluate_individual_records_per_seed(self, x: npt.NDArray[np.float64], seeds: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        costs, records = self.evaluate_population_records_per_seed(x.reshape(1, -1), seeds)
        return costs[0], records[0]


def run_trainer(
    cfg: TrainingConfig,
    problem: FakeProblem,
    *,
    seed: int = 42,
    cwd: str | None = None,
    resume_dir: Path | None = None,
    verbose: bool = False,
    checkpoint_interval: int = 10,
    logger: Any = None,
) -> tuple[Trainer, dict]:
    """`train()` minus config resolution: build the adapter `cfg.optimizer.algorithm`
    selects from the config + fake problem and drive `run_loop` headless."""
    cls = IslandsTrainer if cfg.optimizer.algorithm == "islands" else SingleAlgoTrainer
    rng = np.random.default_rng(seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = cls.from_config(
        cfg,
        problem,
        save_dir,
        toml={},
        cwd=cwd,
        rng=rng,
        resume_dir=resume_dir,
        from_scratch=False,
        corridor_acc=None,
        verbose=verbose,
        checkpoint_interval=checkpoint_interval,
    )
    if logger is None:
        logger = TrainingLogger(scheme=cfg.guidance_type, run=0, output_dir=save_dir, config_hash="fake")
    return trainer, run_loop(trainer, config=cfg, logger=logger, display=NoopDisplay())
