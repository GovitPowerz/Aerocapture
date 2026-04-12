"""Structured per-generation training metrics logger.

Writes one JSON-lines file per training session. Each line is a complete
record of metrics for one generation. The in-memory buffer feeds the
LiveDisplay; the file is the source of truth for post-training reports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aerocapture.training.metrics import capture_rate, cost_stats, population_diversity

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


class TrainingLogger:
    """Collects per-generation metrics and writes them to a JSONL file."""

    def __init__(self, scheme: str, run: int, output_dir: Path, config_hash: str) -> None:
        self._scheme = scheme
        self._run = run
        self._config_hash = config_hash
        self._buffer: list[dict] = []
        self._best_cost = float("inf")

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        self._filepath = output_dir / f"run_{run:03d}_{timestamp}.jsonl"
        self._file = open(self._filepath, "a")  # noqa: SIM115

    def log_generation(
        self,
        generation: int,
        population: npt.NDArray[np.float64],
        costs: npt.NDArray[np.float64],
        best_individual: npt.NDArray[np.float64],
        decode_fn: Callable[[npt.NDArray[np.float64]], dict[str, float]] | None,
        weight_stats: dict[str, dict[str, float]] | None = None,
        mc_seed: int | None = None,
        pool_metrics: dict | None = None,
        gen_elapsed_s: float | None = None,
        gen_best_individual: npt.NDArray[np.float64] | None = None,
        validation: dict | None = None,
    ) -> None:
        """Log metrics for one generation."""
        stats = cost_stats(costs)
        # In adaptive-seed mode, use pool's per-seed capture rate (honest metric).
        # The default capture_rate(costs) is meaningless when costs are aggregated fitness.
        cap_rate = pool_metrics["capture_rate"] if pool_metrics is not None and "capture_rate" in pool_metrics else capture_rate(costs)
        diversity = population_diversity(population)

        gen_best = stats["best"]
        improved = gen_best < self._best_cost
        if improved:
            self._best_cost = gen_best

        best_params = decode_fn(best_individual) if decode_fn is not None else None
        gen_best_params = decode_fn(gen_best_individual) if decode_fn is not None and gen_best_individual is not None else None

        constraint_violation_rate = float(np.mean(costs > np.median(costs) * 2)) if len(costs) > 0 else 0.0

        record = {
            "generation": generation,
            "run": self._run,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "best_cost": gen_best,
            "mean_cost": stats["mean"],
            "worst_cost": stats["worst"],
            "median_cost": stats["median"],
            "std_cost": stats["std"],
            "capture_rate": cap_rate,
            "constraint_violation_rate": constraint_violation_rate,
            "population_diversity": diversity,
            "best_params": best_params,
            "gen_best_params": gen_best_params,
            "improvement": improved,
            "scheme": self._scheme,
            "config_hash": self._config_hash,
            "all_costs": costs.tolist(),
        }

        if weight_stats is not None:
            record["weight_stats"] = weight_stats

        if mc_seed is not None:
            record["mc_seed"] = mc_seed

        if pool_metrics is not None:
            record["pool_metrics"] = pool_metrics

        if gen_elapsed_s is not None:
            record["gen_elapsed_s"] = round(gen_elapsed_s, 3)

        if validation is not None:
            record["validation"] = validation

        self._buffer.append(record)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @property
    def buffer(self) -> list[dict]:
        """In-memory metrics buffer for LiveDisplay."""
        return self._buffer

    def close(self) -> None:
        """Close the JSONL file."""
        self._file.close()
