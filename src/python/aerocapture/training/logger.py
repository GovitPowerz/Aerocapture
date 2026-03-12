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
        populations: list[npt.NDArray[np.int8]],
        costs: list[npt.NDArray[np.float64]],
        best_chromosome: npt.NDArray[np.int8],
        decode_fn: Callable[[npt.NDArray[np.int8]], dict[str, float]] | None,
        weight_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Log metrics for one generation."""
        all_chroms = np.vstack(populations)
        all_costs = np.concatenate(costs)

        stats = cost_stats(all_costs)
        cap_rate = capture_rate(all_costs)
        diversity = population_diversity(all_chroms)

        gen_best = stats["best"]
        improved = gen_best < self._best_cost
        if improved:
            self._best_cost = gen_best

        best_params = decode_fn(best_chromosome) if decode_fn is not None else None

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
            "population_diversity": diversity,
            "best_params": best_params,
            "improvement": improved,
            "scheme": self._scheme,
            "config_hash": self._config_hash,
        }

        if weight_stats is not None:
            record["weight_stats"] = weight_stats

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
