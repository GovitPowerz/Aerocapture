"""Tests for TrainingLogger — JSONL metrics logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.logger import TrainingLogger


@pytest.fixture
def logger(tmp_path: Path) -> TrainingLogger:
    return TrainingLogger(scheme="equilibrium_glide", run=0, output_dir=tmp_path, config_hash="abc123")


def _make_populations(n_pop: int = 10, chrom_len: int = 112) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 2, size=(n_pop, chrom_len), dtype=np.int8)]


def _make_costs(n_pop: int = 10) -> list[np.ndarray]:
    return [np.arange(1.0, n_pop + 1, dtype=np.float64) * 100]


def _decode_fn(chrom: np.ndarray) -> dict[str, float]:
    return {"param_a": 1.0, "param_b": 2.0}


class TestTrainingLogger:
    def test_creates_jsonl_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        assert "run_000_" in jsonl_files[0].name

    def test_jsonl_record_fields(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        record = logger.buffer[0]
        required_fields = {
            "generation",
            "run",
            "timestamp",
            "best_cost",
            "mean_cost",
            "worst_cost",
            "median_cost",
            "std_cost",
            "capture_rate",
            "population_diversity",
            "best_params",
            "improvement",
            "scheme",
            "config_hash",
        }
        assert required_fields.issubset(record.keys())

    def test_multiple_generations_appended(self, logger: TrainingLogger, tmp_path: Path) -> None:
        for gen in range(1, 4):
            logger.log_generation(gen, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # Must be valid JSON

    def test_buffer_matches_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        file_record = json.loads(jsonl_file.read_text().strip())
        assert logger.buffer[0]["generation"] == file_record["generation"]
        assert logger.buffer[0]["best_cost"] == file_record["best_cost"]

    def test_improvement_tracking(self, logger: TrainingLogger) -> None:
        costs_improving = [np.array([500.0, 600.0])]
        costs_worse = [np.array([700.0, 800.0])]
        pop = [np.zeros((2, 112), dtype=np.int8)]
        logger.log_generation(1, pop, costs_improving, np.zeros(112, dtype=np.int8), _decode_fn)
        logger.log_generation(2, pop, costs_worse, np.zeros(112, dtype=np.int8), _decode_fn)
        assert logger.buffer[0]["improvement"] is True  # First gen always improves (from inf)
        assert logger.buffer[1]["improvement"] is False

    def test_none_decode_fn_for_nn(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_populations(), _make_costs(), np.zeros(112, dtype=np.int8), None)
        assert logger.buffer[0]["best_params"] is None

    def test_multi_subpop_concatenation(self, logger: TrainingLogger) -> None:
        pops = _make_populations(5) + _make_populations(5)
        costs = _make_costs(5) + _make_costs(5)
        logger.log_generation(1, pops, costs, np.zeros(112, dtype=np.int8), _decode_fn)
        assert 0.0 <= logger.buffer[0]["population_diversity"] <= 1.0
