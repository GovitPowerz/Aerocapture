"""Tests for TrainingLogger -- JSONL metrics logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.logger import TrainingLogger


@pytest.fixture
def logger(tmp_path: Path) -> TrainingLogger:
    return TrainingLogger(scheme="equilibrium_glide", run=0, output_dir=tmp_path, config_hash="abc123")


def _make_population(n_pop: int = 10, n_params: int = 7) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n_pop, n_params))


def _make_costs(n_pop: int = 10) -> np.ndarray:
    return np.arange(1.0, n_pop + 1, dtype=np.float64) * 100


def _decode_fn(x: np.ndarray) -> dict[str, float]:
    return {"param_a": 1.0, "param_b": 2.0}


class TestTrainingLogger:
    def test_creates_jsonl_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        logger.close()
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        assert "run_000_" in jsonl_files[0].name

    def test_jsonl_record_fields(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
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
            "cost_outlier_rate",
            "best_params",
            "improvement",
            "scheme",
            "config_hash",
        }
        assert required_fields.issubset(record.keys())

    def test_cost_outlier_rate_is_twice_median_share(self, logger: TrainingLogger) -> None:
        # median 2.5: only 10.0 exceeds 5.0; 5.0 itself is not an outlier
        logger.log_generation(1, _make_population(n_pop=4), np.array([1.0, 2.0, 3.0, 10.0]), np.full(7, 0.5), _decode_fn)
        logger.log_generation(2, _make_population(n_pop=4), np.array([1.0, 2.0, 3.0, 5.0]), np.full(7, 0.5), _decode_fn)
        assert logger.buffer[0]["cost_outlier_rate"] == 0.25
        assert logger.buffer[1]["cost_outlier_rate"] == 0.0
        assert "constraint_violation_rate" not in logger.buffer[0]
        logger.close()

    def test_cost_outlier_rate_uses_the_finite_median(self, logger: TrainingLogger) -> None:
        # Failed batch evals and the resume prologue log inf costs: finite median 1.5, both inf exceed 3.0
        logger.log_generation(1, _make_population(n_pop=4), np.array([1.0, 2.0, np.inf, np.inf]), np.full(7, 0.5), _decode_fn)
        logger.log_generation(2, _make_population(n_pop=4), np.full(4, np.inf), np.full(7, 0.5), _decode_fn)
        assert logger.buffer[0]["cost_outlier_rate"] == 0.5
        assert logger.buffer[1]["cost_outlier_rate"] == 0.0
        logger.close()

    def test_multiple_generations_appended(self, logger: TrainingLogger, tmp_path: Path) -> None:
        for gen in range(1, 4):
            logger.log_generation(gen, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # Must be valid JSON

    def test_buffer_matches_file(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        file_record = json.loads(jsonl_file.read_text().strip())
        assert logger.buffer[0]["generation"] == file_record["generation"]
        assert logger.buffer[0]["best_cost"] == file_record["best_cost"]

    def test_improvement_tracking(self, logger: TrainingLogger) -> None:
        costs_improving = np.array([500.0, 600.0])
        costs_worse = np.array([700.0, 800.0])
        pop = np.random.default_rng(0).random((2, 7))
        logger.log_generation(1, pop, costs_improving, np.full(7, 0.5), _decode_fn)
        logger.log_generation(2, pop, costs_worse, np.full(7, 0.5), _decode_fn)
        assert logger.buffer[0]["improvement"] is True  # First gen always improves (from inf)
        assert logger.buffer[1]["improvement"] is False

    def test_none_decode_fn_for_nn(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), None)
        assert logger.buffer[0]["best_params"] is None

    def test_weight_stats_recorded(self, logger: TrainingLogger) -> None:
        stats = {
            "layer_0_w": {"min": -0.38, "max": 0.41, "mean": 0.01, "std": 0.22},
            "layer_0_b": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
        }
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, weight_stats=stats)
        assert logger.buffer[0]["weight_stats"] == stats
        logger.close()

    def test_weight_stats_none_by_default(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        assert "weight_stats" not in logger.buffer[0]
        logger.close()

    def test_mc_seed_recorded(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, mc_seed=99)
        assert logger.buffer[0]["mc_seed"] == 99
        logger.close()

    def test_mc_seed_absent_by_default(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        assert "mc_seed" not in logger.buffer[0]
        logger.close()

    def test_mc_seed_written_to_jsonl(self, logger: TrainingLogger, tmp_path: Path) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, mc_seed=77)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        record = json.loads(jsonl_file.read_text().strip())
        assert record["mc_seed"] == 77

    def test_validation_recorded(self, logger: TrainingLogger) -> None:
        val = {
            "mean_cost": 42.3,
            "median_cost": 38.1,
            "std_cost": 15.2,
            "p95_cost": 112.5,
            "worst_cost": 8432.0,
            "capture_rate": 0.97,
            "n_sims": 1000,
        }
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, validation=val)
        assert logger.buffer[0]["validation"] == val
        logger.close()

    def test_validation_payload_has_one_shape_for_both_adapters(self, logger: TrainingLogger) -> None:
        """`trainer._build_validation_payload` is the only emitter of the run log's
        validation sub-record: the islands path (logged with `island_name`) carries
        the same keys as the single-algo path (#221)."""
        from aerocapture.training.trainer import _build_validation_payload

        costs = np.array([10.0, 20.0, 30.0, 40.0])
        metrics, summary = _build_validation_payload(costs, None, 4, {"cost_transform": "linear"}, feasible=True, violation_rates={"heat_load": 0.0})
        assert summary is None  # no final records
        expected_keys = ["rms_cost", "mean_cost", "median_cost", "std_cost", "p95_cost", "worst_cost", "capture_rate", "n_sims", "feasible", "violation_rates"]
        assert list(metrics) == expected_keys
        assert metrics["rms_cost"] == float(np.sqrt(np.mean(costs**2))) and metrics["n_sims"] == 4 and metrics["feasible"] is True

        # The islands adapter's own emit, not a hand-fed payload.
        import types

        from aerocapture.training.champion import Champion
        from aerocapture.training.island_model import Island
        from aerocapture.training.trainer import IslandsTrainer

        X, F = _make_population(), _make_costs().reshape(-1, 1)
        island = Island(name="ga", algorithm=types.SimpleNamespace(pop=types.SimpleNamespace(get={"X": X, "F": F}.__getitem__)))
        island.champion = Champion(individual=np.full(7, 0.5), val_rms=30.0)
        t = IslandsTrainer.__new__(IslandsTrainer)  # bypass __init__ (pymoo setup)
        model = types.SimpleNamespace(islands=[island], migration_log=[], latest_migration_summary=None, latest_migration_gen=None, origin_stats_cache=None)
        t.island_model = model  # type: ignore[assignment]
        t._val_records = [
            {"island": "ga", "validated": True, "promoted": False, "feasible": True, "violation_rates": {"heat_load": 0.0}, "argmin_train_cost": 1.0}
            | {"val_rms": metrics["rms_cost"], "val_costs": costs, "val_records": None, "stagnation": 1}
        ]
        t.val_seeds = [1, 2, 3, 4]
        t.problem = types.SimpleNamespace(cost_kwargs={"cost_transform": "linear"})  # type: ignore[assignment]
        t.decode_fn = _decode_fn
        t._events = []
        t.config = types.SimpleNamespace(optimizer=types.SimpleNamespace(n_gen=10))  # type: ignore[assignment]
        t.verbose = False
        t.emit(1, logger, types.SimpleNamespace(update=lambda *a, **kw: None, is_live=True))  # type: ignore[arg-type]
        (record,) = logger.buffer
        assert record["island_name"] == "ga"
        assert record["validation"] == metrics
        logger.close()

    def test_validation_absent_by_default(self, logger: TrainingLogger) -> None:
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn)
        assert "validation" not in logger.buffer[0]
        logger.close()

    def test_validation_written_to_jsonl(self, logger: TrainingLogger, tmp_path: Path) -> None:
        val = {"mean_cost": 42.3, "capture_rate": 0.97, "n_sims": 1000}
        logger.log_generation(1, _make_population(), _make_costs(), np.full(7, 0.5), _decode_fn, validation=val)
        logger.close()
        jsonl_file = list(tmp_path.glob("*.jsonl"))[0]
        record = json.loads(jsonl_file.read_text().strip())
        assert record["validation"]["mean_cost"] == 42.3
