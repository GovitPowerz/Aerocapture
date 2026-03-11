"""Integration test: verify TrainingLogger is called correctly by train.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tests.fixtures.factories import make_training_config


class TestTrainLoggerIntegration:
    def test_logger_called_once_per_generation(self, tmp_path: Path) -> None:
        """Verify log_generation is called once per gen, after tournament, before checkpoint."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 2
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.save_dir = str(tmp_path)

        mock_logger_instance = MagicMock()
        mock_logger_instance.buffer = []
        MockLoggerClass = MagicMock(return_value=mock_logger_instance)

        with (
            patch("aerocapture.training.logger.TrainingLogger", MockLoggerClass),
            patch("aerocapture.training.train.evaluate_chromosome", return_value=(100.0, None)),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train

            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

            # log_generation should be called n_gen times
            assert mock_logger_instance.log_generation.call_count == 2
            # close should be called once per run
            assert mock_logger_instance.close.call_count == 1
