"""Integration test: the loop calls TrainingLogger once per generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aerocapture.training.param_spaces import PARAM_SPACES

from tests.fixtures.factories import make_training_config
from tests.fixtures.fake_problem import FakeProblem, run_trainer


class TestTrainLoggerIntegration:
    def test_logger_called_once_per_generation(self, tmp_path: Path) -> None:
        """Verify log_generation is called once per gen via pymoo stepping."""
        config = make_training_config("equilibrium_glide")
        config.optimizer.n_gen = 2
        config.optimizer.n_pop = 4
        config.save_dir = str(tmp_path)

        mock_logger = MagicMock()
        mock_logger.buffer = []
        # toml_path "" = no validation pool (no prologue record), as train() without a TOML.
        problem = FakeProblem(list(PARAM_SPACES["equilibrium_glide"]), seeds=[42], toml_path="")

        run_trainer(config, problem, cwd=str(tmp_path), logger=mock_logger)

        # log_generation should be called n_gen times
        assert mock_logger.log_generation.call_count == 2
        # close should be called once
        assert mock_logger.close.call_count == 1
