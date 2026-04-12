"""Integration test: verify TrainingLogger is called correctly by train.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tests.fixtures.factories import make_training_config


@pytest.fixture(autouse=True)
def _dummy_executable(tmp_path: Path) -> None:
    """Create a dummy executable so train()'s binary check passes."""
    exe = tmp_path / "dummy"
    exe.touch()
    exe.chmod(0o755)


class TestTrainLoggerIntegration:
    def test_logger_called_once_per_generation(self, tmp_path: Path) -> None:
        """Verify log_generation is called once per gen via pymoo stepping."""
        config = make_training_config("equilibrium_glide")
        config.optimizer.n_gen = 2
        config.optimizer.n_pop = 4
        config.save_dir = str(tmp_path)

        mock_logger_instance = MagicMock()
        mock_logger_instance.buffer = []
        MockLoggerClass = MagicMock(return_value=mock_logger_instance)

        # Mock the AerocaptureProblem._evaluate to avoid running Rust
        def mock_evaluate(self_prob, X, out, *args, **kwargs):  # type: ignore[no-untyped-def]
            out["F"] = np.random.default_rng(42).random((X.shape[0], 1)) * 1000

        with (
            patch("aerocapture.training.logger.TrainingLogger", MockLoggerClass),
            patch("aerocapture.training.problem.AerocaptureProblem._evaluate", mock_evaluate),
        ):
            from aerocapture.training.train import train

            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # log_generation should be called n_gen times
        assert mock_logger_instance.log_generation.call_count == 2
        # close should be called once
        assert mock_logger_instance.close.call_count == 1
