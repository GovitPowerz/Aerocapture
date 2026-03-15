"""Tests for graceful keyboard interrupt handling in train()."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aerocapture.training.config import TrainingConfig
from aerocapture.training.train import train


class TestKeyboardInterrupt:
    """Tests that Ctrl+C saves checkpoint and returns cleanly."""

    def test_interrupt_returns_interrupted_flag(self, tmp_path: Path) -> None:
        """train() returns interrupted=True on KeyboardInterrupt."""
        # Create a dummy executable so the fast-fail check passes.
        exe_path = tmp_path / "src" / "rust" / "target" / "release"
        exe_path.mkdir(parents=True)

        # Create the NN model directory so save_checkpoint can write the best model.
        nn_dir = tmp_path / "data" / "neural_network"
        nn_dir.mkdir(parents=True)
        dummy_exe = exe_path / "aerocapture"
        dummy_exe.write_text("#!/bin/sh\nexit 0\n")
        dummy_exe.chmod(dummy_exe.stat().st_mode | stat.S_IEXEC)

        cfg = TrainingConfig()
        cfg.ga.n_gen = 100
        cfg.ga.n_pop = 2
        cfg.ga.n_runs = 1
        cfg.save_dir = str(tmp_path / "training_output")

        call_count = 0

        def mock_evaluate(*args: object, **kwargs: object) -> tuple[float, None]:
            nonlocal call_count
            call_count += 1
            if call_count > 10:
                raise KeyboardInterrupt
            return 1e6 + call_count, None

        mock_pop = np.zeros((2, cfg.chrom_length), dtype=np.int8)
        mock_costs = np.array([1e6, 1e6 + 1])
        mock_offspring = np.ones((2, cfg.chrom_length), dtype=np.int8)

        with (
            patch("aerocapture.training.train.create_initial_population", return_value=(mock_pop, mock_costs)),
            patch("aerocapture.training.train.crossover_and_mutate", return_value=mock_offspring),
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=mock_evaluate),
        ):
            result = train(cfg, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        assert result["interrupted"] is True
        assert result["best_cost"] < float("inf")
