"""Tests for graceful keyboard interrupt handling in train()."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# train() checkpoints via write_nn_json, which (post Phase 1 Task 9) routes NN
# chromosome serialization through aerocapture_rs.flat_weights_to_json -- the
# Rust-side LayerWeights trait is the single source of truth. Skip when the
# PyO3 bindings aren't installed (matches test_rl_parse_network_v2.py).
pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import TrainingConfig  # noqa: E402
from aerocapture.training.optimizer import OptimizerConfig  # noqa: E402
from aerocapture.training.problem import AerocaptureProblem  # noqa: E402
from aerocapture.training.train import train  # noqa: E402


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

        cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="adaptive"))
        cfg.optimizer.n_gen = 100
        cfg.optimizer.n_pop = 4
        cfg.save_dir = str(tmp_path / "training_output")

        call_count = 0

        def mock_evaluate(self_prob, X, out, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise KeyboardInterrupt
            out["F"] = np.random.default_rng(call_count).random((X.shape[0], 1)) * 1e6

        with (
            patch("aerocapture.training.problem.AerocaptureProblem._evaluate", mock_evaluate),
            patch.object(AerocaptureProblem, "_run_batch", return_value=np.full(cfg.optimizer.n_pop, 1000.0)),
        ):
            result = train(cfg, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        assert result["interrupted"] is True
        assert result["best_cost"] < float("inf")
