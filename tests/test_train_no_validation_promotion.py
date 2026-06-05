"""Regression: single-algo training with validation_n_sims=0 must promote a
later generation's best, not freeze the gen-0 argmin (defect D1)."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import TrainingConfig  # noqa: E402
from aerocapture.training.optimizer import OptimizerConfig  # noqa: E402
from aerocapture.training.problem import AerocaptureProblem  # noqa: E402
from aerocapture.training.train import train  # noqa: E402


def test_no_validation_promotes_later_generation(tmp_path: Path) -> None:
    exe_path = tmp_path / "src" / "rust" / "target" / "release"
    exe_path.mkdir(parents=True)
    (tmp_path / "data" / "neural_network").mkdir(parents=True)
    dummy_exe = exe_path / "aerocapture"
    dummy_exe.write_text("#!/bin/sh\nexit 0\n")
    dummy_exe.chmod(dummy_exe.stat().st_mode | stat.S_IEXEC)

    cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"))
    cfg.optimizer.n_gen = 4
    cfg.optimizer.n_pop = 4
    cfg.optimizer.validation_n_sims = 0  # validation gate OFF -> exercises the D1 path
    cfg.save_dir = str(tmp_path / "training_output")

    gen0_min = 1000.0
    call_count = 0

    def mock_run_batch(self_prob, X):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        base = max(gen0_min - 100.0 * (call_count - 1), 10.0)
        return base + np.arange(X.shape[0], dtype=np.float64)

    with patch.object(AerocaptureProblem, "_run_batch", mock_run_batch):
        result = train(cfg, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

    assert result["interrupted"] is False
    assert result["best_cost"] < gen0_min - 100.0
