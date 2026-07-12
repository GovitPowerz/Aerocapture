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


class TestResumePreservesCheckpointedBest:
    """Regression: resume must not overwrite the checkpointed best individual
    when the resumed population has a lower training cost under the current
    seed list (the two costs aren't comparable under adaptive/rotating seeds).
    End-to-end through train()."""

    def test_resume_keeps_checkpointed_best_individual(self, tmp_path: Path) -> None:
        from aerocapture.training.param_spaces import PARAM_SPACES
        from aerocapture.training.train import save_checkpoint

        # Fast-fail exe + NN dir, matching the sibling test.
        exe_path = tmp_path / "src" / "rust" / "target" / "release"
        exe_path.mkdir(parents=True)
        (tmp_path / "data" / "neural_network").mkdir(parents=True)
        dummy_exe = exe_path / "aerocapture"
        dummy_exe.write_text("#!/bin/sh\nexit 0\n")
        dummy_exe.chmod(dummy_exe.stat().st_mode | stat.S_IEXEC)

        save_dir = tmp_path / "training_output"
        save_dir.mkdir(parents=True)
        cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"))
        cfg.guidance_type = "equilibrium_glide"  # non-NN: skips write_nn_json
        cfg.optimizer.n_pop = 6
        cfg.optimizer.n_gen = 5
        cfg.save_dir = str(save_dir)

        param_specs = PARAM_SPACES[cfg.guidance_type]
        n_params = len(param_specs)

        # Craft a checkpoint where best_individual (the validated best) has
        # a HIGH training cost (simulating an old promotion whose then-active
        # seeds produced a high RMS). Another individual in the population
        # has a LOWER training cost under the current seeds -- not actually
        # better, just evaluated on a different seed set. The buggy resume
        # would swap best_individual to this population argmin.
        rng_ck = np.random.default_rng(0)
        population = rng_ck.random((cfg.optimizer.n_pop, n_params))
        costs = np.array([100.0, 500.0, 500.0, 800.0, 500.0, 500.0])
        checkpointed_best_individual = population[3].copy()
        checkpointed_best_cost = 800.0

        save_checkpoint(
            save_dir,
            generation=10,
            population=population,
            costs=costs,
            best_cost=checkpointed_best_cost,
            best_individual=checkpointed_best_individual,
            cost_history=[float(c) for c in costs],
            rng=rng_ck,
            config=cfg,
            cwd=None,
            param_specs=param_specs,
            best_val_cost=150.0,
        )

        # Mock _evaluate to raise KeyboardInterrupt on the first gen call so
        # train() exits immediately after the resume-init block we care about.
        def mock_evaluate(self_prob, X, out, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        # Also mock _run_batch in case pymoo triggers it during setup.
        with (
            patch("aerocapture.training.problem.AerocaptureProblem._evaluate", mock_evaluate),
            patch.object(AerocaptureProblem, "_run_batch", return_value=np.full(cfg.optimizer.n_pop, 1000.0)),
        ):
            result = train(
                cfg,
                seed=42,
                cwd=str(tmp_path),
                resume_dir=str(save_dir),
                verbose=False,
                no_tui=True,
            )

        assert result["best_individual"] is not None
        assert np.array_equal(result["best_individual"], checkpointed_best_individual), (
            "resume overwrote checkpointed best with population argmin -- regression of the adaptive-seed training-cost incomparability bug"
        )
        assert result["best_cost"] == checkpointed_best_cost


class TestResumeGrowsPopulation:
    """End-to-end: resuming with a larger [optimizer] n_pop grows the population
    through the wired single-algo resume path (not just the resize_population
    helper in isolation)."""

    def test_resume_with_larger_n_pop_grows_and_preserves_best(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from aerocapture.training.param_spaces import PARAM_SPACES
        from aerocapture.training.train import save_checkpoint

        exe_path = tmp_path / "src" / "rust" / "target" / "release"
        exe_path.mkdir(parents=True)
        (tmp_path / "data" / "neural_network").mkdir(parents=True)
        dummy_exe = exe_path / "aerocapture"
        dummy_exe.write_text("#!/bin/sh\nexit 0\n")
        dummy_exe.chmod(dummy_exe.stat().st_mode | stat.S_IEXEC)

        save_dir = tmp_path / "training_output"
        save_dir.mkdir(parents=True)
        cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"))
        cfg.guidance_type = "equilibrium_glide"  # non-NN: skips write_nn_json
        cfg.optimizer.n_pop = 8  # RESUME target (checkpoint below is 4)
        cfg.optimizer.n_gen = 1  # +resumed_gen(2) -> runs exactly gen 2
        cfg.optimizer.validation_n_sims = 0  # no validation gate (keeps best verbatim)
        cfg.save_dir = str(save_dir)

        param_specs = PARAM_SPACES[cfg.guidance_type]
        n_params = len(param_specs)

        # Checkpoint at the SMALL size (n_pop=4).
        rng_ck = np.random.default_rng(0)
        small_pop = rng_ck.random((4, n_params))
        costs = np.array([100.0, 500.0, 800.0, 500.0])
        checkpointed_best = small_pop[0].copy()
        save_checkpoint(
            save_dir,
            generation=2,
            population=small_pop,
            costs=costs,
            best_cost=100.0,
            best_individual=checkpointed_best,
            cost_history=[float(c) for c in costs],
            rng=rng_ck,
            config=cfg,
            cwd=None,
            param_specs=param_specs,
            best_val_cost=100.0,
        )

        # Size-aware mock: the grown pop (8) and the small pop (4) both get costs.
        with patch.object(AerocaptureProblem, "_run_batch", side_effect=lambda X: np.full(X.shape[0], 1000.0)):
            result = train(
                cfg,
                seed=42,
                cwd=str(tmp_path),
                resume_dir=str(save_dir),
                verbose=True,
                no_tui=True,
            )

        out = capsys.readouterr().out
        assert "Resizing resumed population 4 -> 8" in out, f"resize wiring did not fire; stdout was:\n{out}"
        # Checkpointed best is preserved (validation off, None-gate stays closed).
        assert result["best_individual"] is not None
        assert np.array_equal(result["best_individual"], checkpointed_best)
        assert not result.get("interrupted", False)


class TestCheckpointAtomicity:
    """save_checkpoint writes via tempfile+rename; load_checkpoint falls back
    past corrupt pairs left by pre-atomic-write crashes."""

    def _make_cfg(self, save_dir: Path) -> TrainingConfig:
        cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"))
        cfg.guidance_type = "equilibrium_glide"  # non-NN: skips write_nn_json
        cfg.optimizer.n_pop = 4
        cfg.save_dir = str(save_dir)
        return cfg

    def _save(self, save_dir: Path, cfg: TrainingConfig, generation: int) -> np.ndarray:
        from aerocapture.training.param_spaces import PARAM_SPACES
        from aerocapture.training.train import save_checkpoint

        param_specs = PARAM_SPACES[cfg.guidance_type]
        rng = np.random.default_rng(generation)
        population = rng.random((cfg.optimizer.n_pop, len(param_specs)))
        save_checkpoint(
            save_dir,
            generation=generation,
            population=population,
            costs=np.full(cfg.optimizer.n_pop, 42.0),
            best_cost=42.0,
            best_individual=population[0].copy(),
            cost_history=[42.0],
            rng=rng,
            config=cfg,
            cwd=None,
            param_specs=param_specs,
        )
        return population

    def test_save_leaves_no_tmp_files_and_round_trips(self, tmp_path: Path) -> None:
        from aerocapture.training.train import load_checkpoint

        save_dir = tmp_path / "out"
        save_dir.mkdir()
        cfg = self._make_cfg(save_dir)
        population = self._save(save_dir, cfg, generation=10)

        assert not list(save_dir.glob(".tmp_*")), "temp files must not survive a completed save"
        loaded = load_checkpoint(save_dir)
        assert loaded is not None
        assert loaded["generation"] == 10
        assert np.array_equal(loaded["population"], population)

    def test_load_falls_back_past_corrupt_latest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from aerocapture.training.train import load_checkpoint

        save_dir = tmp_path / "out"
        save_dir.mkdir()
        cfg = self._make_cfg(save_dir)
        population = self._save(save_dir, cfg, generation=10)

        # A truncated pair from a crash mid-write (pre-atomicity artifact).
        (save_dir / "checkpoint_g00020.json").write_text('{"generation": 2')
        (save_dir / "checkpoint_g00020.npz").write_bytes(b"not a zip")
        # A json whose npz never landed.
        (save_dir / "checkpoint_g00030.json").write_text("{}")

        loaded = load_checkpoint(save_dir)
        assert loaded is not None, "corrupt latest checkpoints must not brick resume"
        assert loaded["generation"] == 10
        assert np.array_equal(loaded["population"], population)
        out = capsys.readouterr().out
        assert "Skipping" in out

    def test_load_returns_none_when_all_corrupt(self, tmp_path: Path) -> None:
        from aerocapture.training.train import load_checkpoint

        save_dir = tmp_path / "out"
        save_dir.mkdir()
        (save_dir / "checkpoint_g00005.json").write_text("{broken")
        (save_dir / "checkpoint_g00005.npz").write_bytes(b"junk")
        assert load_checkpoint(save_dir) is None
