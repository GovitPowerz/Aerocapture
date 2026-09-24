"""Graceful keyboard interrupt, resume-preserves-champion, and checkpoint atomicity,
driven through the trainer seam against a fake evaluator (no simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.checkpoint import save_checkpoint
from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.param_spaces import PARAM_SPACES
from aerocapture.training.seed_curator import SeedCurator

from tests.fixtures.fake_problem import FakeProblem, run_trainer

SCHEME = "equilibrium_glide"  # non-NN: best artifacts need no PyO3 extension


def _cfg(save_dir: Path, *, strategy: str = "fixed") -> TrainingConfig:
    cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy=strategy), guidance_type=SCHEME)
    cfg.save_dir = str(save_dir)
    return cfg


class TestKeyboardInterrupt:
    """Tests that Ctrl+C saves checkpoint and returns cleanly."""

    def test_interrupt_returns_interrupted_flag(self, tmp_path: Path) -> None:
        """run_loop returns interrupted=True on KeyboardInterrupt."""
        cfg = _cfg(tmp_path / "training_output", strategy="adaptive")
        cfg.optimizer.n_gen = 100
        cfg.optimizer.n_pop = 4

        # Adaptive seeds + validation on: the 4th per-seed evaluation lands mid-loop.
        problem = FakeProblem(list(PARAM_SPACES[SCHEME]), seeds=[42], interrupt_after=3)
        _, result = run_trainer(cfg, problem, cwd=str(tmp_path))

        assert result["interrupted"] is True
        assert result["best_cost"] < float("inf")
        assert list(Path(cfg.save_dir).glob("checkpoint_g*.json"))


class TestResumePreservesCheckpointedBest:
    """Regression: resume must not overwrite the checkpointed best individual
    when the resumed population has a lower training cost under the current
    seed list (the two costs aren't comparable under adaptive/rotating seeds).
    End-to-end through the adapter's resume path (`from_config`)."""

    def test_resume_keeps_checkpointed_best_individual(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "training_output"
        save_dir.mkdir(parents=True)
        cfg = _cfg(save_dir)
        cfg.optimizer.n_pop = 6
        cfg.optimizer.n_gen = 5

        param_specs = PARAM_SPACES[SCHEME]
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

        # No validation pool (toml_path ""), so the first evaluation is gen 0's
        # pymoo step: interrupt there, right after the resume-init block we care about.
        problem = FakeProblem(list(param_specs), seeds=[42], toml_path="", interrupt_after=0)
        _, result = run_trainer(cfg, problem, cwd=str(tmp_path), resume_dir=save_dir)

        assert result["interrupted"] is True
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
        save_dir = tmp_path / "training_output"
        save_dir.mkdir(parents=True)
        cfg = _cfg(save_dir)
        cfg.optimizer.n_pop = 8  # RESUME target (checkpoint below is 4)
        cfg.optimizer.n_gen = 1  # +resumed_gen(2) -> runs exactly gen 2
        cfg.optimizer.validation_n_sims = 0  # no validation gate (keeps best verbatim)

        param_specs = PARAM_SPACES[SCHEME]
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

        # Constant costs: the grown pop (8) and the small pop (4) both score 1000.
        problem = FakeProblem(list(param_specs), seeds=[42], constant_cost=1000.0)
        _, result = run_trainer(cfg, problem, cwd=str(tmp_path), resume_dir=save_dir, verbose=True)

        out = capsys.readouterr().out
        assert "Resizing resumed population 4 -> 8" in out, f"resize wiring did not fire; stdout was:\n{out}"
        # Checkpointed best is preserved (validation off, None-gate stays closed).
        assert result["best_individual"] is not None
        assert np.array_equal(result["best_individual"], checkpointed_best)
        assert not result.get("interrupted", False)


class TestResumeSeedListWidth:
    """Regression: a checkpointed adaptive seed list narrower/wider than the
    resumed `training_n_sims` must be dropped at restore and the curation clock
    reset, so the first resumed generation flies an n_sims-wide bootstrap list
    and the periodic trigger re-curates right away instead of rotating at the
    old width (or on random draws) until the old clock's next interval."""

    def test_resume_with_changed_n_sims_bootstraps_n_sims_wide_seeds(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        save_dir = tmp_path / "training_output"
        save_dir.mkdir(parents=True)
        cfg = _cfg(save_dir, strategy="adaptive")
        cfg.optimizer.n_pop = 4
        cfg.optimizer.n_gen = 1  # +resumed_gen(2) -> runs exactly gen 2
        cfg.optimizer.training_n_sims = 10  # checkpoint below carries a 2-seed list
        cfg.optimizer.validation_n_sims = 0  # no promotion-triggered curation
        cfg.optimizer.seed_pool_interval = 3  # fires at gen 2 only through the reset clock: 2 - (-1) >= 3, kept clock 2 - 1 = 1 would not

        param_specs = PARAM_SPACES[SCHEME]
        rng_ck = np.random.default_rng(0)
        population = rng_ck.random((4, len(param_specs)))
        # n_bins = 2: what a checkpoint written under training_n_sims = 2 carries (curate yields n_bins-wide lists).
        checkpointed_curator = SeedCurator(sample_size=cfg.optimizer.curation_sample_size, n_bins=2, excluded_seeds=set(), rng=rng_ck)
        checkpointed_curator.seed_list = [7, 11]
        checkpointed_curator.last_curation_gen = 1
        save_checkpoint(
            save_dir,
            generation=2,
            population=population,
            costs=np.full(4, 42.0),
            best_cost=42.0,
            best_individual=population[0].copy(),
            cost_history=[42.0],
            rng=rng_ck,
            config=cfg,
            cwd=None,
            param_specs=param_specs,
            seed_curator=checkpointed_curator,
        )

        seen_widths: list[int] = []
        problem = FakeProblem(list(param_specs), seeds=[7, 11], constant_cost=1000.0)
        original = problem.evaluate_population_per_seed

        def recording(X: np.ndarray, seeds: list[int]) -> np.ndarray:
            seen_widths.append(len(seeds))
            return original(X, seeds)

        problem.evaluate_population_per_seed = recording  # type: ignore[method-assign]
        trainer, result = run_trainer(cfg, problem, cwd=str(tmp_path), resume_dir=save_dir, verbose=True)

        assert len(problem.seeds) == 10
        assert set(problem.seeds) != {7, 11}
        out = capsys.readouterr().out
        assert "seed list from checkpoint dropped: 2 seeds != training_n_sims 10" in out, f"width notice did not fire; stdout was:\n{out}"
        assert "seed-curator knobs from TOML override checkpoint: n_bins 2 -> 10" in out
        # Exactly: the re-eval after the bootstrap draw, pymoo's step (both 10 seeds), then the
        # curation probe the reset clock fires at gen 2 over the sample pool.
        assert seen_widths == [10, 10, cfg.optimizer.curation_sample_size], seen_widths
        assert trainer.seed_curator is not None
        assert trainer.seed_curator.last_curation_gen == 2
        assert trainer.seed_curator.seed_list is not None and len(trainer.seed_curator.seed_list) == 10
        assert problem.seeds == trainer.seed_curator.seed_list
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
        from aerocapture.training.checkpoint import save_checkpoint
        from aerocapture.training.param_spaces import PARAM_SPACES

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
        from aerocapture.training.checkpoint import load_checkpoint

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
        from aerocapture.training.checkpoint import load_checkpoint

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
        from aerocapture.training.checkpoint import load_checkpoint

        save_dir = tmp_path / "out"
        save_dir.mkdir()
        (save_dir / "checkpoint_g00005.json").write_text("{broken")
        (save_dir / "checkpoint_g00005.npz").write_bytes(b"junk")
        assert load_checkpoint(save_dir) is None
