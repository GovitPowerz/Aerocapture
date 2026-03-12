"""Integration test: verify TrainingLogger is called correctly by train.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


class TestRotateSeedsIntegration:
    def test_evaluate_called_with_mc_seed_when_rotate_enabled(self, tmp_path: Path) -> None:
        """When rotate_seeds=True, evaluate_chromosome receives mc_seed arg."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.save_dir = str(tmp_path)
        config.sim.toml_config = "dummy.toml"

        # Create a dummy TOML with [monte_carlo].seed
        dummy_toml = tmp_path / "dummy.toml"
        dummy_toml.write_text("[monte_carlo]\nseed = 10\n")

        mock_eval_calls: list[dict] = []

        def tracking_eval(*args: object, **kwargs: object) -> tuple[float, None]:
            mock_eval_calls.append(kwargs.copy())  # type: ignore[arg-type]
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=tracking_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train

            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # All evaluate calls should have mc_seed set
        assert len(mock_eval_calls) > 0
        for call in mock_eval_calls:
            assert "mc_seed" in call
            assert call["mc_seed"] == 10 + 0  # base_mc_seed(10) + gen(0)

    def test_evaluate_called_without_mc_seed_when_rotate_disabled(self, tmp_path: Path) -> None:
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = False
        config.save_dir = str(tmp_path)

        mock_eval_calls: list[dict] = []

        def tracking_eval(*args: object, **kwargs: object) -> tuple[float, None]:
            mock_eval_calls.append(kwargs.copy())  # type: ignore[arg-type]
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=tracking_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train

            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # No evaluate call should have mc_seed
        for call in mock_eval_calls:
            assert call.get("mc_seed") is None

    def test_parents_reevaluated_when_rotate_enabled(self, tmp_path: Path) -> None:
        """With rotate_seeds, total evals per gen = 2*n_pop (offspring + parents)."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.save_dir = str(tmp_path)
        config.sim.toml_config = "dummy.toml"

        dummy_toml = tmp_path / "dummy.toml"
        dummy_toml.write_text("[monte_carlo]\nseed = 10\n")

        eval_count = 0

        def counting_eval(*args: object, **kwargs: object) -> tuple[float, None]:
            nonlocal eval_count
            eval_count += 1
            return 100.0, None

        with (
            patch("aerocapture.training.train.evaluate_chromosome", side_effect=counting_eval),
            patch("aerocapture.training.train.create_initial_population") as mock_init,
        ):
            rng = np.random.default_rng(0)
            pop = rng.integers(0, 2, size=(4, 112), dtype=np.int8)
            costs = np.array([100.0, 200.0, 300.0, 400.0])
            mock_init.return_value = (pop, costs)

            from aerocapture.training.train import train

            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

        # 1 gen, 1 subpop: 4 offspring + 4 parents = 8 evals
        assert eval_count == 8

    def test_rotate_seeds_requires_toml_config(self, tmp_path: Path) -> None:
        """rotate_seeds=True without toml_config raises ValueError."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.sim.toml_config = None
        config.save_dir = str(tmp_path)

        from aerocapture.training.train import train

        with pytest.raises(ValueError, match="rotate_seeds requires a TOML config"):
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

    def test_rotate_seeds_requires_mc_seed_in_toml(self, tmp_path: Path) -> None:
        """rotate_seeds=True with TOML missing [monte_carlo].seed raises ValueError."""
        config = make_training_config("equilibrium_glide")
        config.ga.n_gen = 1
        config.ga.n_pop = 4
        config.ga.n_runs = 1
        config.ga.rotate_seeds = True
        config.sim.toml_config = "dummy.toml"
        config.save_dir = str(tmp_path)

        dummy_toml = tmp_path / "dummy.toml"
        dummy_toml.write_text('[mission]\ntype = "aerocapture"\n')

        from aerocapture.training.train import train

        with pytest.raises(ValueError, match=r"rotate_seeds requires \[monte_carlo\]\.seed"):
            train(config, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)
