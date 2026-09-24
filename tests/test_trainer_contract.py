"""The trainer seam through its interface: both adapters, driven by `run_loop`
against a fake per-seed evaluator (no simulator, no patched privates).

Pins the per-path conventions the `trainer.py` docstring promises: JSONL labels
(`gen + 1` single-algo, `gen` islands), checkpoint labels and cadence,
`interrupted: True` after Ctrl+C with a checkpoint on disk, and the
final-selection sidecar after `finalize`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
from aerocapture.training.param_spaces import PARAM_SPACES
from aerocapture.training.trainer import Trainer

from tests.fixtures.fake_problem import FakeProblem, run_trainer

SCHEME = "equilibrium_glide"


def _config(save_dir: Path, *, algorithm: str = "ga", n_gen: int = 3, validation_n_sims: int = 4) -> TrainingConfig:
    cfg = TrainingConfig(
        optimizer=OptimizerConfig(
            algorithm=algorithm,
            seed_strategy="fixed",
            n_pop=6,
            n_gen=n_gen,
            training_n_sims=2,
            validation_n_sims=validation_n_sims,
            islands=IslandSettings(k_top=1, k_period=1),
        ),
        guidance_type=SCHEME,
        save_dir=str(save_dir),
    )
    return cfg


def _run(cfg: TrainingConfig, save_dir: Path, *, interrupt_after: int | None = None, checkpoint_interval: int = 2) -> tuple[Trainer, dict]:
    problem = FakeProblem(list(PARAM_SPACES[SCHEME]), seeds=[42], interrupt_after=interrupt_after)
    return run_trainer(cfg, problem, seed=0, checkpoint_interval=checkpoint_interval)


def _jsonl_records(save_dir: Path) -> list[dict]:
    files = sorted(save_dir.glob("run_*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


class TestSingleAlgo:
    def test_conventions(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        _, result = _run(cfg, tmp_path)

        assert result["interrupted"] is False
        assert result["best_individual"] is not None
        # Prologue logs the start gen, then every gen is labeled gen + 1.
        assert [r["generation"] for r in _jsonl_records(tmp_path)] == [0, 1, 2, 3]
        # Cadence (gen+1) % 2 -> g00002; the final checkpoint carries the last gen that ran.
        assert sorted(p.name for p in tmp_path.glob("checkpoint_g*.json")) == ["checkpoint_g00002.json", "checkpoint_g00003.json"]
        assert (tmp_path / "final_selection.json").exists()
        assert (tmp_path / "best_params.json").exists()

    def test_interrupt_saves_checkpoint(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, n_gen=50)
        _, result = _run(cfg, tmp_path, interrupt_after=4)

        assert result["interrupted"] is True
        assert result["best_cost"] < float("inf")
        assert list(tmp_path.glob("checkpoint_g*.json")), "Ctrl+C must leave a checkpoint"
        assert not (tmp_path / "final_selection.json").exists()


class TestIslands:
    def test_conventions(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, algorithm="islands")
        _, result = _run(cfg, tmp_path)

        assert result["interrupted"] is False
        assert result["winner"] is not None
        records = _jsonl_records(tmp_path)
        # Islands log `gen` (no prologue record), one record per island per gen.
        assert sorted(r["generation"] for r in records) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert {r["island_name"] for r in records} == {"pso", "ga", "de"}
        # (gen+1) % 2 == 0 or last gen, labeled `gen`.
        assert sorted(p.name for p in tmp_path.glob("checkpoint_g*.npz")) == ["checkpoint_g00001.npz", "checkpoint_g00002.npz"]
        assert (tmp_path / "final_selection.json").exists()

    def test_interrupt_saves_checkpoint(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, algorithm="islands", n_gen=50)
        _, result = _run(cfg, tmp_path, interrupt_after=6)

        assert result["interrupted"] is True
        assert result["winner"] is None
        assert list(tmp_path.glob("checkpoint_g*.npz")), "Ctrl+C must leave a checkpoint"


@pytest.mark.parametrize("algorithm", ["ga", "islands"])
def test_adapters_satisfy_the_protocol(tmp_path: Path, algorithm: str) -> None:
    cfg = _config(tmp_path, algorithm=algorithm, n_gen=1)
    trainer, _ = _run(cfg, tmp_path)
    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.finalize_in_display_scope, bool)
    assert isinstance(trainer.start_gen, int)
