"""The trainer's deploy path is run-local (#170).

A run writes the TOML's `[data] neural_network` path only when it resolves
into the run's own output dir. A config that base-inherits another cell's TOML
keeps that cell's path, so before this rule every `--output-dir` run of the
family overwrote the cell's `best_model.json` and left its `best_params.json`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.artifacts import owned_deploy_path, write_best_artifacts
from aerocapture.training.config import NetworkConfig, TrainingConfig
from aerocapture.training.training_config import _setup_param_specs

REPO = Path(__file__).resolve().parents[1]
ROOT = "training_output/__pytest_deploy_path__"


def _nn_config(nn_param_file: str) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.sim.nn_param_file = nn_param_file
    return cfg


class TestOwnedDeployPath:
    def test_inside_save_dir(self, tmp_path: Path) -> None:
        cfg = _nn_config("runs/cell/best_model.json")
        assert owned_deploy_path(cfg, tmp_path / "runs/cell", tmp_path) == (tmp_path / "runs/cell/best_model.json").resolve()

    def test_other_filename_in_save_dir(self, tmp_path: Path) -> None:
        cfg = _nn_config("runs/cell/deployed.json")
        assert owned_deploy_path(cfg, tmp_path / "runs/cell", tmp_path) == (tmp_path / "runs/cell/deployed.json").resolve()

    def test_output_dir_above_the_cells_owns_none_of_them(self, tmp_path: Path) -> None:
        cfg = _nn_config("runs/cells/cell/best_model.json")
        assert owned_deploy_path(cfg, tmp_path / "runs", tmp_path) is None

    def test_other_run_dir(self, tmp_path: Path) -> None:
        cfg = _nn_config("runs/cell/best_model.json")
        assert owned_deploy_path(cfg, tmp_path / "runs/cell_s2", tmp_path) is None

    def test_string_prefix_sibling_is_not_inside(self, tmp_path: Path) -> None:
        cfg = _nn_config("runs/cell_s2/best_model.json")
        assert owned_deploy_path(cfg, tmp_path / "runs/cell", tmp_path) is None

    def test_relative_save_dir_resolves_against_process_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _nn_config("runs/cell/best_model.json")
        assert owned_deploy_path(cfg, Path("runs/cell"), ".") == (tmp_path / "runs/cell/best_model.json").resolve()

    def test_no_cwd_never_deploys(self, tmp_path: Path) -> None:
        cfg = _nn_config(str(tmp_path / "runs/cell/best_model.json"))
        assert owned_deploy_path(cfg, tmp_path / "runs/cell", None) is None


class TestWriteBestArtifactsDeploy:
    def _write(self, tmp_path: Path, nn_param_file: str) -> Path:
        pytest.importorskip("aerocapture_rs")
        arch = [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "asinh"}]
        cfg = TrainingConfig(network=NetworkConfig(architecture=arch, input_mask=[0, 1, 2]))
        cfg.guidance_type = "neural_network"
        cfg.sim.nn_param_file = nn_param_file
        specs = _setup_param_specs(cfg, {}, verbose=False)
        save_dir = tmp_path / "run"
        save_dir.mkdir()
        write_best_artifacts(np.full(len(specs), 0.25), cfg, specs, save_dir, cwd=tmp_path, deploy_to_cwd=True)
        return save_dir

    def test_owned_path_gets_the_model(self, tmp_path: Path) -> None:
        save_dir = self._write(tmp_path, "run/deployed.json")
        assert (save_dir / "deployed.json").read_bytes() == (save_dir / "best_model.json").read_bytes()

    def test_another_runs_path_is_never_written(self, tmp_path: Path) -> None:
        save_dir = self._write(tmp_path, "cell/best_model.json")
        assert (save_dir / "best_model.json").exists()
        assert not (tmp_path / "cell").exists()


@pytest.fixture
def scratch_root() -> Iterator[Path]:
    root = REPO / ROOT
    shutil.rmtree(root, ignore_errors=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _train(toml: Path, *extra: str) -> str:
    cmd = [sys.executable, "-u", "-m", "aerocapture.training.train", str(toml), "--no-tui", "--skip-report", *extra]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{' '.join(cmd)} exited {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    return proc.stdout


@pytest.mark.slow
def test_output_dir_runs_never_write_each_others_model(tmp_path: Path, scratch_root: Path) -> None:
    """Two `--output-dir` trainings of one TOML leave each other's best_model.json
    byte-identical to what each wrote itself; a plain run still deploys to the TOML path."""
    pytest.importorskip("aerocapture_rs")
    base = REPO / "configs/training/msr_aller_nn_atan2_train.toml"
    cell = scratch_root / "cell"
    toml = tmp_path / "deploy_path.toml"
    toml.write_text(
        f"""
base = ["{base.as_posix()}"]

[data]
neural_network = "{ROOT}/cell/best_model.json"

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 4
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "asinh"

[optimizer]
algorithm = "pso"
n_pop = 8
n_gen = 2
training_n_sims = 2
validation_n_sims = 2
seed_strategy = "fixed"
"""
    )
    other = tmp_path / "other"

    _train(toml, "--output-dir", f"{ROOT}/cell", "--seed", "1")
    cell_model = (cell / "best_model.json").read_bytes()

    out = _train(toml, "--output-dir", str(other), "--seed", "2")
    other_model = (other / "best_model.json").read_bytes()
    assert other_model != cell_model, "seeds 1 and 2 trained the same model; the clobber check below would be vacuous"
    assert (cell / "best_model.json").read_bytes() == cell_model, "the --output-dir run overwrote the TOML path's model"
    notices = [line for line in out.splitlines() if "not written" in line]
    assert len(notices) == 1, notices
    assert f"{ROOT}/cell/best_model.json" in notices[0]

    _train(toml, "--seed", "3", "--from-scratch")
    assert (cell / "best_model.json").read_bytes() not in (cell_model, other_model), "the plain run did not deploy to the TOML path"
    assert (other / "best_model.json").read_bytes() == other_model
