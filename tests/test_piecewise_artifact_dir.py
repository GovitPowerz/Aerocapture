"""A piecewise_constant run writes its corridor and reference into the mission dir only from its canonical dir or as `reference_only`.

`training_output/<mission>/ref_trajectory.dat` is the tracked reference every
ref-tracking scheme flies; an `--output-dir` piecewise_constant run (a retune
under another regime) must not replace it. A `reference_only` run exists to write
that reference and does so from any dir (`msr_aller_pc_ref_train.toml` runs with
`--output-dir training_output/pc_ref`).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from aerocapture.training.artifacts import piecewise_artifact_dir

REPO = Path(__file__).resolve().parents[1]
ROOT = "training_output/__pytest_piecewise_artifact_dir__"
MISSION_FILES = (REPO / "training_output/mars/ref_trajectory.dat", REPO / "training_output/mars/corridor_boundaries.npz")


def test_canonical_run_writes_the_mission_dir(tmp_path: Path) -> None:
    mission = tmp_path / "training_output/mars"
    assert piecewise_artifact_dir(Path("training_output/piecewise_constant"), mission, tmp_path, reference_only=False) == mission


def test_canonical_run_given_as_output_dir_writes_the_mission_dir(tmp_path: Path) -> None:
    mission = tmp_path / "training_output/mars"
    assert piecewise_artifact_dir(tmp_path / "training_output/piecewise_constant/", mission, tmp_path, reference_only=False) == mission


def test_output_dir_run_stays_run_local(tmp_path: Path) -> None:
    save_dir = Path("training_output/ou_marginal/classical/piecewise_constant")
    assert piecewise_artifact_dir(save_dir, tmp_path / "training_output/mars", tmp_path, reference_only=False) == (tmp_path / save_dir).resolve()


def test_reference_only_run_writes_the_mission_dir_from_any_dir(tmp_path: Path) -> None:
    mission = tmp_path / "training_output/mars"
    assert piecewise_artifact_dir(Path("training_output/pc_ref"), mission, tmp_path, reference_only=True) == mission


@pytest.fixture
def scratch_root() -> Iterator[Path]:
    root = REPO / ROOT
    shutil.rmtree(root, ignore_errors=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mission_snapshot() -> Iterator[dict[Path, bytes | None]]:
    """The mission artifacts before the run; put back afterwards if the run changed them."""
    before = {p: p.read_bytes() if p.exists() else None for p in MISSION_FILES}
    yield before
    for p, data in before.items():
        if data is None:
            p.unlink(missing_ok=True)
        elif p.read_bytes() != data:
            p.write_bytes(data)


@pytest.mark.slow
def test_output_dir_run_writes_only_its_own_dir(tmp_path: Path, scratch_root: Path, mission_snapshot: dict[Path, bytes | None]) -> None:
    """train.py end to end: the wiring, not the helper. An `--output-dir` piecewise_constant
    run lands corridor_boundaries.npz and ref_trajectory.dat in its own dir and leaves
    every byte of training_output/mars/ as it found it."""
    pytest.importorskip("aerocapture_rs")
    base = REPO / "configs/training/msr_aller_piecewise_constant_train.toml"
    toml = tmp_path / "pc_retune.toml"
    toml.write_text(
        f"""
base = ["{base.as_posix()}"]

[optimizer]
n_pop = 4
n_gen = 1
training_n_sims = 1
validation_n_sims = 1
seed_strategy = "fixed"
"""
    )
    cell = scratch_root / "cell"
    cmd = [sys.executable, "-u", "-m", "aerocapture.training.train", str(toml), "--no-tui", "--skip-report", "--sim-timeout", "5"]
    cmd += ["--output-dir", f"{ROOT}/cell", "--seed", "1"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{' '.join(cmd)} exited {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert (cell / "corridor_boundaries.npz").exists()
    assert (cell / "ref_trajectory.dat").exists()
    for p, data in mission_snapshot.items():
        assert (p.read_bytes() if p.exists() else None) == data, f"{p} was rewritten by an --output-dir run"
