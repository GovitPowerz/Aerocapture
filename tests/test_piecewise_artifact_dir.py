"""A piecewise_constant run writes its corridor and reference into the mission dir only from its canonical dir or as `reference_only`.

`training_output/<mission>/ref_trajectory.dat` is the tracked reference every
ref-tracking scheme flies; an `--output-dir` piecewise_constant run (a retune
under another regime) must not replace it. A `reference_only` run exists to write
that reference and does so from any dir (`msr_aller_pc_ref_train.toml` runs with
`--output-dir training_output/pc_ref`).
"""

from __future__ import annotations

from pathlib import Path

from aerocapture.training.artifacts import piecewise_artifact_dir


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
