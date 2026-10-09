"""experiments/paper/17_quantization.sh: the resumable QAT phases, dry-run.

A fake champion dir and PTQ verdict stand in for a campaign; a `uv` shim on PATH records every
`aerocapture.training.train` launch instead of running it and forwards everything else (the verdict
pre-flight loads the real v4 configs) to the real uv.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/paper/17_quantization.sh"
CHAMP_GEN = 50


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Path]:
    real_uv = shutil.which("uv")
    assert real_uv is not None
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "uv").write_text(
        f'#!/bin/bash\ncase " $* " in *" aerocapture.training.train "*) printf \'%s\\n\' "$@" >> "$TRAIN_LOG"; exit 0;; esac\nexec "{real_uv}" "$@"\n'
    )
    (shim / "uv").chmod(0o755)
    champ = tmp_path / "champ"
    champ.mkdir()
    (champ / f"checkpoint_g{CHAMP_GEN:05d}.json").write_text("{}")
    (champ / f"checkpoint_g{CHAMP_GEN:05d}.npz").write_bytes(b"npz")
    quant = tmp_path / "quant_v4"
    (quant / "ptq_sweep").mkdir(parents=True)
    (quant / "ptq_sweep" / "quantization_results.json").write_text(
        json.dumps({"noise_seeding": "per_draw", "verdict": {"bits": 4, "granularity": "per_channel", "tensor_policy": "proj_only"}})
    )
    return {"shim": shim, "champ": champ, "quant": quant, "log": tmp_path / "train.log"}


def _run(c: dict[str, Path], *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PATH": f"{c['shim']}:{os.environ['PATH']}", "TRAIN_LOG": str(c["log"]), "CHAMPION_DIR": str(c["champ"]), "QUANT_DIR": str(c["quant"])}
    return subprocess.run([str(RUNNER), "v4", *args], capture_output=True, text=True, env=env, cwd=REPO)


def _launches(c: dict[str, Path]) -> list[str]:
    return c["log"].read_text().splitlines() if c["log"].exists() else []


def test_qat_finetune_seeds_both_checkpoint_files_and_adds_3000_gens(campaign: dict[str, Path]) -> None:
    res = _run(campaign, "qat_finetune")
    assert res.returncode == 0, res.stdout + res.stderr
    arm = campaign["quant"] / "qat4_finetune"
    assert (arm / f"checkpoint_g{CHAMP_GEN:05d}.json").exists() and (arm / f"checkpoint_g{CHAMP_GEN:05d}.npz").exists()
    launch = _launches(campaign)
    assert "configs/training/quant/v4_qat4_finetune.toml" in launch
    assert launch[launch.index("--n-gen") + 1] == "3000"
    assert launch[launch.index("--output-dir") + 1] == str(arm)
    assert "--from-scratch" not in launch


def test_qat_finetune_resumes_the_remainder_and_stops_at_target(campaign: dict[str, Path]) -> None:
    arm = campaign["quant"] / "qat4_finetune"
    arm.mkdir()
    (arm / f"checkpoint_g{CHAMP_GEN + 1000:05d}.json").write_text("{}")
    res = _run(campaign, "qat_finetune")
    assert res.returncode == 0, res.stdout + res.stderr
    assert not (arm / f"checkpoint_g{CHAMP_GEN:05d}.json").exists()  # an arm with checkpoints is not re-seeded
    launch = _launches(campaign)
    assert launch[launch.index("--n-gen") + 1] == "2000"
    (arm / f"checkpoint_g{CHAMP_GEN + 3000:05d}.json").write_text("{}")
    res = _run(campaign, "qat_finetune")
    assert res.returncode == 0 and "nothing to do" in res.stdout
    assert len(_launches(campaign)) == len(launch)


def test_qat_scratch_matches_the_champion_budget_then_resumes(campaign: dict[str, Path]) -> None:
    res = _run(campaign, "qat_scratch")
    assert res.returncode == 0, res.stdout + res.stderr
    launch = _launches(campaign)
    assert "--from-scratch" in launch and launch[launch.index("--n-gen") + 1] == str(CHAMP_GEN)
    arm = campaign["quant"] / "qat4_scratch"
    arm.mkdir()
    (arm / "checkpoint_g00020.json").write_text("{}")
    campaign["log"].unlink()
    res = _run(campaign, "qat_scratch")
    assert res.returncode == 0, res.stdout + res.stderr
    launch = _launches(campaign)
    assert "--from-scratch" not in launch and launch[launch.index("--n-gen") + 1] == str(CHAMP_GEN - 20)


def test_verdict_gate_refuses_a_ptq_file_from_the_other_regime(campaign: dict[str, Path]) -> None:
    ptq = campaign["quant"] / "ptq_sweep" / "quantization_results.json"
    ptq.write_text(json.dumps({**json.loads(ptq.read_text()), "noise_seeding": "legacy"}))
    res = _run(campaign, "qat_finetune")
    assert res.returncode != 0 and "scored under legacy" in res.stdout + res.stderr
    assert not _launches(campaign)


def test_verdict_gate_refuses_a_config_off_the_verdict_cell(campaign: dict[str, Path]) -> None:
    ptq = campaign["quant"] / "ptq_sweep" / "quantization_results.json"
    ptq.write_text(json.dumps({"noise_seeding": "per_draw", "verdict": {"bits": 4, "granularity": "per_tensor", "tensor_policy": "all"}}))
    res = _run(campaign, "qat_scratch")
    assert res.returncode != 0 and "!= PTQ verdict" in res.stdout + res.stderr
    assert not _launches(campaign)


def test_runner_syntax() -> None:
    assert subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True).returncode == 0
