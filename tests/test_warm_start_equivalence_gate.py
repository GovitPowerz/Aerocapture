"""magnitude_only warm-start with supervisor_schemes=['ftc'] should be at
least as good as the pre-refactor pipeline (within 5% slack on validation
RMS after 20 GA generations under a fixed seed).

Slow / E2E: requires Rust + a trained FTC scheme available at
training_output/ftc/best_params.json, AND a pre-recorded baseline at
tests/fixtures/warm_start_magonly_baseline.json. Skipped if either is missing.

Recording the baseline:
  Run the pre-refactor pipeline (the existing code on `main` before this branch
  is merged) with --n-gen 20 --n-pop 32 under a fixed seed; extract best
  validation RMS; write to tests/fixtures/warm_start_magonly_baseline.json:

      {
        "val_rms_after_20_gens": <measured_value>,
        "recorded_at": "YYYY-MM-DD",
        "config": "msr_aller_nn_train_consolidated.toml",
        "n_gen": 20,
        "n_pop": 32
      }

  The 5% absolute slack allows for the magnitude_only target signal change
  (pre_lateral_magnitude -> abs(final_signed_bank)) introduced by this refactor.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("training_output/ftc/best_params.json").exists(),
    reason="requires trained FTC scheme at training_output/ftc/best_params.json",
)
def test_magnitude_only_at_least_as_good(tmp_path):
    """20-gen GA warm-start run achieves RMS within 5% of recorded baseline."""
    baseline_path = Path("tests/fixtures/warm_start_magonly_baseline.json")
    if not baseline_path.exists():
        pytest.skip("baseline snapshot missing; record one with the pre-refactor pipeline first (see this test file's module docstring)")
    baseline = json.loads(baseline_path.read_text())["val_rms_after_20_gens"]

    out_dir = tmp_path / "training_run"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "aerocapture.training.train",
        "configs/training/msr_aller_nn_train_consolidated.toml",
        "--n-gen",
        "20",
        "--n-pop",
        "32",
        "--no-tui",
        "--skip-report",
        "--output-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, f"train.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Parse best validation RMS from JSONL log
    jsonl_paths = sorted(out_dir.glob("*.jsonl"))
    assert jsonl_paths, f"no JSONL log in {out_dir}"
    best_rms = float("inf")
    for line in jsonl_paths[0].read_text().splitlines():
        rec = json.loads(line)
        rms = rec.get("validation", {}).get("rms_cost")
        if rms is not None and rms < best_rms:
            best_rms = rms
    assert best_rms != float("inf"), "no validation RMS in log"
    assert best_rms <= baseline * 1.05, f"warm-start regression: post-refactor RMS {best_rms:.3f} > baseline {baseline:.3f} + 5% slack"
