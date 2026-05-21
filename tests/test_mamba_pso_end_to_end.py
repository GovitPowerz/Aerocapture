"""Phase 4a end-to-end PSO training smoke test for Mamba.

Unlike `test_mamba_pso_smoke.py` (which only exercises init + serialization),
this test actually runs the full training pipeline via subprocess:
  python -m aerocapture.training.train <toml> --no-tui --skip-report --output-dir <tmp>

This is the test that would have caught the `846cedd` bug: MambaSpec pydantic
validator resolves optional `dt_rank`, but `NetworkConfig.architecture` holds
raw pre-validation dicts from TOML. When the initial-validation pass tries
to serialize the best individual via `flat_weights_to_json`, the Rust side
deserializes into `LayerSpec::Mamba { dt_rank: usize }` and rejects the dict
with "missing field `dt_rank`".

We run with reduced-scope settings (n_pop=4, n_gen=1, training_n_sims=2,
validation_n_sims=2, seed_strategy=fixed) so the test takes <60s in CI while
still exercising:
  - TOML -> NetworkConfig.__post_init__ dt_rank normalization
  - AerocaptureProblem.evaluate_individual_per_seed -> write_nn_json
  - write_nn_json -> aerocapture_rs.flat_weights_to_json (the load-bearing
    PyO3 serialization path)
  - Rust runtime forward with a Mamba architecture under real MC dispersions

This test is @slow and gated to the python-pyo3 CI job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("aerocapture_rs")

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_BASE = REPO_ROOT / "configs" / "training" / "common.toml"
MISSION_BASE = REPO_ROOT / "configs" / "missions" / "mars.toml"
# The train.py CLI requires [data] neural_network to live under training_output/.
# Use a pytest-specific subdir so we can clean up without touching real output.
TRAINING_OUTPUT_DIR = REPO_ROOT / "training_output" / "__pytest_mamba_e2e__"


@pytest.mark.slow
def test_mamba_pso_end_to_end_single_generation(tmp_path: Path) -> None:
    # Minimal Mamba-containing TOML: Dense(21->8,tanh) -> Mamba(8, 4, 1) -> Dense(8->2,linear)
    # n_pop=4, n_gen=1, training_n_sims=2, validation_n_sims=2, seed_strategy=fixed
    # -> roughly 4 individuals * (2 training + 2 validation) = ~16 MC sims total
    config = tmp_path / "mamba_e2e.toml"
    # [data] neural_network must live under training_output/ per train.py's check.
    nn_deploy_path = TRAINING_OUTPUT_DIR.relative_to(REPO_ROOT) / "best_model.json"
    config.write_text(
        textwrap.dedent(f"""
        base = ["{MISSION_BASE.as_posix()}", "{COMMON_BASE.as_posix()}"]

        [guidance]
        type = "neural_network"

        [data]
        neural_network = "{nn_deploy_path.as_posix()}"
        results_suffix = ".test_mamba_e2e"

        [network]
        input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        [[network.architecture]]
        type = "dense"
        input_size = 21
        output_size = 8
        activation = "tanh"

        [[network.architecture]]
        type = "mamba"
        input_size = 8
        d_state = 4
        # dt_rank omitted -> max(1, 8/16) = 1. This is the specific path
        # that 846cedd fixed.

        [[network.architecture]]
        type = "dense"
        input_size = 8
        output_size = 2
        activation = "linear"

        [optimizer]
        algorithm = "pso"
        n_pop = 4
        n_gen = 1
        seed_strategy = "fixed"
        training_n_sims = 2
        validation_n_sims = 2
        """).strip()
        + "\n"
    )

    # Clean up any leftover from previous runs; also registers cleanup for this run.
    if TRAINING_OUTPUT_DIR.exists():
        shutil.rmtree(TRAINING_OUTPUT_DIR)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aerocapture.training.train",
                str(config),
                "--no-tui",
                "--skip-report",
                "--from-scratch",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            pytest.fail(f"train exited with code {result.returncode}\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        best_model = TRAINING_OUTPUT_DIR / "best_model.json"
        assert best_model.exists(), f"best_model.json missing after train()\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        payload = json.loads(best_model.read_text())
        assert payload["format_version"] == 2
        types = [layer["type"] for layer in payload["architecture"]]
        assert types == ["dense", "mamba", "dense"], f"unexpected architecture: {types}"

        # The serialized Mamba layer must have dt_rank resolved (846cedd invariant).
        mamba_spec = payload["architecture"][1]
        assert "dt_rank" in mamba_spec, f"best_model.json Mamba spec must include resolved dt_rank: {mamba_spec}"
        assert mamba_spec["dt_rank"] == 1, f"expected dt_rank=1 for input_size=8, got {mamba_spec['dt_rank']}"
    finally:
        # Clean up the test-specific training_output subdir.
        if TRAINING_OUTPUT_DIR.exists():
            shutil.rmtree(TRAINING_OUTPUT_DIR)
