"""2-gen PSO training on a minimal GRU config. Not a convergence test -- just
verifies the full stack (config parse, architecture construction, PSO eval,
Rust runtime, JSON write) runs end-to-end without error.

Runs in the python-pyo3 CI job (bindings required).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_gru_pso_smoke_2_gens(tmp_path: Path) -> None:
    from aerocapture.training.config import NetworkConfig, SimConfig, TrainingConfig
    from aerocapture.training.optimizer import OptimizerConfig, PSOSettings
    from aerocapture.training.train import train

    architecture = [
        {"type": "dense", "input_size": 16, "output_size": 8, "activation": "tanh"},
        {"type": "gru", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    save_dir = tmp_path / "neural_network_gru_pso_smoke"

    nn_cfg = NetworkConfig(
        architecture=architecture,
        input_mask=list(range(16)),
    )
    sim_cfg = SimConfig(
        executable="src/rust/target/release/aerocapture",
        nn_param_file=str(save_dir / "best_model.json"),
        toml_config="configs/training/msr_aller_gru_pso_train.toml",
        n_sims=2,
    )
    optimizer = OptimizerConfig(
        algorithm="pso",
        n_pop=8,
        n_gen=2,
        seed_strategy="fixed",
        training_n_sims=2,
        validation_n_sims=2,
        pso=PSOSettings(),
    )
    cfg = TrainingConfig(
        network=nn_cfg,
        optimizer=optimizer,
        sim=sim_cfg,
        save_dir=str(save_dir),
        guidance_type="neural_network",
    )

    result = train(cfg, seed=1, cwd=".", verbose=False, no_tui=True, from_scratch=True)
    assert result is not None
    assert not result.get("interrupted", False)
    assert result.get("best_individual") is not None

    # train() writes best_model.json inside save_dir.
    best_model = save_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {save_dir}"

    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    layer_types = [entry["type"] for entry in raw["architecture"]]
    assert layer_types == ["dense", "gru", "dense"], f"unexpected arch: {layer_types}"

    # Rust nn_forward consumes the produced JSON.
    zeros_input = [0.0] * 16
    output = aerocapture_rs.nn_forward(str(best_model), zeros_input)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
