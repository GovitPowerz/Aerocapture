"""5-update PPO-LSTM smoke test. Verifies end-to-end: TOML parse, V2Policy
instantiation, rollout collect with tuple-state threading, chunked BPTT update
with _zero_entry tuple dispatch, validation promotion, v2 JSON export with
lstm, Rust nn_forward consumes it.

Runs in the python-pyo3 CI job (bindings required). Not a convergence test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_lstm_ppo_smoke_5_updates(tmp_path: Path) -> None:
    import tomli_w
    from aerocapture.training.rl.config import RLConfig
    from aerocapture.training.rl.display import make_display
    from aerocapture.training.rl.logger import RLLogger
    from aerocapture.training.rl.train import _generate_seed_model, _run_ppo
    from aerocapture.training.toml_utils import load_toml_with_bases

    # Load the full PPO-LSTM config with base inheritance resolved, then shrink
    # every dimension so the smoke test fits in CI (~60-90s).
    resolved = load_toml_with_bases(Path("configs/training/msr_aller_lstm_ppo_train.toml"))

    # Shrink the architecture: Dense(21->8) -> LSTM(8, hidden=8) -> Dense(8->2).
    resolved["network"]["architecture"] = [
        {"type": "dense", "input_size": 21, "output_size": 8, "activation": "tanh"},
        {"type": "lstm", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]

    # Reduce RL dimensions. n_envs=4 * rollout_steps=64 * 5 updates = 1280 env-steps.
    rl_section: dict[str, Any] = resolved.setdefault("rl", {})
    rl_section["n_envs"] = 4
    rl_section["total_env_steps"] = 4 * 64 * 5
    rl_section["validation_n_sims"] = 4
    rl_section["validation_interval_updates"] = 5
    rl_section["checkpoint_interval_updates"] = 5

    ppo_section: dict[str, Any] = rl_section.setdefault("ppo", {})
    ppo_section["rollout_steps"] = 64
    ppo_section["bptt_length"] = 16  # 64 / 16 = 4 chunks -- exercises tuple-state detach
    ppo_section["update_epochs"] = 2
    ppo_section["minibatches"] = 2

    data_section: dict[str, Any] = resolved.setdefault("data", {})
    seed_model_path = tmp_path / "seed_model.json"
    data_section["neural_network"] = str(seed_model_path)

    resolved.pop("base", None)

    smoke_toml = tmp_path / "smoke.toml"
    smoke_toml.write_bytes(tomli_w.dumps(resolved).encode())

    output_dir = tmp_path / "neural_network_lstm_ppo_smoke"
    output_dir.mkdir()

    cfg = RLConfig.from_toml(smoke_toml)

    _generate_seed_model(cfg, seed_model_path)
    env_overrides = {"data.neural_network": str(seed_model_path)}

    logger = RLLogger(output_dir, config_hash="smoke")
    display = make_display(cfg.total_env_steps, enabled=False)
    interrupted = {"v": False}

    try:
        _run_ppo(
            cfg,
            smoke_toml,
            output_dir,
            logger,
            display,
            interrupted,
            None,
            env_overrides,
            None,
        )
    finally:
        display.close()
        logger.close()

    best_model = output_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {output_dir}"

    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    layer_types = [entry["type"] for entry in raw["architecture"]]
    assert layer_types == ["dense", "lstm", "dense"], f"unexpected arch: {layer_types}"
    assert "lstm" in layer_types

    # Rust nn_forward consumes the produced JSON and returns a finite 2-tuple.
    output = aerocapture_rs.nn_forward(str(best_model), [0.0] * 21)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
