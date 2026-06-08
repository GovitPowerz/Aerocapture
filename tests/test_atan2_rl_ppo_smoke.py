"""5-update dense-PPO smoke on the atan2 DV-reward config. Exercises the full
path: TOML parse, V2Policy + atan2 head, rollout with (N,5) DV aux, DV-reward
potential, BPTT update, validation, v2 JSON export, Rust nn_forward consumes it.

Runs in the python-pyo3 CI job (bindings required). Not a convergence test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_atan2_rl_ppo_smoke_5_updates(tmp_path: Path) -> None:
    import tomli_w
    from aerocapture.training.rl.config import RLConfig
    from aerocapture.training.rl.display import make_display
    from aerocapture.training.rl.logger import RLLogger
    from aerocapture.training.rl.train import _generate_seed_model, _run_ppo
    from aerocapture.training.toml_utils import load_toml_with_bases

    resolved = load_toml_with_bases(Path("configs/training/msr_aller_nn_atan2_ppo_train.toml"))

    # Shrink RL dimensions for CI. n_envs=4 * rollout_steps=64 * 5 updates = 1280 steps.
    rl_section: dict[str, Any] = resolved.setdefault("rl", {})
    rl_section["n_envs"] = 4
    rl_section["total_env_steps"] = 4 * 64 * 5
    rl_section["validation_n_sims"] = 4
    rl_section["validation_interval_updates"] = 5
    rl_section["checkpoint_interval_updates"] = 5

    ppo_section: dict[str, Any] = rl_section.setdefault("ppo", {})
    ppo_section["rollout_steps"] = 64
    ppo_section["bptt_length"] = 64  # dense: one chunk
    ppo_section["update_epochs"] = 2
    ppo_section["minibatches"] = 2

    data_section: dict[str, Any] = resolved.setdefault("data", {})
    seed_model_path = tmp_path / "seed_model.json"
    data_section["neural_network"] = str(seed_model_path)

    resolved.pop("base", None)
    smoke_toml = tmp_path / "smoke.toml"
    smoke_toml.write_bytes(tomli_w.dumps(resolved).encode())

    output_dir = tmp_path / "neural_network_atan2_rl_smoke"
    output_dir.mkdir()

    cfg = RLConfig.from_toml(smoke_toml)
    assert cfg.reward.potential == "dv"

    _generate_seed_model(cfg, seed_model_path)
    env_overrides = {"data.neural_network": str(seed_model_path)}

    logger = RLLogger(output_dir, config_hash="smoke")
    display = make_display(cfg.total_env_steps, enabled=False)
    interrupted = {"v": False}

    try:
        _run_ppo(cfg, smoke_toml, output_dir, logger, display, interrupted, None, env_overrides, None)
    finally:
        display.close()
        logger.close()

    best_model = output_dir / "best_model.json"
    assert best_model.exists(), f"best_model.json missing under {output_dir}"

    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    layer_types = [entry["type"] for entry in raw["architecture"]]
    assert layer_types == ["dense", "dense", "dense"], f"unexpected arch: {layer_types}"

    output = aerocapture_rs.nn_forward(str(best_model), [0.0] * 35)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
