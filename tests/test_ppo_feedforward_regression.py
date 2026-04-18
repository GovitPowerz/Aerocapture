"""Feedforward PPO regression: V2Policy with a dense-only arch runs through the
ppo_update_bptt machinery with bptt_length = rollout_steps (single chunk) and
produces a functional trained policy.

Phase 1.5 success criterion #2: feedforward PPO behavior is preserved through
the V2Policy + ppo_update_bptt migration. This is a functional gate (file
exists + Rust can load the output + output is finite), not a bit-identity gate
against a pre-migration baseline (which we do not have frozen on disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_feedforward_ppo_regression(tmp_path: Path) -> None:
    import tomli_w
    import torch
    from aerocapture.training.rl.config import RLConfig
    from aerocapture.training.rl.display import make_display
    from aerocapture.training.rl.logger import RLLogger
    from aerocapture.training.rl.train import _generate_seed_model, _run_ppo
    from aerocapture.training.toml_utils import load_toml_with_bases

    # Load the existing feedforward PPO config with base inheritance resolved,
    # then shrink every dimension so the regression test fits in CI (<= 90s).
    # This config keeps the v1 `layer_sizes` + `activations` dense-only arch.
    resolved = load_toml_with_bases(Path("configs/training/msr_aller_rl_train.toml"))

    # Reduce RL dimensions. n_envs=4 * rollout_steps=64 * 5 updates = 1280 env-steps.
    rl_section: dict[str, Any] = resolved.setdefault("rl", {})
    rl_section["n_envs"] = 4
    rl_section["total_env_steps"] = 4 * 64 * 5
    rl_section["validation_n_sims"] = 4
    rl_section["validation_interval_updates"] = 5
    rl_section["checkpoint_interval_updates"] = 5

    ppo_section: dict[str, Any] = rl_section.setdefault("ppo", {})
    ppo_section["rollout_steps"] = 64
    # bptt_length == rollout_steps: single chunk, stateless-equivalent feedforward PPO.
    ppo_section["bptt_length"] = 64
    ppo_section["update_epochs"] = 2
    ppo_section["minibatches"] = 2

    # Point data.neural_network inside tmp_path so no repo pollution.
    data_section: dict[str, Any] = resolved.setdefault("data", {})
    seed_model_path = tmp_path / "seed_model.json"
    data_section["neural_network"] = str(seed_model_path)

    # Strip `base` so the rewritten file is self-contained (no further resolution).
    resolved.pop("base", None)

    smoke_toml = tmp_path / "smoke.toml"
    smoke_toml.write_bytes(tomli_w.dumps(resolved).encode())

    output_dir = tmp_path / "neural_network_ppo_ff_smoke"
    output_dir.mkdir()

    # Reproducibility: seed the RNGs before seed-model + policy construction.
    torch.manual_seed(2026)
    np.random.seed(2026)

    cfg = RLConfig.from_toml(smoke_toml)

    # Mirror main()'s from-scratch init: generate a random-weight seed model so
    # BatchedSimulation has a valid NN JSON to load at env construction.
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
    # Dense-only: v1 [23, 16, 8, 4, 2] layer_sizes round-trip to 4 dense layers in v2.
    assert set(layer_types) == {"dense"}, f"expected dense-only arch, got: {layer_types}"

    # Rust nn_forward consumes the produced JSON and returns a finite 2-tuple.
    output = aerocapture_rs.nn_forward(str(best_model), [0.0] * 23)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
    assert all(np.isfinite(v) for v in output)
