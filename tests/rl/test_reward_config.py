"""[rl.reward] parsing + atan2 RL config load."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w
from aerocapture.training.rl.config import RLConfig


def test_reward_config_parses_dv_fields(tmp_path: Path) -> None:
    p = tmp_path / "rl.toml"
    p.write_bytes(tomli_w.dumps({"rl": {"algorithm": "ppo", "reward": {"potential": "dv", "dv2_weight": 2.0}}}).encode())
    cfg = RLConfig.from_toml(p)
    assert cfg.reward.potential == "dv"
    assert cfg.reward.dv2_weight == 2.0
    assert cfg.reward.dv1_weight == 1.0  # default
    assert cfg.reward.dv3_weight == 1.0  # default


def test_reward_config_default_potential_is_phase_aware(tmp_path: Path) -> None:
    p = tmp_path / "rl.toml"
    p.write_bytes(tomli_w.dumps({"rl": {"algorithm": "ppo"}}).encode())
    cfg = RLConfig.from_toml(p)
    assert cfg.reward.potential == "phase_aware"


def test_atan2_rl_config_loads() -> None:
    # rl.train transitively imports aerocapture_rs via env.py's top-level import.
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.rl.train import _parse_network_config

    cfg = RLConfig.from_toml(Path("configs/training/msr_aller_nn_atan2_ppo_train.toml"))
    assert cfg.algorithm == "ppo"
    assert cfg.reward.potential == "dv"
    # dv weights are a tuning knob; assert they parse as positive + uniform, not a pinned value.
    assert cfg.reward.dv1_weight > 0
    assert cfg.reward.dv2_weight == cfg.reward.dv1_weight
    assert cfg.reward.dv3_weight == cfg.reward.dv1_weight
    # n_envs: leaf overrides rl_common's 64 for throughput.
    assert cfg.n_envs == 256
    assert cfg.reward.normalize_obs is False  # warm-start fix: redundant ObsNormalizer disabled
    input_mask, architecture, input_dim = _parse_network_config(cfg)
    assert len(input_mask) == 17
    assert input_dim == 17
    assert {32, 33, 34}.issubset(set(input_mask))
    # atan2_signed requires a 2-output last layer.
    assert architecture[-1].output_size == 2
