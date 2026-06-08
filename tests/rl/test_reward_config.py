"""[rl.reward] parsing + atan2 RL config load."""

from __future__ import annotations

from pathlib import Path

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
