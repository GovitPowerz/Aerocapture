"""RL config parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.rl.config import RLConfig


def test_loads_common_defaults() -> None:
    cfg = RLConfig.from_toml(Path("configs/training/msr_aller_rl_train.toml"))
    assert cfg.algorithm == "ppo"
    assert cfg.n_envs == 64
    assert cfg.seed_base == 3_000_000
    assert cfg.ppo.learning_rate == 3.0e-4
    assert cfg.reward.shaping_enabled is True


def test_cli_override() -> None:
    cfg = RLConfig.from_toml(
        Path("configs/training/msr_aller_rl_train.toml"),
        overrides={"algorithm": "sac", "total_env_steps": 1_000},
    )
    assert cfg.algorithm == "sac"
    assert cfg.total_env_steps == 1_000


def test_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        RLConfig.from_toml(
            Path("configs/training/msr_aller_rl_train.toml"),
            overrides={"algorithm": "dqn"},
        )
