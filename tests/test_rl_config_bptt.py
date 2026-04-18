"""bptt_length config surface + divisibility guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.rl.config import PPOConfig, RLConfig


def test_ppo_config_default_bptt_length_is_32() -> None:
    c = PPOConfig()
    assert c.bptt_length == 32


def test_ppo_config_bptt_length_must_divide_rollout_steps(tmp_path: Path) -> None:
    toml = tmp_path / "bad.toml"
    toml.write_text(
        """
[rl.ppo]
rollout_steps = 100
bptt_length = 32
""".lstrip()
    )
    with pytest.raises(ValueError, match="rollout_steps"):
        RLConfig.from_toml(toml)


def test_ppo_config_bptt_length_evenly_divides_ok(tmp_path: Path) -> None:
    toml = tmp_path / "good.toml"
    toml.write_text(
        """
[rl.ppo]
rollout_steps = 256
bptt_length = 32
""".lstrip()
    )
    c = RLConfig.from_toml(toml)
    assert c.ppo.rollout_steps == 256
    assert c.ppo.bptt_length == 32
