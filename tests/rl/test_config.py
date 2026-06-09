"""RL config parser tests -- use a hermetic fixture instead of a live leaf TOML."""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.rl.config import RLConfig


@pytest.fixture
def fixture_toml(tmp_path: Path) -> Path:
    """A minimal hermetic TOML with all the fields `from_toml` exercises."""
    toml = tmp_path / "rl_fixture.toml"
    toml.write_text(
        """
[rl]
algorithm = "ppo"
total_env_steps = 500000
n_envs = 64
seed_base = 3000000
validation_n_sims = 100
validation_interval_updates = 5
checkpoint_interval_updates = 10

[rl.reward]
corridor_weight = 0.1
energy_rate_weight = 0.05
constraint_weight = 0.2
apoapsis_weight = 0.2
eccentricity_weight = 0.1
energy_scale = 1.0e6
normalize_returns = true
normalize_obs = true
norm_warmup_steps = 1000

[rl.ppo]
learning_rate = 3.0e-4
rollout_steps = 2048
update_epochs = 10
minibatches = 32
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_grad_norm = 0.5
initial_log_std = -0.5
min_log_std = -2.0
lr_anneal_start = 0.7
target_kl = 0.03

[rl.sac]
learning_rate = 3.0e-4
buffer_size = 1000000
batch_size = 256
gamma = 0.99
tau = 0.005
train_every = 1
gradient_steps = 1
target_entropy = "auto"
initial_alpha = 0.2
warmup_steps = 50000
"""
    )
    return toml


def test_loads_defaults_from_fixture(fixture_toml: Path) -> None:
    cfg = RLConfig.from_toml(fixture_toml)
    assert cfg.algorithm == "ppo"
    assert cfg.n_envs == 64
    assert cfg.seed_base == 3_000_000
    assert cfg.ppo.learning_rate == 3.0e-4
    assert cfg.reward.corridor_weight == 0.1
    assert cfg.reward.normalize_returns is True
    assert cfg.sac.warmup_steps == 50_000
    assert cfg.ppo.target_kl == 0.03


def test_cli_override(fixture_toml: Path) -> None:
    cfg = RLConfig.from_toml(fixture_toml, overrides={"algorithm": "sac", "total_env_steps": 1_000})
    assert cfg.algorithm == "sac"
    assert cfg.total_env_steps == 1_000


def test_rejects_unknown_algorithm(fixture_toml: Path) -> None:
    with pytest.raises(ValueError, match="algorithm"):
        RLConfig.from_toml(fixture_toml, overrides={"algorithm": "dqn"})


def test_ppo_override(fixture_toml: Path) -> None:
    cfg = RLConfig.from_toml(fixture_toml, ppo_overrides={"rollout_steps": 128})
    assert cfg.ppo.rollout_steps == 128


def test_sac_warmup_steps_dataclass_matches_toml_default() -> None:
    """SACConfig dataclass default must match the canonical TOML default
    so that `SACConfig()` in tests doesn't silently drift from production behavior.
    """
    from aerocapture.training.rl.config import SACConfig

    assert SACConfig().warmup_steps == 50_000


def test_ppo_log_std_ceiling_and_entropy_anneal_defaults() -> None:
    """max_log_std defaults to a generous safety rail; entropy anneal off by default."""
    from aerocapture.training.rl.config import PPOConfig

    p = PPOConfig()
    assert p.max_log_std == 2.0
    assert p.entropy_anneal_start == 1.0


def test_ppo_overrides_max_log_std_and_entropy_anneal(fixture_toml: Path) -> None:
    cfg = RLConfig.from_toml(fixture_toml, ppo_overrides={"max_log_std": 0.0, "entropy_anneal_start": 0.5})
    assert cfg.ppo.max_log_std == 0.0
    assert cfg.ppo.entropy_anneal_start == 0.5
