"""CheckpointConfig defaults + TrainingConfig integration."""

from __future__ import annotations

from aerocapture.training.config import CheckpointConfig, TrainingConfig


def test_defaults_disable_pruning() -> None:
    """No keep_last means legacy behavior: every checkpoint is retained."""
    cfg = CheckpointConfig()
    assert cfg.keep_last is None


def test_training_config_carries_checkpoints_field() -> None:
    cfg = TrainingConfig()
    assert isinstance(cfg.checkpoints, CheckpointConfig)
    assert cfg.checkpoints.keep_last is None


def test_keep_last_explicit_value() -> None:
    cfg = TrainingConfig(checkpoints=CheckpointConfig(keep_last=10))
    assert cfg.checkpoints.keep_last == 10
