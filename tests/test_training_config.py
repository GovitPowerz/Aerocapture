"""Tests for training configuration dataclasses."""

from __future__ import annotations

from aerocapture.training.config import GAConfig


def test_ga_config_rotate_seeds_default_false() -> None:
    ga = GAConfig()
    assert ga.rotate_seeds is False
