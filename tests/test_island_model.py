"""Unit tests for the 3-island PSO/GA/DE evolutionary trainer.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

import pytest

from aerocapture.training.optimizer import IslandSettings, OptimizerConfig


def test_optimizer_config_islands_parses() -> None:
    d = {
        "algorithm": "islands",
        "seed_strategy": "adaptive",
        "n_pop": 64,
        "training_n_sims": 20,
        "islands": {
            "enabled": True,
            "k_period": 25,
            "k_top": 3,
            "pso_inject_velocity_scale": 0.05,
        },
        "ga": {"crossover_eta": 15.0, "mutation_eta": 20.0},
        "pso": {"w": 0.7, "c1": 1.5, "c2": 1.5},
        "de": {"variant": "DE/rand/1/bin", "crossover_prob": 0.8, "scaling_factor": 0.6},
    }
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.algorithm == "islands"
    assert cfg.islands.enabled is True
    assert cfg.islands.k_period == 25
    assert cfg.islands.k_top == 3
    assert cfg.islands.pso_inject_velocity_scale == 0.05


def test_optimizer_config_islands_default_values() -> None:
    d = {"algorithm": "islands", "seed_strategy": "fixed"}
    cfg = OptimizerConfig.from_dict(d)
    assert cfg.islands.enabled is True
    assert cfg.islands.k_period == 25
    assert cfg.islands.k_top == 3
    assert cfg.islands.pso_inject_velocity_scale == 0.05


def test_optimizer_config_islands_invalid_k_top_raises() -> None:
    with pytest.raises(ValueError, match="k_top"):
        IslandSettings(k_top=0)


def test_optimizer_config_islands_invalid_k_period_raises() -> None:
    with pytest.raises(ValueError, match="k_period"):
        IslandSettings(k_period=0)


def test_optimizer_config_islands_invalid_velocity_scale_raises() -> None:
    with pytest.raises(ValueError, match="pso_inject_velocity_scale"):
        IslandSettings(pso_inject_velocity_scale=-0.01)


def test_create_algorithm_raises_for_islands_value() -> None:
    """Direct create_algorithm() call with 'islands' must fail loudly with a
    pointer to IslandModel. The islands path goes through IslandModel.__init__,
    not through create_algorithm."""
    from aerocapture.training.optimizer import create_algorithm

    cfg = OptimizerConfig(algorithm="islands", seed_strategy="fixed")
    with pytest.raises(ValueError, match="IslandModel"):
        create_algorithm(cfg, n_params=10)
