"""WarmStartConfig defaults + TOML parsing."""

import pytest
from aerocapture.training.config import TrainingConfig, WarmStartConfig


def test_defaults():
    cfg = WarmStartConfig()
    assert cfg.enabled is False  # bare construction does NOT flip the gate
    assert cfg.supervisor_schemes == [
        "ftc",
        "equilibrium_glide",
        "energy_controller",
        "pred_guid",
        "fnpag",
    ]
    assert cfg.bptt_length == 32
    assert cfg.n_warm_seeds == 200
    assert cfg.n_epochs == 10
    assert cfg.bound_multiplier == 4.0
    assert cfg.jitter == 0.02
    assert cfg.cmaes_sigma0 == 0.1
    assert cfg.params_paths == {}


def test_training_config_has_warm_start_field():
    cfg = TrainingConfig()
    assert isinstance(cfg.warm_start, WarmStartConfig)
    assert cfg.warm_start.enabled is False  # default TrainingConfig: warm-start off


def test_from_dict():
    d = {
        "supervisor_schemes": ["ftc", "fnpag"],
        "bptt_length": 16,
        "n_warm_seeds": 100,
        "params_paths": {"ftc": "/some/path/best_params.json"},
    }
    cfg = WarmStartConfig.from_dict(d)
    assert cfg.enabled is True  # presence of the TOML block flips the gate on
    assert cfg.supervisor_schemes == ["ftc", "fnpag"]
    assert cfg.bptt_length == 16
    assert cfg.n_warm_seeds == 100
    assert cfg.params_paths == {"ftc": "/some/path/best_params.json"}
    # Unspecified keys use defaults
    assert cfg.bound_multiplier == 4.0


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        WarmStartConfig.from_dict({"typo_key": 5})
