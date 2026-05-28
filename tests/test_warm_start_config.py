"""WarmStartConfig defaults + TOML parsing."""

import pytest
from aerocapture.training.config import AdamConfig, TrainingConfig, WarmStartConfig


def test_defaults() -> None:
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
    assert cfg.minibatch_size == 128
    assert cfg.eval_interval == 0  # disabled by default; opt in via TOML
    assert cfg.bound_multiplier == 4.0
    assert cfg.jitter == 0.02
    assert cfg.cmaes_sigma0 == 0.1
    assert cfg.params_paths == {}


def test_training_config_has_warm_start_field() -> None:
    cfg = TrainingConfig()
    assert isinstance(cfg.warm_start, WarmStartConfig)
    assert cfg.warm_start.enabled is False  # default TrainingConfig: warm-start off


def test_from_dict() -> None:
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


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        WarmStartConfig.from_dict({"typo_key": 5})


def test_adam_defaults_match_torch() -> None:
    """AdamConfig defaults must match torch.optim.Adam's defaults so a bare
    `[warm_start]` block (no `[warm_start.adam]`) doesn't silently change
    optimizer behavior vs the pre-existing hardcoded `lr=1e-3`."""
    adam = AdamConfig()
    assert adam.lr == 1e-3
    assert adam.beta1 == 0.9
    assert adam.beta2 == 0.999
    assert adam.eps == 1e-8
    assert adam.weight_decay == 0.0
    assert adam.amsgrad is False


def test_warm_start_config_carries_adam_field() -> None:
    cfg = WarmStartConfig()
    assert isinstance(cfg.adam, AdamConfig)


def test_from_dict_parses_nested_adam_block() -> None:
    cfg = WarmStartConfig.from_dict({"adam": {"lr": 0.01, "beta1": 0.8, "amsgrad": True}})
    assert cfg.enabled is True
    assert isinstance(cfg.adam, AdamConfig)
    assert cfg.adam.lr == 0.01
    assert cfg.adam.beta1 == 0.8
    assert cfg.adam.beta2 == 0.999  # defaulted
    assert cfg.adam.amsgrad is True


def test_from_dict_rejects_unknown_adam_keys() -> None:
    with pytest.raises(ValueError, match=r"unknown \[warm_start.adam\] keys"):
        WarmStartConfig.from_dict({"adam": {"learning_rate": 0.01}})  # typo for `lr`


def test_from_dict_rejects_non_table_adam() -> None:
    with pytest.raises(ValueError, match=r"\[warm_start.adam\] must be a table"):
        WarmStartConfig.from_dict({"adam": 0.01})
