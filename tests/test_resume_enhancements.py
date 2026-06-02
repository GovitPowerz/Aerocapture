import json
from pathlib import Path

import numpy as np
from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.train import load_checkpoint, save_checkpoint


def _make_config() -> TrainingConfig:
    return TrainingConfig(guidance_type="equilibrium_glide")


def test_grow_fresh_fraction_default() -> None:
    cfg = OptimizerConfig(seed_strategy="fixed")
    assert cfg.grow_fresh_fraction == 0.2


def test_grow_fresh_fraction_from_dict() -> None:
    cfg = OptimizerConfig.from_dict({"seed_strategy": "fixed", "grow_fresh_fraction": 0.5})
    assert cfg.grow_fresh_fraction == 0.5


def test_checkpoint_persists_cost_transform(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5)]
    pop = rng.random((3, 1))
    save_checkpoint(
        tmp_path,
        generation=2,
        population=pop,
        costs=np.zeros(3),
        best_cost=1.0,
        best_individual=pop[0],
        cost_history=[1.0],
        rng=rng,
        config=_make_config(),
        cwd=None,
        param_specs=specs,
        cost_transform="log",
    )
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded["cost_transform"] == "log"


def test_load_checkpoint_legacy_cost_transform_defaults_none(tmp_path: Path) -> None:
    # Hand-write a checkpoint pair with NO cost_transform key (legacy).
    (tmp_path / "checkpoint_g00000.json").write_text(
        json.dumps({"generation": 0, "best_cost": 1.0, "best_val_cost": 1.0, "cost_history": [], "rng_state": None})
    )
    np.savez(tmp_path / "checkpoint_g00000.npz", population=np.zeros((2, 1)), costs=np.zeros(2))
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded["cost_transform"] is None
