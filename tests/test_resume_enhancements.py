from aerocapture.training.optimizer import OptimizerConfig


def test_grow_fresh_fraction_default() -> None:
    cfg = OptimizerConfig(seed_strategy="fixed")
    assert cfg.grow_fresh_fraction == 0.2


def test_grow_fresh_fraction_from_dict() -> None:
    cfg = OptimizerConfig.from_dict({"seed_strategy": "fixed", "grow_fresh_fraction": 0.5})
    assert cfg.grow_fresh_fraction == 0.5
