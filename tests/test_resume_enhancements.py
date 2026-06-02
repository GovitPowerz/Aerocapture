import json
import types
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


def test_islands_from_checkpoint_returns_cost_transform(tmp_path: Path) -> None:
    # Verify the islands npz carries cost_transform at top level.
    import pickle

    npz = tmp_path / "checkpoint_g00000.npz"
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        version=2,
        generation=0,
        base_mc_seed=42,
        cost_transform="log",
        island_states=np.array(pickle.dumps([]), dtype=object),
        migration_log=np.array(pickle.dumps([]), dtype=object),
        rng_state=np.array(pickle.dumps(np.random.default_rng(0).bit_generator.state), dtype=object),
        seed_curator_state=np.array(pickle.dumps(None), dtype=object),
    )
    tmp.rename(npz)

    with np.load(npz, allow_pickle=True) as data:
        assert "cost_transform" in data
        assert str(data["cost_transform"]) == "log"


class _FakeProblem:
    def __init__(self, rms: float) -> None:
        self._rms = rms
        self.cost_kwargs = {"cost_transform": "linear"}

    def evaluate_individual_records_per_seed(self, x, seeds):  # type: ignore[no-untyped-def]
        # costs whose RMS == self._rms regardless of x
        return np.full(len(seeds), self._rms, dtype=np.float64), [{} for _ in seeds]


def _fake_island(name: str, best_indiv, best_val_cost: float):  # type: ignore[no-untyped-def]
    return types.SimpleNamespace(
        name=name,
        best_overall_individual=best_indiv,
        best_val_cost=best_val_cost,
        last_validated_individual=None,
    )


def test_revalidate_each_recomputes_best_val_cost() -> None:
    from aerocapture.training.island_model import IslandModel

    model = IslandModel.__new__(IslandModel)  # bypass __init__
    model.islands = [
        _fake_island("pso", np.array([0.1, 0.2]), best_val_cost=999.0),
        _fake_island("ga", None, best_val_cost=999.0),  # no best -> skipped
    ]
    model.problem = _FakeProblem(rms=3.5)
    model.validation_seeds = [1, 2, 3]

    model.revalidate_each()

    assert model.islands[0].best_val_cost == 3.5
    last_validated = model.islands[0].last_validated_individual
    assert last_validated is not None
    assert np.array_equal(last_validated, np.array([0.1, 0.2]))
    # Island with no best_overall_individual is untouched.
    assert model.islands[1].best_val_cost == 999.0
    assert model.islands[1].last_validated_individual is None
