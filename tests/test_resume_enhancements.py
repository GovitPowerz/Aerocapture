import json
import types
from pathlib import Path

import numpy as np
from aerocapture.training.champion import Champion
from aerocapture.training.checkpoint import load_checkpoint, save_checkpoint
from aerocapture.training.config import TrainingConfig
from aerocapture.training.optimizer import OptimizerConfig
from aerocapture.training.param_spaces import ParamSpec


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
        champion=Champion(individual=pop[0], train_cost=1.0),
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


def test_islands_checkpoint_npz_schema_has_cost_transform(tmp_path: Path) -> None:
    # Schema canary: from_checkpoint's read is covered by test_islands_resume_grow_and_revalidate.
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


def test_revalidate_each_wiring(capsys) -> None:  # type: ignore[no-untyped-def]
    """`revalidate_each` re-anchors only the islands holding a champion, under the
    config ceiling, announcing with the island label; without validation seeds it is a no-op."""
    from aerocapture.training import charts
    from aerocapture.training.island_model import Island, IslandModel

    class _Problem:
        cost_kwargs = {"cost_transform": "linear", "heat_load_limit": 25000.0}

        def evaluate_individual_records_per_seed(self, x, seeds):  # type: ignore[no-untyped-def]
            records = np.zeros((len(seeds), 52))
            records[0, charts._FR_INTEGRATED_FLUX] = 26.0  # MJ/m^2 -> 26000 kJ > limit on one draw of three
            return np.full(len(seeds), 3.5), records

    model = IslandModel.__new__(IslandModel)  # bypass __init__
    model.islands = [Island("pso", algorithm=None, champion=Champion(individual=np.array([0.1, 0.2]), val_rms=999.0)), Island("ga", algorithm=None)]
    model.problem = _Problem()  # type: ignore[assignment]
    model.validation_seeds = [1, 2, 3]
    model.config = OptimizerConfig(seed_strategy="fixed", max_violation_rate=0.5)

    model.revalidate_each()
    pso = model.islands[0].champion
    assert pso.trusted and pso.val_rms == 3.5
    assert pso.last_validated is not None and np.array_equal(pso.last_validated, [0.1, 0.2])
    assert model.islands[1].champion.individual is None and model.islands[1].champion.last_validated is None  # no champion: skipped
    assert capsys.readouterr().out == ""

    model.config = OptimizerConfig(seed_strategy="fixed")  # ceiling 0.0: the same flight is now infeasible
    model.revalidate_each()
    assert not pso.trusted and pso.val_rms == float("inf")
    assert "pso:champion" in capsys.readouterr().out

    model.validation_seeds = []
    pso.val_rms = 999.0
    model.revalidate_each()
    assert pso.val_rms == 999.0  # no pool: untouched


def test_resize_populations_grows_each_island() -> None:
    from aerocapture.training.island_model import IslandModel
    from pymoo.core.population import Population

    class _P:
        def __init__(self) -> None:
            self.cost_kwargs = {"cost_transform": "linear"}

        seeds = [0]

        def evaluate_population_per_seed(self, X, seeds):  # type: ignore[no-untyped-def]
            return (np.arange(X.shape[0], dtype=np.float64))[:, None]

    rng = np.random.default_rng(0)

    def _island(name: str):  # type: ignore[no-untyped-def]
        pop = Population.new("X", rng.random((4, 2)))
        pop.set("F", np.arange(4.0).reshape(-1, 1))
        algo = types.SimpleNamespace(pop=pop)
        return types.SimpleNamespace(name=name, algorithm=algo)

    model = IslandModel.__new__(IslandModel)
    model.islands = [_island("ga"), _island("de")]
    model.problem = _P()
    model.n_params = 2

    changed = model.resize_populations(target_n=10, rng=rng, fresh_fraction=0.2, velocity_scale=0.05)
    assert changed is True
    for isl in model.islands:
        assert isl.algorithm.pop.get("X").shape == (10, 2)
        assert isl.algorithm.pop.get("F").shape[0] == 10


def test_resize_populations_noop_when_size_matches() -> None:
    from aerocapture.training.island_model import IslandModel
    from pymoo.core.population import Population

    class _P:
        cost_kwargs = {"cost_transform": "linear"}

        seeds = [0]

        def evaluate_population_per_seed(self, X, seeds):  # type: ignore[no-untyped-def]
            return (np.zeros(X.shape[0]))[:, None]

    rng = np.random.default_rng(0)
    pop = Population.new("X", rng.random((5, 1)))
    pop.set("F", np.zeros((5, 1)))
    island = types.SimpleNamespace(name="ga", algorithm=types.SimpleNamespace(pop=pop))
    model = IslandModel.__new__(IslandModel)
    model.islands = [island]  # type: ignore[list-item]
    model.problem = _P()
    model.n_params = 1

    changed = model.resize_populations(target_n=5, rng=rng, fresh_fraction=0.2, velocity_scale=0.05)
    assert changed is False


def test_islands_resume_grow_and_revalidate(tmp_path: Path) -> None:
    from aerocapture.training.island_model import IslandModel
    from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
    from pymoo.core.population import Population

    class _P:
        def __init__(self) -> None:
            self.cost_kwargs = {"cost_transform": "linear"}

        seeds = [0]

        def evaluate_population_per_seed(self, X, seeds):  # type: ignore[no-untyped-def]
            return (np.linspace(1.0, 2.0, X.shape[0]))[:, None]

        def evaluate_individual_records_per_seed(self, x, seeds):  # type: ignore[no-untyped-def]
            return np.full(len(seeds), 1.23, dtype=np.float64), [{} for _ in seeds]

    rng = np.random.default_rng(0)

    # k_top=1 so k_top*(n_islands-1)=2 <= n_pop=4 (IslandModel.__init__ guard).
    cfg = OptimizerConfig(seed_strategy="fixed", algorithm="islands", n_pop=4, validation_n_sims=3, islands=IslandSettings(k_top=1))
    model = IslandModel(
        config=cfg,
        problem=_P(),
        n_params=2,
        validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12],
        base_mc_seed=42,
        rng=rng,
    )
    # Seed each island with a real pop so checkpoint has something to write.
    for isl in model.islands:
        pop = Population.new("X", rng.random((4, 2)))
        pop.set("F", np.arange(4.0).reshape(-1, 1))
        isl.algorithm.pop = pop
        isl.algorithm.is_initialized = True
        isl.algorithm.n_iter = 1
        isl.champion = Champion(individual=pop.get("X")[0].copy(), val_rms=999.0)
    ckpt = tmp_path / "checkpoint_g00000.npz"
    model.checkpoint(ckpt, generation=0)

    # Resume into a BIGGER model with a changed transform.
    cfg2 = OptimizerConfig(seed_strategy="fixed", algorithm="islands", n_pop=12, validation_n_sims=3, islands=IslandSettings(k_top=1))
    p2 = _P()
    p2.cost_kwargs = {"cost_transform": "log"}
    model2 = IslandModel(
        config=cfg2,
        problem=p2,
        n_params=2,
        validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12],
        base_mc_seed=42,
        rng=np.random.default_rng(1),
    )
    gen, _curator, saved_transform = model2.from_checkpoint(ckpt)
    assert gen == 0
    assert saved_transform == "linear"

    model2.resize_populations(target_n=12, rng=np.random.default_rng(2), fresh_fraction=0.2, velocity_scale=0.05)
    model2.revalidate_each()

    for isl in model2.islands:
        assert isl.algorithm.pop.get("X").shape == (12, 2)
        assert isl.champion.val_rms == 1.23  # re-validated under new metric, not the stale 999.0

    from pymoo.algorithms.soo.nonconvex.pso import PSO

    for isl in model2.islands:
        if isinstance(isl.algorithm, PSO):
            assert isl.algorithm.particles.get("X").shape == (12, 2)
            assert isl.algorithm.particles.get("V").shape == (12, 2)


def test_resize_populations_shrinks_each_island() -> None:
    from aerocapture.training.island_model import IslandModel
    from pymoo.core.population import Population

    class _P:
        cost_kwargs = {"cost_transform": "linear"}

        seeds = [0]

        def evaluate_population_per_seed(self, X, seeds):  # type: ignore[no-untyped-def]
            return (np.zeros(X.shape[0]))[:, None]

    rng = np.random.default_rng(0)

    def _island(name: str):  # type: ignore[no-untyped-def]
        pop = Population.new("X", rng.random((6, 2)))
        pop.set("F", np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.0]).reshape(-1, 1))
        return types.SimpleNamespace(name=name, algorithm=types.SimpleNamespace(pop=pop))

    model = IslandModel.__new__(IslandModel)
    model.islands = [_island("ga")]
    model.problem = _P()
    model.n_params = 2

    changed = model.resize_populations(target_n=3, rng=rng, fresh_fraction=0.2, velocity_scale=0.05)
    assert changed is True
    assert model.islands[0].algorithm.pop.get("X").shape == (3, 2)


def test_grow_fresh_fraction_out_of_range_raises() -> None:
    import pytest
    from aerocapture.training.optimizer import OptimizerConfig

    with pytest.raises(ValueError, match="grow_fresh_fraction"):
        OptimizerConfig(seed_strategy="fixed", grow_fresh_fraction=1.5)
    with pytest.raises(ValueError, match="grow_fresh_fraction"):
        OptimizerConfig(seed_strategy="fixed", grow_fresh_fraction=-0.1)


def test_islands_from_checkpoint_legacy_missing_cost_transform(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from aerocapture.training.island_model import IslandModel
    from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
    from pymoo.core.population import Population

    class _P:
        cost_kwargs = {"cost_transform": "linear"}

        seeds = [0]

        def evaluate_population_per_seed(self, X, seeds):  # type: ignore[no-untyped-def]
            return (np.linspace(1.0, 2.0, X.shape[0]))[:, None]

        def evaluate_individual_records_per_seed(self, x, seeds):  # type: ignore[no-untyped-def]
            return np.full(len(seeds), 1.0, dtype=np.float64), [{} for _ in seeds]

    rng = np.random.default_rng(0)
    cfg = OptimizerConfig(seed_strategy="fixed", algorithm="islands", n_pop=4, validation_n_sims=3, islands=IslandSettings(k_top=1))
    model = IslandModel(
        config=cfg,
        problem=_P(),
        n_params=2,
        validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12],
        base_mc_seed=42,
        rng=rng,
    )
    for isl in model.islands:
        pop = Population.new("X", rng.random((4, 2)))
        pop.set("F", np.arange(4.0).reshape(-1, 1))
        isl.algorithm.pop = pop
        isl.algorithm.is_initialized = True
        isl.algorithm.n_iter = 1
        isl.champion = Champion(individual=pop.get("X")[0].copy())
    ckpt = tmp_path / "checkpoint_g00000.npz"
    model.checkpoint(ckpt, generation=0)

    # Strip the cost_transform key to simulate a legacy (pre-feature) checkpoint.
    with np.load(ckpt, allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files if k != "cost_transform"}
    legacy = tmp_path / "checkpoint_legacy.npz"
    np.savez_compressed(legacy, **arrays)

    model2 = IslandModel(
        config=cfg,
        problem=_P(),
        n_params=2,
        validation_seeds=[1, 2, 3],
        final_eval_seeds=[10, 11, 12],
        base_mc_seed=42,
        rng=np.random.default_rng(1),
    )
    _gen, _curator, saved_transform = model2.from_checkpoint(legacy)
    assert saved_transform is None
