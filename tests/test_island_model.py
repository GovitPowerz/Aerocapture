"""Unit tests for the 3-island PSO/GA/DE evolutionary trainer.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aerocapture.training.island_model import Island, IslandModel, MigrationEvent, inject_into_pso, migrate
from aerocapture.training.optimizer import DESettings, GASettings, IslandSettings, OptimizerConfig, PSOSettings
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem


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


class _FakeAlgo:
    """Minimal stand-in for pymoo Algorithm — only `.pop` is touched by migrate()."""

    def __init__(self, pop: Population) -> None:
        self.pop = pop


def _make_pop(X: np.ndarray, F: np.ndarray) -> Population:
    pop = Population.new("X", X)
    pop.set("F", F.reshape(-1, 1))
    return pop


def _make_island(name: str, X: np.ndarray, F: np.ndarray) -> Island:
    return Island(
        name=name,
        algorithm=_FakeAlgo(_make_pop(X, F)),
        last_validated_individual=None,
        best_overall_individual=None,
        best_overall_cost=float("inf"),
        best_val_cost=float("inf"),
        stagnation_counter=0,
    )


def test_migrate_top_k_selection() -> None:
    # Source island A: F = [10, 1, 5, 3, 20]; top-2 are indices 1 (F=1) and 3 (F=3)
    X_a = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    F_a = np.array([10.0, 1.0, 5.0, 3.0, 20.0])
    # Destination B: F = [100, 50, 200, 300, 75]; worst-2 are indices 3 (F=300) and 2 (F=200)
    X_b = np.array([[1.1], [1.2], [1.3], [1.4], [1.5]])
    F_b = np.array([100.0, 50.0, 200.0, 300.0, 75.0])
    islands = [_make_island("A", X_a, F_a), _make_island("B", X_b, F_b)]

    rng = np.random.default_rng(42)
    events = migrate(islands, k_top=2, current_gen=10, rng=rng)

    # B receives 2 from A (no self-migration), A receives 2 from B.
    assert len(events) == 4
    a_to_b = [e for e in events if e.src_island == "A" and e.dst_island == "B"]
    assert len(a_to_b) == 2
    # B's worst slots (sorted by ascending F) are 3 then 2; migrants overwrite them.
    new_F_b = islands[1].algorithm.pop.get("F").flatten()
    # B's slots 3 and 2 should now hold the values from A's top-2 (F=1.0 and F=3.0)
    migrant_F_values = sorted([float(new_F_b[3]), float(new_F_b[2])])
    assert migrant_F_values == [1.0, 3.0]


def test_migrate_no_self_migration() -> None:
    X = np.random.default_rng(0).random((10, 4))
    F = np.arange(10, dtype=float)
    islands = [
        _make_island("PSO", X.copy(), F.copy()),
        _make_island("GA", X.copy(), F.copy()),
        _make_island("DE", X.copy(), F.copy()),
    ]
    events = migrate(islands, k_top=3, current_gen=5, rng=np.random.default_rng(0))
    for e in events:
        assert e.src_island != e.dst_island


def test_migrate_determinism_with_fixed_rng() -> None:
    def run_once() -> list[MigrationEvent]:
        X = np.linspace(0.0, 1.0, 20).reshape(5, 4)
        F = np.array([2.0, 1.0, 3.0, 5.0, 4.0])
        islands = [
            _make_island("A", X.copy(), F.copy()),
            _make_island("B", X.copy() + 0.1, F.copy() + 10.0),
            _make_island("C", X.copy() + 0.2, F.copy() + 20.0),
        ]
        return migrate(islands, k_top=2, current_gen=7, rng=np.random.default_rng(123))

    e1, e2 = run_once(), run_once()
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2, strict=True):
        assert a.src_island == b.src_island
        assert a.dst_island == b.dst_island
        assert a.slot_idx == b.slot_idx
        assert a.F_migrant == b.F_migrant


def test_migrate_three_islands_total_event_count() -> None:
    # With 3 islands and k_top=3: each destination receives top-3 from each of 2
    # sources = 6 migrants. Total events = 3 * 6 = 18.
    X = np.random.default_rng(0).random((10, 4))
    F = np.arange(10, dtype=float)
    islands = [
        _make_island("PSO", X.copy(), F.copy()),
        _make_island("GA", X.copy(), F.copy() + 100.0),
        _make_island("DE", X.copy(), F.copy() + 200.0),
    ]
    events = migrate(islands, k_top=3, current_gen=1, rng=np.random.default_rng(0))
    assert len(events) == 18


def test_migrate_logs_f_displaced() -> None:
    X = np.zeros((5, 2))
    F_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    F_b = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    islands = [_make_island("A", X.copy(), F_a), _make_island("B", X.copy(), F_b)]
    events = migrate(islands, k_top=2, current_gen=0, rng=np.random.default_rng(0))
    # Migrants A->B replaced the 2 worst of B (originally F=500, F=400).
    a_to_b = [e for e in events if e.src_island == "A" and e.dst_island == "B"]
    f_displaced_set = sorted(e.F_displaced for e in a_to_b)
    assert f_displaced_set == [400.0, 500.0]


class _UnitCubeProblem(Problem):
    """Trivial problem: f(x) = sum(x). 4 dims, [0,1] bounds, single objective."""

    def __init__(self) -> None:
        super().__init__(n_var=4, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        out["F"] = X.sum(axis=1).reshape(-1, 1)

    def evaluate_individual_per_seed(
        self, X: np.ndarray, seeds: list[int]
    ) -> np.ndarray:
        base = float(np.sum(X))
        return np.array([base + 0.01 * s for s in seeds], dtype=np.float64)


def _make_real_pso() -> PSO:
    """Construct and run-once a small pymoo PSO so V and pop[slot].X/.F are populated."""
    problem = _UnitCubeProblem()
    pso = PSO(pop_size=10, w=0.7, c1=1.5, c2=1.5)
    pso.setup(problem, seed=0)
    pso.next()  # advance one gen so V and personal-best slots exist
    return pso


def test_inject_into_pso_writes_velocity_in_range() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=3, X=X_new, F=0.42, velocity_scale=0.05, rng=rng)

    V = pso.particles.get("V")
    assert V[3].shape == (4,)
    assert np.all(np.abs(V[3]) <= 0.05)


def test_inject_into_pso_sets_pbest_to_current_position() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    X_new = np.array([0.11, 0.22, 0.33, 0.44])
    inject_into_pso(pso, slot=5, X=X_new, F=1.23, velocity_scale=0.05, rng=rng)

    np.testing.assert_array_equal(pso.pop[5].X, X_new)
    assert float(pso.pop[5].F[0]) == 1.23


def test_inject_into_pso_does_not_corrupt_other_slots() -> None:
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    V_before = pso.particles.get("V").copy()
    X_before = [pso.pop[i].X.copy() for i in range(10)]
    F_before = [pso.pop[i].F.copy() for i in range(10)]
    particles_X_before = [pso.particles[i].X.copy() for i in range(10)]
    particles_F_before = [pso.particles[i].F.copy() for i in range(10)]

    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=7, X=X_new, F=0.42, velocity_scale=0.05, rng=rng)

    V_after = pso.particles.get("V")
    for i in range(10):
        if i == 7:
            continue
        np.testing.assert_array_equal(V_before[i], V_after[i])
        np.testing.assert_array_equal(X_before[i], pso.pop[i].X)
        np.testing.assert_array_equal(F_before[i], pso.pop[i].F)
        np.testing.assert_array_equal(pso.particles[i].X, particles_X_before[i])
        np.testing.assert_array_equal(pso.particles[i].F, particles_F_before[i])


def test_inject_into_pso_velocity_seeded_rng_deterministic() -> None:
    pso = _make_real_pso()
    X_new = np.array([0.5, 0.5, 0.5, 0.5])
    inject_into_pso(pso, slot=0, X=X_new, F=0.0, velocity_scale=0.05,
                    rng=np.random.default_rng(42))
    V_first = pso.particles.get("V")[0].copy()

    pso2 = _make_real_pso()
    inject_into_pso(pso2, slot=0, X=X_new, F=0.0, velocity_scale=0.05,
                    rng=np.random.default_rng(42))
    V_second = pso2.particles.get("V")[0]
    np.testing.assert_array_equal(V_first, V_second)


def test_inject_into_pso_writes_both_particles_and_pop() -> None:
    """After inject, BOTH algorithm.particles[slot].X and algorithm.pop[slot].X must equal X_new.

    pymoo PSO has separate references for current swarm position vs personal best,
    so the rescue mechanism requires writing both.
    """
    pso = _make_real_pso()
    rng = np.random.default_rng(0)
    X_new = np.array([0.91, 0.92, 0.93, 0.94])
    inject_into_pso(pso, slot=2, X=X_new, F=0.0, velocity_scale=0.05, rng=rng)
    np.testing.assert_array_equal(pso.pop[2].X, X_new)
    np.testing.assert_array_equal(pso.particles[2].X, X_new)
    assert float(pso.pop[2].F[0]) == 0.0
    assert float(pso.particles[2].F[0]) == 0.0


class _MockProblem:
    """Stand-in for AerocaptureProblem that does deterministic per-seed eval."""

    def __init__(self, n_var: int = 4) -> None:
        self.n_var = n_var

    def evaluate_individual_per_seed(
        self, X: np.ndarray, seeds: list[int]
    ) -> np.ndarray:
        # F = sum(X) + 0.01 * seed_idx (so different islands get different rms).
        base = float(np.sum(X))
        return np.array([base + 0.01 * s for s in seeds], dtype=np.float64)

    # AerocaptureProblem also exposes these — IslandModel.__init__ may call them.
    n_obj = 1
    n_ieq_constr = 0
    n_eq_constr = 0
    xl = None
    xu = None


def _make_islands_cfg() -> OptimizerConfig:
    return OptimizerConfig(
        algorithm="islands",
        seed_strategy="fixed",
        n_pop=8,
        n_gen=5,
        training_n_sims=2,
        validation_n_sims=4,
        ga=GASettings(),
        pso=PSOSettings(),
        de=DESettings(),
        islands=IslandSettings(k_period=2, k_top=2),
    )


def test_island_model_init_creates_three_named_islands() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg,
        problem=problem,
        n_params=4,
        validation_seeds=[100, 101, 102, 103],
        final_eval_seeds=[200, 201, 202, 203],
        base_mc_seed=42,
        rng=np.random.default_rng(0),
    )
    names = [i.name for i in model.islands]
    assert names == ["pso", "ga", "de"]


def test_island_model_final_eval_picks_lowest_rms_winner() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg,
        problem=problem,
        n_params=4,
        validation_seeds=[100, 101, 102, 103],
        final_eval_seeds=[200, 201, 202, 203],
        base_mc_seed=42,
        rng=np.random.default_rng(0),
    )
    # Hand-set each island's best_overall_individual so final_eval has work to do.
    model.islands[0].best_overall_individual = np.array([0.1, 0.1, 0.1, 0.1])  # sum=0.4
    model.islands[1].best_overall_individual = np.array([0.5, 0.5, 0.5, 0.5])  # sum=2.0
    model.islands[2].best_overall_individual = np.array([0.2, 0.2, 0.2, 0.2])  # sum=0.8

    results = model.final_eval()
    assert len(results) == 3
    rms_by_island = {r["island"]: r["rms"] for r in results}
    assert rms_by_island["pso"] < rms_by_island["de"] < rms_by_island["ga"]


def test_island_model_final_eval_skips_islands_without_best() -> None:
    cfg = _make_islands_cfg()
    problem = _MockProblem(n_var=4)
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    model.islands[0].best_overall_individual = np.array([0.1, 0.2, 0.3, 0.4])
    # ga and de have no best_overall — they should be skipped.
    results = model.final_eval()
    assert len(results) == 1
    assert results[0]["island"] == "pso"


def test_island_model_step_advances_all_three_islands() -> None:
    """One step() call must invoke .next() on each island's algorithm."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()  # real pymoo problem so .next() works
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)

    model.step(current_gen=0)
    # Each island should have a populated pop.
    for island in model.islands:
        assert island.algorithm.pop is not None
        assert len(island.algorithm.pop) == cfg.n_pop


def test_island_model_step_fires_migration_at_k_period() -> None:
    cfg = _make_islands_cfg()
    cfg.islands.k_period = 2  # migrate every 2 gens
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)

    # gens 0, 1: no migration. gen 2: migration fires.
    model.step(0)
    model.step(1)
    assert len(model.migration_log) == 0
    model.step(2)
    # k_top=2 with 3 islands -> 3 * 2 * 2 = 12 events.
    assert len(model.migration_log) == 12


def test_island_model_step_disabled_migration_never_fires() -> None:
    cfg = _make_islands_cfg()
    cfg.islands.enabled = False
    cfg.islands.k_period = 1  # would migrate every gen if enabled
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    for g in range(5):
        model.step(g)
    assert model.migration_log == []


def test_validate_each_fires_only_when_argmin_changes() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    # Advance once so each island's pop has valid F.
    model.step(current_gen=0)

    # First call: every island's argmin differs from None -> all 3 validate.
    metrics = model.validate_each(current_gen=0)
    assert len(metrics) == 3
    for m in metrics:
        assert m["validated"] is True

    # Second call without changing pop: argmin unchanged -> no validation.
    metrics2 = model.validate_each(current_gen=1)
    assert all(m["validated"] is False for m in metrics2)


def test_validate_each_promotes_best_overall_on_rms_improvement() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[1, 2, 3], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    # Run first validation.
    model.validate_each(current_gen=0)
    initial_costs = [island.best_val_cost for island in model.islands]
    assert all(c < float("inf") for c in initial_costs)
    assert all(island.best_overall_individual is not None for island in model.islands)


def test_pool_top_k_across_islands_unions_populations() -> None:
    """`pool_top_k_X` returns the K lowest-F individuals from the UNION of all
    island populations (search-space-wide signal, no per-island silo)."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    pooled = model.pool_top_k_X(k=5)
    assert pooled.shape == (5, 4)
    # The pooled cost values must be monotonically <= the worst individual in any
    # single island (by definition of pool-then-rank-then-take-top-K).
    pooled_costs = np.asarray([float(np.sum(x)) for x in pooled])
    assert pooled_costs.shape == (5,)
    assert np.all(pooled_costs == np.sort(pooled_costs))


def test_re_evaluate_all_populations_updates_f() -> None:
    """re_evaluate_all_populations calls problem._run_batch directly,
    overwriting each island's pop.F with fresh values."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    # Inject a _run_batch stub that returns zeros.
    def _stub_run_batch(X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=np.float64)

    problem._run_batch = _stub_run_batch  # type: ignore[attr-defined]

    model.re_evaluate_all_populations()
    for island in model.islands:
        F = island.algorithm.pop.get("F").flatten()
        assert np.all(F == 0.0)


def test_logger_writes_island_name_field_when_provided() -> None:
    """Backwards-compatible: when island_name is provided, it appears in the JSONL record."""
    from aerocapture.training.logger import TrainingLogger

    with tempfile.TemporaryDirectory() as td:
        logger = TrainingLogger(
            scheme="islands", run=0, output_dir=Path(td), config_hash="dummy",
        )
        X = np.zeros((4, 2))
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        best = np.array([0.5, 0.5])
        logger.log_generation(
            generation=0,
            population=X,
            costs=costs,
            best_individual=best,
            decode_fn=None,
            island_name="pso",
        )
        logger.close()
        jsonl_files = list(Path(td).glob("*.jsonl"))
        assert len(jsonl_files) == 1
        record = json.loads(jsonl_files[0].read_text().strip().splitlines()[-1])
        assert record["island_name"] == "pso"


def test_logger_omits_island_name_when_not_provided() -> None:
    """Backwards-compatible: when island_name is omitted, no island_name key is emitted."""
    from aerocapture.training.logger import TrainingLogger

    with tempfile.TemporaryDirectory() as td:
        logger = TrainingLogger(
            scheme="ftc", run=0, output_dir=Path(td), config_hash="dummy",
        )
        logger.log_generation(
            generation=0,
            population=np.zeros((4, 2)),
            costs=np.array([1.0, 2.0, 3.0, 4.0]),
            best_individual=np.array([0.5, 0.5]),
            decode_fn=None,
        )
        logger.close()
        jsonl_files = list(Path(td).glob("*.jsonl"))
        record = json.loads(jsonl_files[0].read_text().strip().splitlines()[-1])
        # No island_name key when not provided (backwards compatibility).
        assert "island_name" not in record


def test_island_model_checkpoint_roundtrip() -> None:
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100, 101], final_eval_seeds=[200, 201],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)
    model.validate_each(current_gen=0)
    # Force a fake migration event into the log.
    model.migration_log.append(MigrationEvent(
        gen=1, src_island="ga", dst_island="pso", slot_idx=0,
        F_migrant=0.1, F_displaced=10.0,
    ))

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00005.npz"
        model.checkpoint(ckpt_path, generation=5)
        assert ckpt_path.exists()

        # Build a fresh model and restore from checkpoint.
        restored = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100, 101], final_eval_seeds=[200, 201],
            base_mc_seed=42, rng=np.random.default_rng(99),
        )
        for island in restored.islands:
            island.algorithm.setup(problem, seed=0)
        gen_restored, curator_state = restored.from_checkpoint(ckpt_path)
        assert gen_restored == 5
        assert curator_state is None  # we didn't pass a curator state at save time

    # Verify per-island state matches.
    for orig, rest in zip(model.islands, restored.islands, strict=True):
        assert orig.name == rest.name
        assert orig.best_val_cost == rest.best_val_cost
        assert orig.stagnation_counter == rest.stagnation_counter
        if orig.best_overall_individual is None:
            assert rest.best_overall_individual is None
        else:
            np.testing.assert_array_equal(
                orig.best_overall_individual, rest.best_overall_individual,
            )
    assert len(restored.migration_log) == 1
    assert restored.migration_log[0].src_island == "ga"


def test_island_model_resume_preserves_best_overall_per_island() -> None:
    """Regression guard: cross-gen training-cost incomparability rule must apply
    per-island. Restoring a checkpoint must NOT overwrite best_overall_* with
    the resumed population's gen-0 argmin (see project memory
    project_resume_cost_incomparability)."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    # Stamp each island's best_overall as a sentinel different from any pop member.
    sentinel = np.array([0.99, 0.99, 0.99, 0.99])
    for island in model.islands:
        island.best_overall_individual = sentinel.copy()
        island.best_val_cost = 0.123
        island.best_overall_cost = 0.456

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00010.npz"
        model.checkpoint(ckpt_path, generation=10)

        restored = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100], final_eval_seeds=[200],
            base_mc_seed=42, rng=np.random.default_rng(0),
        )
        for island in restored.islands:
            island.algorithm.setup(problem, seed=0)
        # Advance to a fresh pop with potentially-better argmin.
        restored.step(current_gen=0)
        gen_restored, _ = restored.from_checkpoint(ckpt_path)
        assert gen_restored == 10

    # The sentinel must survive across the resume; the restored model must NOT
    # have replaced it with the gen-0 argmin.
    for island in restored.islands:
        np.testing.assert_array_equal(island.best_overall_individual, sentinel)
        assert island.best_val_cost == 0.123


def test_island_model_checkpoint_roundtrips_seed_curator_state() -> None:
    """checkpoint() + from_checkpoint() round-trip the optional seed_curator_state dict."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    curator_state = {
        "sample_size": 100,
        "n_bins": 5,
        "seed_list": [1, 2, 3, 4, 5],
        "last_curation_gen": 7,
    }

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00010.npz"
        model.checkpoint(ckpt_path, generation=10, seed_curator_state=curator_state)

        restored = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100], final_eval_seeds=[200],
            base_mc_seed=42, rng=np.random.default_rng(0),
        )
        for island in restored.islands:
            island.algorithm.setup(problem, seed=0)
        gen_restored, restored_curator_state = restored.from_checkpoint(ckpt_path)

    assert gen_restored == 10
    assert restored_curator_state == curator_state


def test_from_checkpoint_raises_on_base_mc_seed_mismatch() -> None:
    """ValueError (not AssertionError) when base_mc_seed disagrees on resume."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    model = IslandModel(
        config=cfg, problem=problem, n_params=4,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=42, rng=np.random.default_rng(0),
    )
    for island in model.islands:
        island.algorithm.setup(problem, seed=0)
    model.step(current_gen=0)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_g00001.npz"
        model.checkpoint(ckpt_path, generation=1)

        wrong_seed_model = IslandModel(
            config=cfg, problem=problem, n_params=4,
            validation_seeds=[100], final_eval_seeds=[200],
            base_mc_seed=999,  # different from saved 42
            rng=np.random.default_rng(0),
        )
        for island in wrong_seed_model.islands:
            island.algorithm.setup(problem, seed=0)

        with pytest.raises(ValueError, match="base_mc_seed"):
            wrong_seed_model.from_checkpoint(ckpt_path)


def test_islands_use_warm_started_pop_array() -> None:
    """With warm-start active, the (jittered) starting population is fanned
    out to all 3 islands. Each island then sees the same X[0], same X[1], etc."""
    cfg = _make_islands_cfg()
    problem = _UnitCubeProblem()
    n_pop = cfg.n_pop
    n_params = 4
    rng = np.random.default_rng(42)
    pop_array = rng.uniform(0.0, 1.0, size=(n_pop, n_params))  # simulates jittered warm-start output

    model = IslandModel(
        config=cfg, problem=problem, n_params=n_params,
        validation_seeds=[100], final_eval_seeds=[200],
        base_mc_seed=0, rng=np.random.default_rng(0),
    )

    for island in model.islands:
        init_pop = Population.new("X", pop_array.copy())
        Evaluator().eval(problem, init_pop)
        island.algorithm.setup(problem, pop=init_pop)

    # All 3 islands' pre-`next()` X must equal pop_array.
    for island in model.islands:
        X = island.algorithm.pop.get("X")
        np.testing.assert_array_equal(X, pop_array)
