"""Unit tests for the 3-island PSO/GA/DE evolutionary trainer.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.island_model import Island, MigrationEvent, migrate
from aerocapture.training.optimizer import IslandSettings, OptimizerConfig
from pymoo.core.population import Population


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
