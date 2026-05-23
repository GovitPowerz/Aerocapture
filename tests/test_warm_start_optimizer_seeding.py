"""Per-algorithm warm-start seeding contract: GA/DE/PSO replicate+jitter,
CMA-ES seeds mean (no jitter)."""

import numpy as np
import pytest
from aerocapture.training.train import _seed_initial_population


def test_ga_replicate_and_jitter() -> None:
    chromo = np.full(50, 0.5)
    n_pop = 30
    rng = np.random.default_rng(0)
    pop = _seed_initial_population(
        algorithm_name="ga",
        chromosome=chromo,
        n_pop=n_pop,
        jitter=0.02,
        rng=rng,
    )
    assert pop.shape == (n_pop, 50)
    assert pop.mean(axis=0) == pytest.approx(0.5, abs=0.01)
    assert pop.std(axis=0).mean() == pytest.approx(0.02, abs=0.005)
    assert (pop >= 0.0).all() and (pop <= 1.0).all()


def test_cma_es_singleton_seeded() -> None:
    """CMA-ES seeding tiles the chromosome without jitter; pymoo uses the
    population mean as initial mean. sigma0 applied separately via OptimizerConfig."""
    chromo = np.full(50, 0.5)
    rng = np.random.default_rng(0)
    pop = _seed_initial_population(
        algorithm_name="cma_es",
        chromosome=chromo,
        n_pop=20,
        jitter=0.02,
        rng=rng,
    )
    assert pop.shape[0] >= 1
    # All rows equal the chromosome -- no jitter for CMA-ES.
    assert np.allclose(pop[0], chromo)
    assert np.allclose(pop[-1], chromo)


def test_de_and_pso_match_ga_contract() -> None:
    chromo = np.full(20, 0.7)
    n_pop = 10
    for algo in ("de", "pso"):
        rng = np.random.default_rng(0)
        pop = _seed_initial_population(algo, chromo, n_pop, jitter=0.02, rng=rng)
        assert pop.shape == (n_pop, 20)
        assert pop.mean(axis=0) == pytest.approx(0.7, abs=0.02)


def test_n_weights_kwarg_restricts_jitter_to_weight_slab() -> None:
    """When n_weights < chromosome.size, jitter only applies to the first n_weights
    columns (scaffolding tail is left intact for the caller to overwrite)."""
    chromo = np.full(50, 0.5)
    chromo[40:] = 0.9  # last 10 are scaffolding
    rng = np.random.default_rng(0)
    pop = _seed_initial_population(
        algorithm_name="ga",
        chromosome=chromo,
        n_pop=20,
        jitter=0.05,
        rng=rng,
        n_weights=40,
    )
    # First 40 columns jittered
    assert pop[:, :40].std(axis=0).mean() == pytest.approx(0.05, abs=0.01)
    # Last 10 columns unchanged
    assert np.allclose(pop[:, 40:], 0.9)


def test_unknown_algorithm_raises() -> None:
    chromo = np.full(10, 0.5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown algorithm"):
        _seed_initial_population("nonexistent", chromo, 5, jitter=0.02, rng=rng)
