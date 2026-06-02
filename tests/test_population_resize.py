import numpy as np
from aerocapture.training.population import resize_population


def test_equal_target_is_identity() -> None:
    rng = np.random.default_rng(0)
    pop = rng.random((5, 3))
    out = resize_population(pop, np.arange(5.0), 5, rng)
    assert np.array_equal(out, pop)


def test_grow_preserves_resumed_individuals_verbatim() -> None:
    rng = np.random.default_rng(1)
    pop = rng.random((4, 3))
    out = resize_population(pop, np.arange(4.0), 10, rng, fresh_fraction=0.2)
    assert out.shape == (10, 3)
    assert np.array_equal(out[:4], pop)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_grow_fresh_fraction_split() -> None:
    rng = np.random.default_rng(2)
    pop = np.full((10, 2), 0.5)
    # 10 new slots, fresh_fraction 0.2 -> round(0.2*10)=2 fresh, 8 clone+jitter.
    out = resize_population(pop, np.zeros(10), 20, rng, fresh_fraction=0.2, jitter_sigma=0.02)
    new = out[10:]
    # 5*jitter_sigma = 0.1: clones stay inside this band, fresh-randoms spread wider.
    near = np.abs(new - 0.5).max(axis=1) < 0.1
    assert near.sum() == 8
    assert (~near).sum() == 2


def test_shrink_keeps_best_by_cost() -> None:
    rng = np.random.default_rng(3)
    pop = rng.random((6, 2))
    costs = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 0.0])
    out = resize_population(pop, costs, 3, rng)
    assert out.shape == (3, 2)
    # Best 3 by cost are indices 5 (0.0), 1 (1.0), 3 (2.0).
    expected = pop[[5, 1, 3]]
    assert np.array_equal(out, expected)


def test_shrink_none_costs_keeps_first_n() -> None:
    rng = np.random.default_rng(4)
    pop = rng.random((6, 2))
    out = resize_population(pop, None, 3, rng)
    assert np.array_equal(out, pop[:3])
