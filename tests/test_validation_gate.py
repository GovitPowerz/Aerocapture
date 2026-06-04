"""Unit tests for the shared `run_validation_gate` (H8).

The gate is the guarded gen-best selection + identity-trigger validation core
shared by the single-algorithm training loop and the islands trainer.

Behavior changes vs the pre-refactor single-algo path (bare np.argmin):
  1. All-inf population is now guarded: pop[0] is no longer validated/promoted.
  2. Mixed finite+NaN population: the gate selects the finite minimum instead of
     the NaN-indexed individual that bare np.argmin returns.  This closes a latent
     single-algo bug and aligns it with the islands trainer's already-NaN-safe path.
"""

from __future__ import annotations

import numpy as np
from aerocapture.training.evaluate import GateStatus, run_validation_gate


class _FakeProblem:
    """Minimal stand-in: returns deterministic per-seed costs keyed off the
    individual's first element, plus a dummy (n_seeds, 1) records matrix.
    Records which individual was evaluated so tests can assert no junk eval."""

    def __init__(self, val_cost: float) -> None:
        self.val_cost = val_cost
        self.evaluated: list[np.ndarray] = []

    def evaluate_individual_records_per_seed(self, x, seeds):  # type: ignore[no-untyped-def]
        self.evaluated.append(np.asarray(x).copy())
        n = len(seeds)
        costs = np.full(n, self.val_cost, dtype=np.float64)
        records = np.zeros((n, 1), dtype=np.float64)
        return costs, records


_SEEDS = [1, 2, 3]


def test_all_inf_population_is_skipped_not_promoted() -> None:
    # THE FIX: an all-inf F must not select/validate/promote pop[0].
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    F = np.array([[np.inf], [np.inf]], dtype=np.float64)
    prob = _FakeProblem(val_cost=10.0)

    result = run_validation_gate(X, F, last_validated=None, best_val_cost=1e9, problem=prob, val_seeds=_SEEDS)

    assert result.status is GateStatus.SKIP_ALL_INF
    assert result.individual is None
    assert result.promoted is False
    assert result.argmin_cost == float("inf")
    assert result.val_rms is None
    assert prob.evaluated == []  # never ran the validation MC on junk


def test_finite_population_selects_guarded_argmin_and_promotes() -> None:
    # Normal path: guarded argmin == bare argmin when a finite value exists;
    # val_rms < best_val_cost -> promoted.
    X = np.array([[0.5, 0.6], [0.1, 0.2], [0.9, 0.8]], dtype=np.float64)
    F = np.array([[5.0], [2.0], [7.0]], dtype=np.float64)
    prob = _FakeProblem(val_cost=3.0)

    result = run_validation_gate(X, F, last_validated=None, best_val_cost=1e9, problem=prob, val_seeds=_SEEDS)

    assert result.status is GateStatus.VALIDATED
    assert result.individual is not None
    np.testing.assert_array_equal(result.individual, X[1])  # the argmin row
    assert result.argmin_cost == 2.0
    assert result.val_rms == 3.0  # sqrt(mean([3,3,3]^2)) = 3
    assert result.promoted is True
    assert len(prob.evaluated) == 1


def test_finite_population_no_promotion_when_val_rms_not_better() -> None:
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    F = np.array([[2.0], [9.0]], dtype=np.float64)
    prob = _FakeProblem(val_cost=50.0)

    result = run_validation_gate(X, F, last_validated=None, best_val_cost=10.0, problem=prob, val_seeds=_SEEDS)

    assert result.status is GateStatus.VALIDATED
    assert result.val_rms == 50.0
    assert result.promoted is False  # 50 >= 10


def test_unchanged_argmin_skips_revalidation() -> None:
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    F = np.array([[2.0], [9.0]], dtype=np.float64)
    prob = _FakeProblem(val_cost=3.0)
    last = X[0].copy()  # equals the guarded argmin row

    result = run_validation_gate(X, F, last_validated=last, best_val_cost=1e9, problem=prob, val_seeds=_SEEDS)

    assert result.status is GateStatus.SKIP_UNCHANGED
    assert result.individual is not None
    np.testing.assert_array_equal(result.individual, X[0])
    assert result.argmin_cost == 2.0
    assert result.promoted is False
    assert prob.evaluated == []  # no re-validation of an unchanged individual


def test_partial_inf_picks_finite_minimum() -> None:
    # Guarded selection ignores +inf rows.  For a pure +inf/finite mix (no NaN),
    # the result equals bare argmin -- both pick index 1.  The NaN case diverges
    # from bare argmin and is covered by test_mixed_finite_and_nan_picks_finite_minimum.
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    F = np.array([[np.inf], [4.0], [np.inf]], dtype=np.float64)
    prob = _FakeProblem(val_cost=1.0)

    result = run_validation_gate(X, F, last_validated=None, best_val_cost=1e9, problem=prob, val_seeds=_SEEDS)

    assert result.status is GateStatus.VALIDATED
    np.testing.assert_array_equal(result.individual, X[1])
    assert result.argmin_cost == 4.0


def test_F_accepts_2d_column_shape() -> None:
    # pop.get("F") is (n_pop, 1); the gate must reshape it like np.argmin(costs).
    X = np.array([[0.1], [0.2]], dtype=np.float64)
    F_col = np.array([[3.0], [1.0]], dtype=np.float64)
    F_flat = np.array([3.0, 1.0], dtype=np.float64)
    prob_a = _FakeProblem(val_cost=2.0)
    prob_b = _FakeProblem(val_cost=2.0)

    r_col = run_validation_gate(X, F_col, None, 1e9, prob_a, _SEEDS)
    r_flat = run_validation_gate(X, F_flat, None, 1e9, prob_b, _SEEDS)

    assert r_col.status is r_flat.status is GateStatus.VALIDATED
    np.testing.assert_array_equal(r_col.individual, r_flat.individual)
    assert r_col.argmin_cost == r_flat.argmin_cost == 1.0


def test_mixed_finite_and_nan_picks_finite_minimum() -> None:
    # NaN costs can occur when a Rust sim leaks a NaN state (e.g. no sim_timeout_secs).
    # The gate's nanargmin selects the finite minimum, NOT the NaN-indexed individual
    # that bare np.argmin would have returned.
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    F = np.array([[0.5], [np.nan], [0.3], [0.8]], dtype=np.float64)
    prob = _FakeProblem(val_cost=0.1)

    result = run_validation_gate(X, F, last_validated=None, best_val_cost=1e9, problem=prob, val_seeds=_SEEDS)

    # Gate must select the finite minimum (0.3 at index 2), not the NaN at index 1.
    assert result.status is GateStatus.VALIDATED
    np.testing.assert_array_equal(result.individual, X[2])
    assert result.argmin_cost == 0.3
    assert result.promoted is True

    # Document the divergence: bare np.argmin returns the NaN index (1).
    bare_argmin_idx = int(np.argmin(F.reshape(-1)))
    assert bare_argmin_idx == 1  # NaN propagation makes np.argmin pick the NaN row
    assert result.individual is not None
    assert not np.array_equal(result.individual, X[bare_argmin_idx])
