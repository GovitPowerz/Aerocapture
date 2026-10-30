"""The champion record (#221): the five transitions, both wire maps, the legacy
checkpoint cases, and `revalidate` (the block the prologue, `revalidate_each` and
the retro CLI used to copy). Pure numpy, no simulator."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training import charts
from aerocapture.training.champion import Champion, KnownCandidate, revalidate
from aerocapture.training.evaluate import GateResult, GateStatus

COST_KWARGS = {"heat_flux_limit": 200.0, "g_load_limit": 4.0, "heat_load_limit": 25000.0}
X = np.array([0.3, 0.7])


def _records(n: int, n_hl_viol: int = 0) -> np.ndarray:
    fr = np.zeros((n, 52))
    fr[:, charts._FR_MAX_HEAT_FLUX] = 150.0
    fr[:, charts._FR_MAX_G_LOAD] = 3.0
    fr[:, charts._FR_INTEGRATED_FLUX] = 20.0
    fr[:n_hl_viol, charts._FR_INTEGRATED_FLUX] = 26.0
    return fr


def _validated(individual: np.ndarray, val_rms: float, *, feasible: bool, champion_rms: float) -> GateResult:
    return GateResult(
        status=GateStatus.VALIDATED,
        argmin_cost=12.5,
        individual=individual,
        val_costs=np.full(3, val_rms),
        val_records=_records(3),
        val_rms=val_rms,
        promoted=(val_rms < champion_rms) and feasible,
        feasible=feasible,
        violation_rates={"heat_load": 0.0 if feasible else 0.5},
    )


class _FakeProblem:
    def __init__(self, costs: np.ndarray, records: np.ndarray) -> None:
        self._costs, self._records = costs, records

    def evaluate_individual_records_per_seed(self, x: np.ndarray, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        return self._costs, self._records


class TestTrusted:
    def test_empty_and_untrusted_and_trusted(self) -> None:
        assert not Champion().trusted
        assert not Champion(individual=X).trusted  # inf val_rms: untrusted
        assert Champion(individual=X, val_rms=50.0).trusted


class TestConsider:
    def test_feasible_strictly_lower_rms_displaces(self) -> None:
        c = Champion(individual=X, train_cost=99.0, val_rms=60.0)
        cand = np.array([0.1, 0.2])
        assert c.consider(_validated(cand, 50.0, feasible=True, champion_rms=c.val_rms)) is True
        assert c.individual is not None and np.array_equal(c.individual, cand) and c.individual is not cand
        assert c.val_rms == 50.0 and c.train_cost == 12.5
        assert c.last_validated is cand
        assert c.trusted

    def test_tie_keeps_incumbent_but_marks_last_validated(self) -> None:
        c = Champion(individual=X, train_cost=99.0, val_rms=50.0)
        cand = np.array([0.1, 0.2])
        assert c.consider(_validated(cand, 50.0, feasible=True, champion_rms=c.val_rms)) is False
        assert c.individual is X and c.val_rms == 50.0 and c.train_cost == 99.0
        assert c.last_validated is cand  # the gate's identity trigger moves even without promotion

    def test_infeasible_never_displaces(self) -> None:
        c = Champion(individual=X, val_rms=60.0)
        assert c.consider(_validated(np.array([0.1, 0.2]), 1.0, feasible=False, champion_rms=c.val_rms)) is False
        assert c.individual is X and c.val_rms == 60.0

    @pytest.mark.parametrize("status", [GateStatus.SKIP_ALL_INF, GateStatus.SKIP_UNCHANGED])
    def test_skips_are_no_ops(self, status: GateStatus) -> None:
        c = Champion(individual=X, val_rms=60.0, last_validated=X)
        assert c.consider(GateResult(status=status, argmin_cost=1.0, individual=np.array([0.9, 0.9]))) is False
        assert c.last_validated is X and c.val_rms == 60.0


class TestReanchor:
    def test_feasible_anchors_rms_and_last_validated(self) -> None:
        c = Champion(individual=X, train_cost=99.0)
        assert c.reanchor(42.0, {"heat_load": 0.0}, ceiling=0.0) is True
        assert c.val_rms == 42.0 and c.trusted and c.train_cost == 99.0
        assert c.last_validated is not None and np.array_equal(c.last_validated, X) and c.last_validated is not X

    def test_infeasible_sets_inf_but_keeps_individual_as_last_validated(self) -> None:
        c = Champion(individual=X, val_rms=42.0)
        assert c.reanchor(10.0, {"heat_load": 0.1}, ceiling=0.0) is False
        assert c.val_rms == float("inf") and not c.trusted and c.individual is X
        assert c.last_validated is not None and np.array_equal(c.last_validated, X)
        assert c.violation_rates == {"heat_load": 0.1}

    def test_ceiling_admits_rate_at_or_below(self) -> None:
        c = Champion(individual=X)
        assert c.reanchor(10.0, {"heat_load": 0.1}, ceiling=0.1) is True

    def test_non_finite_rms_never_anchors(self) -> None:
        """A NaN anchor would block every later promotion (`x < nan` is False)."""
        c = Champion(individual=X, val_rms=42.0)
        assert c.reanchor(float("nan"), None, ceiling=0.0) is True  # the verdict is feasibility
        assert c.val_rms == float("inf") and not c.trusted
        assert c.consider(_validated(np.array([0.1, 0.2]), 50.0, feasible=True, champion_rms=c.val_rms)) is True


class TestAdoptUnvalidated:
    def test_untrusted_champion(self) -> None:
        c = Champion(individual=X, train_cost=1.0, val_rms=5.0, violation_rates={"heat_load": 0.0})
        cand = np.array([0.1, 0.2])
        c.adopt_unvalidated(cand, 0.5)
        assert c.individual is cand and c.train_cost == 0.5
        assert c.val_rms == float("inf") and c.violation_rates is None and not c.trusted


class TestAdoptSelection:
    def test_with_train_cost(self) -> None:
        c = Champion(individual=X, train_cost=99.0, val_rms=60.0, last_validated=X)
        cand = np.array([0.1, 0.2])
        c.adopt_selection(cand, 40.0, 7.0, feasible=True)
        assert c.individual is not None and np.array_equal(c.individual, cand) and c.individual is not cand
        assert c.val_rms == 40.0 and c.train_cost == 7.0 and c.last_validated is X

    def test_none_keeps_recorded_train_cost(self) -> None:
        c = Champion(individual=X, train_cost=99.0, val_rms=60.0)
        c.adopt_selection(np.array([0.1, 0.2]), 40.0, None, feasible=True)
        assert c.train_cost == 99.0 and c.val_rms == 40.0

    def test_infeasible_fallback_is_untrusted(self) -> None:
        """The no-champion all-infeasible fallback deploys but must not become a trusted bar."""
        c = Champion(individual=X, train_cost=99.0, val_rms=60.0, violation_rates={"heat_load": 0.0})
        c.adopt_selection(np.array([0.1, 0.2]), 40.0, 7.0, feasible=False)
        assert c.individual is not None and np.array_equal(c.individual, [0.1, 0.2]) and c.train_cost == 7.0
        assert c.val_rms == float("inf") and not c.trusted and c.violation_rates is None
        assert c.single_meta()["best_val_cost"] == float("inf")


class TestPresentation:
    def test_trusted_is_known_not_a_row(self) -> None:
        c = Champion(individual=X, val_rms=50.0)
        known = c.known()
        assert isinstance(known, KnownCandidate) and np.array_equal(known.x, X) and known.provenance == "champion" and known.val_rms == 50.0
        assert c.known("ga:") is not None and c.known("ga:").provenance == "ga:champion"  # type: ignore[union-attr]
        assert c.untrusted_row() is None

    def test_untrusted_is_a_row_not_known(self) -> None:
        c = Champion(individual=X)
        assert c.known() is None
        row = c.untrusted_row("pso:")
        assert row is not None and np.array_equal(row[0], X) and row[1] == "pso:champion[infeasible]"

    def test_empty_is_neither(self) -> None:
        assert Champion().known() is None and Champion().untrusted_row() is None


class TestWireFormat:
    def test_single_round_trip(self) -> None:
        c = Champion(individual=X, train_cost=3.0, val_rms=2.0, last_validated=np.array([0.9, 0.9]))
        assert c.single_meta() == {"best_cost": 3.0, "best_val_cost": 2.0}
        back = Champion.from_single(c.single_meta(), c.individual)
        assert back.individual is X and back.train_cost == 3.0 and back.val_rms == 2.0 and back.trusted
        # The single format does not persist the last-validated individual: it is seeded from the champion.
        assert back.last_validated is not None and np.array_equal(back.last_validated, X) and back.last_validated is not X

    def test_single_legacy_cases(self) -> None:
        assert Champion.from_single({"best_cost": 1.0, "best_val_cost": 2.5}, X).trusted
        untrusted = Champion.from_single({"best_cost": 1.0, "best_val_cost": float("inf")}, X)
        assert untrusted.individual is X and not untrusted.trusted
        no_val_key = Champion.from_single({"best_cost": 1.0}, X)  # pre-validation-gate json
        assert no_val_key.train_cost == 1.0 and not no_val_key.trusted
        empty = Champion.from_single({"best_cost": 1.0, "best_val_cost": 2.5}, None)  # no best_individual array
        assert empty.individual is None and empty.last_validated is None and not empty.trusted
        assert empty.single_meta()["best_cost"] == 1.0  # a write-back (patch_checkpoint) keeps the stored training cost

    def test_islands_round_trip_keeps_key_order(self) -> None:
        last = np.array([0.9, 0.9])
        c = Champion(individual=X, train_cost=3.0, val_rms=2.0, last_validated=last)
        state = c.island_state()
        assert list(state) == ["last_validated_individual", "best_overall_individual", "best_overall_cost", "best_val_cost"]
        back = Champion.from_island_state({"name": "ga", "pop_X": None, **state, "stagnation_counter": 2})
        assert back.individual is X and back.last_validated is last and back.train_cost == 3.0 and back.val_rms == 2.0

    def test_islands_legacy_cases(self) -> None:
        trusted = Champion.from_island_state({"best_overall_individual": X, "best_val_cost": 2.0})  # no cost / last-validated keys
        assert trusted.trusted and trusted.train_cost == float("inf") and trusted.last_validated is None
        assert not Champion.from_island_state({"best_overall_individual": X, "best_val_cost": float("inf")}).trusted
        assert Champion.from_island_state({"best_overall_individual": None, "best_val_cost": 2.0}).individual is None


class TestRevalidate:
    def test_feasible_anchors_and_returns_the_arrays(self, capsys: pytest.CaptureFixture[str]) -> None:
        c = Champion(individual=X, train_cost=9.0)
        costs, records, feasible = revalidate(c, _FakeProblem(np.full(10, 50.0), _records(10)), list(range(10)), COST_KWARGS, 0.0, verbose=True)
        assert feasible and c.val_rms == 50.0 and c.trusted and c.train_cost == 9.0
        assert c.violation_rates == {"heat_flux": 0.0, "g_load": 0.0, "heat_load": 0.0}
        assert np.array_equal(costs, np.full(10, 50.0)) and records.shape == (10, 52)
        assert c.last_validated is not None and np.array_equal(c.last_validated, X)
        assert capsys.readouterr().out == ""

    def test_infeasible_is_untrusted_and_announced(self, capsys: pytest.CaptureFixture[str]) -> None:
        c = Champion(individual=X, val_rms=0.1)
        problem = _FakeProblem(np.full(10, 50.0), _records(10, n_hl_viol=1))
        _, _, feasible = revalidate(c, problem, list(range(10)), COST_KWARGS, 0.0, verbose=True, label="pso:")
        assert not feasible and c.val_rms == float("inf") and not c.trusted
        assert c.violation_rates == {"heat_flux": 0.0, "g_load": 0.0, "heat_load": 0.1}
        assert c.last_validated is not None and np.array_equal(c.last_validated, X)
        out = capsys.readouterr().out
        assert "pso:champion" in out and "INFEASIBLE" in out and "heat_load=10.000%" in out
        assert c.untrusted_row("pso:") is not None

    def test_quiet_without_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        revalidate(Champion(individual=X), _FakeProblem(np.full(10, 50.0), _records(10, n_hl_viol=1)), list(range(10)), COST_KWARGS, 0.0, verbose=False)
        assert capsys.readouterr().out == ""

    def test_no_limits_means_feasible(self) -> None:
        c = Champion(individual=X)
        revalidate(c, _FakeProblem(np.full(10, 50.0), _records(10, n_hl_viol=10)), list(range(10)), None, 0.0, verbose=False)
        assert c.trusted and c.violation_rates is None
