"""Feasibility-aware validation gate + final selection (IMPROVEMENTS 9.14)."""

import numpy as np
import pytest
from aerocapture.training import charts
from aerocapture.training.evaluate import (
    GateStatus,
    constraint_violation_rates,
    run_validation_gate,
)
from aerocapture.training.final_select import KnownCandidate, select_final_individual

COST_KWARGS = {"heat_flux_limit": 200.0, "g_load_limit": 4.0, "heat_load_limit": 25000.0}


def _records(n: int, n_flux_viol: int = 0, n_g_viol: int = 0, n_hl_viol: int = 0) -> np.ndarray:
    """Synthetic (n, 52) final records with the requested violation counts."""
    fr = np.zeros((n, 52))
    fr[:, charts._FR_MAX_HEAT_FLUX] = 150.0  # kW/m^2, under the 200 limit
    fr[:, charts._FR_MAX_G_LOAD] = 3.0  # g, under 4
    fr[:, charts._FR_INTEGRATED_FLUX] = 20.0  # MJ/m^2 -> 20000 kJ, under 25000
    fr[:n_flux_viol, charts._FR_MAX_HEAT_FLUX] = 220.0
    fr[:n_g_viol, charts._FR_MAX_G_LOAD] = 4.5
    fr[:n_hl_viol, charts._FR_INTEGRATED_FLUX] = 26.0
    return fr


class TestViolationRates:
    def test_counts_each_constraint(self) -> None:
        rates = constraint_violation_rates(_records(10, n_flux_viol=1, n_g_viol=2, n_hl_viol=3), COST_KWARGS)
        assert rates == {"heat_flux": 0.1, "g_load": 0.2, "heat_load": 0.3}

    def test_heat_load_unit_conversion(self) -> None:
        """Records carry MJ/m^2; the limit is in kJ/m^2 (read_cost_kwargs contract)."""
        fr = _records(4)
        fr[:, charts._FR_INTEGRATED_FLUX] = 25.1  # MJ -> 25100 kJ > 25000
        rates = constraint_violation_rates(fr, COST_KWARGS)
        assert rates is not None and rates["heat_load"] == 1.0

    def test_no_limits_returns_none(self) -> None:
        assert constraint_violation_rates(_records(5), {}) is None
        assert constraint_violation_rates(_records(5), None) is None

    def test_partial_limits(self) -> None:
        rates = constraint_violation_rates(_records(10, n_flux_viol=5), {"heat_flux_limit": 200.0})
        assert rates == {"heat_flux": 0.5}


class _FakeProblem:
    """Per-seed evaluator returning canned costs + records."""

    def __init__(self, costs: np.ndarray, records: np.ndarray) -> None:
        self._costs = costs
        self._records = records

    def evaluate_individual_records_per_seed(self, x: np.ndarray, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        return self._costs, self._records

    def evaluate_population_records_per_seed(self, X: np.ndarray, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        return np.tile(self._costs, (n, 1)), np.tile(self._records[None], (n, 1, 1))


class TestGateFeasibility:
    X = np.array([[0.3, 0.7]])
    F = np.array([100.0])
    SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def _gate(self, records: np.ndarray, ceiling: float, best_val: float = 1e9):  # type: ignore[no-untyped-def]
        problem = _FakeProblem(np.full(10, 50.0), records)
        return run_validation_gate(
            self.X,
            self.F,
            None,
            best_val,
            problem,
            self.SEEDS,
            max_violation_rate=ceiling,
            cost_kwargs=COST_KWARGS,
        )

    def test_infeasible_blocks_promotion_despite_better_rms(self) -> None:
        gate = self._gate(_records(10, n_hl_viol=2), ceiling=0.0)
        assert gate.status is GateStatus.VALIDATED
        assert gate.val_rms is not None and gate.val_rms < 1e9
        assert not gate.feasible
        assert not gate.promoted
        assert gate.violation_rates == {"heat_flux": 0.0, "g_load": 0.0, "heat_load": 0.2}

    def test_feasible_promotes(self) -> None:
        gate = self._gate(_records(10), ceiling=0.0)
        assert gate.feasible and gate.promoted

    def test_ceiling_allows_rate_at_or_below(self) -> None:
        gate = self._gate(_records(10, n_flux_viol=1), ceiling=0.1)
        assert gate.feasible and gate.promoted
        gate = self._gate(_records(10, n_flux_viol=2), ceiling=0.1)
        assert not gate.feasible

    def test_no_cost_kwargs_means_no_check(self) -> None:
        problem = _FakeProblem(np.full(10, 50.0), _records(10, n_flux_viol=10))
        gate = run_validation_gate(self.X, self.F, None, 1e9, problem, self.SEEDS)
        assert gate.feasible and gate.promoted and gate.violation_rates is None

    def test_default_ceiling_is_strict_zero(self) -> None:
        problem = _FakeProblem(np.full(10, 50.0), _records(10, n_g_viol=1))
        gate = run_validation_gate(self.X, self.F, None, 1e9, problem, self.SEEDS, cost_kwargs=COST_KWARGS)
        assert not gate.feasible and not gate.promoted


class TestFinalSelectionFeasibility:
    SEEDS = list(range(10))

    def test_infeasible_fresh_cannot_displace_champion(self) -> None:
        champion = KnownCandidate(x=np.array([0.5, 0.5]), provenance="champion", val_rms=60.0)
        problem = _FakeProblem(np.full(10, 50.0), _records(10, n_hl_viol=3))  # fresh rms 50 < 60 but infeasible
        sel = select_final_individual(
            problem,
            np.array([[0.1, 0.9]]),
            ["last_gen[0]"],
            [champion],
            self.SEEDS,
            max_violation_rate=0.0,
            cost_kwargs=COST_KWARGS,
        )
        assert sel.provenance == "champion"
        assert not sel.promoted
        assert sel.winner_feasible
        fresh = [r for r in sel.candidate_rms if r["provenance"] == "last_gen[0]"][0]
        assert fresh["feasible"] is False

    def test_feasible_fresh_wins(self) -> None:
        champion = KnownCandidate(x=np.array([0.5, 0.5]), provenance="champion", val_rms=60.0)
        problem = _FakeProblem(np.full(10, 50.0), _records(10))
        sel = select_final_individual(
            problem,
            np.array([[0.1, 0.9]]),
            ["last_gen[0]"],
            [champion],
            self.SEEDS,
            max_violation_rate=0.0,
            cost_kwargs=COST_KWARGS,
        )
        assert sel.promoted and sel.provenance == "last_gen[0]" and sel.winner_feasible

    def test_no_champion_all_infeasible_falls_back_with_flag(self) -> None:
        problem = _FakeProblem(np.full(10, 50.0), _records(10, n_flux_viol=5))
        sel = select_final_individual(
            problem,
            np.array([[0.1, 0.9], [0.2, 0.8]]),
            ["last_gen[0]", "last_gen[1]"],
            [],
            self.SEEDS,
            max_violation_rate=0.0,
            cost_kwargs=COST_KWARGS,
        )
        assert sel.promoted and not sel.winner_feasible


class TestOptimizerConfigKnob:
    def test_parses_and_validates(self) -> None:
        from aerocapture.training.optimizer import OptimizerConfig

        cfg = OptimizerConfig.from_dict({"seed_strategy": "fixed", "max_violation_rate": 0.02})
        assert cfg.max_violation_rate == 0.02
        assert OptimizerConfig.from_dict({"seed_strategy": "fixed"}).max_violation_rate == 0.0
        with pytest.raises(ValueError, match="max_violation_rate"):
            OptimizerConfig.from_dict({"seed_strategy": "fixed", "max_violation_rate": 1.5})
