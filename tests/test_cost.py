"""Tests for compute_cost edge cases and invariants.

Column layout of final_conditions (0-indexed):
    8  = energy (MJ/kg), >0 → hyperbolic
    10 = eccentricity, >1 → hyperbolic
    28 = sim_time (s)
    30 = periapsis_err (km)
    31 = apoapsis_err (km)
    42 = dv_total (m/s)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.evaluate import compute_cost
from hypothesis import given, settings
from hypothesis import strategies as st

N_COLS = 53


def _make_row(
    *,
    energy: float = -1.0,
    ecc: float = 0.5,
    sim_time: float = 300.0,
    peri_err: float = 0.0,
    apo_err: float = 0.0,
    dv_total: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Build a single-row final_conditions array with the given values."""
    row = np.zeros((1, N_COLS))
    row[0, 8] = energy
    row[0, 10] = ecc
    row[0, 28] = sim_time
    row[0, 30] = peri_err
    row[0, 31] = apo_err
    row[0, 42] = dv_total
    return row


class TestCostEdgeCases:
    def test_nan_energy_returns_finite_cost(self) -> None:
        """NaN energy/ecc: NaN comparisons are False, so orbit falls into the captured path.

        Both (nan > 1.0) and (nan > 0) evaluate to False in numpy, so the row is
        classified as captured with zero orbit error → cost=0. Key invariant: the
        result must be finite and non-negative regardless of NaN inputs.
        """
        row = _make_row(energy=float("nan"), ecc=float("nan"))
        cost = compute_cost(row)
        assert np.isfinite(cost), "cost must be finite even for NaN inputs"
        assert cost >= 0.0, "cost must be non-negative even for NaN inputs"

    def test_parabolic_energy_zero_classified_as_captured(self) -> None:
        """Energy=0 and ecc=1 both fail the strict > comparisons.

        compute_cost uses (ecc > 1.0) | (energy > 0) — neither is satisfied at the
        boundary, so the row is classified as captured (orbit_err=0, dv=0) → cost=0.
        """
        row = _make_row(energy=0.0, ecc=1.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12), f"Parabolic boundary values classified as captured with zero errors, expected cost≈0, got {cost}"

    def test_strictly_hyperbolic_penalized(self) -> None:
        """energy > 0 AND ecc > 1 → Level 0 penalty well above 1e6."""
        row = _make_row(energy=5.0, ecc=2.0)
        cost = compute_cost(row)
        assert cost > 1e6, f"Hyperbolic escape should be penalized above 1e6, got {cost}"

    def test_captured_orbit_reasonable_cost(self) -> None:
        """Negative energy + ecc<1 + small errors → cost below 1e4."""
        row = _make_row(energy=-1.5, ecc=0.3, peri_err=2.0, apo_err=3.0, dv_total=50.0)
        cost = compute_cost(row)
        assert cost < 1e4, f"Captured orbit with small errors should have low cost, got {cost}"

    def test_cost_always_nonnegative(self) -> None:
        """Cost must always be non-negative regardless of input."""
        rows = [
            _make_row(energy=-5.0, ecc=0.1, peri_err=-50.0, apo_err=-50.0),
            _make_row(energy=10.0, ecc=2.0),
            _make_row(energy=-0.01, ecc=0.99, dv_total=1e30),
        ]
        for row in rows:
            cost = compute_cost(row)
            assert cost >= 0.0, f"cost={cost} is negative for row {row}"

    def test_multi_sim_rms_equals_single_row(self) -> None:
        """Stacking identical rows produces the same cost as a single row."""
        row = _make_row(energy=-2.0, ecc=0.4, peri_err=5.0, apo_err=8.0, dv_total=100.0)
        stacked = np.tile(row, (5, 1))

        cost_single = compute_cost(row)
        cost_multi = compute_cost(stacked)

        assert abs(cost_single - cost_multi) < 1e-9, f"cost mismatch: single={cost_single}, multi={cost_multi}"

    def test_perfect_orbit_has_low_cost(self) -> None:
        """Zero orbital errors and zero dv → cost is exactly 0."""
        row = _make_row(energy=-2.0, ecc=0.4, peri_err=0.0, apo_err=0.0, dv_total=0.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_hyperbolic_higher_cost_than_captured(self) -> None:
        """Hyperbolic escape should always cost more than a well-captured orbit."""
        hyperbolic = _make_row(energy=1.0, ecc=1.5)
        captured = _make_row(energy=-1.0, ecc=0.5, peri_err=5.0, apo_err=5.0)
        assert compute_cost(hyperbolic) > compute_cost(captured)


class TestCostProperties:
    @given(
        energy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        ecc=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
        peri_err=st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
        apo_err=st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
        dv_total=st.floats(0.0, 1e4, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_cost_always_finite_for_finite_inputs(
        self,
        energy: float,
        ecc: float,
        peri_err: float,
        apo_err: float,
        dv_total: float,
    ) -> None:
        """For any finite inputs, compute_cost must return a finite, non-negative value."""
        row = _make_row(energy=energy, ecc=ecc, peri_err=peri_err, apo_err=apo_err, dv_total=dv_total)
        cost = compute_cost(row)
        assert np.isfinite(cost), f"cost is not finite: {cost}"
        assert cost >= 0.0, f"cost is negative: {cost}"
