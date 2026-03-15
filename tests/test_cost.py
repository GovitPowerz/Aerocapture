"""Tests for compute_cost: delta-V primary with normalized constraint penalties.

Column layout of final_conditions (0-indexed, 52-column):
    7  = energy (MJ/kg), >0 → hyperbolic
    9  = eccentricity, >1 → hyperbolic
    16 = max heat flux (kW/m²)
    17 = max g-load (g)
    27 = sim_time (s)
    41 = dv_total (m/s)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.evaluate import compute_cost
from hypothesis import given, settings
from hypothesis import strategies as st

N_COLS = 52


def _make_row(
    *,
    energy: float = -1.0,
    ecc: float = 0.5,
    sim_time: float = 300.0,
    dv_total: float = 0.0,
    g_max: float = 0.0,
    q_max: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Build a single-row final_conditions array with the given values."""
    row = np.zeros((1, N_COLS))
    row[0, 7] = energy
    row[0, 9] = ecc
    row[0, 16] = q_max
    row[0, 17] = g_max
    row[0, 27] = sim_time
    row[0, 41] = dv_total
    return row


class TestCostDeltaVPrimary:
    def test_zero_dv_zero_cost(self) -> None:
        """Captured with zero delta-V and no constraint violations → cost = 0."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12)

    def test_dv_is_primary_cost(self) -> None:
        """For captured trajectory, cost ≈ delta-V when no constraints violated."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=150.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(150.0, abs=1e-6)

    def test_dv_clipped_at_10000(self) -> None:
        """Delta-V above 10000 m/s is clipped."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=50000.0)
        cost = compute_cost(row)
        assert cost <= 10001.0

    def test_bogus_dv_treated_as_noncapture(self) -> None:
        """dv_total > 1e10 (bogus Fortran value) → non-capture penalty path."""
        row = _make_row(energy=-1.0, ecc=0.5, dv_total=1e30)
        cost = compute_cost(row)
        assert cost > 1e6, f"Bogus dv should trigger non-capture penalty, got {cost}"


class TestCostConstraintPenalties:
    def test_gload_below_limit_no_penalty(self) -> None:
        """G-load below limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, g_max=14.0)
        cost = compute_cost(row, g_load_limit=15.0)
        assert cost == pytest.approx(100.0, abs=1e-6)

    def test_gload_at_limit_no_penalty(self) -> None:
        """G-load exactly at limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, g_max=15.0)
        cost = compute_cost(row, g_load_limit=15.0)
        assert cost == pytest.approx(100.0, abs=1e-9)

    def test_gload_above_limit_adds_penalty(self) -> None:
        """G-load above limit adds quadratic normalized penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, g_max=16.5)
        cost_with = compute_cost(row, g_load_limit=15.0, g_load_weight=1000.0)
        cost_without = compute_cost(row, g_load_limit=15.0, g_load_weight=0.0)
        assert cost_with > cost_without

    def test_heat_flux_below_limit_no_penalty(self) -> None:
        """Heat flux below limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, q_max=180.0)
        cost = compute_cost(row, heat_flux_limit=200.0)
        assert cost == pytest.approx(100.0, abs=1e-6)

    def test_heat_flux_at_limit_no_penalty(self) -> None:
        """Heat flux exactly at limit contributes zero penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, q_max=200.0)
        cost = compute_cost(row, heat_flux_limit=200.0)
        assert cost == pytest.approx(100.0, abs=1e-9)

    def test_heat_flux_above_limit_adds_penalty(self) -> None:
        """Heat flux above limit adds quadratic normalized penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0, q_max=250.0)
        cost_with = compute_cost(row, heat_flux_limit=200.0, heat_flux_weight=1000.0)
        cost_without = compute_cost(row, heat_flux_limit=200.0, heat_flux_weight=0.0)
        assert cost_with > cost_without

    def test_normalized_exceedance_symmetry(self) -> None:
        """10% g-load exceedance = 10% heat flux exceedance at equal weights."""
        row_g = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0, g_max=11.0)
        cost_g = compute_cost(row_g, g_load_limit=10.0, g_load_weight=1000.0, heat_flux_weight=0.0)
        row_q = _make_row(energy=-2.0, ecc=0.4, dv_total=0.0, q_max=110.0)
        cost_q = compute_cost(row_q, heat_flux_limit=100.0, heat_flux_weight=1000.0, g_load_weight=0.0)
        assert cost_g == pytest.approx(cost_q, rel=1e-10)

    def test_weight_zero_disables_penalty(self) -> None:
        """Setting weight to 0 disables that constraint penalty."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=50.0, g_max=100.0, q_max=1000.0)
        cost = compute_cost(row, g_load_weight=0.0, heat_flux_weight=0.0)
        assert cost == pytest.approx(50.0, abs=1e-6)


class TestCostHyperbolic:
    def test_hyperbolic_penalized(self) -> None:
        """energy > 0 AND ecc > 1 → Level 0 penalty above 1e6."""
        row = _make_row(energy=5.0, ecc=2.0)
        cost = compute_cost(row)
        assert cost > 1e6

    def test_hyperbolic_higher_than_captured(self) -> None:
        """Hyperbolic always costs more than a well-captured orbit."""
        hyperbolic = _make_row(energy=1.0, ecc=1.5)
        captured = _make_row(energy=-1.0, ecc=0.5, dv_total=500.0)
        assert compute_cost(hyperbolic) > compute_cost(captured)

    def test_parabolic_boundary_classified_as_captured(self) -> None:
        """Energy=0, ecc=1 (strict >) → classified as captured."""
        row = _make_row(energy=0.0, ecc=1.0, dv_total=0.0)
        cost = compute_cost(row)
        assert cost == pytest.approx(0.0, abs=1e-12)


class TestCostAggregation:
    def test_multi_sim_rms(self) -> None:
        """Stacking identical rows produces the same cost as a single row."""
        row = _make_row(energy=-2.0, ecc=0.4, dv_total=100.0)
        stacked = np.tile(row, (5, 1))
        assert compute_cost(row) == pytest.approx(compute_cost(stacked), abs=1e-9)


class TestCostProperties:
    @given(
        energy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        ecc=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
        dv_total=st.floats(0.0, 1e4, allow_nan=False, allow_infinity=False),
        g_max=st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False),
        q_max=st.floats(0.0, 1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_cost_always_finite_nonneg(
        self, energy: float, ecc: float, dv_total: float, g_max: float, q_max: float,
    ) -> None:
        """For any finite inputs, compute_cost returns a finite, non-negative value."""
        row = _make_row(energy=energy, ecc=ecc, dv_total=dv_total, g_max=g_max, q_max=q_max)
        cost = compute_cost(row)
        assert np.isfinite(cost), f"cost is not finite: {cost}"
        assert cost >= 0.0, f"cost is negative: {cost}"
