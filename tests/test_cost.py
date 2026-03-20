"""Tests for the unified cost function with log-cap compression."""

import numpy as np
from aerocapture.training.evaluate import compute_cost, log_cap
from hypothesis import given, settings
from hypothesis import strategies as st


class TestLogCap:
    """Tests for the C1-continuous log-cap function."""

    def test_linear_below_threshold(self) -> None:
        dv = np.array([100.0, 500.0, 999.0])
        result = log_cap(dv, threshold=1000.0)
        np.testing.assert_array_almost_equal(result, dv)

    def test_log_above_threshold(self) -> None:
        dv = np.array([2000.0, 5000.0, 10000.0])
        result = log_cap(dv, threshold=1000.0)
        expected = 1000.0 * (1.0 + np.log(dv / 1000.0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_c0_continuity_at_threshold(self) -> None:
        t = 1000.0
        below = log_cap(np.array([t - 1e-10]), threshold=t)[0]
        above = log_cap(np.array([t + 1e-10]), threshold=t)[0]
        assert abs(below - above) < 1e-6

    def test_c1_continuity_at_threshold(self) -> None:
        t = 1000.0
        eps = 1e-6
        left_deriv = (log_cap(np.array([t]), t)[0] - log_cap(np.array([t - eps]), t)[0]) / eps
        right_deriv = (log_cap(np.array([t + eps]), t)[0] - log_cap(np.array([t]), t)[0]) / eps
        assert abs(left_deriv - 1.0) < 1e-3
        assert abs(right_deriv - 1.0) < 1e-3

    def test_safety_floor(self) -> None:
        result = log_cap(np.array([0.0, -1.0]), threshold=1000.0)
        assert np.all(np.isfinite(result))

    @given(st.floats(min_value=0.01, max_value=1e6))
    @settings(max_examples=200)
    def test_monotonically_increasing(self, dv: float) -> None:
        eps = 1.0
        v1 = log_cap(np.array([dv]), threshold=1000.0)[0]
        v2 = log_cap(np.array([dv + eps]), threshold=1000.0)[0]
        assert v2 >= v1


class TestUnifiedComputeCost:
    """Tests for the unified compute_cost function."""

    @staticmethod
    def _make_final(n: int, dv: float = 200.0, g: float = 5.0, q: float = 100.0) -> np.ndarray:
        arr = np.zeros((n, 52))
        arr[:, 41] = dv
        arr[:, 17] = g
        arr[:, 16] = q
        return arr

    def test_good_capture_cost_equals_dv(self) -> None:
        final = self._make_final(5, dv=200.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert abs(cost - 200.0) < 1.0

    def test_bad_capture_log_compressed(self) -> None:
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert 2500 < cost < 2700

    def test_crash_dv_produces_high_cost(self) -> None:
        final = self._make_final(5, dv=20000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert 3900 < cost < 4100

    def test_cost_ordering(self) -> None:
        good = compute_cost(self._make_final(5, dv=200.0))
        bad = compute_cost(self._make_final(5, dv=5000.0))
        hyper = compute_cost(self._make_final(5, dv=10500.0))
        crash = compute_cost(self._make_final(5, dv=19000.0))
        assert good < bad < hyper < crash

    def test_g_load_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, g=10.0))
        with_penalty = compute_cost(self._make_final(5, dv=200.0, g=20.0))
        assert with_penalty > no_penalty

    def test_heat_flux_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, q=100.0))
        with_penalty = compute_cost(self._make_final(5, dv=200.0, q=300.0))
        assert with_penalty > no_penalty

    def test_custom_dv_threshold(self) -> None:
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost_low_t = compute_cost(final, dv_threshold=500.0)
        cost_high_t = compute_cost(final, dv_threshold=2000.0)
        assert cost_low_t < cost_high_t

    @given(st.floats(min_value=1.0, max_value=50000.0))
    @settings(max_examples=100)
    def test_cost_always_finite(self, dv: float) -> None:
        final = self._make_final(3, dv=dv, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert np.isfinite(cost)
        assert cost >= 0
