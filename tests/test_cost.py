"""Tests for the unified cost function with quadratic-penalty DV compression."""

import numpy as np
from aerocapture.training.evaluate import compute_cost, dv_cost, log_cap
from hypothesis import given, settings
from hypothesis import strategies as st


class TestLogCap:
    """Tests for the legacy C1-continuous log-cap function (deprecated)."""

    def test_linear_below_threshold(self) -> None:
        dv = np.array([100.0, 500.0, 999.0])
        result = log_cap(dv, threshold=1000.0)
        np.testing.assert_array_almost_equal(result, dv)

    def test_log_above_threshold(self) -> None:
        dv = np.array([2000.0, 5000.0, 10000.0])
        result = log_cap(dv, threshold=1000.0)
        expected = 1000.0 * (1.0 + np.log(dv / 1000.0))
        np.testing.assert_array_almost_equal(result, expected)

    def test_safety_floor(self) -> None:
        result = log_cap(np.array([0.0, -1.0]), threshold=1000.0)
        assert np.all(np.isfinite(result))


class TestDvCost:
    """Tests for the C1-continuous quadratic-penalty DV cost function."""

    def test_linear_below_threshold(self) -> None:
        dv = np.array([100.0, 500.0, 999.0])
        result = dv_cost(dv, threshold=1000.0)
        np.testing.assert_array_almost_equal(result, dv)

    def test_above_threshold(self) -> None:
        dv = np.array([10000.0])
        result = dv_cost(dv, threshold=1000.0)[0]
        x = 9000.0
        expected = 1000.0 + np.sqrt(x + x**2 / 20000.0)
        assert abs(result - expected) < 1e-6

    def test_c0_continuity_at_threshold(self) -> None:
        t = 1000.0
        below = dv_cost(np.array([t - 1e-4]), threshold=t)[0]
        above = dv_cost(np.array([t + 1e-4]), threshold=t)[0]
        assert abs(below - above) < 0.1

    def test_c0_but_not_c1_at_threshold(self) -> None:
        """sqrt formula is C0 continuous but has infinite right derivative at threshold."""
        t = 1000.0
        eps = 1e-6
        left_deriv = (dv_cost(np.array([t]), t)[0] - dv_cost(np.array([t - eps]), t)[0]) / eps
        assert abs(left_deriv - 1.0) < 1e-3
        # Right derivative is very large (sqrt singularity at x=0)
        right_deriv = (dv_cost(np.array([t + eps]), t)[0] - dv_cost(np.array([t]), t)[0]) / eps
        assert right_deriv > 100  # infinite in the limit

    def test_safety_floor(self) -> None:
        result = dv_cost(np.array([0.0, -1.0]), threshold=1000.0)
        assert np.all(np.isfinite(result))

    def test_nonzero_gradient_at_non_capture(self) -> None:
        """Slope should be positive everywhere above threshold."""
        eps = 1.0
        for dv in [2000.0, 5000.0, 10000.0, 20000.0]:
            v1 = dv_cost(np.array([dv]))[0]
            v2 = dv_cost(np.array([dv + eps]))[0]
            assert v2 > v1

    def test_cost_spread_for_non_captures(self) -> None:
        """Non-capture range should have meaningful cost spread."""
        barely_hyper = dv_cost(np.array([10000.0]))[0]
        early_crash = dv_cost(np.array([20000.0]))[0]
        assert early_crash > barely_hyper

    @given(st.floats(min_value=0.01, max_value=1e6))
    @settings(max_examples=200)
    def test_monotonically_increasing(self, dv: float) -> None:
        eps = 1.0
        v1 = dv_cost(np.array([dv]), threshold=1000.0)[0]
        v2 = dv_cost(np.array([dv + eps]), threshold=1000.0)[0]
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

    def test_non_capture_has_high_cost(self) -> None:
        final = self._make_final(5, dv=10000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        # dv_cost(10000) = 1000 + sqrt(9000 + 9000^2/20000) = 1000 + sqrt(13050) ~ 1114
        assert 1100 < cost < 1200

    def test_crash_dv_produces_very_high_cost(self) -> None:
        final = self._make_final(5, dv=20000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        # dv_cost(20000) = 1000 + sqrt(19000 + 19000^2/20000) = 1000 + sqrt(37050) ~ 1192
        assert 1150 < cost < 1250

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

    def test_heat_load_penalty_applied_when_exceeded(self) -> None:
        """Cost increases when integrated heat load exceeds limit."""
        fc = self._make_final(10, dv=100.0)
        fc[:, 28] = 60.0  # 60 MJ/m² = 60000 kJ/m²
        cost_under = compute_cost(fc, heat_load_limit=100000.0)  # well under
        cost_over = compute_cost(fc, heat_load_limit=10000.0)  # well over
        assert cost_over > cost_under

    def test_heat_load_penalty_zero_when_under_limit(self) -> None:
        """No penalty when heat load is under limit."""
        fc = self._make_final(10, dv=100.0)
        fc[:, 28] = 10.0  # 10 MJ/m² = 10000 kJ/m²
        cost_no_hl = compute_cost(fc, heat_load_weight=0.0)
        cost_with_hl = compute_cost(fc, heat_load_limit=50000.0, heat_load_weight=1000.0)
        assert abs(cost_no_hl - cost_with_hl) < 1e-10

    def test_custom_dv_threshold(self) -> None:
        """Higher threshold means more of the DV is in the linear regime."""
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost_low_t = compute_cost(final, dv_threshold=500.0)
        cost_high_t = compute_cost(final, dv_threshold=2000.0)
        assert cost_low_t < cost_high_t

    def test_zero_dv_produces_finite_cost(self) -> None:
        """DV=0 (safety floor) should produce a near-zero finite cost."""
        final = self._make_final(3, dv=0.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert np.isfinite(cost)
        assert cost < 1.0

    @given(st.floats(min_value=1.0, max_value=50000.0))
    @settings(max_examples=100)
    def test_cost_always_finite(self, dv: float) -> None:
        final = self._make_final(3, dv=dv, g=5.0, q=50.0)
        cost = compute_cost(final)
        assert np.isfinite(cost)
        assert cost >= 0


class TestSentinelOverrides:
    """Tests for sentinel chromosome override construction."""

    def test_sentinel_bank_angles_coverage(self) -> None:
        from aerocapture.training.train import _SENTINEL_BANK_ANGLES

        assert len(_SENTINEL_BANK_ANGLES) == 11
        assert _SENTINEL_BANK_ANGLES[0] == 0
        assert _SENTINEL_BANK_ANGLES[-1] == 180
        # Uniform 18-degree spacing
        for i in range(len(_SENTINEL_BANK_ANGLES) - 1):
            assert _SENTINEL_BANK_ANGLES[i + 1] - _SENTINEL_BANK_ANGLES[i] == 18

    def test_sentinel_override_construction(self) -> None:
        from aerocapture.training.train import _SENTINEL_BANK_ANGLES

        section = "piecewise_constant"
        sentinel_overrides: list[dict[str, object]] = []
        for bank in _SENTINEL_BANK_ANGLES:
            ovr: dict[str, object] = {f"guidance.{section}.bank_angle_{i}": float(bank) for i in range(10)}
            ovr["guidance.type"] = "piecewise_constant"
            ovr["simulation.n_sims"] = 1
            sentinel_overrides.append(ovr)

        assert len(sentinel_overrides) == 11
        # Each override has 10 bank angles + guidance.type + simulation.n_sims = 12 keys
        for ovr in sentinel_overrides:
            assert len(ovr) == 12

        # First sentinel: all bank angles = 0.0 (full lift-up)
        for i in range(10):
            assert sentinel_overrides[0][f"guidance.piecewise_constant.bank_angle_{i}"] == 0.0

        # Last sentinel: all bank angles = 180.0 (full lift-down)
        for i in range(10):
            assert sentinel_overrides[-1][f"guidance.piecewise_constant.bank_angle_{i}"] == 180.0

        # Middle sentinel (index 5): all bank angles = 90.0
        for i in range(10):
            assert sentinel_overrides[5][f"guidance.piecewise_constant.bank_angle_{i}"] == 90.0
