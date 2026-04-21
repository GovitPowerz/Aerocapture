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
    """Tests for the C-infinity softplus-quadratic DV cost function."""

    def test_captures_nearly_untouched(self) -> None:
        """Low DV values (captures) should be barely affected by the softplus tail."""
        for dv_val in [50.0, 100.0, 200.0]:
            result = dv_cost(np.array([dv_val]), threshold=1000.0)[0]
            assert abs(result - dv_val) < 1.0

    def test_strong_gradient_above_threshold(self) -> None:
        """Non-capture cost should be much higher than threshold."""
        result = dv_cost(np.array([10000.0]), threshold=1000.0)[0]
        assert result > 20000  # ~23050 expected

    def test_c_infinity_at_threshold(self) -> None:
        """Slope should be continuous through the threshold (no kink)."""
        t = 1000.0
        eps = 0.01
        left_slope = (dv_cost(np.array([t]), t)[0] - dv_cost(np.array([t - eps]), t)[0]) / eps
        right_slope = (dv_cost(np.array([t + eps]), t)[0] - dv_cost(np.array([t]), t)[0]) / eps
        assert abs(left_slope - right_slope) < 0.01  # smooth transition

    def test_safety_floor(self) -> None:
        result = dv_cost(np.array([0.0, -1.0]), threshold=1000.0)
        assert np.all(np.isfinite(result))

    def test_strong_far_gradient(self) -> None:
        """Slope at dv=10000 should be ~2.9, much stronger than log_cap's 0.1."""
        eps = 1.0
        v1 = dv_cost(np.array([10000.0]))[0]
        v2 = dv_cost(np.array([10000.0 + eps]))[0]
        slope = v2 - v1
        assert slope > 2.5

    def test_wide_cost_spread_for_non_captures(self) -> None:
        """Non-capture range should span >30000 cost (vs ~700 for log_cap)."""
        barely_hyper = dv_cost(np.array([10000.0]))[0]
        early_crash = dv_cost(np.array([20000.0]))[0]
        assert early_crash - barely_hyper > 30000

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
        # dv_cost(10000) ~ 23050 (softplus-quadratic)
        assert 22000 < cost < 24000

    def test_crash_dv_produces_very_high_cost(self) -> None:
        final = self._make_final(5, dv=20000.0, g=5.0, q=50.0)
        cost = compute_cost(final)
        # dv_cost(20000) ~ 57050 (softplus-quadratic)
        assert 56000 < cost < 58000

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
        """Lower threshold activates penalty earlier, so cost is higher."""
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost_low_t = compute_cost(final, dv_threshold=500.0)
        cost_high_t = compute_cost(final, dv_threshold=2000.0)
        assert cost_low_t > cost_high_t

    def test_cost_transforms_preserve_ordering(self) -> None:
        """All transforms are monotonic: good < crash regardless of transform."""
        good = self._make_final(5, dv=200.0)
        crash = self._make_final(5, dv=20000.0)
        for transform in ("linear", "sqrt", "squared", "cubed"):
            g = compute_cost(good, cost_transform=transform)
            c = compute_cost(crash, cost_transform=transform)
            assert g < c, f"ordering broken under {transform}"

    def test_sqrt_compresses_squared_and_cubed_expand_tail(self) -> None:
        """sqrt shrinks crash/good ratio; squared and cubed blow it up."""
        good = self._make_final(5, dv=200.0)
        crash = self._make_final(5, dv=20000.0)
        lin = compute_cost(crash) / compute_cost(good)
        sqrt_r = compute_cost(crash, cost_transform="sqrt") / compute_cost(good, cost_transform="sqrt")
        sq_r = compute_cost(crash, cost_transform="squared") / compute_cost(good, cost_transform="squared")
        cb_r = compute_cost(crash, cost_transform="cubed") / compute_cost(good, cost_transform="cubed")
        assert sqrt_r < lin < sq_r < cb_r

    def test_cost_transform_identities_n1(self) -> None:
        """For n=1 records, transform equals the elementwise op on the linear cost."""
        final = self._make_final(1, dv=500.0)
        lin = compute_cost(final)
        assert abs(compute_cost(final, cost_transform="sqrt") - np.sqrt(lin)) < 1e-6
        assert abs(compute_cost(final, cost_transform="squared") - lin**2) < 1e-3
        assert abs(compute_cost(final, cost_transform="cubed") - lin**3) < 1.0

    def test_unknown_cost_transform_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unknown cost_transform"):
            compute_cost(self._make_final(1, dv=200.0), cost_transform="cbrt")

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
