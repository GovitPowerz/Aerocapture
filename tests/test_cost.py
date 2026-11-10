"""Tests for the unified cost function with quadratic-penalty DV compression."""

from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.cost import CostSettings, compute_cost, dv_cost, per_sim_cost
from hypothesis import given, settings
from hypothesis import strategies as st


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
        cost = compute_cost(final, CostSettings())
        assert abs(cost - 200.0) < 1.0

    def test_non_capture_has_high_cost(self) -> None:
        final = self._make_final(5, dv=10000.0, g=5.0, q=50.0)
        cost = compute_cost(final, CostSettings())
        # dv_cost(10000) ~ 23050 (softplus-quadratic)
        assert 22000 < cost < 24000

    def test_crash_dv_produces_very_high_cost(self) -> None:
        final = self._make_final(5, dv=20000.0, g=5.0, q=50.0)
        cost = compute_cost(final, CostSettings())
        # dv_cost(20000) ~ 57050 (softplus-quadratic)
        assert 56000 < cost < 58000

    def test_cost_ordering(self) -> None:
        good = compute_cost(self._make_final(5, dv=200.0), CostSettings())
        bad = compute_cost(self._make_final(5, dv=5000.0), CostSettings())
        hyper = compute_cost(self._make_final(5, dv=10500.0), CostSettings())
        crash = compute_cost(self._make_final(5, dv=19000.0), CostSettings())
        assert good < bad < hyper < crash

    def test_g_load_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, g=10.0), CostSettings())
        with_penalty = compute_cost(self._make_final(5, dv=200.0, g=20.0), CostSettings())
        assert with_penalty > no_penalty

    def test_heat_flux_penalty(self) -> None:
        no_penalty = compute_cost(self._make_final(5, dv=200.0, q=100.0), CostSettings())
        with_penalty = compute_cost(self._make_final(5, dv=200.0, q=300.0), CostSettings())
        assert with_penalty > no_penalty

    def test_heat_load_penalty_applied_when_exceeded(self) -> None:
        """Cost increases when integrated heat load exceeds limit."""
        fc = self._make_final(10, dv=100.0)
        fc[:, 28] = 60.0  # 60 MJ/m² = 60000 kJ/m²
        cost_under = compute_cost(fc, CostSettings(heat_load_limit=100000.0))  # well under
        cost_over = compute_cost(fc, CostSettings(heat_load_limit=10000.0))  # well over
        assert cost_over > cost_under

    def test_heat_load_penalty_zero_when_under_limit(self) -> None:
        """No penalty when heat load is under limit."""
        fc = self._make_final(10, dv=100.0)
        fc[:, 28] = 10.0  # 10 MJ/m² = 10000 kJ/m²
        cost_no_hl = compute_cost(fc, CostSettings(heat_load_weight=0.0))
        cost_with_hl = compute_cost(fc, CostSettings(heat_load_limit=50000.0, heat_load_weight=1000.0))
        assert abs(cost_no_hl - cost_with_hl) < 1e-10

    def test_custom_dv_threshold(self) -> None:
        """Lower threshold activates penalty earlier, so cost is higher."""
        final = self._make_final(5, dv=5000.0, g=5.0, q=50.0)
        cost_low_t = compute_cost(final, CostSettings(dv_threshold=500.0))
        cost_high_t = compute_cost(final, CostSettings(dv_threshold=2000.0))
        assert cost_low_t > cost_high_t

    def test_cost_transforms_preserve_ordering(self) -> None:
        """All transforms are monotonic: good < crash regardless of transform."""
        good = self._make_final(5, dv=200.0)
        crash = self._make_final(5, dv=20000.0)
        for transform in ("linear", "sqrt", "log", "squared", "cubed"):
            g = compute_cost(good, CostSettings(cost_transform=transform))
            c = compute_cost(crash, CostSettings(cost_transform=transform))
            assert g < c, f"ordering broken under {transform}"

    def test_sqrt_compresses_squared_and_cubed_expand_tail(self) -> None:
        """log < sqrt shrinks crash/good ratio; squared and cubed blow it up."""
        good = self._make_final(5, dv=200.0)
        crash = self._make_final(5, dv=20000.0)
        lin = compute_cost(crash, CostSettings()) / compute_cost(good, CostSettings())
        log_r = compute_cost(crash, CostSettings(cost_transform="log")) / compute_cost(good, CostSettings(cost_transform="log"))
        sqrt_r = compute_cost(crash, CostSettings(cost_transform="sqrt")) / compute_cost(good, CostSettings(cost_transform="sqrt"))
        sq_r = compute_cost(crash, CostSettings(cost_transform="squared")) / compute_cost(good, CostSettings(cost_transform="squared"))
        cb_r = compute_cost(crash, CostSettings(cost_transform="cubed")) / compute_cost(good, CostSettings(cost_transform="cubed"))
        assert log_r < sqrt_r < lin < sq_r < cb_r

    def test_cost_transform_identities_n1(self) -> None:
        """For n=1 records, transform equals the elementwise op on the linear cost."""
        final = self._make_final(1, dv=500.0)
        lin = compute_cost(final, CostSettings())
        assert abs(compute_cost(final, CostSettings(cost_transform="sqrt")) - np.sqrt(lin)) < 1e-6
        assert abs(compute_cost(final, CostSettings(cost_transform="log")) - np.log1p(lin)) < 1e-6
        assert abs(compute_cost(final, CostSettings(cost_transform="squared")) - lin**2) < 1e-3
        assert abs(compute_cost(final, CostSettings(cost_transform="cubed")) - lin**3) < 1.0

    def test_log_transform_handles_zero_cost(self) -> None:
        """log1p(0) = 0; the safety floor must not produce NaN/Inf."""
        final = self._make_final(3, dv=0.0)
        c = compute_cost(final, CostSettings(cost_transform="log"))
        assert np.isfinite(c)
        assert c >= 0.0

    def test_unknown_cost_transform_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unknown cost_transform"):
            compute_cost(self._make_final(1, dv=200.0), CostSettings(cost_transform="cbrt"))

    def test_zero_dv_produces_finite_cost(self) -> None:
        """DV=0 (safety floor) should produce a near-zero finite cost."""
        final = self._make_final(3, dv=0.0, g=5.0, q=50.0)
        cost = compute_cost(final, CostSettings())
        assert np.isfinite(cost)
        assert cost < 1.0

    @given(st.floats(min_value=1.0, max_value=50000.0))
    @settings(max_examples=100)
    def test_cost_always_finite(self, dv: float) -> None:
        final = self._make_final(3, dv=dv, g=5.0, q=50.0)
        cost = compute_cost(final, CostSettings())
        assert np.isfinite(cost)
        assert cost >= 0


class TestSentinelOverrides:
    """Tests for sentinel chromosome override construction."""

    def test_sentinel_bank_angles_coverage(self) -> None:
        from aerocapture.training.corridor import _SENTINEL_BANK_ANGLES

        assert len(_SENTINEL_BANK_ANGLES) == 11
        assert _SENTINEL_BANK_ANGLES[0] == 0
        assert _SENTINEL_BANK_ANGLES[-1] == 180
        # Uniform 18-degree spacing
        for i in range(len(_SENTINEL_BANK_ANGLES) - 1):
            assert _SENTINEL_BANK_ANGLES[i + 1] - _SENTINEL_BANK_ANGLES[i] == 18

    def test_sentinel_override_construction(self) -> None:
        from aerocapture.training.corridor import _SENTINEL_BANK_ANGLES

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


class TestPiecewiseSegmentResolution:
    """Resolve n_segments from TOML the same way Rust does."""

    def test_default_when_unspecified(self) -> None:
        from aerocapture.training.training_config import _resolve_piecewise_n_segments

        assert _resolve_piecewise_n_segments({"guidance": {"piecewise_constant": {}}}) == 10
        assert _resolve_piecewise_n_segments({}) == 10

    def test_explicit_n_segments_wins(self) -> None:
        from aerocapture.training.training_config import _resolve_piecewise_n_segments

        toml = {"guidance": {"piecewise_constant": {"n_segments": 7}}}
        assert _resolve_piecewise_n_segments(toml) == 7

    def test_derived_from_bank_angles_array(self) -> None:
        from aerocapture.training.training_config import _resolve_piecewise_n_segments

        toml = {"guidance": {"piecewise_constant": {"bank_angles": [10.0, 20.0, 30.0, 40.0, 50.0]}}}
        assert _resolve_piecewise_n_segments(toml) == 5

    def test_derived_from_individual_keys(self) -> None:
        from aerocapture.training.training_config import _resolve_piecewise_n_segments

        toml = {"guidance": {"piecewise_constant": {f"bank_angle_{i}": 65.0 for i in range(4)}}}
        assert _resolve_piecewise_n_segments(toml) == 4

    def test_rejects_zero_segments(self) -> None:
        import pytest
        from aerocapture.training.training_config import _resolve_piecewise_n_segments

        with pytest.raises(ValueError, match="n_segments must be >= 1"):
            _resolve_piecewise_n_segments({"guidance": {"piecewise_constant": {"n_segments": 0}}})

    def test_make_specs_factory(self) -> None:
        from aerocapture.training.param_spaces import _SHAPING_PARAMS, make_piecewise_constant_specs

        specs = make_piecewise_constant_specs(5)
        bank_specs = [s for s in specs if s.name.startswith("bank_angle_")]
        assert len(bank_specs) == 5
        assert [s.name for s in bank_specs] == [f"bank_angle_{i}" for i in range(5)]
        assert len(specs) == 5 + len(_SHAPING_PARAMS)


class TestSoftplusStability:
    """_softplus must be warning-free on both tails (the old np.where form
    evaluated exp(kx) on the discarded branch, overflowing for kx > ~709)."""

    def test_no_overflow_far_above_knee(self) -> None:
        from aerocapture.training.cost import _softplus

        # kx = 5000 with the constraint knee k=100: the old form overflowed here.
        with np.errstate(over="raise"):
            out = _softplus(np.array([50.0]), 100.0)
        assert np.isclose(out[0], 50.0)  # asymptotically linear

    def test_matches_naive_form_in_safe_range(self) -> None:
        from aerocapture.training.cost import _softplus

        x = np.linspace(-3.0, 3.0, 31)
        k = 2.0
        np.testing.assert_allclose(_softplus(x, k), np.log1p(np.exp(k * x)) / k, rtol=1e-12)


class TestCostSettings:
    """One `[cost_function]` reader; a misspelled weight must not train at the 1000.0 default silently."""

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValueError, match=r"unknown \[cost_function\] keys: \['heat_load_weigth'\]"):
            CostSettings.from_toml({"cost_function": {"heat_load_weigth": 1.0}})

    def test_reads_weights_and_limits(self) -> None:
        cs = CostSettings.from_toml({"cost_function": {"heat_load_weight": 1, "cost_transform": "cubed"}, "flight": {"constraints": {"max_heat_load": 1234.0}}})
        assert cs == CostSettings(heat_load_weight=1.0, cost_transform="cubed", heat_load_limit=1234.0)
        assert isinstance(cs.heat_load_weight, float)  # an integer TOML value is coerced

    def test_one_default_table(self) -> None:
        """#241: an empty config scores at the field defaults; the function has no second table."""
        assert CostSettings.from_toml({}) == CostSettings()
        cs = CostSettings()
        assert (cs.g_load_weight, cs.heat_flux_weight, cs.heat_load_weight) == (1000.0, 1000.0, 1000.0)
        assert (cs.dv_threshold, cs.g_load_limit, cs.heat_flux_limit, cs.heat_load_limit, cs.cost_transform) == (1000.0, 15.0, 200.0, 25000.0, "linear")

    def test_report_reader_is_the_same_function(self, tmp_path: Path) -> None:
        from aerocapture.training.report import read_cost_settings

        toml = tmp_path / "c.toml"
        toml.write_text("[cost_function]\nheat_flux_weigth = 2.0\n")
        with pytest.raises(ValueError, match=r"unknown \[cost_function\] keys"):
            read_cost_settings(toml)


# (dv m/s, g-load, heat flux kW/m^2, heat load MJ/m^2) -> the per-simulation cost under
# `linear` and `cubed`, at common.toml's weights (1.0). Captured from the pre-#241
# `compute_cost` on 1-row slices.
_KNOWN_COSTS = {
    "capture": ((212.5, 4.2, 96.0, 14.0), 212.53800574493766, 9600852.636642784),
    "near_knee": ((1040.0, 6.0, 150.0, 20.0), 1131.7183236655737, 1449489396.9760826),
    "hyperbolic": ((10850.0, 3.1, 80.0, 9.0), 25551.125, 16681307180449.08),
    "crash": ((27400.0, 31.0, 410.0, 61.0), 88651.55666666667, 696721315970531.9),
    "g_violation": ((180.0, 16.5, 120.0, 12.0), 180.12746207765446, 5844398.089172706),
    "heat_flux_violation": ((180.0, 5.0, 230.0, 12.0), 180.1774616267245, 5849266.281725992),
    "heat_load_violation": ((180.0, 5.0, 120.0, 27.5), 180.12746207765446, 5844398.089172706),
    "floor": ((0.0, 0.0, 0.0, 0.0), 0.004540890997614375, 9.363176947259404e-08),
}
_KNOWN_RMS = {"linear": 32621.660370602942, "cubed": 246398776919412.56}


def _record(dv: float, g: float, q: float, heat_load: float) -> np.ndarray:
    r = np.zeros(52)
    r[41], r[17], r[16], r[28] = dv, g, q, heat_load
    return r


def _outcome_grid() -> np.ndarray:
    """(48, 25, 52) final records: captures, the knee, hyperbolic exits, crashes, and every
    constraint on both sides of its limit (the known rows included)."""
    rng = np.random.default_rng(241)
    n = 48 * 25
    dv = np.concatenate(
        [rng.uniform(0.0, 900.0, n // 2), rng.uniform(900.0, 1100.0, n // 8), rng.uniform(3000.0, 53000.0, n // 4), rng.uniform(10000.0, 14000.0, n // 8)]
    )
    grid = np.zeros((n, 52))
    grid[:, 41] = rng.permutation(dv)
    grid[:, 17] = rng.uniform(0.0, 40.0, n)
    grid[:, 16] = rng.uniform(0.0, 500.0, n)
    grid[:, 28] = rng.uniform(0.0, 70.0, n)
    for i, (rec, _, _) in enumerate(_KNOWN_COSTS.values()):
        grid[i] = _record(*rec)
    return grid.reshape(48, 25, 52)


class TestPerSimCost:
    """The per-simulation cost over any leading shape; the RMS is the caller's reduction."""

    @pytest.mark.parametrize(("transform", "column"), [("linear", 1), ("cubed", 2)])
    def test_known_costs(self, transform: str, column: int) -> None:
        cs = CostSettings(g_load_weight=1.0, heat_flux_weight=1.0, heat_load_weight=1.0, cost_transform=transform)
        records = np.array([_record(*row[0]) for row in _KNOWN_COSTS.values()])
        np.testing.assert_allclose(per_sim_cost(records, cs), [row[column] for row in _KNOWN_COSTS.values()], rtol=1e-13)
        assert compute_cost(records, cs) == pytest.approx(_KNOWN_RMS[transform], rel=1e-13)

    @pytest.mark.parametrize("transform", ["linear", "sqrt", "log", "squared", "cubed"])
    @pytest.mark.parametrize("weight", [1.0, 1000.0])
    def test_array_form_equals_the_one_row_loop_bit_for_bit(self, transform: str, weight: float) -> None:
        """#241: the grid in one call is the f64 the training loop used to get from
        `compute_cost` on each 1-row slice (which also took `sqrt(c**2)` of every cost)."""
        cs = CostSettings(g_load_weight=weight, heat_flux_weight=weight, heat_load_weight=weight, cost_transform=transform)
        grid = _outcome_grid()
        loop = np.array([[compute_cost(grid[i, k].reshape(1, 52), cs) for k in range(grid.shape[1])] for i in range(grid.shape[0])])
        assert np.array_equal(per_sim_cost(grid, cs), loop)
        assert np.array_equal(per_sim_cost(grid[3], cs), loop[3])
        # One record at a time (the RL terminal cost): numpy scalar math would take pow() for x**2.
        cells = np.array([[per_sim_cost(grid[i, k], cs) for k in range(grid.shape[1])] for i in range(grid.shape[0])])
        assert np.array_equal(cells, loop)
