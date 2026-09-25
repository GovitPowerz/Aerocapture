"""World-model experiment seams (#113): scoring metrics, plant replay, planner (experiments/world_model/)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

# wm_plant (and wm_planner through it) drive the extension; the whole module needs it.
aerocapture_rs = pytest.importorskip("aerocapture_rs")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/world_model"))

import torch  # noqa: E402
import wm_metrics  # noqa: E402
import wm_model  # noqa: E402
import wm_planner  # noqa: E402
import wm_plant  # noqa: E402


def _crps_gaussian(mu: float, sigma: float, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007, eq. 21): the ensemble estimator's oracle."""
    z = (y - mu) / sigma
    return np.asarray(sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) - 1.0 / np.sqrt(np.pi)))


class TestCrps:
    def test_the_oracle_at_the_mean_of_a_unit_normal(self) -> None:
        # 2*phi(0) - 1/sqrt(pi)
        assert _crps_gaussian(0.0, 1.0, np.array(0.0)) == pytest.approx(0.2336949, abs=1e-7)

    def test_ensemble_crps_is_the_crps_of_the_empirical_cdf(self) -> None:
        # samples {0, 1} at y = 0: integral of (F_n - 1{x >= 0})^2 = (0.5 - 1)^2 over [0, 1)
        assert wm_metrics.crps_ensemble(np.array([[0.0], [1.0]]), np.array([0.0])) == pytest.approx([0.25])

    def test_ensemble_crps_converges_to_the_gaussian_closed_form(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(0.3, 2.0, size=(20_000, 3))
        y = np.array([-1.0, 0.3, 4.0])
        closed = _crps_gaussian(0.3, 2.0, y)
        np.testing.assert_allclose(wm_metrics.crps_ensemble(samples, y), closed, rtol=0.02)


class TestCoverage:
    def test_a_calibrated_ensemble_covers_at_the_nominal_rate(self) -> None:
        rng = np.random.default_rng(1)
        samples, y = rng.normal(size=(400, 5000)), rng.normal(size=5000)
        np.testing.assert_allclose(wm_metrics.central_coverage(samples, y, np.array([0.5, 0.9])), [0.5, 0.9], atol=0.02)

    def test_an_overconfident_ensemble_undercovers(self) -> None:
        # half-width sigma: the 90% interval spans |z| < 1.645 * 0.5, which holds 2*Phi(0.8224) - 1 = 0.589 of N(0, 1)
        rng = np.random.default_rng(2)
        samples, y = rng.normal(scale=0.5, size=(400, 5000)), rng.normal(size=5000)
        assert wm_metrics.central_coverage(samples, y, np.array([0.9]))[0] == pytest.approx(0.589, abs=0.02)


class TestRocAuc:
    def test_the_textbook_example(self) -> None:
        # scikit-learn's roc_auc_score docstring example: 3 of 4 positive-negative pairs ordered correctly
        assert wm_metrics.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) == pytest.approx(0.75)

    def test_ties_count_half_and_infinite_scores_rank_first(self) -> None:
        assert wm_metrics.roc_auc(np.array([1.0, 1.0]), np.array([False, True])) == pytest.approx(0.5)
        assert wm_metrics.roc_auc(np.array([np.inf, 5.0, 1.0]), np.array([True, False, False])) == pytest.approx(1.0)


WM_MEDIUM = "experiments/world_model/configs/wm_medium.toml"


@pytest.fixture(scope="module")
def stub(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("wm") / "stub_model.json"
    wm_plant.write_stub_model(WM_MEDIUM, path)
    return path


class TestPlantReplay:
    def test_a_shared_action_prefix_replays_bit_identically_then_diverges(self, stub: Path) -> None:
        t = 40
        base = wm_plant.bank_schedule(123, 1500)
        other = base.copy()
        other[t:] += np.deg2rad(10.0)
        a, b = wm_plant.fly(WM_MEDIUM, [7, 7], np.stack([base, other]), stub=stub)
        # step t flies a_t and returns the nav one tick later: rows before t match, row t already differs
        np.testing.assert_array_equal(a.obs[:t], b.obs[:t])
        np.testing.assert_array_equal(a.aux[:t], b.aux[:t])
        assert not np.array_equal(a.obs[t, :22], b.obs[t, :22])

    def test_an_episode_does_not_depend_on_its_batch_neighbours(self, stub: Path) -> None:
        acts = np.stack([wm_plant.bank_schedule(s, 1500) for s in (1, 2, 3)])
        alone = wm_plant.fly(WM_MEDIUM, [11], acts[:1], stub=stub)[0]
        batched = wm_plant.fly(WM_MEDIUM, [11, 12, 13], acts, stub=stub)[0]
        np.testing.assert_array_equal(alone.obs, batched.obs)
        assert alone.dv_m_s == batched.dv_m_s

    def test_the_prev_bank_input_records_the_shaped_command(self, stub: Path) -> None:
        # a constant -0.5 rad command from an entry bank near +1.2 rad: the shaper ramps toward it, then holds it
        ep = wm_plant.fly(WM_MEDIUM, [5], np.full((1, 1500), -0.5), stub=stub)[0]
        prev_bank = wm_plant.raw_input(ep.obs[:, 22], wm_plant.load_normalization(stub)[22])
        assert prev_bank[0] > 0.0
        assert np.all(np.diff(prev_bank[:10]) <= 0.0)
        np.testing.assert_allclose(prev_bank[10:], -0.5, atol=1e-6)


class TestBankSchedule:
    def test_deterministic_per_seed_and_within_the_bank_range(self) -> None:
        a, b, c = wm_plant.bank_schedule(9, 800), wm_plant.bank_schedule(9, 800), wm_plant.bank_schedule(10, 800)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
        assert np.abs(a).max() <= np.pi


# the Rust unit tests' active_params() (src/rust/src/gnc/guidance/lateral.rs), threshold in rad, energies in J/kg
_LATERAL = wm_planner.LateralParams(tau=15.0, threshold=0.01, min_reversal_interval=5.0, activation=0.0, inhibition=-1e12, max_reversals=5)


def _seeded(params: wm_planner.LateralParams, err: float) -> wm_planner.Lateral:
    lat = wm_planner.Lateral(params)
    assert not lat.update(err, -1e6, 1.0, 0.0)  # the first tick only seeds the rate estimate
    return lat


class TestLateral:
    def test_a_positive_projected_error_reverses_to_a_negative_roll(self) -> None:
        lat = _seeded(_LATERAL, 0.5)
        assert lat.update(0.5, -1e6, 1.0, 1.0)
        assert (lat.sign, lat.n_reversals) == (-1.0, 1)

    def test_an_error_inside_the_threshold_holds_the_sign(self) -> None:
        lat = _seeded(_LATERAL, 0.001)
        assert not lat.update(0.001, -1e6, 1.0, 1.0)
        assert lat.sign == 1.0

    def test_reversals_respect_the_minimum_interval(self) -> None:
        lat = _seeded(_LATERAL, 0.5)
        assert lat.update(0.5, -1e6, 1.0, 1.0)
        assert not lat.update(-0.5, -1e6, 1.0, 3.0)
        assert lat.update(-0.5, -1e6, 1.0, 7.0)
        assert lat.n_reversals == 2

    def test_reversals_respect_the_budget(self) -> None:
        params = wm_planner.LateralParams(tau=15.0, threshold=0.01, min_reversal_interval=0.0, activation=0.0, inhibition=-1e12, max_reversals=1)
        lat = _seeded(params, 0.5)
        assert lat.update(0.5, -1e6, 1.0, 1.0)
        assert not lat.update(-0.5, -1e6, 1.0, 10.0)

    def test_no_reversal_outside_the_energy_window(self) -> None:
        lat = _seeded(_LATERAL, 0.5)
        assert not lat.update(0.5, 1e6, 1.0, 1.0)
        assert lat.n_reversals == 0


class TestBisection:
    def test_bisection_finds_the_bank_that_hits_the_target_apoapsis(self) -> None:
        # apoapsis decreases with bank; the target sits at 1.1 rad in flight 0 and 0.4 rad in flight 1
        root = np.array([1.1, 0.4])

        def residual(rows: np.ndarray, bank: np.ndarray) -> np.ndarray:
            return np.asarray(3.0e5 * (root[rows] - bank))

        bank = wm_planner.bisect_bank(residual, np.array([0.3, 0.3]), np.array([1.9, 1.9]), tol=1.0)
        np.testing.assert_allclose(bank, root, atol=(1.9 - 0.3) / 2**9)

    def test_bisection_saturates_when_the_target_is_out_of_reach(self) -> None:
        # flight 0 over-dissipates even at the least bank, flight 1 still escapes at the most bank
        def residual(rows: np.ndarray, bank: np.ndarray) -> np.ndarray:
            return np.asarray(np.array([-1.0e5, 1.0e5])[rows])

        np.testing.assert_array_equal(wm_planner.bisect_bank(residual, np.array([0.3, 0.3]), np.array([1.9, 1.9]), tol=1.0), [0.3, 1.9])

    def test_bisection_predicts_only_for_flights_still_bracketed(self) -> None:
        # flight 0 saturates at lo, flight 1 is within tol at the first midpoint, flight 2 (a non-dyadic root) needs all
        # 8 halvings: fnpag.rs predicts the endpoints for all, the midpoint and the halvings only where the bracket straddles
        calls: list[np.ndarray] = []

        def residual(rows: np.ndarray, bank: np.ndarray) -> np.ndarray:
            calls.append(rows.copy())
            return np.asarray(np.where(rows == 0, -1.0, 3.0e5 * (np.array([0.0, 1.1, 0.5123])[rows] - bank)))

        wm_planner.bisect_bank(residual, np.full(3, 0.3), np.full(3, 1.9), tol=1.0)
        assert [c.tolist() for c in calls[:3]] == [[0, 1, 2], [0, 1, 2], [1, 2]]
        assert all(c.tolist() == [2] for c in calls[3:])
        assert len(calls) == 3 + wm_planner.N_BISECT_STEPS

    def test_a_non_finite_prediction_reads_as_unbound_like_fnpag_rs(self, stub: Path) -> None:
        ro = wm_planner.Readout(wm_plant.load_normalization(stub), 120.0)
        diverged = np.full((1, 3, wm_model.N_X), np.nan, np.float32)
        assert ro.apoapsis(diverged).tolist() == [wm_planner.UNBOUND_APOAPSIS_M]
        assert ro.dv(diverged).tolist() == [np.inf]


class TestPredictorTiming:
    def test_step_from_the_filtered_state_reproduces_the_teacher_forced_prediction(self) -> None:
        # h[k] has consumed (x_0, a_1) .. (x_k, a_{k+1}); a planner at row t0 holds h[t0 - 1] and steps (x_t0, a_{t0+1})
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        norm = wm_model.Normalizer(rng.normal(size=wm_model.N_X), rng.uniform(0.5, 2.0, wm_model.N_X), np.zeros(wm_model.N_X), np.ones(wm_model.N_X))
        x = rng.normal(size=(12, wm_model.N_X)).astype(np.float32)
        banks = rng.uniform(-np.pi, np.pi, 12)
        for kind in ("gru", "mlp"):
            pred = wm_model.Predictor(wm_model.Dynamics(kind), norm)
            mu, _, hs = pred.teacher_forced(x, wm_model.action_features(banks[1:]))
            for t0 in (0, 5, 10):
                h0 = None if hs is None else hs[t0 - 1][None] if t0 > 0 else np.zeros((1, wm_model.HIDDEN), np.float32)
                h1, x1 = pred.step(h0, x[t0][None], banks[t0 + 1 : t0 + 2])
                np.testing.assert_allclose(x1[0], mu[t0], rtol=1e-5, atol=1e-5)
                if hs is not None:
                    assert h1 is not None
                    np.testing.assert_allclose(h1[0], hs[t0], rtol=1e-5, atol=1e-5)
                free = pred.free_run(h0, x[t0][None], banks[None, t0 + 1 : t0 + 2])
                np.testing.assert_allclose(free[0, 0], x1[0])
