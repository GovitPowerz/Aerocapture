"""Tests for potential-based (PBRS) per-step reward calculator."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.rl.rewards import StepRewardCalculator, compute_terminal_cost


@pytest.fixture
def default_calc() -> StepRewardCalculator:
    return StepRewardCalculator(
        input_mask=list(range(23)),
        gamma=0.99,
        corridor_weight=0.1,
        energy_rate_weight=0.05,
        constraint_weight=0.2,
        apoapsis_weight=0.2,
        eccentricity_weight=0.1,
        energy_scale=1.0e6,
    )


def _make_obs(n: int = 1, **overrides: float) -> np.ndarray:
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 6] = -1.0  # heat_flux_frac = 0
    obs[:, 7] = -1.0  # heat_load_frac = 0
    for k, v in overrides.items():
        obs[:, int(k)] = v
    return obs


def test_pbrs_identity_gives_minus_one_minus_gamma_phi(default_calc: StepRewardCalculator) -> None:
    """When obs_next == obs_cur, step_reward = (gamma - 1) * Phi(obs_cur).

    Phi is negative (penalty sum), so (gamma-1)*Phi is positive -- there is
    no state change, but PBRS pays a small constant premium per step; this is
    a telescoping offset that does not affect the optimum.
    """
    obs = _make_obs(n=4, **{"15": -1.0, "19": 0.5})  # capture + pdyn_error
    aux = np.zeros((4, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, obs, aux, aux)
    phi = default_calc._potential(obs, aux)
    expected = default_calc.gamma * phi - phi
    assert np.allclose(r, expected, atol=1e-8)


def test_pbrs_improvement_gives_positive_reward(default_calc: StepRewardCalculator) -> None:
    """Transition from worse to better state (Phi increases) -> reward > 0."""
    obs_bad = _make_obs(n=1, **{"15": -1.0, "19": 1.0})  # big pdyn error
    obs_good = _make_obs(n=1, **{"15": -1.0, "19": 0.0})  # zero pdyn error
    aux = np.zeros((1, 2), dtype=np.float32)
    r = default_calc.step_reward(obs_bad, obs_good, aux, aux)
    assert r[0] > 0


def test_pbrs_degradation_gives_negative_reward(default_calc: StepRewardCalculator) -> None:
    """Transition from better to worse (Phi decreases) -> reward < 0."""
    obs_good = _make_obs(n=1, **{"15": -1.0, "19": 0.0})
    obs_bad = _make_obs(n=1, **{"15": -1.0, "19": 1.0})
    aux = np.zeros((1, 2), dtype=np.float32)
    r = default_calc.step_reward(obs_good, obs_bad, aux, aux)
    assert r[0] < 0


def test_potential_phase_gating_capture(default_calc: StepRewardCalculator) -> None:
    """Capture phase: pdyn_error contributes to Phi, sma_error does not."""
    obs = _make_obs(n=1, **{"15": -1.0, "19": 1.0, "13": 1.0})
    aux = np.zeros((1, 2), dtype=np.float32)
    phi = default_calc._potential(obs, aux)
    # Only capture terms active. Phi = -(corridor + constraint). Constraint=0 here.
    expected = -default_calc.corridor_weight * 1.0**2
    assert np.isclose(phi[0], expected, atol=1e-8)


def test_potential_phase_gating_exit(default_calc: StepRewardCalculator) -> None:
    """Exit phase: sma_error contributes to Phi, pdyn_error does not."""
    obs = _make_obs(n=1, **{"15": 1.0, "19": 1.0, "13": 1.0})
    aux = np.zeros((1, 2), dtype=np.float32)
    phi = default_calc._potential(obs, aux)
    # Only exit terms active.
    expected = -default_calc.apoapsis_weight * 1.0**2
    assert np.isclose(phi[0], expected, atol=1e-8)


def test_missing_mask_raises() -> None:
    with pytest.raises(ValueError, match="missing required indices"):
        StepRewardCalculator(input_mask=[0, 1, 2])


def test_terminal_cost_matches_evaluate_module() -> None:
    from aerocapture.training.evaluate import compute_cost

    fc = np.zeros((1, 52))
    fc[0, 41] = 100.0
    fc[0, 17] = 5.0
    fc[0, 16] = 150.0
    fc[0, 28] = 10.0
    expected = compute_cost(fc)
    actual = compute_terminal_cost(fc[0])
    assert abs(actual - expected) < 1e-9
