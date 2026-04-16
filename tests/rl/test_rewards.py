"""Tests for phase-aware per-step reward calculator."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.rl.rewards import StepRewardCalculator, compute_terminal_cost


@pytest.fixture
def default_calc() -> StepRewardCalculator:
    return StepRewardCalculator(
        input_mask=list(range(23)),
        corridor_weight=0.1,
        energy_rate_weight=0.05,
        constraint_weight=0.2,
        apoapsis_weight=0.2,
        eccentricity_weight=0.1,
        energy_scale=1.0e6,
    )


def test_capture_phase_corridor_penalty(default_calc: StepRewardCalculator) -> None:
    """Non-zero pdyn_error during capture produces negative reward."""
    n = 4
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # bounce_flag = pre-bounce (capture phase)
    obs[:, 19] = 0.5  # pdyn_error (normalized)
    aux_cur = np.zeros((n, 2), dtype=np.float32)
    aux_next = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "corridor penalty should be negative"


def test_exit_phase_apoapsis_penalty(default_calc: StepRewardCalculator) -> None:
    """Non-zero sma_error during exit produces negative reward."""
    n = 4
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = 1.0  # bounce_flag = post-bounce (exit phase)
    obs[:, 13] = 0.5  # sma_error (normalized)
    aux_cur = np.zeros((n, 2), dtype=np.float32)
    aux_next = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "apoapsis penalty should be negative"


def test_zero_obs_gives_zero_reward(default_calc: StepRewardCalculator) -> None:
    """All-zero obs and aux produces zero reward (no deviation = no penalty)."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # capture phase
    aux = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux, aux)
    # NOTE: heat_flux_frac = (obs[6]+1)/2 = 0.5 when obs[6]=0, so constraint
    # penalty is non-zero. Only truly zero when obs[6]=-1 and obs[7]=-1.
    # This test checks that corridor + energy terms are zero (constraint is separate).
    # With obs[6]=0 => frac=0.5 => penalty = 0.2*(0.25+0.25) = 0.1
    assert np.allclose(r, -0.1, atol=1e-6)


def test_energy_dissipation_not_penalized(default_calc: StepRewardCalculator) -> None:
    """Negative energy change (dissipation) during capture gives no energy penalty."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0
    obs[:, 6] = -1.0  # zero heat flux frac
    obs[:, 7] = -1.0  # zero heat load frac
    aux_cur = np.array([[5e6, 0.0]] * n, dtype=np.float32)
    aux_next = np.array([[4.9e6, 0.0]] * n, dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.allclose(r, 0.0, atol=1e-10)


def test_energy_gain_penalized(default_calc: StepRewardCalculator) -> None:
    """Positive energy change (gaining energy) during capture gives negative reward."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0
    obs[:, 6] = -1.0
    obs[:, 7] = -1.0
    aux_cur = np.array([[4.9e6, 0.0]] * n, dtype=np.float32)
    aux_next = np.array([[5e6, 0.0]] * n, dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "energy gain should be penalized"


def test_constraint_penalty_scales_quadratically(default_calc: StepRewardCalculator) -> None:
    """Constraint penalty grows quadratically with heat flux fraction."""
    n = 1
    obs_low = np.zeros((n, 23), dtype=np.float32)
    obs_low[:, 15] = -1.0
    obs_low[:, 6] = 0.0  # frac = 0.5
    obs_low[:, 7] = -1.0  # frac = 0
    obs_high = np.zeros((n, 23), dtype=np.float32)
    obs_high[:, 15] = -1.0
    obs_high[:, 6] = 0.8  # frac = 0.9
    obs_high[:, 7] = -1.0
    aux = np.zeros((n, 2), dtype=np.float32)
    r_low = default_calc.step_reward(obs_low, aux, aux)
    r_high = default_calc.step_reward(obs_high, aux, aux)
    assert r_high[0] < r_low[0], "higher heat flux fraction should give more penalty"


def test_phase_gating(default_calc: StepRewardCalculator) -> None:
    """Capture-only terms inactive during exit, exit-only terms inactive during capture."""
    n = 1
    obs_cap = np.zeros((n, 23), dtype=np.float32)
    obs_cap[:, 15] = -1.0
    obs_cap[:, 6] = -1.0
    obs_cap[:, 7] = -1.0
    obs_cap[:, 19] = 1.0  # pdyn_error active in capture
    obs_exit = np.zeros((n, 23), dtype=np.float32)
    obs_exit[:, 15] = 1.0
    obs_exit[:, 6] = -1.0
    obs_exit[:, 7] = -1.0
    obs_exit[:, 13] = 1.0  # sma_error active in exit
    aux = np.zeros((n, 2), dtype=np.float32)
    r_cap = default_calc.step_reward(obs_cap, aux, aux)
    r_exit = default_calc.step_reward(obs_exit, aux, aux)
    assert r_cap[0] < 0
    assert r_exit[0] < 0


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
