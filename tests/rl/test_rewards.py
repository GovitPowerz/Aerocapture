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
    aux = np.zeros((4, 7), dtype=np.float32)
    r = default_calc.step_reward(obs, obs, aux, aux)
    phi = default_calc._potential(obs, aux)
    expected = default_calc.gamma * phi - phi
    assert np.allclose(r, expected, atol=1e-8)


def test_pbrs_improvement_gives_positive_reward(default_calc: StepRewardCalculator) -> None:
    """Transition from worse to better state (Phi increases) -> reward > 0."""
    obs_bad = _make_obs(n=1, **{"15": -1.0, "19": 1.0})  # big pdyn error
    obs_good = _make_obs(n=1, **{"15": -1.0, "19": 0.0})  # zero pdyn error
    aux = np.zeros((1, 7), dtype=np.float32)
    r = default_calc.step_reward(obs_bad, obs_good, aux, aux)
    assert r[0] > 0


def test_pbrs_degradation_gives_negative_reward(default_calc: StepRewardCalculator) -> None:
    """Transition from better to worse (Phi decreases) -> reward < 0."""
    obs_good = _make_obs(n=1, **{"15": -1.0, "19": 0.0})
    obs_bad = _make_obs(n=1, **{"15": -1.0, "19": 1.0})
    aux = np.zeros((1, 7), dtype=np.float32)
    r = default_calc.step_reward(obs_good, obs_bad, aux, aux)
    assert r[0] < 0


def test_potential_phase_gating_capture(default_calc: StepRewardCalculator) -> None:
    """Capture phase: pdyn_error contributes to Phi, sma_error does not."""
    obs = _make_obs(n=1, **{"15": -1.0, "19": 1.0, "13": 1.0})
    aux = np.zeros((1, 7), dtype=np.float32)
    phi = default_calc._potential(obs, aux)
    # Only capture terms active. Phi = -(corridor + constraint). Constraint=0 here.
    expected = -default_calc.corridor_weight * 1.0**2
    assert np.isclose(phi[0], expected, atol=1e-8)


def test_potential_phase_gating_exit(default_calc: StepRewardCalculator) -> None:
    """Exit phase: sma_error contributes to Phi, pdyn_error does not."""
    obs = _make_obs(n=1, **{"15": 1.0, "19": 1.0, "13": 1.0})
    aux = np.zeros((1, 7), dtype=np.float32)
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


# ---------------------------------------------------------------------------
# DV-correction potential mode
# ---------------------------------------------------------------------------


def _dv_calc(dv1_weight: float = 1.0, dv2_weight: float = 1.0, dv3_weight: float = 1.0) -> StepRewardCalculator:
    return StepRewardCalculator(
        input_mask=list(range(23)),
        potential="dv",
        constraint_weight=0.2,
        gamma=0.99,
        dv1_weight=dv1_weight,
        dv2_weight=dv2_weight,
        dv3_weight=dv3_weight,
    )


def _aux(n: int = 1, dv1: float = 0.0, dv2: float = 0.0, dv3: float = 0.0, hf: float = 0.0, hl: float = 0.0) -> np.ndarray:
    aux = np.zeros((n, 7), dtype=np.float32)
    aux[:, 2] = dv1
    aux[:, 3] = dv2
    aux[:, 4] = dv3
    aux[:, 5] = hf
    aux[:, 6] = hl
    return aux


def test_dv_potential_value() -> None:
    calc = _dv_calc(dv1_weight=1.0, dv2_weight=1.0, dv3_weight=1.0)
    obs = _make_obs(n=1)  # hf_frac = hl_frac = 0
    phi = calc._potential(obs, _aux(dv1=100.0, dv2=20.0, dv3=5.0))
    assert np.isclose(phi[0], -(100.0 + 20.0 + 5.0), atol=1e-6)
    # dv1 is physically sign-changing (v_cur - v_tgt); negative dv1 -> positive Phi.
    phi_neg = calc._potential(obs, _aux(dv1=-50.0))
    assert np.isclose(phi_neg[0], 50.0, atol=1e-6)


def test_dv_potential_weights_linear() -> None:
    calc = _dv_calc(dv1_weight=1.0, dv2_weight=2.0, dv3_weight=0.0)
    obs = _make_obs(n=1)
    phi = calc._potential(obs, _aux(dv1=10.0, dv2=10.0, dv3=10.0))
    assert np.isclose(phi[0], -(10.0 + 20.0 + 0.0), atol=1e-6)


def test_dv_potential_keeps_thermal_term() -> None:
    calc = _dv_calc()
    obs = _make_obs(n=1)
    phi = calc._potential(obs, _aux(hf=1.0, hl=1.0))  # dv = 0, hf_frac = hl_frac = 1 (raw)
    assert np.isclose(phi[0], -0.2 * (1.0 + 1.0), atol=1e-6)


def test_dv_reward_positive_when_dv_decreases() -> None:
    calc = _dv_calc()
    obs = _make_obs(n=1)
    r = calc.step_reward(obs, obs, _aux(dv1=200.0), _aux(dv1=100.0))
    # gamma*Phi(next) - Phi(cur) = 0.99*(-100) - (-200) = 101 > 0
    assert np.isclose(r[0], 101.0, atol=1e-4)


def test_dv_mode_requires_no_obs_indices() -> None:
    # dv mode reads only the aux channel (DV + raw thermal fractions); any mask is legal.
    StepRewardCalculator(input_mask=[0, 1, 2], potential="dv")


def test_wrong_aux_width_raises() -> None:
    calc = _dv_calc()
    with pytest.raises(ValueError, match="aux must have 7 columns"):
        calc._potential(_make_obs(n=1), np.zeros((1, 5), dtype=np.float32))


def _normalize(raw: float, spec: dict) -> float:
    """Mirror of Rust apply_norm: transform((raw - center) / scale)."""
    v = (raw - spec["center"]) / spec["scale"]
    if spec["transform"] == "asinh":
        return float(np.arcsinh(v))
    if spec["transform"] == "tanh":
        return float(np.tanh(v))
    return float(v)


@pytest.mark.parametrize("hf_raw, hl_raw", [(0.0, 0.0), (0.37, 0.81), (1.0, 1.0), (1.3, 0.05)])
def test_thermal_term_uses_raw_aux_under_default_normalization(hf_raw: float, hl_raw: float) -> None:
    """Obs carries the fractions through the sim's DEFAULT normalization; the reward must
    still see the raw values (it reads aux, so the normalization cannot drift it)."""
    aerocapture_rs = pytest.importorskip("aerocapture_rs")
    norm = aerocapture_rs.default_normalization()
    calc = _dv_calc()
    obs = _make_obs(n=1, **{"6": _normalize(hf_raw, norm[6]), "7": _normalize(hl_raw, norm[7])})
    phi = calc._potential(obs, _aux(hf=hf_raw, hl=hl_raw))
    assert np.isclose(phi[0], -calc.constraint_weight * (hf_raw**2 + hl_raw**2), atol=1e-6)


@pytest.mark.parametrize("hf_raw, hl_raw", [(0.0, 0.0), (0.37, 0.81), (1.0, 1.0)])
def test_thermal_term_uses_raw_aux_under_toml_override(hf_raw: float, hl_raw: float) -> None:
    """Same with an asinh [network.normalization] override (the atan2 PPO config's shape)."""
    override_hf = {"transform": "asinh", "scale": 0.339, "center": 0.400}
    override_hl = {"transform": "asinh", "scale": 0.301, "center": 0.377}
    calc = _dv_calc()
    obs = _make_obs(n=1, **{"6": _normalize(hf_raw, override_hf), "7": _normalize(hl_raw, override_hl)})
    phi = calc._potential(obs, _aux(hf=hf_raw, hl=hl_raw))
    assert np.isclose(phi[0], -calc.constraint_weight * (hf_raw**2 + hl_raw**2), atol=1e-6)
    # And the obs columns are irrelevant: flipping them leaves Phi unchanged.
    obs_flipped = _make_obs(n=1, **{"6": -obs[0, 6], "7": -obs[0, 7]})
    assert np.isclose(calc._potential(obs_flipped, _aux(hf=hf_raw, hl=hl_raw))[0], phi[0], atol=1e-9)


def test_invalid_potential_raises() -> None:
    with pytest.raises(ValueError, match="potential must be"):
        StepRewardCalculator(input_mask=list(range(23)), potential="bogus")  # type: ignore[arg-type]
