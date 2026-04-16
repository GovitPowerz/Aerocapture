"""Reward shaping tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest  # noqa: F401
from aerocapture.training.rl.rewards import PBRSShaper, compute_terminal_cost


def test_disabled_shaper_returns_zero() -> None:
    shaper = PBRSShaper(enabled=False)
    aux = np.zeros((4, 2), dtype=np.float32)
    aux_next = np.ones((4, 2), dtype=np.float32)
    r = shaper.step_reward(aux, aux_next, gamma=0.99)
    assert np.allclose(r, 0.0)


def test_enabled_shaper_telescoping_identity() -> None:
    """Sum of step rewards with gamma=1 telescopes to phi(s_n) - phi(s_0).

    Potential-based shaping guarantees:
        sum_t (gamma * phi(s_{t+1}) - phi(s_t)) = phi(s_n) - phi(s_0)  for gamma=1
    """
    rng = np.random.default_rng(0)
    n_steps = 20
    # aux sequence: (n_steps+1, 2) with [energy, pdyn]
    aux_seq = rng.standard_normal((n_steps + 1, 2)).astype(np.float32)

    shaper = PBRSShaper(
        enabled=True,
        alpha=1.0,
        pdyn_scale=1.0,
        ref_fn=lambda e: np.zeros_like(e),
    )

    def phi(a: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        # matches shaper.phi with pdyn_ref=0: -|pdyn|
        return -np.abs(a[..., 1].astype(np.float64))

    total = np.zeros(1)
    for t in range(n_steps):
        total += shaper.step_reward(aux_seq[t : t + 1], aux_seq[t + 1 : t + 2], gamma=1.0)
    expected = phi(aux_seq[n_steps]) - phi(aux_seq[0])
    assert np.allclose(total, expected, atol=1e-6)


def test_terminal_cost_matches_evaluate_module() -> None:
    from aerocapture.training.evaluate import compute_cost

    fc = np.zeros((1, 52))
    fc[0, 41] = 100.0  # dv_total
    fc[0, 17] = 5.0  # g-load
    fc[0, 16] = 150.0  # peak heat flux
    fc[0, 28] = 10.0  # heat load MJ/m2
    expected = compute_cost(fc)
    actual = compute_terminal_cost(fc[0])
    assert abs(actual - expected) < 1e-9
