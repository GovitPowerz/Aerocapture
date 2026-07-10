"""init_v2_population centers for the cfc/slstm/mlstm probe layers."""

from __future__ import annotations

import numpy as np
from aerocapture.training.initialization_v2 import init_v2_population


def _arch(mid: dict) -> list[dict]:
    return [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        mid,
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]


def test_cfc_population_finite_and_shaped() -> None:
    arch = _arch({"type": "cfc", "input_size": 4, "hidden_size": 4, "backbone_units": 5})
    pop = init_v2_population(arch, n_pop=8, bound_multiplier=2.0, rng=np.random.default_rng(0))
    # dense 16, cfc 5*8+5+4*(4*5+4) = 141, dense 10 -> 167  (dense1 = 3*4+4 = 16, dense2 = 4*2+2 = 10)
    assert pop.shape == (8, 16 + 141 + 10)
    assert np.all(np.isfinite(pop))


def test_slstm_forget_bias_centered_at_two() -> None:
    h, i = 4, 4
    arch = _arch({"type": "slstm", "input_size": i, "hidden_size": h})
    pop = init_v2_population(arch, n_pop=64, bound_multiplier=2.0, rng=np.random.default_rng(1))
    dense1 = 3 * 4 + 4
    b0 = dense1 + 4 * h * i + 4 * h * h  # bias start inside the slstm slab
    forget = pop[:, b0 + h : b0 + 2 * h]
    other = pop[:, b0 : b0 + h]
    assert abs(float(forget.mean()) - 2.0) < 0.1
    assert abs(float(other.mean())) < 0.1


def test_mlstm_forget_bias_centered_at_two() -> None:
    h, i = 4, 4
    arch = _arch({"type": "mlstm", "input_size": i, "hidden_size": h})
    pop = init_v2_population(arch, n_pop=64, bound_multiplier=2.0, rng=np.random.default_rng(2))
    dense1 = 3 * 4 + 4
    b_f_idx = dense1 + 4 * (h * i + h) + 2 * (i + 1) - 1  # last mlstm element
    assert abs(float(pop[:, b_f_idx].mean()) - 2.0) < 0.1
    b_i_idx = dense1 + 4 * (h * i + h) + i  # w_i (I) then b_i
    assert abs(float(pop[:, b_i_idx].mean())) < 0.1
