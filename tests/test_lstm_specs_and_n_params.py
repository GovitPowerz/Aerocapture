"""LSTM integration: _lstm_specs produces correct bounds/order; _layer_n_params + _layer_output_size handle LSTM."""

from __future__ import annotations

import math

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _lstm_specs, nn_param_specs_from_v2
from aerocapture.training.rl.schemas import LstmSpec


def test_layer_n_params_lstm() -> None:
    entry = {"type": "lstm", "input_size": 32, "hidden_size": 32}
    # 4H*I + 4H*H + 8H = 4*32*32 + 4*32*32 + 8*32 = 4096 + 4096 + 256 = 8448
    assert _layer_n_params(entry) == 8448


def test_layer_output_size_lstm() -> None:
    entry = {"type": "lstm", "input_size": 32, "hidden_size": 16}
    assert _layer_output_size(entry) == 16


def test_lstm_specs_count_matches_flat_weights() -> None:
    spec = LstmSpec(type="lstm", input_size=5, hidden_size=3)
    specs = _lstm_specs(spec, layer_idx=1, bound_multiplier=1.0)
    # 4H*I + 4H*H + 2*4H = 60 + 36 + 24 = 120
    assert len(specs) == 120


def test_lstm_specs_flat_order_matches_gru_pattern() -> None:
    """First 4H*I specs are weight_ih; next 4H*H are weight_hh; next 4H are bias_ih; last 4H are bias_hh."""
    spec = LstmSpec(type="lstm", input_size=4, hidden_size=2)
    specs = _lstm_specs(spec, layer_idx=0, bound_multiplier=1.0)
    hidden = 2
    n_in = 4
    four_h = 4 * hidden
    # weight_ih block: first 4H*I = 32 entries, name prefix "w_ih"
    for j in range(four_h * n_in):
        assert specs[j].name == f"w_ih0_{j}"
    # weight_hh block: next 4H*H = 16 entries, name prefix "w_hh"
    offset = four_h * n_in
    for j in range(four_h * hidden):
        assert specs[offset + j].name == f"w_hh0_{j}"
    # bias_ih block
    offset = four_h * n_in + four_h * hidden
    for j in range(four_h):
        assert specs[offset + j].name == f"b_ih0_{j}"
    # bias_hh block
    offset = four_h * n_in + four_h * hidden + four_h
    for j in range(four_h):
        assert specs[offset + j].name == f"b_hh0_{j}"


def test_lstm_specs_bounds_are_tanh_xavier() -> None:
    n_in, hidden = 5, 4
    spec = LstmSpec(type="lstm", input_size=n_in, hidden_size=hidden)
    specs = _lstm_specs(spec, layer_idx=0, bound_multiplier=1.0)

    # First weight spec: bounds symmetric around 0
    ps = specs[0]
    assert ps.p_min < 0
    assert ps.p_max > 0
    assert math.isclose(ps.p_min, -ps.p_max)

    # Last bias spec: tighter bounds (0.1 * bound_multiplier)
    bias_spec = specs[-1]
    assert math.isclose(bias_spec.p_max, 0.1, abs_tol=1e-12)


def test_nn_param_specs_from_v2_dispatches_lstm() -> None:
    architecture = [
        LstmSpec(type="lstm", input_size=4, hidden_size=2),
    ]
    specs = nn_param_specs_from_v2(architecture, bound_multiplier=1.0)
    # Lstm: 4*2*4 + 4*2*2 + 8*2 = 32 + 16 + 16 = 64
    assert len(specs) == 64
