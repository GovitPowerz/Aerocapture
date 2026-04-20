"""init_v2_population produces activation-aware initial chromosomes per layer type.

Tests per-layer statistics: Xavier bounds for dense/gru/lstm weights, small bias
noise, and LSTM forget-bias-1 init (Jozefowicz et al 2015).
"""

from __future__ import annotations

import numpy as np
from aerocapture.training.config import _layer_n_params
from aerocapture.training.initialization_v2 import init_v2_population


def _layer_offsets(architecture: list[dict]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for entry in architecture:
        n = _layer_n_params(entry)
        offsets.append((cursor, cursor + n))
        cursor += n
    return offsets


def test_init_v2_population_shape() -> None:
    architecture = [
        {"type": "dense", "input_size": 16, "output_size": 32, "activation": "tanh"},
        {"type": "lstm", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(0)
    pop = init_v2_population(architecture, n_pop=50, bound_multiplier=1.0, rng=rng)
    n_expected = 544 + 8448 + 66  # = 9058
    assert pop.shape == (50, n_expected)
    assert np.all(np.isfinite(pop))


def test_init_v2_population_forget_bias_slice_init_to_one() -> None:
    architecture = [
        {"type": "dense", "input_size": 16, "output_size": 32, "activation": "tanh"},
        {"type": "lstm", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(0)
    pop = init_v2_population(architecture, n_pop=1024, bound_multiplier=1.0, rng=rng)

    offsets = _layer_offsets(architecture)
    lstm_start, _ = offsets[1]
    H, in_size = 32, 32
    four_h = 4 * H

    bias_ih_start = lstm_start + four_h * in_size + four_h * H
    bias_hh_start = bias_ih_start + four_h

    # Forget slice on bias_ih is rows [H:2H] of the 4H axis.
    forget_ih = pop[:, bias_ih_start + H : bias_ih_start + 2 * H]
    forget_hh = pop[:, bias_hh_start + H : bias_hh_start + 2 * H]

    # Forget bias on ih should mean ~1.0 with small noise
    assert 0.9 < float(forget_ih.mean()) < 1.1, f"forget_ih mean {float(forget_ih.mean())}"
    assert 0.005 < float(forget_ih.std()) < 0.02, f"forget_ih std {float(forget_ih.std())}"

    # Forget bias on hh stays near 0 (+1 is only on ih; gate sum is ih + hh)
    assert -0.01 < float(forget_hh.mean()) < 0.01, f"forget_hh mean {float(forget_hh.mean())}"


def test_init_v2_population_non_forget_biases_small() -> None:
    architecture = [
        {"type": "lstm", "input_size": 8, "hidden_size": 8},
    ]
    rng = np.random.default_rng(1)
    pop = init_v2_population(architecture, n_pop=1024, bound_multiplier=1.0, rng=rng)
    H, in_size = 8, 8
    four_h = 4 * H

    bias_ih_start = four_h * in_size + four_h * H
    # Non-forget gates: i (rows [0:H]), g (rows [2H:3H]), o (rows [3H:4H])
    i_slice = pop[:, bias_ih_start : bias_ih_start + H]
    g_slice = pop[:, bias_ih_start + 2 * H : bias_ih_start + 3 * H]
    o_slice = pop[:, bias_ih_start + 3 * H : bias_ih_start + 4 * H]

    for s, name in [(i_slice, "i"), (g_slice, "g"), (o_slice, "o")]:
        assert abs(float(s.mean())) < 0.005, f"{name}-gate bias mean drifted from 0"
        assert 0.005 < float(s.std()) < 0.02, f"{name}-gate bias std out of range"


def test_init_v2_population_dense_bounds_respected() -> None:
    """Dense layer with tanh activation: weights within Xavier bound."""
    in_size, out_size = 10, 20
    architecture = [
        {"type": "dense", "input_size": in_size, "output_size": out_size, "activation": "tanh"},
    ]
    rng = np.random.default_rng(2)
    pop = init_v2_population(architecture, n_pop=2048, bound_multiplier=1.0, rng=rng)
    # Xavier uniform bound for tanh: sqrt(6/(fan_in + fan_out)) = sqrt(6/30) ~= 0.447
    # Draws are uniform in [-0.447, 0.447], so magnitude <= 0.447.
    assert np.all(np.isfinite(pop))
    assert float(np.abs(pop).max()) <= 0.5  # slightly loose to allow rounding


def test_init_v2_population_gru_weight_bounds_respected() -> None:
    """GRU weight_ih block: Xavier bound = sqrt(6 / (I + 3H))."""
    in_size, H = 16, 32
    architecture = [
        {"type": "gru", "input_size": in_size, "hidden_size": H},
    ]
    rng = np.random.default_rng(3)
    pop = init_v2_population(architecture, n_pop=256, bound_multiplier=1.0, rng=rng)
    three_h = 3 * H
    n_w_ih = three_h * in_size
    n_w_hh = three_h * H
    weight_ih = pop[:, :n_w_ih]
    weight_hh = pop[:, n_w_ih : n_w_ih + n_w_hh]
    # Xavier for (I=16, 3H=96): sqrt(6 / 112) ~= 0.231
    # Xavier for (H=32, 3H=96): sqrt(6 / 128) ~= 0.217
    assert float(np.abs(weight_ih).max()) <= 0.3
    assert float(np.abs(weight_hh).max()) <= 0.3


def test_init_v2_population_dispatches_by_type_not_input_order() -> None:
    """Dense after LSTM, LSTM after Dense: per-layer offsets stay correct."""
    architecture = [
        {"type": "lstm", "input_size": 4, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 3, "activation": "linear"},
    ]
    rng = np.random.default_rng(4)
    pop = init_v2_population(architecture, n_pop=16, bound_multiplier=1.0, rng=rng)
    expected_n = _layer_n_params(architecture[0]) + _layer_n_params(architecture[1])
    assert pop.shape[1] == expected_n
    assert np.all(np.isfinite(pop))


def test_init_v2_population_unknown_type_raises() -> None:
    architecture = [{"type": "mamba", "input_size": 4, "hidden_size": 4}]
    rng = np.random.default_rng(5)
    try:
        init_v2_population(architecture, n_pop=4, bound_multiplier=1.0, rng=rng)
    except ValueError as e:
        assert "mamba" in str(e) or "unknown" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on unknown layer type")


def test_build_initial_population_for_v2_normalizes_to_unit_cube() -> None:
    """Wrapper in train.py produces [0, 1] values and passes forget-bias through."""
    from aerocapture.training.encoding import _lstm_specs
    from aerocapture.training.rl.schemas import LstmSpec
    from aerocapture.training.train import build_initial_population_for_v2

    architecture = [
        {"type": "lstm", "input_size": 8, "hidden_size": 8},
    ]
    # Build ParamSpecs that match the architecture's widened-forget-bias bounds.
    spec = LstmSpec(type="lstm", input_size=8, hidden_size=8)
    param_specs = _lstm_specs(spec, layer_idx=0, bound_multiplier=2.0)

    rng = np.random.default_rng(42)
    pop = build_initial_population_for_v2(
        architecture=architecture,
        n_pop=64,
        bound_multiplier=2.0,
        rng=rng,
        param_specs=param_specs,
    )
    assert pop.shape == (64, len(param_specs))
    assert float(pop.min()) >= 0.0
    assert float(pop.max()) <= 1.0
    assert np.all(np.isfinite(pop))

    # Decode a sample forget-bias position and confirm it lands near 1.0.
    # Forget slice on bias_ih is rows [H:2H] of 4H.
    H, in_size = 8, 8
    four_h = 4 * H
    bias_ih_start = four_h * in_size + four_h * H  # within the LSTM block (single-layer arch)
    forget_idx = bias_ih_start + H  # first f-gate entry
    # Decode: physical = p_min + normalized * (p_max - p_min)
    s = param_specs[forget_idx]
    decoded = s.p_min + pop[:, forget_idx] * (s.p_max - s.p_min)
    # Should cluster near 1.0 with small std (~ 0.01 * bound_multiplier = 0.02)
    assert 0.9 < float(decoded.mean()) < 1.1, f"forget-bias decoded mean {float(decoded.mean())}"


def test_build_initial_population_for_v2_does_not_saturate_search_box() -> None:
    """Regression guard for the bound_multiplier mismatch bug.

    If train.py calls ``nn_param_specs_from_v2`` with a different
    ``bound_multiplier`` than ``build_initial_population_for_v2``, ~49% of
    initial values clip to the [0, 1] search-box edges, silently defeating
    the activation-aware init. Assert the observed distribution stays
    inside the box with room to spare.
    """
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.rl.schemas import LayerSpec
    from aerocapture.training.train import build_initial_population_for_v2
    from pydantic import TypeAdapter

    # Mirror the real train.py call path for a dense+lstm architecture,
    # ensuring nn_param_specs_from_v2 and build_initial_population_for_v2
    # use the same bound_multiplier.
    architecture = [
        {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
        {"type": "lstm", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    validated = TypeAdapter(list[LayerSpec]).validate_python(architecture)
    param_specs = nn_param_specs_from_v2(validated, bound_multiplier=2.0)

    rng = np.random.default_rng(123)
    pop = build_initial_population_for_v2(
        architecture=architecture,
        n_pop=64,
        bound_multiplier=2.0,
        rng=rng,
        param_specs=param_specs,
    )

    # Boundary saturation (values stuck at exactly 0.0 or 1.0) must be
    # negligible -- if this rises above a few percent, the Xavier + forget-
    # bias-1 init has been silently clipped by a bound mismatch.
    saturated = ((pop == 0.0) | (pop == 1.0)).mean()
    assert saturated < 0.01, (
        f"{saturated:.1%} of init values saturate at search-box edges; "
        "bound_multiplier likely mismatched between "
        "nn_param_specs_from_v2 and build_initial_population_for_v2"
    )

    # Empirical std of a correctly sized uniform [0, 1] draw is ~0.289
    # (= 1/sqrt(12)). Tolerate a modest drift (forget-bias slice pulls the
    # global mean slightly off 0.5 because it sits near normalized 0.625).
    empirical_std = float(pop.std())
    assert 0.24 < empirical_std < 0.32, f"init std {empirical_std:.3f} outside expected uniform-[0,1] range"
