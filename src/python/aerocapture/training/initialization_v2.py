"""Activation-aware initialization for v2 architectures.

Dispatches per layer type to produce initial PSO chromosomes that match
theoretical Xavier/He/LeCun bounds per activation, plus LSTM forget-bias
init to 1.0 (Jozefowicz, Zaremba & Sutskever 2015).

Flat-weight layout per layer must match the Rust LayerWeights trait:
  - Dense: row-major W (O*I) + b (O)
  - Gru:   row-major weight_ih (3H*I) + weight_hh (3H*H) + bias_ih (3H) + bias_hh (3H)
  - Lstm:  row-major weight_ih (4H*I) + weight_hh (4H*H) + bias_ih (4H) + bias_hh (4H)

Gate order on the multi-H axis:
  - Gru:  (r, z, n)
  - Lstm: (i, f, g, o)  -- forget slice is rows [H:2H] of the 4H axis.

Bias init convention:
  - Dense biases: uniform in [-bound, +bound] (matches existing dense path).
  - Gru biases:   N(0, 0.01 * bound_multiplier).
  - Lstm i/g/o biases: N(0, 0.01 * bound_multiplier).
  - Lstm forget-bias slice on bias_ih: 1.0 + N(0, 0.01 * bound_multiplier).
  - Lstm bias_hh forget slice: N(0, 0.01 * bound_multiplier)  -- forget contribution
    is on bias_ih only (gate is sigmoid(bias_ih + bias_hh + ...); doubling would
    give sigmoid(2.0) which is too strong a "remember").
"""

from __future__ import annotations

import numpy as np

from aerocapture.training.config import _layer_n_params
from aerocapture.training.initialization import compute_layer_bound

BIAS_NOISE_STD = 0.01


def init_v2_population(
    architecture: list[dict],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_pop, n_params) initial chromosomes for the PSO v2 path."""
    n_params = sum(_layer_n_params(entry) for entry in architecture)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    cursor = 0
    for entry in architecture:
        n = _layer_n_params(entry)
        slab = pop[:, cursor : cursor + n]
        _fill_layer(entry, slab, bound_multiplier, rng)
        cursor += n

    return pop


def _fill_layer(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    t = entry["type"]
    if t == "dense":
        _fill_dense(entry, slab, bound_multiplier, rng)
    elif t == "gru":
        _fill_gru(entry, slab, bound_multiplier, rng)
    elif t == "lstm":
        _fill_lstm(entry, slab, bound_multiplier, rng)
    else:
        raise ValueError(f"init_v2_population: unknown layer type {t!r}")


def _fill_dense(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    fan_in = int(entry["input_size"])
    fan_out = int(entry["output_size"])
    activation = entry["activation"]
    bound = bound_multiplier * compute_layer_bound(fan_in, fan_out, activation)

    n_w = fan_out * fan_in
    n_b = fan_out
    slab[:, :n_w] = rng.uniform(-bound, bound, size=(slab.shape[0], n_w))
    slab[:, n_w : n_w + n_b] = rng.uniform(-bound, bound, size=(slab.shape[0], n_b))


def _fill_gru(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    fan_in = int(entry["input_size"])
    hidden = int(entry["hidden_size"])
    three_h = 3 * hidden
    n_w_ih = three_h * fan_in
    n_w_hh = three_h * hidden

    w_ih_bound = bound_multiplier * compute_layer_bound(fan_in, three_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(hidden, three_h, "tanh")
    bias_std = BIAS_NOISE_STD * bound_multiplier

    pop_n = slab.shape[0]
    slab[:, :n_w_ih] = rng.uniform(-w_ih_bound, w_ih_bound, size=(pop_n, n_w_ih))
    slab[:, n_w_ih : n_w_ih + n_w_hh] = rng.uniform(-w_hh_bound, w_hh_bound, size=(pop_n, n_w_hh))
    bias_start = n_w_ih + n_w_hh
    slab[:, bias_start : bias_start + three_h] = rng.normal(0.0, bias_std, size=(pop_n, three_h))
    slab[:, bias_start + three_h : bias_start + 2 * three_h] = rng.normal(0.0, bias_std, size=(pop_n, three_h))


def _fill_lstm(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    fan_in = int(entry["input_size"])
    hidden = int(entry["hidden_size"])
    four_h = 4 * hidden
    n_w_ih = four_h * fan_in
    n_w_hh = four_h * hidden

    w_ih_bound = bound_multiplier * compute_layer_bound(fan_in, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(hidden, four_h, "tanh")
    bias_std = BIAS_NOISE_STD * bound_multiplier

    pop_n = slab.shape[0]
    slab[:, :n_w_ih] = rng.uniform(-w_ih_bound, w_ih_bound, size=(pop_n, n_w_ih))
    slab[:, n_w_ih : n_w_ih + n_w_hh] = rng.uniform(-w_hh_bound, w_hh_bound, size=(pop_n, n_w_hh))

    bias_ih_start = n_w_ih + n_w_hh
    bias_hh_start = bias_ih_start + four_h

    # Start with small Gaussian noise on all biases
    slab[:, bias_ih_start : bias_ih_start + four_h] = rng.normal(0.0, bias_std, size=(pop_n, four_h))
    slab[:, bias_hh_start : bias_hh_start + four_h] = rng.normal(0.0, bias_std, size=(pop_n, four_h))

    # Override forget slice (rows [H:2H] of the 4H axis) on bias_ih to ~1.0.
    # Forget contribution is put on bias_ih ONLY; bias_hh forget stays ~ 0
    # because the gate is sigmoid(bias_ih + bias_hh + ...) and we don't want
    # forget = sigmoid(2.0).
    slab[:, bias_ih_start + hidden : bias_ih_start + 2 * hidden] = 1.0 + rng.normal(0.0, bias_std, size=(pop_n, hidden))
