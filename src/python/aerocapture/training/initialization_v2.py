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

import math
from typing import Any

import numpy as np

from aerocapture.training.config import _layer_n_params
from aerocapture.training.encoding import _MAMBA_DT_BIAS_SEED
from aerocapture.training.initialization import compute_layer_bound

BIAS_NOISE_STD = 0.01
_INIT_JITTER_STD = 0.01


def init_v2_population(
    architecture: list[Any],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_pop, n_params) initial chromosomes for the PSO v2 path."""
    n_params = sum(_layer_n_params(entry) for entry in architecture)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    cursor = 0
    for layer_idx, entry in enumerate(architecture):
        n = _layer_n_params(entry)
        slab = pop[:, cursor : cursor + n]
        _fill_layer(entry, slab, bound_multiplier, rng, layer_idx=layer_idx)
        cursor += n

    return pop


def _fill_layer(entry: Any, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator, layer_idx: int = 0) -> None:
    from aerocapture.training.rl.schemas import TransformerSpec

    if isinstance(entry, TransformerSpec):
        _fill_transformer(entry, slab, bound_multiplier, rng, layer_idx=layer_idx)
        return

    # Normalise Pydantic models to plain dicts so downstream helpers can use [].
    if hasattr(entry, "model_dump"):
        entry = entry.model_dump()

    t = entry["type"]
    if t == "dense":
        _fill_dense(entry, slab, bound_multiplier, rng)
    elif t == "gru":
        _fill_gru(entry, slab, bound_multiplier, rng)
    elif t == "lstm":
        _fill_lstm(entry, slab, bound_multiplier, rng)
    elif t == "window":
        # Zero trainable params: slab has width 0, nothing to fill. The outer
        # cursor advanced by _layer_n_params(window) == 0, so this branch only
        # exists to stay off the "raise ValueError" path.
        assert slab.shape[1] == 0, f"window slab expected 0-width, got {slab.shape[1]}"
    elif t == "transformer":
        # Should not reach here: TransformerSpec Pydantic path is handled above,
        # but a raw dict entry with type="transformer" is also valid.
        _fill_transformer_dict(entry, slab, bound_multiplier, rng, layer_idx=layer_idx)
    elif t == "mamba":
        _fill_mamba(entry, slab, bound_multiplier, rng, layer_idx=layer_idx)
    elif t == "mamba3":
        _fill_mamba3(entry, slab, bound_multiplier, rng, layer_idx=layer_idx)
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


def _fill_transformer(entry: object, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator, layer_idx: int = 0) -> None:
    """Fill a TransformerSpec Pydantic object by delegating to _transformer_specs ParamSpec bounds."""
    from aerocapture.training.encoding import _transformer_specs

    specs = _transformer_specs(entry, layer_idx, bound_multiplier)  # type: ignore[arg-type]
    pop_n = slab.shape[0]
    for j, ps in enumerate(specs):
        slab[:, j] = rng.uniform(ps.p_min, ps.p_max, size=pop_n)


def _fill_transformer_dict(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator, layer_idx: int = 0) -> None:
    """Fill a raw dict transformer entry by converting to TransformerSpec and delegating."""
    from aerocapture.training.rl.schemas import TransformerSpec

    spec = TransformerSpec(**entry)
    _fill_transformer(spec, slab, bound_multiplier, rng, layer_idx=layer_idx)


def _fill_mamba(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator, layer_idx: int = 0) -> None:
    """Fill a Mamba layer slab in-place.

    Canonical flat order (matches Rust `LayerWeights for MambaLayer::to_flat`
    and `_mamba_specs` in encoding.py):
      1. x_proj_w  [(dt_rank + 2*d_state), d_inner] row-major -- Xavier uniform around 0
      2. dt_proj_w [d_inner, dt_rank]                row-major -- Xavier * dt_rank^{-0.5} around 0
      3. dt_proj_b [d_inner]                                   -- shared center + per-individual jitter
      4. a_log     [d_inner, d_state]                row-major -- HiPPO log(n+1) + per-individual jitter
      5. d_skip    [d_inner]                                   -- 1.0 + per-individual jitter

    Per-individual `N(0, _INIT_JITTER_STD * bound_multiplier)` jitter on slices 3-5
    ensures PSO population diversity on the otherwise-shared-center parameters.
    Sub-RNG seed for dt_proj_b mixes `_MAMBA_DT_BIAS_SEED` with `layer_idx` so
    each stacked Mamba layer gets its own dt-bias centers (different timescales
    per layer) while still agreeing with `_mamba_specs` bounds for the same layer.
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    d_inner = int(entry["input_size"])
    d_state = int(entry["d_state"])
    dt_rank = resolve_mamba_dt_rank(entry)
    pop_n = slab.shape[0]
    jitter_std = _INIT_JITTER_STD * bound_multiplier

    # Shared per-channel dt_proj_b centers (same sub-RNG seed as _mamba_specs).
    # Mix layer_idx into the seed so stacked Mamba layers diverge at init.
    local_rng = np.random.default_rng(_MAMBA_DT_BIAS_SEED ^ layer_idx)
    dt_bias_centers = np.log(np.expm1(local_rng.uniform(1e-3, 1e-1, size=d_inner)))

    # HiPPO a_log centers: outer d_inner, inner d_state (row-major)
    a_log_centers = np.tile(np.log(np.arange(d_state) + 1.0), d_inner)  # shape (d_inner*d_state,)

    # Cursor positions within the slab
    n_xp = (dt_rank + 2 * d_state) * d_inner
    n_dw = d_inner * dt_rank
    n_db = d_inner
    n_al = d_inner * d_state
    n_ds = d_inner

    c0 = 0
    c1 = c0 + n_xp
    c2 = c1 + n_dw
    c3 = c2 + n_db
    c4 = c3 + n_al
    # c4 + n_ds == slab.shape[1]

    # 1. x_proj_w: Xavier uniform around 0 (per-individual, no shared center)
    fan_in_xp = d_inner
    fan_out_xp = dt_rank + 2 * d_state
    bound_xp = math.sqrt(6.0 / (fan_in_xp + fan_out_xp)) * bound_multiplier
    slab[:, c0:c1] = rng.uniform(-bound_xp, bound_xp, size=(pop_n, n_xp))

    # 2. dt_proj_w: Xavier * dt_rank^{-0.5} around 0 (per-individual)
    fan_in_dt = dt_rank
    fan_out_dt = d_inner
    bound_dt = math.sqrt(6.0 / (fan_in_dt + fan_out_dt)) / math.sqrt(max(dt_rank, 1)) * bound_multiplier
    slab[:, c1:c2] = rng.uniform(-bound_dt, bound_dt, size=(pop_n, n_dw))

    # 3. dt_proj_b: shared centers + per-individual jitter
    slab[:, c2:c3] = dt_bias_centers + rng.normal(0.0, jitter_std, size=(pop_n, n_db))

    # 4. a_log: HiPPO centers + per-individual jitter
    slab[:, c3:c4] = a_log_centers + rng.normal(0.0, jitter_std, size=(pop_n, n_al))

    # 5. d_skip: 1.0 + per-individual jitter
    slab[:, c4 : c4 + n_ds] = 1.0 + rng.normal(0.0, jitter_std, size=(pop_n, n_ds))


def _fill_mamba3(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator, layer_idx: int = 0) -> None:
    """Fill a Mamba-3 layer slab in-place.

    Base blocks (x_proj_w, dt_proj_w, dt_proj_b, a_log) mirror `_fill_mamba`, with
    two conditional blocks inserted before d_skip (canonical flat order, matching
    Rust `Mamba3Layer::to_flat` and `_mamba3_specs`):
      a_imag       [d_inner, d_state] -- S4D-Lin ramp pi*(n+1)/(d_state+1) + jitter [iff complex]
      lambda_logit [d_inner]          -- center +4 (near-euler) + jitter            [iff trapezoidal]
    """
    from aerocapture.training.config import resolve_mamba_dt_rank

    d_inner = int(entry["input_size"])
    d_state = int(entry["d_state"])
    dt_rank = resolve_mamba_dt_rank(entry)
    complex_mode = entry.get("state_mode", "real") == "complex"
    trapezoidal = entry.get("discretization", "euler") == "trapezoidal"
    pop_n = slab.shape[0]
    jitter_std = _INIT_JITTER_STD * bound_multiplier

    local_rng = np.random.default_rng(_MAMBA_DT_BIAS_SEED ^ layer_idx)
    dt_bias_centers = np.log(np.expm1(local_rng.uniform(1e-3, 1e-1, size=d_inner)))
    a_log_centers = np.tile(np.log(np.arange(d_state) + 1.0), d_inner)

    n_xp = (dt_rank + 2 * d_state) * d_inner
    n_dw = d_inner * dt_rank
    n_db = d_inner
    n_al = d_inner * d_state

    c = 0
    fan_out_xp = dt_rank + 2 * d_state
    bound_xp = math.sqrt(6.0 / (d_inner + fan_out_xp)) * bound_multiplier
    slab[:, c : c + n_xp] = rng.uniform(-bound_xp, bound_xp, size=(pop_n, n_xp))
    c += n_xp

    bound_dt = math.sqrt(6.0 / (dt_rank + d_inner)) / math.sqrt(max(dt_rank, 1)) * bound_multiplier
    slab[:, c : c + n_dw] = rng.uniform(-bound_dt, bound_dt, size=(pop_n, n_dw))
    c += n_dw

    slab[:, c : c + n_db] = dt_bias_centers + rng.normal(0.0, jitter_std, size=(pop_n, n_db))
    c += n_db

    slab[:, c : c + n_al] = a_log_centers + rng.normal(0.0, jitter_std, size=(pop_n, n_al))
    c += n_al

    if complex_mode:
        # S4D-Lin rotation-frequency ramp pi*(n+1)/(d_state+1), tiled over d_inner.
        a_imag_centers = np.tile(math.pi * (np.arange(d_state) + 1.0) / (d_state + 1.0), d_inner)
        slab[:, c : c + n_al] = a_imag_centers + rng.normal(0.0, jitter_std, size=(pop_n, n_al))
        c += n_al

    if trapezoidal:
        # lambda_logit center +4 (sigmoid ~ 0.98 == near-euler start).
        slab[:, c : c + d_inner] = 4.0 + rng.normal(0.0, jitter_std, size=(pop_n, d_inner))
        c += d_inner

    slab[:, c : c + d_inner] = 1.0 + rng.normal(0.0, jitter_std, size=(pop_n, d_inner))
    c += d_inner
