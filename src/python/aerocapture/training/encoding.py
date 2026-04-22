"""Real-valued encoding/decoding for optimizer parameters.

All algorithms work on normalized np.ndarray[float64] in [0, 1].
Decoding to physical values happens at evaluation time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from aerocapture.training.initialization import compute_layer_bound
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.rl.schemas import (
    DenseSpec,
    GruSpec,
    LayerSpec,
    LstmSpec,
    TransformerSpec,
    WindowSpec,
)


def decode_normalized(x: npt.NDArray[np.float64], specs: list[ParamSpec]) -> dict[str, float]:
    """Decode a normalized [0,1] vector to physical parameter values.

    Linear params:    value = p_min + x * (p_max - p_min)
    Log-scale params: value = 10^(log10(p_min) + x * (log10(p_max) - log10(p_min)))
    """
    result: dict[str, float] = {}
    for i, s in enumerate(specs):
        xi = float(x[i])
        if s.log_scale:
            log_min = math.log10(s.p_min)
            log_max = math.log10(s.p_max)
            result[s.name] = 10.0 ** (log_min + xi * (log_max - log_min))
        else:
            result[s.name] = s.p_min + xi * (s.p_max - s.p_min)
    return result


def encode_to_normalized(params: dict[str, float], specs: list[ParamSpec]) -> npt.NDArray[np.float64]:
    """Encode physical parameter values to normalized [0,1] vector."""
    x = np.empty(len(specs), dtype=np.float64)
    for i, s in enumerate(specs):
        v = params[s.name]
        if s.log_scale:
            log_min = math.log10(s.p_min)
            log_max = math.log10(s.p_max)
            x[i] = (math.log10(v) - log_min) / (log_max - log_min)
        else:
            x[i] = (v - s.p_min) / (s.p_max - s.p_min)
    return x


def decode_normalized_array(X: npt.NDArray[np.float64], specs: list[ParamSpec]) -> list[dict[str, float]]:
    """Decode a population matrix (n_pop, n_params) to a list of param dicts."""
    return [decode_normalized(X[i], specs) for i in range(X.shape[0])]


def nn_param_specs_from_architecture(
    layer_sizes: list[int],
    activations: list[str],
    bound_multiplier: float = 2.0,
) -> list[ParamSpec]:
    """Generate ParamSpec list for NN weights from architecture.

    Each weight gets bounds [-m * scale, +m * scale] where scale is the
    Xavier/He/LeCun bound for its layer and m is bound_multiplier.
    Biases use the same bounds as their layer's weights.
    """
    specs: list[ParamSpec] = []
    for layer_idx in range(len(activations)):
        fan_in = layer_sizes[layer_idx]
        fan_out = layer_sizes[layer_idx + 1]
        bound = bound_multiplier * compute_layer_bound(fan_in, fan_out, activations[layer_idx])

        for j in range(fan_out):
            for k in range(fan_in):
                specs.append(ParamSpec(f"w{layer_idx}_{j}_{k}", -bound, bound, 0.0))
        for j in range(fan_out):
            specs.append(ParamSpec(f"bias{layer_idx}_{j}", -bound, bound, 0.0))

    return specs


def nn_param_specs_from_v2(
    architecture: Sequence[LayerSpec],
    bound_multiplier: float = 1.0,
) -> list[ParamSpec]:
    """Generate per-parameter ParamSpecs from a v2 architecture list.

    Dispatches per layer type. Phase 0 implements only `dense`.
    For v2 all-dense architectures, output must be numerically identical to
    nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier).
    """
    specs: list[ParamSpec] = []
    for layer_idx, layer in enumerate(architecture):
        specs.extend(_layer_param_specs(layer, layer_idx, bound_multiplier))
    return specs


def _layer_param_specs(layer: LayerSpec, layer_idx: int = 0, bound_multiplier: float = 1.0) -> list[ParamSpec]:
    if isinstance(layer, DenseSpec):
        return _dense_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, GruSpec):
        return _gru_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, LstmSpec):
        return _lstm_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, WindowSpec):
        return []  # zero trainable parameters
    if isinstance(layer, TransformerSpec):
        return _transformer_specs(layer, layer_idx, bound_multiplier)
    msg = f"Unknown layer type for PSO specs: {layer!r}"
    raise ValueError(msg)


def _dense_specs(layer: DenseSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    # Mirrors nn_param_specs_from_architecture: activation-aware bound via
    # compute_layer_bound (Xavier/He/LeCun), biases use the same bound as weights.
    fan_in = layer.input_size
    fan_out = layer.output_size
    bound = bound_multiplier * compute_layer_bound(fan_in, fan_out, layer.activation)

    specs: list[ParamSpec] = []
    for j in range(fan_out):
        for k in range(fan_in):
            specs.append(ParamSpec(f"w{layer_idx}_{j}_{k}", -bound, bound, 0.0))
    for j in range(fan_out):
        specs.append(ParamSpec(f"bias{layer_idx}_{j}", -bound, bound, 0.0))
    return specs


def _gru_specs(layer: GruSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Flat-weight spec order matches the Rust `LayerWeights for GruLayer`:
    weight_ih (row-major [3H, I]) -> weight_hh (row-major [3H, H]) -> bias_ih -> bias_hh.
    """
    h = layer.hidden_size
    three_h = 3 * h
    w_ih_bound = bound_multiplier * compute_layer_bound(layer.input_size, three_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(h, three_h, "tanh")
    b_bound = 0.1 * bound_multiplier

    specs: list[ParamSpec] = []
    for j in range(three_h * layer.input_size):
        specs.append(ParamSpec(f"w_ih{layer_idx}_{j}", -w_ih_bound, w_ih_bound, 0.0))
    for j in range(three_h * h):
        specs.append(ParamSpec(f"w_hh{layer_idx}_{j}", -w_hh_bound, w_hh_bound, 0.0))
    for j in range(three_h):
        specs.append(ParamSpec(f"b_ih{layer_idx}_{j}", -b_bound, b_bound, 0.0))
    for j in range(three_h):
        specs.append(ParamSpec(f"b_hh{layer_idx}_{j}", -b_bound, b_bound, 0.0))
    return specs


def _lstm_specs(layer: LstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Flat-weight spec order matches the Rust `LayerWeights for LstmLayer`:
    weight_ih (row-major [4H, I]) -> weight_hh (row-major [4H, H]) -> bias_ih -> bias_hh.

    Gate ordering on the 4H axis: (i, f, g, o). The forget-gate slice on bias_ih
    (rows [H:2H]) uses a wider ParamSpec bound (2.0 * bound_multiplier) to
    accommodate the Jozefowicz forget-bias-1 init (value ~1.0) inside PSO's
    search box. All other biases use the tight 0.1 * bound_multiplier bound.
    """
    h = layer.hidden_size
    four_h = 4 * h
    w_ih_bound = bound_multiplier * compute_layer_bound(layer.input_size, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(h, four_h, "tanh")
    tight_bias_bound = 0.1 * bound_multiplier
    forget_bias_bound = 2.0 * bound_multiplier

    specs: list[ParamSpec] = []
    for j in range(four_h * layer.input_size):
        specs.append(ParamSpec(f"w_ih{layer_idx}_{j}", -w_ih_bound, w_ih_bound, 0.0))
    for j in range(four_h * h):
        specs.append(ParamSpec(f"w_hh{layer_idx}_{j}", -w_hh_bound, w_hh_bound, 0.0))
    # bias_ih: forget slice (rows [H:2H]) uses wider bound; rest tight.
    for j in range(four_h):
        if h <= j < 2 * h:
            specs.append(ParamSpec(f"b_ih{layer_idx}_{j}", -forget_bias_bound, forget_bias_bound, 0.0))
        else:
            specs.append(ParamSpec(f"b_ih{layer_idx}_{j}", -tight_bias_bound, tight_bias_bound, 0.0))
    # bias_hh: all gates use tight bound.
    for j in range(four_h):
        specs.append(ParamSpec(f"b_hh{layer_idx}_{j}", -tight_bias_bound, tight_bias_bound, 0.0))
    return specs


def _transformer_specs(layer: TransformerSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """ParamSpec list in canonical flat order matching Rust TransformerLayer::to_flat.

    INVARIANT: ordering MUST match Rust's to_flat / from_flat cursor advance order:
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1, b_ffn1, w_ffn2, b_ffn2,
    ln1_gamma, ln1_beta, ln2_gamma, ln2_beta.

    Bounds:
      - Projection matrices (Q/K/V/O): Xavier uniform sqrt(6 / (2*d_model)) * mul
      - FFN1/FFN2:                     Xavier uniform sqrt(6 / (d_model + d_ffn)) * mul
      - Biases:                        tight uniform [-0.1*mul, 0.1*mul]
      - LN gamma:                      uniform [1 - 0.01*mul, 1 + 0.01*mul]
      - LN beta:                       uniform [-0.01*mul, 0.01*mul]
    """
    from math import sqrt

    d = layer.d_model
    f = layer.d_ffn
    mul = bound_multiplier
    li = layer_idx

    proj_bound = sqrt(6.0 / (2.0 * d)) * mul
    ffn_bound = sqrt(6.0 / (d + f)) * mul
    bias_bound = 0.1 * mul
    gamma_lo, gamma_hi = 1.0 - 0.01 * mul, 1.0 + 0.01 * mul
    beta_bound = 0.01 * mul

    specs: list[ParamSpec] = []
    # 4 projection matrices: w_q/b_q, w_k/b_k, w_v/b_v, w_o/b_o  (each [d,d] + [d])
    for proj_name, bias_name in (("w_q", "b_q"), ("w_k", "b_k"), ("w_v", "b_v"), ("w_o", "b_o")):
        for j in range(d):
            for k in range(d):
                specs.append(ParamSpec(f"{proj_name}{li}_{j}_{k}", -proj_bound, proj_bound, 0.0))
        for j in range(d):
            specs.append(ParamSpec(f"{bias_name}{li}_{j}", -bias_bound, bias_bound, 0.0))
    # w_ffn1 [f, d] + b_ffn1 [f]
    for j in range(f):
        for k in range(d):
            specs.append(ParamSpec(f"w_ffn1_{li}_{j}_{k}", -ffn_bound, ffn_bound, 0.0))
    for j in range(f):
        specs.append(ParamSpec(f"b_ffn1_{li}_{j}", -bias_bound, bias_bound, 0.0))
    # w_ffn2 [d, f] + b_ffn2 [d]
    for j in range(d):
        for k in range(f):
            specs.append(ParamSpec(f"w_ffn2_{li}_{j}_{k}", -ffn_bound, ffn_bound, 0.0))
    for j in range(d):
        specs.append(ParamSpec(f"b_ffn2_{li}_{j}", -bias_bound, bias_bound, 0.0))
    # LN1: gamma [d] + beta [d]
    for j in range(d):
        specs.append(ParamSpec(f"ln1_gamma{li}_{j}", gamma_lo, gamma_hi, 0.0))
    for j in range(d):
        specs.append(ParamSpec(f"ln1_beta{li}_{j}", -beta_bound, beta_bound, 0.0))
    # LN2: gamma [d] + beta [d]
    for j in range(d):
        specs.append(ParamSpec(f"ln2_gamma{li}_{j}", gamma_lo, gamma_hi, 0.0))
    for j in range(d):
        specs.append(ParamSpec(f"ln2_beta{li}_{j}", -beta_bound, beta_bound, 0.0))
    return specs
