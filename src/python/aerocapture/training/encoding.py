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
from aerocapture.training.layer_schema import layer_schema
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.rl.schemas import (
    CfcSpec,
    DenseSpec,
    GruSpec,
    LayerSpec,
    LstmSpec,
    Mamba3Spec,
    MambaSpec,
    MlstmSpec,
    SlstmSpec,
    TransformerSpec,
    WindowSpec,
)

# Deterministic sub-seed for dt_proj_b center draw. Matched between _mamba_specs
# (ParamSpec bounds) and _init_mamba_layer (initial population values) so both
# agree on the center each ParamSpec window is centered around.
_MAMBA_DT_BIAS_SEED: int = 0xDE17A  # arbitrary but stable


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
    """Per-parameter ParamSpecs for one layer, in the layer's tensor-table order.

    Each generator walks `layer_schema(layer)` (the Rust table: names, shapes,
    canonical flat order) and applies its per-tensor bound / center / naming
    rule, so the chromosome order can never drift from the Rust `to_flat`.
    """
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
    if isinstance(layer, MambaSpec):
        return _mamba_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, Mamba3Spec):
        return _mamba3_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, CfcSpec):
        return _cfc_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, SlstmSpec):
        return _slstm_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, MlstmSpec):
        return _mlstm_specs(layer, layer_idx, bound_multiplier)
    msg = f"Unknown layer type for PSO specs: {layer!r}"
    raise ValueError(msg)


def _uniform(prefix: str, shape: tuple[int, ...], bound: float, center: float = 0.0) -> list[ParamSpec]:
    """One ParamSpec per element, flat-indexed (`prefix_j`), symmetric bound around `center`."""
    return [ParamSpec(f"{prefix}_{j}", center - bound, center + bound, center) for j in range(math.prod(shape))]


def _dense_specs(layer: DenseSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    # Mirrors nn_param_specs_from_architecture: activation-aware bound via
    # compute_layer_bound (Xavier/He/LeCun), biases use the same bound as weights.
    bound = bound_multiplier * compute_layer_bound(layer.input_size, layer.output_size, layer.activation)
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        if name == "w":
            specs += [ParamSpec(f"w{layer_idx}_{j}_{k}", -bound, bound, 0.0) for j in range(shape[0]) for k in range(shape[1])]
        else:  # b
            specs += [ParamSpec(f"bias{layer_idx}_{j}", -bound, bound, 0.0) for j in range(shape[0])]
    return specs


_GATED_CELL_NAMES = {"weight_ih": "w_ih", "weight_hh": "w_hh", "bias_ih": "b_ih", "bias_hh": "b_hh", "bias": "b"}


def _gru_specs(layer: GruSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """tanh-Xavier on the 3H-concatenated gate matrices, tight 0.1*mul biases."""
    h = layer.hidden_size
    bounds = {
        "weight_ih": bound_multiplier * compute_layer_bound(layer.input_size, 3 * h, "tanh"),
        "weight_hh": bound_multiplier * compute_layer_bound(h, 3 * h, "tanh"),
        "bias_ih": 0.1 * bound_multiplier,
        "bias_hh": 0.1 * bound_multiplier,
    }
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        specs += _uniform(f"{_GATED_CELL_NAMES[name]}{layer_idx}", shape, bounds[name])
    return specs


def _lstm_specs(layer: LstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """tanh-Xavier on the 4H gate matrices; gate order (i, f, g, o).

    The forget-gate slice of bias_ih (rows [H:2H]) uses a wider bound
    (2.0 * bound_multiplier) to hold the Jozefowicz forget-bias-1 init inside
    PSO's search box. All other biases use the tight 0.1 * bound_multiplier bound.
    """
    h = layer.hidden_size
    bounds = {
        "weight_ih": bound_multiplier * compute_layer_bound(layer.input_size, 4 * h, "tanh"),
        "weight_hh": bound_multiplier * compute_layer_bound(h, 4 * h, "tanh"),
        "bias_ih": 0.1 * bound_multiplier,
        "bias_hh": 0.1 * bound_multiplier,
    }
    forget_bias_bound = 2.0 * bound_multiplier
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        prefix = f"{_GATED_CELL_NAMES[name]}{layer_idx}"
        for j in range(math.prod(shape)):
            bound = forget_bias_bound if name == "bias_ih" and h <= j < 2 * h else bounds[name]
            specs.append(ParamSpec(f"{prefix}_{j}", -bound, bound, 0.0))
    return specs


def _transformer_specs(layer: TransformerSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Bounds:
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
    for name, shape in layer_schema(layer):
        if name in ("w_q", "w_k", "w_v", "w_o"):
            specs += [ParamSpec(f"{name}{li}_{j}_{k}", -proj_bound, proj_bound, 0.0) for j in range(shape[0]) for k in range(shape[1])]
        elif name in ("w_ffn1", "w_ffn2"):
            specs += [ParamSpec(f"{name}_{li}_{j}_{k}", -ffn_bound, ffn_bound, 0.0) for j in range(shape[0]) for k in range(shape[1])]
        elif name in ("b_ffn1", "b_ffn2"):
            specs += [ParamSpec(f"{name}_{li}_{j}", -bias_bound, bias_bound, 0.0) for j in range(shape[0])]
        elif name.startswith("b_"):
            specs += [ParamSpec(f"{name}{li}_{j}", -bias_bound, bias_bound, 0.0) for j in range(shape[0])]
        elif name.endswith("_gamma"):
            specs += [ParamSpec(f"{name}{li}_{j}", gamma_lo, gamma_hi, 0.0) for j in range(shape[0])]
        else:  # ln*_beta
            specs += [ParamSpec(f"{name}{li}_{j}", -beta_bound, beta_bound, 0.0) for j in range(shape[0])]
    return specs


def _ssm_specs(layer: MambaSpec | Mamba3Spec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Mamba / Mamba-3 per-tensor rules:
    x_proj_w     -- Xavier around 0
    dt_proj_w    -- Xavier * dt_rank^{-0.5} around 0
    dt_proj_b    -- inv_softplus(U(1e-3, 1e-1)) centers, drawn from
                    `_MAMBA_DT_BIAS_SEED ^ layer_idx` so stacked layers diverge at
                    init (matches `_init_mamba_layer`)
    a_log        -- HiPPO log(n+1) centers (outer d, inner n)
    a_imag       -- rotation frequency, center 0, +-pi            [Mamba3 complex]
    lambda_logit -- center +4 (near-euler), wide asymmetric search [Mamba3 trapezoidal]
    d_skip       -- 1.0 centers
    """
    d_inner = layer.input_size
    dt_rank = layer.dt_rank
    assert dt_rank is not None  # validator always resolves this
    mul = bound_multiplier
    li = layer_idx

    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        if name == "x_proj_w":
            bound = math.sqrt(6.0 / (d_inner + shape[0])) * mul
            specs += _uniform(f"x_proj_w{li}", shape, bound)
        elif name == "dt_proj_w":
            bound = math.sqrt(6.0 / (dt_rank + d_inner)) / math.sqrt(max(dt_rank, 1)) * mul
            specs += _uniform(f"dt_proj_w{li}", shape, bound)
        elif name == "dt_proj_b":
            dt_draws = np.random.default_rng(_MAMBA_DT_BIAS_SEED ^ li).uniform(1e-3, 1e-1, size=shape[0])
            for d in range(shape[0]):
                center = math.log(math.expm1(float(dt_draws[d])))
                specs.append(ParamSpec(f"dt_proj_b{li}_{d}", center - mul, center + mul, center))
        elif name == "a_log":
            for d in range(shape[0]):
                for n in range(shape[1]):
                    center = math.log(n + 1)
                    specs.append(ParamSpec(f"a_log{li}_{d}_{n}", center - mul, center + mul, center))
        elif name == "a_imag":
            specs += [ParamSpec(f"a_imag{li}_{d}_{n}", -math.pi, math.pi, 0.0) for d in range(shape[0]) for n in range(shape[1])]
        elif name == "lambda_logit":
            specs += [ParamSpec(f"lambda_logit{li}_{d}", -8.0, 12.0, 4.0) for d in range(shape[0])]
        else:  # d_skip
            specs += [ParamSpec(f"d_skip{li}_{d}", 1.0 - mul, 1.0 + mul, 1.0) for d in range(shape[0])]
    return specs


def _mamba_specs(layer: MambaSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    return _ssm_specs(layer, layer_idx, bound_multiplier)


def _mamba3_specs(layer: Mamba3Spec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    return _ssm_specs(layer, layer_idx, bound_multiplier)


def _cfc_specs(layer: CfcSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """tanh-Xavier on w_bb/w_ff1/w_ff2 (feed lecun_tanh / tanh), plain Xavier
    ("linear") on the time heads w_ta/w_tb, tight 0.1*mul biases."""
    i, h, b = layer.input_size, layer.hidden_size, layer.backbone_units
    ff_bound = bound_multiplier * compute_layer_bound(b, h, "tanh")
    t_bound = bound_multiplier * compute_layer_bound(b, h, "linear")
    bounds = {
        "w_bb": bound_multiplier * compute_layer_bound(i + h, b, "tanh"),
        "w_ff1": ff_bound,
        "w_ff2": ff_bound,
        "w_ta": t_bound,
        "w_tb": t_bound,
    }
    bias_bound = 0.1 * bound_multiplier
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        specs += _uniform(f"{name}{layer_idx}", shape, bounds.get(name, bias_bound))
    return specs


def _slstm_specs(layer: SlstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Gate order (i, f, z, o). The forget slice of `bias` (rows [H:2H]) gets the
    wide 3.0*mul bound to hold the +2.0 exp-gating forget-bias init center (LSTM
    forget-bias-1 precedent, scaled for the exponential gate)."""
    h = layer.hidden_size
    bounds = {
        "weight_ih": bound_multiplier * compute_layer_bound(layer.input_size, 4 * h, "tanh"),
        "weight_hh": bound_multiplier * compute_layer_bound(h, 4 * h, "tanh"),
        "bias": 0.1 * bound_multiplier,
    }
    forget = 3.0 * bound_multiplier
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        prefix = f"{_GATED_CELL_NAMES[name]}{layer_idx}"
        for j in range(math.prod(shape)):
            if name == "bias" and h <= j < 2 * h:
                specs.append(ParamSpec(f"{prefix}_{j}", -forget, forget, 2.0))
            else:
                specs.append(ParamSpec(f"{prefix}_{j}", -bounds[name], bounds[name], 0.0))
    return specs


def _mlstm_specs(layer: MlstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Xavier ("linear") on projections and gate vectors; b_f wide (3.0*mul, +2.0
    init center); every other bias tight."""
    i, h = layer.input_size, layer.hidden_size
    proj_bound = bound_multiplier * compute_layer_bound(i, h, "linear")
    gate_bound = bound_multiplier * compute_layer_bound(i, 1, "linear")
    tight = 0.1 * bound_multiplier
    forget = 3.0 * bound_multiplier
    li = layer_idx
    specs: list[ParamSpec] = []
    for name, shape in layer_schema(layer):
        if name == "b_f":
            specs.append(ParamSpec(f"b_f{li}", -forget, forget, 2.0))
        elif name == "b_i":
            specs.append(ParamSpec(f"b_i{li}", -tight, tight, 0.0))
        elif name in ("w_i", "w_f"):
            specs += _uniform(f"{name}{li}", shape, gate_bound)
        elif name.startswith("w_"):
            specs += _uniform(f"{name}{li}", shape, proj_bound)
        else:  # b_q, b_k, b_v, b_o
            specs += _uniform(f"{name}{li}", shape, tight)
    return specs
