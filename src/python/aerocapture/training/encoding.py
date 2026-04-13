"""Real-valued encoding/decoding for optimizer parameters.

All algorithms work on normalized np.ndarray[float64] in [0, 1].
Decoding to physical values happens at evaluation time.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from aerocapture.training.initialization import compute_layer_bound
from aerocapture.training.param_spaces import ParamSpec


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
