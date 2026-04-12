"""Activation-aware weight initialization for GA populations.

Provides Xavier (Glorot), He (Kaiming), and LeCun uniform initialization
bounds, auto-selected by activation function. Generates flat weight vectors
compatible with the real-valued normalized encoding used by the pymoo optimizer.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# Activation -> scheme mapping
_XAVIER_ACTIVATIONS = frozenset({"tanh", "sigmoid", "asinh"})
_HE_ACTIVATIONS = frozenset({"relu"})
_LECUN_ACTIVATIONS = frozenset({"linear"})


def compute_layer_bound(fan_in: int, fan_out: int, activation: str) -> float:
    """Compute uniform initialization bound for a single layer.

    Auto-selects scheme based on activation:
        tanh/sigmoid/asinh -> Xavier: sqrt(6 / (fan_in + fan_out))
        relu               -> He:     sqrt(6 / fan_in)
        linear             -> LeCun:  sqrt(3 / fan_in)

    Args:
        fan_in: Number of input neurons.
        fan_out: Number of output neurons.
        activation: Activation function name.

    Returns:
        Uniform bound: weights should be drawn from U(-bound, +bound).
    """
    if activation in _XAVIER_ACTIVATIONS:
        return math.sqrt(6.0 / (fan_in + fan_out))
    if activation in _HE_ACTIVATIONS:
        return math.sqrt(6.0 / fan_in)
    if activation in _LECUN_ACTIVATIONS:
        return math.sqrt(3.0 / fan_in)
    msg = f"Unknown activation: {activation!r}. Expected one of: tanh, sigmoid, asinh, relu, linear"
    raise ValueError(msg)


def generate_initialized_weights(
    layer_sizes: list[int],
    activations: list[str],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Generate a flat weight vector with per-layer initialization.

    Weight layout matches write_nn_json() / to_flat_weights():
    for each layer: weights (row-major, shape fan_out x fan_in) then biases.

    Args:
        layer_sizes: Network layer sizes, e.g. [16, 24, 2].
        activations: Activation per layer, length = len(layer_sizes) - 1.
        rng: Numpy random generator.

    Returns:
        Flat float64 array of all weights and biases.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        limit = compute_layer_bound(fan_in, fan_out, activations[i])
        w = rng.uniform(-limit, limit, size=(fan_out, fan_in)).ravel()
        b = np.zeros(fan_out, dtype=np.float64)
        parts.append(w)
        parts.append(b)
    return np.concatenate(parts)
