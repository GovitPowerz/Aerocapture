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
_XAVIER_ACTIVATIONS = frozenset({"tanh", "sigmoid", "asinh", "swish", "mish"})
_HE_ACTIVATIONS = frozenset({"relu"})
_LECUN_ACTIVATIONS = frozenset({"linear"})

# Activation-specific calibration factor applied on top of the base
# Xavier/He/LeCun bound. Default (absent key) is 1.0 (no calibration).
#
# Why mish has a calibration and swish does not: mish's origin derivative is
# f'(0) = tanh(ln(2)) approximately 0.60, higher than swish's 0.50. For
# moderate |x| in [0.5, 2] mish(x) is 10-20 percent larger in magnitude than
# swish(x). Gradient-based optimizers (PPO/SAC/BPTT) rescale weights to
# compensate; PSO samples uniformly from the Xavier box and cannot. Result:
# PSO-trained mish nets operate at a "hotter" activation regime than the
# equivalent swish nets, saturating more aggressively through a multi-layer
# stack. The 0.87 factor brings mish's unit-input output down toward swish's:
# swish(1) / mish(1) = 0.731 / 0.865 = 0.845, rounded up to 0.87 for safety.
# Determined empirically 2026-04-17 from training outcome gap on msr_aller NN.
#
# Note: this only affects FUTURE training runs. Existing best_model.json /
# best_params.json artifacts trained with mish keep their original weights
# at inference.
_ACTIVATION_GAIN: dict[str, float] = {
    "mish": 0.87,
}


def compute_layer_bound(fan_in: int, fan_out: int, activation: str) -> float:
    """Compute uniform initialization bound for a single layer.

    Auto-selects scheme based on activation:
        tanh/sigmoid/asinh/swish -> Xavier: sqrt(6 / (fan_in + fan_out))
        mish                      -> Xavier * 0.87 (see _ACTIVATION_GAIN)
        relu                      -> He:     sqrt(6 / fan_in)
        linear                    -> LeCun:  sqrt(3 / fan_in)

    Args:
        fan_in: Number of input neurons.
        fan_out: Number of output neurons.
        activation: Activation function name.

    Returns:
        Uniform bound: weights should be drawn from U(-bound, +bound).
    """
    if activation in _XAVIER_ACTIVATIONS:
        base = math.sqrt(6.0 / (fan_in + fan_out))
    elif activation in _HE_ACTIVATIONS:
        base = math.sqrt(6.0 / fan_in)
    elif activation in _LECUN_ACTIVATIONS:
        base = math.sqrt(3.0 / fan_in)
    else:
        msg = f"Unknown activation: {activation!r}. Expected one of: tanh, sigmoid, asinh, relu, linear, swish, mish"
        raise ValueError(msg)
    return base * _ACTIVATION_GAIN.get(activation, 1.0)


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
