"""Per-layer weight statistics for GA training instrumentation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_weight_stats(
    weights: npt.NDArray[np.float64],
    layer_sizes: list[int],
) -> dict[str, dict[str, float]]:
    """Compute per-layer min/max/mean/std for weights and biases.

    Args:
        weights: Flat weight vector (same layout as write_nn_json / to_flat_weights).
        layer_sizes: Network layer sizes, e.g. [16, 24, 2].

    Returns:
        Dict with keys like "layer_0_w", "layer_0_b", each mapping to
        {"min": ..., "max": ..., "mean": ..., "std": ...}.
    """
    stats: dict[str, dict[str, float]] = {}
    idx = 0
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]

        n_w = fan_in * fan_out
        w = weights[idx : idx + n_w]
        idx += n_w
        stats[f"layer_{i}_w"] = {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
        }

        b = weights[idx : idx + fan_out]
        idx += fan_out
        stats[f"layer_{i}_b"] = {
            "min": float(b.min()),
            "max": float(b.max()),
            "mean": float(b.mean()),
            "std": float(b.std()),
        }

    return stats
