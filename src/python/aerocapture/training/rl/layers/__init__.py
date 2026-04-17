"""Torch mirrors of Rust layer types. One file per layer variant."""

from __future__ import annotations

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.schemas import LayerSpec

__all__ = ["DenseLayer", "build_layer"]


def build_layer(spec: LayerSpec) -> DenseLayer:
    """Dispatch a LayerSpec to its torch module constructor."""
    if spec.type == "dense":
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    # Phase 1+: gru, lstm, attention, layer_norm, ssm, window
    raise ValueError(f"Unknown layer type: {spec.type}")
