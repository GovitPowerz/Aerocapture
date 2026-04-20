"""Torch mirrors of Rust layer types. One file per layer variant."""

from __future__ import annotations

from torch import nn

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer
from aerocapture.training.rl.layers.lstm import LstmLayer
from aerocapture.training.rl.layers.window import WindowLayer
from aerocapture.training.rl.schemas import (
    DenseSpec,
    GruSpec,
    LayerSpec,
    LstmSpec,
    WindowSpec,
)

__all__ = ["DenseLayer", "GruLayer", "LstmLayer", "WindowLayer", "build_layer"]


def build_layer(spec: LayerSpec) -> nn.Module:
    """Dispatch a LayerSpec to its torch module constructor."""
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        raise NotImplementedError("Window-MLP is PSO-only in Phase 2b; PPO use deferred. See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md")
    raise ValueError(f"Unknown layer spec: {spec!r}")
