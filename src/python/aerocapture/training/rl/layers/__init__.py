"""Torch mirrors of Rust layer types. One file per layer variant."""

from __future__ import annotations

from torch import nn

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer
from aerocapture.training.rl.layers.lstm import LstmLayer
from aerocapture.training.rl.layers.mamba import MambaLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer
from aerocapture.training.rl.layers.window import WindowLayer
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

__all__ = [
    "DenseLayer",
    "GruLayer",
    "LstmLayer",
    "MambaLayer",
    "TransformerLayer",
    "WindowLayer",
    "build_layer",
]


def build_layer(spec: LayerSpec) -> nn.Module:
    """Dispatch a LayerSpec to its torch module constructor."""
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        return WindowLayer(spec.input_size, spec.n_steps)
    if isinstance(spec, TransformerSpec):
        return TransformerLayer(spec.d_model, spec.n_heads, spec.d_ffn, spec.n_seq)
    if isinstance(spec, MambaSpec):
        assert spec.dt_rank is not None  # resolved by MambaSpec model_validator
        return MambaLayer(spec.input_size, spec.d_state, spec.dt_rank)
    if isinstance(spec, Mamba3Spec):
        raise NotImplementedError(
            "Mamba3 is PSO-only (ablation spike); the PPO/warm-start V2Policy path is not "
            "implemented. See docs/superpowers/specs/2026-07-07-mamba3-ablation-design.md"
        )
    if isinstance(spec, (CfcSpec, SlstmSpec, MlstmSpec)):
        raise NotImplementedError(
            f"{spec.type} is PSO-only (architecture probe); the PPO/warm-start V2Policy "
            "path is not implemented. See docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md"
        )
    raise ValueError(f"Unknown layer spec: {spec!r}")
