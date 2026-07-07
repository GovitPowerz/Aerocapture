"""Load V2Policy from JSON v2 format.

Shared between RL training (report_rl.py post-training analysis), test code,
and any Python-side consumer that needs the torch model. Rust side uses its own
loader in data/neural.rs; this module is the Python equivalent.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import (
    ArchitectureV2,
    CfcSpec,
    DenseSpec,
    GruSpec,
    LayerWeights,
    LstmSpec,
    Mamba3Spec,
    MambaSpec,
    MlstmSpec,
    SlstmSpec,
    TransformerSpec,
    WindowSpec,
)


@runtime_checkable
class _HasFromFlat(Protocol):
    def from_flat(self, slab: np.ndarray) -> None: ...


def _slab_from_json_weights(layer_spec: DenseSpec | GruSpec | LstmSpec, lw: LayerWeights | None) -> np.ndarray:
    """Reconstruct the canonical flat numpy slab from a JSON layer_weights entry.

    Each layer type's flat order mirrors to_flat() / Rust LayerWeights::to_flat.
    JSON stores Python lists (f64). Returns a 1-D float64 ndarray that can be
    passed directly to module.from_flat().

    Callers: load_policy_from_json only, which already guards against Window /
    Transformer / Mamba specs at line 75, so those types never reach here.
    """

    def arr(x: object) -> np.ndarray:
        return np.array(x, dtype=np.float64).ravel()

    if isinstance(layer_spec, DenseSpec):
        if lw is None or lw.w is None or lw.b is None:
            raise ValueError("Dense layer missing w/b")
        return np.concatenate([arr(lw.w), arr(lw.b)])

    # GruSpec | LstmSpec
    extra = (lw.model_extra if lw is not None else None) or {}
    required = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    missing = [k for k in required if k not in extra]
    if missing:
        raise ValueError(f"{type(layer_spec).__name__} layer missing {missing}")
    return np.concatenate([arr(extra["weight_ih"]), arr(extra["weight_hh"]), arr(extra["bias_ih"]), arr(extra["bias_hh"])])


def load_policy_from_json(path: str, device: str | torch.device = "cpu") -> V2Policy:
    with open(path) as f:
        raw = json.load(f)
    if raw.get("format_version") != 2:
        raise ValueError(f"Expected format_version=2 in {path}, got {raw.get('format_version')}")
    arch = ArchitectureV2.model_validate(raw)

    # Phase 2b / Phase 3a / Phase 4a: Window-MLP, Transformer, and Mamba are PSO-only.
    # V2Policy cannot be built with these layers (build_layer raises NotImplementedError),
    # so we short-circuit here before V2Policy construction would fail opaquely.
    if any(isinstance(spec, (WindowSpec, TransformerSpec, MambaSpec, Mamba3Spec, CfcSpec, SlstmSpec, MlstmSpec)) for spec in arch.architecture):
        raise NotImplementedError(
            "Window-MLP (Phase 2b), Transformer (Phase 3a), Mamba (Phase 4a), Mamba3 "
            "(ablation spike), and CfC/sLSTM/mLSTM (architecture probes) are PSO-only; "
            "load_policy_from_json is a PPO/SAC entry point that cannot construct V2Policy "
            "with these layers. "
            "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md, "
            "docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md, "
            "docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md, "
            "docs/superpowers/specs/2026-07-07-mamba3-ablation-design.md, and "
            "docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md"
        )

    policy = V2Policy(
        architecture=list(arch.architecture),
        input_mask=arch.input_mask,
    ).to(device)

    for i, layer_spec in enumerate(arch.architecture):
        key = f"layer_{i}"
        lw = arch.weights.get(key)  # Window has no weights entry
        assert isinstance(layer_spec, (DenseSpec, GruSpec, LstmSpec)), f"unexpected layer type {type(layer_spec).__name__} past guard"
        slab = _slab_from_json_weights(layer_spec, lw)
        layer = policy.layers[i]
        assert isinstance(layer, _HasFromFlat), f"layer {i} has no from_flat method"
        layer.from_flat(slab)

    return policy
