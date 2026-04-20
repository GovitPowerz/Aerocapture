"""Load V2Policy from JSON v2 format.

Shared between RL training (report_rl.py post-training analysis), test code,
and any Python-side consumer that needs the torch model. Rust side uses its own
loader in data/neural.rs; this module is the Python equivalent.
"""

from __future__ import annotations

import json

import torch

from aerocapture.training.rl.layers import DenseLayer, GruLayer, LstmLayer
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import ArchitectureV2, DenseSpec, GruSpec, LstmSpec


def load_policy_from_json(path: str, device: str | torch.device) -> V2Policy:
    with open(path) as f:
        raw = json.load(f)
    if raw.get("format_version") != 2:
        raise ValueError(f"Expected format_version=2 in {path}, got {raw.get('format_version')}")
    arch = ArchitectureV2.model_validate(raw)
    policy = V2Policy(
        architecture=list(arch.architecture),
        output_interpretation=arch.output_interpretation,
        input_mask=arch.input_mask,
    ).to(device)

    for i, layer_spec in enumerate(arch.architecture):
        key = f"layer_{i}"
        lw = arch.weights[key]
        if isinstance(layer_spec, DenseSpec):
            if lw.w is None or lw.b is None:
                raise ValueError(f"Dense layer {key} missing w/b in {path}")
            # JSON stores Python floats (f64). Load at f64 and let `.copy_`
            # cast to the destination policy's dtype -- preserves precision
            # for f64 policies, safely downcasts for f32 policies.
            w = torch.tensor(lw.w, dtype=torch.float64, device=device)
            b = torch.tensor(lw.b, dtype=torch.float64, device=device)
            layer = policy.layers[i]
            assert isinstance(layer, DenseLayer)
            with torch.no_grad():
                layer.linear.weight.copy_(w)
                layer.linear.bias.copy_(b)
        elif isinstance(layer_spec, GruSpec):
            # Gru weights land in LayerWeights.model_extra (extra="allow").
            extra = lw.model_extra or {}
            required = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            missing = [k for k in required if k not in extra]
            if missing:
                raise ValueError(f"Gru layer {key} missing {missing} in {path}")
            w_ih = torch.tensor(extra["weight_ih"], dtype=torch.float64, device=device)
            w_hh = torch.tensor(extra["weight_hh"], dtype=torch.float64, device=device)
            b_ih = torch.tensor(extra["bias_ih"], dtype=torch.float64, device=device)
            b_hh = torch.tensor(extra["bias_hh"], dtype=torch.float64, device=device)
            layer = policy.layers[i]
            assert isinstance(layer, GruLayer)
            with torch.no_grad():
                layer.weight_ih.copy_(w_ih)
                layer.weight_hh.copy_(w_hh)
                layer.bias_ih.copy_(b_ih)
                layer.bias_hh.copy_(b_hh)
        elif isinstance(layer_spec, LstmSpec):
            # Lstm weights land in LayerWeights.model_extra (extra="allow").
            extra = lw.model_extra or {}
            required = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            missing = [k for k in required if k not in extra]
            if missing:
                raise ValueError(f"Lstm layer {key} missing {missing} in {path}")
            w_ih = torch.tensor(extra["weight_ih"], dtype=torch.float64, device=device)
            w_hh = torch.tensor(extra["weight_hh"], dtype=torch.float64, device=device)
            b_ih = torch.tensor(extra["bias_ih"], dtype=torch.float64, device=device)
            b_hh = torch.tensor(extra["bias_hh"], dtype=torch.float64, device=device)
            layer = policy.layers[i]
            assert isinstance(layer, LstmLayer)
            with torch.no_grad():
                layer.weight_ih.copy_(w_ih)
                layer.weight_hh.copy_(w_hh)
                layer.bias_ih.copy_(b_ih)
                layer.bias_hh.copy_(b_hh)
        else:
            raise ValueError(f"Unknown layer spec type: {type(layer_spec).__name__}")

    return policy
