"""Load V2Policy from JSON v2 format.

Shared between RL training (report_rl.py post-training analysis), test code,
and any Python-side consumer that needs the torch model. Rust side uses its own
loader in data/neural.rs; this module is the Python equivalent.
"""

from __future__ import annotations

import json

import torch

from aerocapture.training.rl.layers import DenseLayer
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import ArchitectureV2


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
        if layer_spec.type == "dense":
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
        # Phase 1+ layer types dispatch here.

    return policy
