"""Export trained PyTorch policies to the NeuralNetModel JSON format.

Rust format (from src/rust/src/data/neural.rs NnJsonFile):
{
  "format_version": 1,
  "architecture": {"layers": [input_dim, hidden1, ..., output_dim], "activations": ["tanh", ...]},
  "weights": {"layer_0": {"w": [[...]], "b": [...]}, "layer_1": {...}, ...},
  "output_interpretation": "atan2",
  "input_mask": [0, ..., N-1]
}
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aerocapture.training.rl.normalizers import ObsNormalizer

import numpy as np
import numpy.typing as npt
import torch

from aerocapture.training.rl.layers import DenseLayer, GruLayer
from aerocapture.training.rl.policy import GaussianPolicy, V2Policy

_ACT_NAMES = {"Tanh": "tanh", "ReLU": "relu", "Sigmoid": "sigmoid", "Identity": "linear", "SiLU": "swish", "Mish": "mish"}


def export_policy_to_json(
    policy: GaussianPolicy,
    output_path: Path,
    input_mask: Sequence[int],
    output_interpretation: str = "atan2",
    obs_normalizer: ObsNormalizer | None = None,
) -> None:
    import copy

    if obs_normalizer is not None:
        trunk = copy.deepcopy(policy.trunk)
        for module in trunk:
            if isinstance(module, torch.nn.Linear):
                obs_normalizer.bake_into_linear(module)
                break
    else:
        trunk = policy.trunk

    layer_sizes: list[int] = [len(input_mask)]
    activations: list[str] = []
    weights_dict: dict[str, dict[str, list[list[float]] | list[float]]] = {}
    layer_idx = 0

    for module in trunk:
        if isinstance(module, torch.nn.Linear):
            layer_sizes.append(module.out_features)
            w = module.weight.detach().cpu().numpy().astype(np.float64)  # (out, in)
            b = module.bias.detach().cpu().numpy().astype(np.float64)  # (out,)
            weights_dict[f"layer_{layer_idx}"] = {
                "w": w.tolist(),
                "b": b.tolist(),
            }
            layer_idx += 1
        else:
            name = type(module).__name__
            activations.append(_ACT_NAMES.get(name, name.lower()))

    doc = {
        "format_version": 1,
        "architecture": {
            "layers": layer_sizes,
            "activations": activations,
        },
        "weights": weights_dict,
        "output_interpretation": output_interpretation,
        "input_mask": list(input_mask),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(doc, f, indent=2)


def export_v2_policy_to_json(
    policy: V2Policy,
    path: str,
    obs_normalizer: ObsNormalizer | None = None,
) -> None:
    """Write a V2Policy as JSON v2.

    Optional `obs_normalizer` bakes the affine transform into the first dense
    layer: `W_new = W / std`, `b_new = b - W @ (mean / std)`. log_std is an
    exploration-noise parameter and is never exported.
    """
    architecture: list[dict[str, object]] = []
    weights: dict[str, dict[str, list[list[float]] | list[float]]] = {}

    for i, layer in enumerate(policy.layers):
        if isinstance(layer, DenseLayer):
            lin = layer.linear
            w = lin.weight.detach().cpu().numpy().astype(np.float64)
            b = lin.bias.detach().cpu().numpy().astype(np.float64)

            if i == 0 and obs_normalizer is not None:
                mean = obs_normalizer._mean.astype(np.float64)
                std = obs_normalizer.std.astype(np.float64)
                w_new = w / std  # broadcasting over columns (inputs)
                b_new = b - w @ (mean / std)
                w, b = w_new, b_new

            architecture.append(
                {
                    "type": "dense",
                    "input_size": lin.in_features,
                    "output_size": lin.out_features,
                    "activation": layer.activation_name,
                }
            )
            weights[f"layer_{i}"] = {
                "w": w.tolist(),
                "b": b.tolist(),
            }
        elif isinstance(layer, GruLayer):
            if i == 0 and obs_normalizer is not None:
                raise NotImplementedError(
                    "obs_normalizer bake-in not supported when layer 0 is Gru. Add a Dense embedding as layer 0 (Phase 0 spec section 3.5 invariant)."
                )
            w_ih = layer.weight_ih.detach().cpu().numpy().astype(np.float64)
            w_hh = layer.weight_hh.detach().cpu().numpy().astype(np.float64)
            b_ih = layer.bias_ih.detach().cpu().numpy().astype(np.float64)
            b_hh = layer.bias_hh.detach().cpu().numpy().astype(np.float64)
            architecture.append(
                {
                    "type": "gru",
                    "input_size": layer.input_size,
                    "hidden_size": layer.hidden_size,
                }
            )
            weights[f"layer_{i}"] = {
                "weight_ih": w_ih.tolist(),
                "weight_hh": w_hh.tolist(),
                "bias_ih": b_ih.tolist(),
                "bias_hh": b_hh.tolist(),
            }
        else:
            raise ValueError(f"Unknown layer type in export: {type(layer).__name__}")

    out = {
        "format_version": 2,
        "architecture": architecture,
        "weights": weights,
        "output_interpretation": policy.output_interpretation,
        "input_mask": policy.input_mask,
        "ablated_input": None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


@dataclass
class _PyNN:
    """Python reimplementation of NeuralNetModel forward used by the roundtrip test."""

    layer_sizes: list[int]
    activations: list[str]
    layer_weights: list[npt.NDArray[np.float64]]
    layer_biases: list[npt.NDArray[np.float64]]
    input_mask: list[int]
    output_interpretation: str

    def _act(self, name: str, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if name == "tanh":
            return np.tanh(x)
        if name == "relu":
            return np.maximum(0.0, x)
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if name in ("linear", "identity"):
            return x
        raise ValueError(f"unknown activation: {name}")

    def forward(self, full_input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = full_input
        for w, b, act in zip(self.layer_weights, self.layer_biases, self.activations, strict=True):
            x = self._act(act, w @ x + b)
        return x

    def forward_bank(self, full_input: npt.NDArray[np.float64]) -> float:
        out = self.forward(full_input)
        if self.output_interpretation == "direct":
            return float(out[0])
        return float(math.atan2(out[0], out[1]))


def load_nn_model_json(path: Path) -> _PyNN:
    with path.open() as f:
        doc = json.load(f)

    # Rust format: {"format_version": 1, "architecture": {"layers": [...], "activations": [...]},
    #               "weights": {"layer_0": {"w": [[...]], "b": [...]}, ...}, ...}
    arch = doc["architecture"]
    layer_sizes: list[int] = arch["layers"][1:]  # drop input dim; keep hidden + output
    activations: list[str] = [a if isinstance(a, str) else a for a in arch["activations"]]
    input_mask: list[int] = doc["input_mask"]

    layer_weights: list[npt.NDArray[np.float64]] = []
    layer_biases: list[npt.NDArray[np.float64]] = []
    for i in range(len(layer_sizes)):
        lw = doc["weights"][f"layer_{i}"]
        layer_weights.append(np.array(lw["w"], dtype=np.float64))
        layer_biases.append(np.array(lw["b"], dtype=np.float64))

    return _PyNN(
        layer_sizes=layer_sizes,
        activations=activations,
        layer_weights=layer_weights,
        layer_biases=layer_biases,
        input_mask=input_mask,
        output_interpretation=doc.get("output_interpretation", "atan2"),
    )
