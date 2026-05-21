"""Export trained PyTorch policies to the NeuralNetModel JSON format.

Rust format (from src/rust/src/data/neural.rs NnJsonFile):
{
  "format_version": 1,
  "architecture": {"layers": [input_dim, hidden1, ..., output_dim], "activations": ["tanh", ...]},
  "weights": {"layer_0": {"w": [[...]], "b": [...]}, "layer_1": {...}, ...},
  "input_mask": [0, ..., N-1]
}

The final layer must produce 2 outputs; bank is `atan2(out[0], out[1])`.
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

from aerocapture.training.rl.layers import DenseLayer, GruLayer, LstmLayer, WindowLayer
from aerocapture.training.rl.layers.mamba import MambaLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer
from aerocapture.training.rl.policy import GaussianPolicy, V2Policy
from aerocapture.training.rl.schemas import MambaSpec, TransformerSpec, WindowSpec

_ACT_NAMES = {"Tanh": "tanh", "ReLU": "relu", "Sigmoid": "sigmoid", "Identity": "linear", "SiLU": "swish", "Mish": "mish"}


def _serialize_mamba_layer(layer: MambaLayer) -> dict:
    """Serialize a MambaLayer to the flat-at-layer-level Mamba weights dict.

    Keys match Rust NnLayerWeights schema: x_proj_w, dt_proj_w, dt_proj_b,
    a_log, d_skip. All arrays are Python lists of f64 (JSON-serializable).
    """

    def to_list(t: torch.Tensor) -> list[float]:
        from typing import cast

        return cast(list[float], t.detach().cpu().numpy().astype(np.float64).tolist())

    return {
        "x_proj_w": to_list(layer.x_proj_w),  # (dt_rank + 2*d_state, input_size)
        "dt_proj_w": to_list(layer.dt_proj_w),  # (input_size, dt_rank)
        "dt_proj_b": to_list(layer.dt_proj_b),  # (input_size,)
        "a_log": to_list(layer.a_log),  # (input_size, d_state)
        "d_skip": to_list(layer.d_skip),  # (input_size,)
    }


def _check_obs_norm_bake_compatibility(
    architecture: list,
    obs_normalizer_active: bool,
) -> None:
    """Raise NotImplementedError when obs-norm bake-in cannot be applied to layer 0.

    Bake-in is only safe when layer 0 is Dense (affine: W/std, b - W@(mean/std)).
    All other layer types are rejected: their nonlinearities don't absorb a linear
    input transform in closed form.

    Accepts either torch module objects (from policy.layers) or schema spec objects
    (MambaSpec, WindowSpec, TransformerSpec -- useful for unit tests).
    """
    if not obs_normalizer_active or not architecture:
        return
    first = architecture[0]
    if isinstance(first, (MambaSpec, MambaLayer)):
        raise NotImplementedError(
            "obs_normalizer bake-in not supported when layer 0 is Mamba. "
            "Mamba's x_proj + softplus + A = -exp(a_log) is nonlinear in x; absorbing "
            "an affine input transform would require shifting dt_proj_b through softplus "
            "(not closed-form). Deferred to Phase 4b. "
            "Add a Dense embedding as layer 0 (Phase 0 spec section 3.5 invariant)."
        )
    if isinstance(first, (WindowSpec, WindowLayer, TransformerSpec, TransformerLayer)):
        raise NotImplementedError(
            f"obs_normalizer bake-in not supported when layer 0 is {type(first).__name__}. "
            "Add a Dense embedding as layer 0 (Phase 0 spec section 3.5 invariant)."
        )
    if isinstance(first, (GruLayer, LstmLayer)):
        raise NotImplementedError(
            f"obs_normalizer bake-in not supported when layer 0 is {type(first).__name__}. "
            "Add a Dense embedding as layer 0 (Phase 0 spec section 3.5 invariant)."
        )


def export_policy_to_json(
    policy: GaussianPolicy,
    output_path: Path,
    input_mask: Sequence[int],
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
        "input_mask": list(input_mask),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(doc, f, indent=2)


def export_v2_policy_to_json(
    policy: V2Policy,
    path: str,
    obs_normalizer: ObsNormalizer | None = None,
    output_param: str | None = None,
) -> None:
    """Write a V2Policy as JSON v2.

    Optional `obs_normalizer` bakes the affine transform into the first dense
    layer: `W_new = W / std`, `b_new = b - W @ (mean / std)`. log_std is an
    exploration-noise parameter and is never exported.

    Optional `output_param` sets the output parameterization field in the JSON
    (e.g. ``"acos_tanh"``). When ``None`` (default), the field is omitted and
    Rust loads it as ``Atan2Signed`` (backward compatible).
    """
    _check_obs_norm_bake_compatibility(
        list(policy.layers),
        obs_normalizer_active=(obs_normalizer is not None),
    )

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
        elif isinstance(layer, LstmLayer):
            w_ih = layer.weight_ih.detach().cpu().numpy().astype(np.float64)
            w_hh = layer.weight_hh.detach().cpu().numpy().astype(np.float64)
            b_ih = layer.bias_ih.detach().cpu().numpy().astype(np.float64)
            b_hh = layer.bias_hh.detach().cpu().numpy().astype(np.float64)
            architecture.append(
                {
                    "type": "lstm",
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
        elif isinstance(layer, WindowLayer):
            architecture.append(
                {
                    "type": "window",
                    "input_size": layer.input_size,
                    "n_steps": layer.n_steps,
                }
            )
            # Window is zero-param: no weights entry for this layer.
        elif isinstance(layer, TransformerLayer):
            architecture.append(
                {
                    "type": "transformer",
                    "d_model": layer.d_model,
                    "n_heads": layer.n_heads,
                    "d_ffn": layer.d_ffn,
                    "n_seq": layer.n_seq,
                }
            )
            weights[f"layer_{i}"] = {
                "w_q": layer.w_q.weight.detach().cpu().tolist(),
                "b_q": layer.w_q.bias.detach().cpu().tolist(),
                "w_k": layer.w_k.weight.detach().cpu().tolist(),
                "b_k": layer.w_k.bias.detach().cpu().tolist(),
                "w_v": layer.w_v.weight.detach().cpu().tolist(),
                "b_v": layer.w_v.bias.detach().cpu().tolist(),
                "w_o": layer.w_o.weight.detach().cpu().tolist(),
                "b_o": layer.w_o.bias.detach().cpu().tolist(),
                "w_ffn1": layer.w_ffn1.weight.detach().cpu().tolist(),
                "b_ffn1": layer.w_ffn1.bias.detach().cpu().tolist(),
                "w_ffn2": layer.w_ffn2.weight.detach().cpu().tolist(),
                "b_ffn2": layer.w_ffn2.bias.detach().cpu().tolist(),
                # Flat LN keys matching Rust NnLayerWeights schema
                "ln1_gamma": layer.ln1_gamma.detach().cpu().tolist(),
                "ln1_beta": layer.ln1_beta.detach().cpu().tolist(),
                "ln2_gamma": layer.ln2_gamma.detach().cpu().tolist(),
                "ln2_beta": layer.ln2_beta.detach().cpu().tolist(),
            }
        elif isinstance(layer, MambaLayer):
            architecture.append(
                {
                    "type": "mamba",
                    "input_size": layer.input_size,
                    "d_state": layer.d_state,
                    "dt_rank": layer.dt_rank,
                }
            )
            weights[f"layer_{i}"] = _serialize_mamba_layer(layer)
        else:
            raise ValueError(f"Unknown layer type in export: {type(layer).__name__}")

    out: dict[str, object] = {
        "format_version": 2,
        "architecture": architecture,
        "weights": weights,
        "input_mask": policy.input_mask,
        "ablated_input": None,
    }
    if output_param is not None:
        out["output_param"] = output_param
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
    )
