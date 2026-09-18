"""Write a `V2Policy` as NN model JSON v2 (the Rust `NnJsonFileV2` format).

Reader counterpart: `aerocapture.training.model_io.load_policy_from_json`
(round-trips bit-for-bit). Tensor names and flat order come from
`aerocapture.training.layer_schema`, the Python view of the Rust tensor table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from torch import nn

from aerocapture.training.layer_schema import layer_schema
from aerocapture.training.torch_mirror.layers import DenseLayer, GruLayer, LstmLayer, WindowLayer
from aerocapture.training.torch_mirror.layers.mamba import MambaLayer
from aerocapture.training.torch_mirror.layers.transformer import TransformerLayer
from aerocapture.training.torch_mirror.policy import V2Policy
from aerocapture.training.torch_mirror.schemas import MambaSpec, TransformerSpec, WindowSpec


class ObsAffine(Protocol):
    """What the obs-normalizer bake-in reads: `rl.normalizers.ObsNormalizer` satisfies it."""

    @property
    def mean(self) -> np.ndarray: ...

    @property
    def std(self) -> np.ndarray: ...


@runtime_checkable
class _HasToFlat(Protocol):
    def to_flat(self) -> np.ndarray: ...


def _spec_entry(layer: nn.Module) -> dict[str, Any]:
    """The v2 architecture entry describing a torch layer module."""
    if isinstance(layer, DenseLayer):
        lin = layer.linear
        return {"type": "dense", "input_size": lin.in_features, "output_size": lin.out_features, "activation": layer.activation_name}
    if isinstance(layer, GruLayer):
        return {"type": "gru", "input_size": layer.input_size, "hidden_size": layer.hidden_size}
    if isinstance(layer, LstmLayer):
        return {"type": "lstm", "input_size": layer.input_size, "hidden_size": layer.hidden_size}
    if isinstance(layer, WindowLayer):
        return {"type": "window", "input_size": layer.input_size, "n_steps": layer.n_steps}
    if isinstance(layer, TransformerLayer):
        return {"type": "transformer", "d_model": layer.d_model, "n_heads": layer.n_heads, "d_ffn": layer.d_ffn, "n_seq": layer.n_seq}
    if isinstance(layer, MambaLayer):
        return {"type": "mamba", "input_size": layer.input_size, "d_state": layer.d_state, "dt_rank": layer.dt_rank}
    raise ValueError(f"Unknown layer type in export: {type(layer).__name__}")


def _split_flat(entry: dict[str, Any], flat: np.ndarray) -> dict[str, np.ndarray]:
    """Named tensors of one layer from its canonical flat slab, in tensor-table order."""
    out: dict[str, np.ndarray] = {}
    off = 0
    for name, shape in layer_schema(entry):
        n = math.prod(shape)
        out[name] = flat[off : off + n].reshape(shape)
        off += n
    if off != flat.size:
        raise ValueError(f"{entry['type']} layer: flat slab has {flat.size} values, schema expects {off}")
    return out


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


def export_v2_policy_to_json(
    policy: V2Policy,
    path: str,
    obs_normalizer: ObsAffine | None = None,
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

    architecture: list[dict[str, Any]] = []
    weights: dict[str, dict[str, Any]] = {}

    for i, layer in enumerate(policy.layers):
        entry = _spec_entry(layer)
        architecture.append(entry)
        assert isinstance(layer, _HasToFlat), f"layer {i} has no to_flat method"
        tensors = _split_flat(entry, layer.to_flat())
        if not tensors:
            continue  # zero-param layer (Window): no weights entry
        if i == 0 and obs_normalizer is not None:
            # Only Dense passes _check_obs_norm_bake_compatibility: fold the affine into W/b.
            mean = obs_normalizer.mean.astype(np.float64)
            std = obs_normalizer.std.astype(np.float64)
            w, b = tensors["w"], tensors["b"]
            tensors = {"w": w / std, "b": b - w @ (mean / std)}
        weights[f"layer_{i}"] = {name: arr.tolist() for name, arr in tensors.items()}

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
