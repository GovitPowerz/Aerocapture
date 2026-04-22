"""Test export_v2_policy_to_json writes the correct Transformer schema."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from aerocapture.training.rl.layers import DenseLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer
from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec


def _make_policy(architecture: list, dtype: torch.dtype = torch.float64) -> MagicMock:
    """Build a V2Policy-compatible mock bypassing build_layer (which rejects Transformer)."""
    layers: list[torch.nn.Module] = []
    for spec in architecture:
        layer: torch.nn.Module
        if isinstance(spec, DenseSpec):
            layer = DenseLayer(spec.input_size, spec.output_size, spec.activation)
        elif isinstance(spec, TransformerSpec):
            layer = TransformerLayer(spec.d_model, spec.n_heads, spec.d_ffn, spec.n_seq)
        else:
            raise TypeError(f"test helper doesn't support {type(spec).__name__}")
        layers.append(layer)

    # Use MagicMock so we don't have to satisfy V2Policy.__init__ signature;
    # export_v2_policy_to_json only reads policy.layers and policy.architecture.
    policy = MagicMock()
    policy.architecture = architecture
    policy.layers = torch.nn.ModuleList([l.to(dtype=dtype) for l in layers])
    policy.input_mask = None
    return policy


def test_export_transformer_writes_flat_ln_keys(tmp_path: Path) -> None:
    from aerocapture.training.rl.export import export_v2_policy_to_json

    architecture = [
        DenseSpec(type="dense", input_size=8, output_size=4, activation="linear"),
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = _make_policy(architecture)
    out_path = tmp_path / "model.json"
    export_v2_policy_to_json(policy, str(out_path), obs_normalizer=None)

    obj = json.loads(out_path.read_text())
    assert obj["format_version"] == 2

    # Architecture entry for the Transformer layer
    arch = obj["architecture"][1]
    assert arch["type"] == "transformer"
    assert arch["d_model"] == 4
    assert arch["n_heads"] == 2
    assert arch["d_ffn"] == 8
    assert arch["n_seq"] == 4

    # Weights dict uses flat LN keys -- NOT nested `ln1: {gamma, beta}`
    layer_1 = obj["weights"]["layer_1"]
    for key in [
        "w_q",
        "b_q",
        "w_k",
        "b_k",
        "w_v",
        "b_v",
        "w_o",
        "b_o",
        "w_ffn1",
        "b_ffn1",
        "w_ffn2",
        "b_ffn2",
        "ln1_gamma",
        "ln1_beta",
        "ln2_gamma",
        "ln2_beta",
    ]:
        assert key in layer_1, f"missing key '{key}' in layer_1 weights"

    # Shape sanity: w_q should be (d_model, d_model) = (4, 4)
    assert len(layer_1["w_q"]) == 4
    assert len(layer_1["w_q"][0]) == 4
    # b_q should be (d_model,) = (4,)
    assert len(layer_1["b_q"]) == 4
    # ln1_gamma / ln1_beta should be (d_model,) = (4,)
    assert len(layer_1["ln1_gamma"]) == 4
    assert len(layer_1["ln1_beta"]) == 4


def test_export_obs_normalizer_rejects_transformer_as_first_layer(tmp_path: Path) -> None:
    from aerocapture.training.rl.export import export_v2_policy_to_json

    architecture = [
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = _make_policy(architecture)
    obs_normalizer = MagicMock()

    with pytest.raises(NotImplementedError):
        export_v2_policy_to_json(policy, str(tmp_path / "model.json"), obs_normalizer=obs_normalizer)
