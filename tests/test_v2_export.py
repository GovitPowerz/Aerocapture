"""Task 9: Python v2 export + load round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.rl.export import export_v2_policy_to_json
from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _dense(policy: V2Policy, i: int) -> DenseLayer:
    """Typed accessor for mypy strict (nn.ModuleList returns Tensor | Module)."""
    layer = policy.layers[i]
    assert isinstance(layer, DenseLayer)
    return layer


def _policy() -> V2Policy:
    architecture = [
        DenseSpec(type="dense", input_size=4, output_size=3, activation="tanh"),
        DenseSpec(type="dense", input_size=3, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    with torch.no_grad():
        _dense(p, 0).linear.weight.data.fill_(0.1)
        _dense(p, 0).linear.bias.data.fill_(0.01)
        _dense(p, 1).linear.weight.data.fill_(-0.1)
        _dense(p, 1).linear.bias.data.fill_(-0.01)
    return p


def test_export_produces_v2_format(tmp_path: Path) -> None:
    p = _policy()
    path = tmp_path / "model.json"
    export_v2_policy_to_json(p, str(path), obs_normalizer=None)
    raw = json.loads(path.read_text())
    assert raw["format_version"] == 2
    assert raw["architecture"][0]["type"] == "dense"
    assert "layer_0" in raw["weights"]
    assert "log_std" not in raw  # log_std is never exported


def test_export_load_roundtrip_preserves_weights(tmp_path: Path) -> None:
    p = _policy()
    path = tmp_path / "model.json"
    export_v2_policy_to_json(p, str(path), obs_normalizer=None)
    q = load_policy_from_json(str(path), device="cpu")

    # Weights match bit-for-bit on the linear layer parameters.
    for i in range(len(p.layers)):
        la = _dense(p, i)
        lb = _dense(q, i)
        torch.testing.assert_close(la.linear.weight, lb.linear.weight, rtol=0, atol=0)
        torch.testing.assert_close(la.linear.bias, lb.linear.bias, rtol=0, atol=0)

    # Forward produces identical output.
    x = torch.randn(1, 4)
    sa = p.new_state(1, "cpu")
    sb = q.new_state(1, "cpu")
    ya, _ = p(x, sa)
    yb, _ = q(x, sb)
    torch.testing.assert_close(ya, yb, rtol=0, atol=0)
