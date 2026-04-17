"""Cross-language equivalence: Rust NeuralNetModel and PyTorch V2Policy produce
the same output on the same input to 1e-10. This is the Phase 0 integration gate.

Subsequent phases extend this test with their new layer types.
"""

from __future__ import annotations

# Phase 0 integration gate MUST NOT skip on missing bindings -- a stale build
# is exactly the failure mode this test exists to catch. Hard import.
import aerocapture_rs
import numpy as np
import torch
from aerocapture.training.rl.export import export_v2_policy_to_json
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _rust_forward_single(json_path: str, inputs: np.ndarray) -> np.ndarray:
    """Load a v2 JSON in Rust and run forward on each input row."""
    return np.array([aerocapture_rs.nn_forward(json_path, input_row.tolist()) for input_row in inputs])


def test_rust_python_dense_equivalence(tmp_path):
    architecture = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(42)
    with torch.no_grad():
        for layer in policy.layers:
            layer.linear.weight.data = torch.randn_like(layer.linear.weight) * 0.3
            layer.linear.bias.data = torch.randn_like(layer.linear.bias) * 0.1

    # Convert to float64 for bitwise equivalence with Rust f64.
    policy.double()

    json_path = tmp_path / "model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    py_out = np.zeros((100, 2), dtype=np.float64)
    state = policy.new_state(1, "cpu")
    for i, x in enumerate(inputs):
        # from_numpy preserves float64 dtype, matching the double()-cast policy.
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    rust_out = _rust_forward_single(str(json_path), inputs)

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"max abs diff {max_diff} exceeds 1e-10"
