"""Cross-language equivalence: Rust NeuralNetModel and PyTorch V2Policy produce
the same output on the same input to 1e-10. This is the Phase 0 integration gate.

Subsequent phases extend this test with their new layer types.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# Skip in environments without the PyO3 bindings installed (standard Python CI
# job). The python-pyo3 CI job builds the bindings via `maturin develop` and
# explicitly runs this test, so a stale build still fails the gate there.
aerocapture_rs = pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.export import export_v2_policy_to_json  # noqa: E402
from aerocapture.training.rl.layers.dense import DenseLayer  # noqa: E402
from aerocapture.training.rl.policy import V2Policy  # noqa: E402
from aerocapture.training.rl.schemas import DenseSpec, GruSpec  # noqa: E402


def _rust_forward_single(json_path: str, inputs: np.ndarray) -> np.ndarray:
    """Load a v2 JSON in Rust and run forward on each input row."""
    return np.array([aerocapture_rs.nn_forward(json_path, input_row.tolist()) for input_row in inputs])


def test_rust_python_dense_equivalence(tmp_path: Path) -> None:
    architecture = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(42)
    with torch.no_grad():
        for layer in policy.layers:
            assert isinstance(layer, DenseLayer)
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


def test_rust_python_gru_equivalence(tmp_path: Path) -> None:
    architecture: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        GruSpec(type="gru", input_size=8, hidden_size=8),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(42)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name == "log_std":
                continue
            p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "gru_model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    # Rust nn_forward is stateless per-call (fresh NnState each call). For a
    # fair Rust<->Python equivalence, reset Python state per step too -- otherwise
    # Python carries hidden state and Rust doesn't.
    py_single_out = np.zeros((100, 2), dtype=np.float64)
    for i, x in enumerate(inputs):
        single_state = policy.new_state(1, "cpu")
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), single_state)
        py_single_out[i] = y.detach().numpy()[0]

    rust_out = _rust_forward_single(str(json_path), inputs)

    max_diff = np.max(np.abs(rust_out - py_single_out))
    assert max_diff < 1e-10, f"gru single-step max abs diff {max_diff}"


def test_rust_python_dense_equivalence_with_input_mask(tmp_path: Path) -> None:
    # Raw input is 5-wide; mask picks 3 indices for a 3-input first-layer Dense.
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=[0, 2, 4])
    torch.manual_seed(7)
    with torch.no_grad():
        for layer in policy.layers:
            assert isinstance(layer, DenseLayer)
            layer.linear.weight.data = torch.randn_like(layer.linear.weight) * 0.3
            layer.linear.bias.data = torch.randn_like(layer.linear.bias) * 0.1
    policy.double()

    json_path = tmp_path / "masked.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(11)
    raw_inputs = rng.standard_normal((50, 5)).astype(np.float64)
    masked_inputs = raw_inputs[:, [0, 2, 4]]

    py_out = np.zeros((50, 2), dtype=np.float64)
    state = policy.new_state(1, "cpu")
    for i, x in enumerate(masked_inputs):
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    # Rust nn_forward takes the RAW input and applies the mask internally.
    rust_out = _rust_forward_single(str(json_path), raw_inputs)

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"dense+mask max abs diff {max_diff}"


def test_rust_python_ppo_gru_export_equivalence(tmp_path: Path) -> None:
    """A V2Policy with GRU, trained under PPO code (simulated by random init here),
    exports to v2 JSON and the Rust runtime's nn_forward matches the Python
    single-step forward at machine epsilon."""
    from aerocapture.training.rl.export import export_v2_policy_to_json
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import DenseSpec, GruSpec

    architecture: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        GruSpec(type="gru", input_size=8, hidden_size=8),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(2026)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name == "log_std":
                continue
            p.data = torch.randn_like(p.data) * 0.2
    policy.double()

    json_path = tmp_path / "ppo_gru_model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(13)
    inputs = rng.standard_normal((50, 5)).astype(np.float64)

    # Stateless comparison: Python resets state per call; Rust's nn_forward is stateless.
    py_out = np.zeros((50, 2), dtype=np.float64)
    for i, x in enumerate(inputs):
        fresh = policy.new_state(1, "cpu")
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), fresh)
        py_out[i] = y.detach().numpy()[0]

    rust_out = np.array([aerocapture_rs.nn_forward(str(json_path), x.tolist()) for x in inputs])

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"ppo-gru export max abs diff {max_diff}"
