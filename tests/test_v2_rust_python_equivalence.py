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
from aerocapture.training.rl.schemas import DenseSpec, GruSpec, LstmSpec  # noqa: E402


def _rust_forward_single(json_path: str, inputs: np.ndarray) -> np.ndarray:
    """Load a v2 JSON in Rust and run forward on each input row."""
    return np.array([aerocapture_rs.nn_forward(json_path, input_row.tolist()) for input_row in inputs])


def test_rust_python_dense_equivalence(tmp_path: Path) -> None:
    architecture = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, input_mask=None)
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
    policy = V2Policy(architecture=architecture, input_mask=None)
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


def test_rust_python_lstm_equivalence(tmp_path: Path) -> None:
    """Dense -> LSTM -> Dense, f64: Rust nn_forward matches PyTorch V2Policy forward
    at machine epsilon. LSTM's (h, c) tuple state is the first multi-tensor state
    exercise of the cross-language contract.
    """
    architecture: list[DenseSpec | LstmSpec] = [
        DenseSpec(type="dense", input_size=5, output_size=4, activation="tanh"),
        LstmSpec(type="lstm", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, input_mask=None)
    torch.manual_seed(1337)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name == "log_std":
                continue
            p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "lstm_model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(17)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    # Rust nn_forward is stateless per-call (fresh NnState each call). For a
    # fair Rust<->Python equivalence, reset Python state per step too.
    py_single_out = np.zeros((100, 2), dtype=np.float64)
    for i, x in enumerate(inputs):
        single_state = policy.new_state(1, "cpu")
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), single_state)
        py_single_out[i] = y.detach().numpy()[0]

    rust_out = _rust_forward_single(str(json_path), inputs)

    max_diff = np.max(np.abs(rust_out - py_single_out))
    assert max_diff < 1e-10, f"lstm single-step max abs diff {max_diff}"


def test_rust_python_lstm_stateful_equivalence(tmp_path: Path) -> None:
    """Stateful Rust<->Python equivalence: state (h, c) persists across 100 steps
    on both sides so the forget-gate, output-gate, and weight_hh all become
    observable. A gate-ordering bug or forget-slice misplacement that survives
    the stateless test would fail here.

    Rationale: the stateless sibling resets state per step, so h_prev = c_prev = 0
    every call, which makes c_new = i*g (forget gate unreachable) and weight_hh
    irrelevant (multiplied by zero). The stateful driver exposes those code paths.
    """
    architecture: list[DenseSpec | LstmSpec] = [
        DenseSpec(type="dense", input_size=5, output_size=4, activation="tanh"),
        LstmSpec(type="lstm", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, input_mask=None)
    torch.manual_seed(2718)
    from aerocapture.training.rl.layers.lstm import LstmLayer

    lstm_layer = policy.layers[1]
    assert isinstance(lstm_layer, LstmLayer)
    H = lstm_layer.hidden_size
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name == "log_std":
                continue
            # Bias the LSTM forget bias toward "remember" so cell state
            # accumulates and the forget-gate path is actually exercised.
            if "bias_ih" in name and p.shape[0] == 4 * H:
                p.data = torch.randn_like(p.data) * 0.1
                p.data[H : 2 * H] = 1.0 + 0.1 * torch.randn_like(p.data[H : 2 * H])
            else:
                p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "lstm_stateful_model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(31)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    # Python: single persistent state across all 100 steps.
    state = policy.new_state(1, "cpu")
    py_out = np.zeros((100, 2), dtype=np.float64)
    for i, x in enumerate(inputs):
        y, state = policy(torch.from_numpy(x).unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    # Rust: single persistent NnState across all 100 steps via nn_forward_sequence.
    rust_out = np.array(aerocapture_rs.nn_forward_sequence(str(json_path), [row.tolist() for row in inputs]))
    assert rust_out.shape == (100, 2)

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"lstm stateful max abs diff {max_diff}"

    # Sanity: the cell state should have diverged substantially from zero by
    # step 99 (if it were still zero, weight_hh and forget gate would remain
    # unobservable). Grab the final c via Python and assert non-trivial magnitude.
    _, final_state = policy(torch.from_numpy(inputs[-1]).unsqueeze(0), policy.new_state(1, "cpu"))
    # This is a sanity check that multi-step state has grown, not a regression guard:
    # run 99 warm-up steps then check c
    state2 = policy.new_state(1, "cpu")
    for x in inputs[:99]:
        _, state2 = policy(torch.from_numpy(x).unsqueeze(0), state2)
    lstm_state = state2[1]  # layer 1 is the LSTM
    assert isinstance(lstm_state, tuple), "LSTM state should be (h, c) tuple"
    _, c99 = lstm_state
    assert float(c99.detach().abs().max()) > 0.1, (
        "Expected LSTM cell state to have non-trivial magnitude after warm-up; if it's near zero, the stateful test is not exercising the forget gate."
    )


def test_rust_python_dense_equivalence_with_input_mask(tmp_path: Path) -> None:
    # Raw input is 5-wide; mask picks 3 indices for a 3-input first-layer Dense.
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, input_mask=[0, 2, 4])
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


@pytest.mark.slow
def test_acos_tanh_rust_python_equivalence(tmp_path: Path) -> None:
    """V2Policy with last-layer (output_size=1, tanh) + output_param='acos_tanh'
    exported to JSON: Rust nn_forward returns the tanh output bit-equivalent to
    the Python V2Policy forward to machine epsilon.

    nn_forward returns the raw NN output vector (length 1 for acos_tanh); the
    acos is applied later in nn_bank_angle (Rust side) and is not tested here.
    """
    architecture = [
        DenseSpec(type="dense", input_size=8, output_size=16, activation="swish"),
        DenseSpec(type="dense", input_size=16, output_size=1, activation="tanh"),
    ]
    torch.manual_seed(0)
    policy = V2Policy(architecture=architecture, input_mask=None)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name == "log_std":
                continue
            p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "acos_tanh_model.json"
    export_v2_policy_to_json(policy, str(json_path), output_param="acos_tanh")

    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((100, 8)).astype(np.float64)

    rust_outputs = np.array([aerocapture_rs.nn_forward(str(json_path), x.tolist()) for x in inputs])

    py_outputs = np.zeros((100, 1), dtype=np.float64)
    for i, x in enumerate(inputs):
        fresh = policy.new_state(1, "cpu")
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), fresh)
        py_outputs[i] = y.detach().numpy()[0]

    diff = np.max(np.abs(rust_outputs - py_outputs))
    assert diff < 1e-10, f"acos_tanh max abs diff {diff} exceeds 1e-10"


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
    policy = V2Policy(architecture=architecture, input_mask=None)
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
