"""Cross-language Transformer equivalence (Rust runtime vs Python mirror).

Architecture: Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2,
d_ffn=32, n_seq=8) -> Dense(16 -> 2, linear). Runs 100 f64 inputs (more than
n_seq=8) to exercise KV-cache growth AND eviction. Both sides thread a single
NnState / (k_cache, v_cache) across the full sequence. Asserts max abs diff
< 1e-10; target machine epsilon.

JSON is written manually (bypassing V2Policy / build_layer / export_v2_policy_to_json)
because build_layer(TransformerSpec) raises NotImplementedError -- the production
PPO path does not support Transformer yet. The manual JSON uses flat LN keys
(ln1_gamma, ln1_beta, ln2_gamma, ln2_beta) to match the Rust NnLayerWeights
schema from Task 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]
from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer


@pytest.mark.slow
def test_transformer_rust_python_equivalence(tmp_path: Path) -> None:
    d_model = 16
    n_heads = 2
    d_ffn = 32
    n_seq = 8

    # 1. Build Python layers in f64. bypass build_layer which rejects Transformer.
    torch.manual_seed(0)
    dense_in = DenseLayer(input_size=8, output_size=16, activation="linear").double()
    transformer = TransformerLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_seq=n_seq).double()
    dense_out = DenseLayer(input_size=16, output_size=2, activation="linear").double()

    # Small range to keep outputs bounded and avoid softmax saturation that
    # could mask numerical drift.
    with torch.no_grad():
        for lin in [
            dense_in.linear,
            transformer.w_q,
            transformer.w_k,
            transformer.w_v,
            transformer.w_o,
            transformer.w_ffn1,
            transformer.w_ffn2,
            dense_out.linear,
        ]:
            torch.nn.init.uniform_(lin.weight, -0.1, 0.1)
            torch.nn.init.uniform_(lin.bias, -0.05, 0.05)
        torch.nn.init.uniform_(transformer.ln1_gamma, 0.9, 1.1)
        torch.nn.init.uniform_(transformer.ln1_beta, -0.05, 0.05)
        torch.nn.init.uniform_(transformer.ln2_gamma, 0.9, 1.1)
        torch.nn.init.uniform_(transformer.ln2_beta, -0.05, 0.05)

    # 2. Serialize to v2 JSON with flat LN keys matching NnLayerWeights in Rust.
    #    Rust save_json writes w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1,
    #    b_ffn1, w_ffn2, b_ffn2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta at the
    #    top level of the layer_N dict (not nested).
    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 16, "activation": "linear"},
            {"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq},
            {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
                "w": dense_in.linear.weight.detach().tolist(),
                "b": dense_in.linear.bias.detach().tolist(),
            },
            "layer_1": {
                "w_q": transformer.w_q.weight.detach().tolist(),
                "b_q": transformer.w_q.bias.detach().tolist(),
                "w_k": transformer.w_k.weight.detach().tolist(),
                "b_k": transformer.w_k.bias.detach().tolist(),
                "w_v": transformer.w_v.weight.detach().tolist(),
                "b_v": transformer.w_v.bias.detach().tolist(),
                "w_o": transformer.w_o.weight.detach().tolist(),
                "b_o": transformer.w_o.bias.detach().tolist(),
                "w_ffn1": transformer.w_ffn1.weight.detach().tolist(),
                "b_ffn1": transformer.w_ffn1.bias.detach().tolist(),
                "w_ffn2": transformer.w_ffn2.weight.detach().tolist(),
                "b_ffn2": transformer.w_ffn2.bias.detach().tolist(),
                "ln1_gamma": transformer.ln1_gamma.detach().tolist(),
                "ln1_beta": transformer.ln1_beta.detach().tolist(),
                "ln2_gamma": transformer.ln2_gamma.detach().tolist(),
                "ln2_beta": transformer.ln2_beta.detach().tolist(),
            },
            "layer_2": {
                "w": dense_out.linear.weight.detach().tolist(),
                "b": dense_out.linear.bias.detach().tolist(),
            },
        },
    }
    model_path = tmp_path / "transformer_model.json"
    model_path.write_text(json.dumps(model_json))

    # 3. 100 random f64 inputs (100 > n_seq=8 exercises cache growth + eviction).
    rng = np.random.default_rng(seed=42)
    inputs = rng.standard_normal((100, 8)).astype(np.float64)

    # 4. Rust: single NnState threaded across all 100 steps via nn_forward_sequence.
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2)

    # 5. Python: thread state through the three layers.
    #    - Dense is stateless (state=None), forward returns (y, None).
    #    - Transformer carries (k_cache, v_cache) tuple state.
    py_outs = np.empty((100, 2), dtype=np.float64)
    t_state = transformer.new_state(batch_size=1)
    dense_in.eval()
    transformer.eval()
    dense_out.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t : t + 1], dtype=torch.float64)  # (1, 8)
            h, _ = dense_in(x, None)
            h, t_state = transformer(h, t_state)
            y, _ = dense_out(h, None)
            py_outs[t] = y.squeeze(0).numpy()

    # 6. Assert equivalence.
    diff = np.abs(rust_outs - py_outs)
    max_diff = float(diff.max())
    print(f"Transformer cross-language max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-10, (
        f"cross-language mismatch: max diff = {max_diff:.2e}. "
        "Likely suspects: softmax reduction order, LN variance formula, "
        "GELU erf vs tanh-approx, PE iteration order, flat vs nested LN keys."
    )
