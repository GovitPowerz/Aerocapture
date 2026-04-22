"""Transformer cache grows organically from 0 to n_seq, then saturates.

Verifies (a) no zero-padding during warm-up (cache length < n_seq works),
(b) output is deterministic across repeated runs, (c) Python mirror and
Rust runtime agree even during the growth phase.

Architecture: Transformer(d_model=8, n_heads=2, d_ffn=16, n_seq=4) ->
Dense(8 -> 2, linear). The nn_forward_sequence API enforces output_size==2
(atan2 contract), so the trailing Dense is the minimal required wrapper.
Cache-length assertions target only the Transformer state, as intended.
"""

from __future__ import annotations

import json
from pathlib import Path

import aerocapture_rs  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch
from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer


@pytest.mark.slow
def test_transformer_cache_warmup(tmp_path: Path) -> None:
    d_model, n_heads, d_ffn, n_seq = 8, 2, 16, 4
    torch.manual_seed(1)
    transformer = TransformerLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_ffn=d_ffn,
        n_seq=n_seq,
    ).double()
    dense_out = DenseLayer(input_size=d_model, output_size=2, activation="linear").double()
    with torch.no_grad():
        for lin in [transformer.w_q, transformer.w_k, transformer.w_v, transformer.w_o, transformer.w_ffn1, transformer.w_ffn2]:
            torch.nn.init.uniform_(lin.weight, -0.1, 0.1)
            torch.nn.init.uniform_(lin.bias, -0.05, 0.05)
        torch.nn.init.uniform_(dense_out.linear.weight, -0.1, 0.1)
        torch.nn.init.uniform_(dense_out.linear.bias, -0.05, 0.05)
        # Default LN gamma=1, beta=0 -- leave unchanged.

    # Two-layer v2 JSON: Transformer -> Dense(8->2)
    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq},
            {"type": "dense", "input_size": d_model, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
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
            "layer_1": {
                "w": dense_out.linear.weight.detach().tolist(),
                "b": dense_out.linear.bias.detach().tolist(),
            },
        },
    }
    model_path = tmp_path / "m.json"
    model_path.write_text(json.dumps(model_json))

    # Drive only 3 steps (fewer than n_seq=4 -- warm-up phase)
    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((3, d_model))

    rust_outs_1 = np.asarray(aerocapture_rs.nn_forward_sequence(str(model_path), inputs.tolist()))

    # Python reference, tracking transformer KV-cache length at each step
    t_state = transformer.new_state(batch_size=1)
    py_outs = np.empty_like(rust_outs_1)
    transformer.eval()
    dense_out.eval()
    with torch.no_grad():
        for t in range(3):
            x = torch.tensor(inputs[t : t + 1], dtype=torch.float64)
            t_out, t_state = transformer(x, t_state)
            out, _ = dense_out(t_out, None)
            py_outs[t] = out.squeeze(0).numpy()
            # Cache grows organically: after step t, len == t+1 (still < n_seq=4).
            assert t_state[0].shape[1] == t + 1, f"step {t}: expected cache_len={t + 1}, got {t_state[0].shape[1]} (zero-padding leaked?)"
            assert t_state[1].shape[1] == t + 1

    # Cross-language agreement during warm-up
    warmup_diff = float(np.abs(rust_outs_1 - py_outs).max())
    assert warmup_diff < 1e-10, f"warm-up cross-language mismatch: {warmup_diff:.2e}"

    # Determinism: run the same sequence again, expect bit-identical output
    rust_outs_2 = np.asarray(aerocapture_rs.nn_forward_sequence(str(model_path), inputs.tolist()))
    assert np.array_equal(rust_outs_1, rust_outs_2), "Rust forward is not deterministic"
