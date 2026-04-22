"""Smoke test for aerocapture_rs.flat_weights_to_json with a Transformer layer.

Verifies Rust path end-to-end: flat -> from_flat_weights_v2 -> save_json.
"""

import json
from pathlib import Path

import aerocapture_rs
import numpy as np


def test_flat_weights_to_json_transformer_roundtrip(tmp_path: Path) -> None:
    architecture = [
        {"type": "dense", "input_size": 8, "output_size": 4, "activation": "linear"},
        {"type": "transformer", "d_model": 4, "n_heads": 2, "d_ffn": 8, "n_seq": 3},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    # Transformer params: 4*16 + 2*32 + 8 + 36 = 172
    # Dense 0: 8*4 + 4 = 36
    # Dense 2: 4*2 + 2 = 10
    # Total: 218
    rng = np.random.default_rng(0)
    flat = rng.standard_normal(218)

    out_path = tmp_path / "model.json"
    aerocapture_rs.flat_weights_to_json(flat.tolist(), json.dumps(architecture), str(out_path), None)

    with out_path.open() as f:
        obj = json.load(f)

    assert obj["format_version"] == 2
    assert len(obj["architecture"]) == 3
    assert obj["architecture"][1]["type"] == "transformer"
    assert obj["architecture"][1]["d_model"] == 4
    assert "layer_1" in obj["weights"]
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
        assert key in layer_1, f"missing key {key}"

    # Verify nn_forward loads + runs
    y = aerocapture_rs.nn_forward(str(out_path), [0.1] * 8)
    assert len(y) == 2
    assert all(np.isfinite(v) for v in y)
