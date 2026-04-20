"""End-to-end verification that aerocapture_rs.flat_weights_to_json handles
architectures containing LSTM layers. The PyO3 helper delegates to
NeuralNetModel::from_flat_weights_v2 + save_json which were extended for
LSTM in Task 4; this test confirms the PyO3 path carries the LSTM layer
through to a valid JSON v2 file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


def test_flat_weights_to_json_lstm_roundtrip(tmp_path: Path) -> None:
    architecture = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        {"type": "lstm", "input_size": 4, "hidden_size": 2},
        {"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"},
    ]
    # Param counts:
    # Dense(3, 4): 3*4 + 4 = 16
    # Lstm(4, 2):  4*2*4 (weight_ih) + 4*2*2 (weight_hh) + 2*4*2 (biases) = 32 + 16 + 16 = 64
    # Dense(2, 2): 2*2 + 2 = 6
    # Total = 86
    n_params = 16 + 64 + 6
    flat = (np.arange(n_params, dtype=np.float64) * 0.001).tolist()

    out_path = tmp_path / "lstm_model.json"
    aerocapture_rs.flat_weights_to_json(
        flat,
        json.dumps(architecture),
        str(out_path),
        "atan2",
        None,  # input_mask
    )
    assert out_path.exists()

    payload = json.loads(out_path.read_text())

    assert payload["format_version"] == 2
    assert len(payload["architecture"]) == 3
    assert payload["architecture"][1]["type"] == "lstm"
    assert payload["architecture"][1]["input_size"] == 4
    assert payload["architecture"][1]["hidden_size"] == 2

    # LSTM weights live under "layer_1" key
    lstm_weights = payload["weights"]["layer_1"]
    assert "weight_ih" in lstm_weights
    assert "weight_hh" in lstm_weights
    assert "bias_ih" in lstm_weights
    assert "bias_hh" in lstm_weights

    # 4H=8 rows in both weight matrices
    assert len(lstm_weights["weight_ih"]) == 8
    assert len(lstm_weights["weight_ih"][0]) == 4  # input_size
    assert len(lstm_weights["weight_hh"]) == 8
    assert len(lstm_weights["weight_hh"][0]) == 2  # hidden_size
    assert len(lstm_weights["bias_ih"]) == 8
    assert len(lstm_weights["bias_hh"]) == 8
