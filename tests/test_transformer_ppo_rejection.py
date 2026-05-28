"""Transformer is PSO-only in Phase 3a; load_policy_from_json must reject cleanly.

build_layer now constructs TransformerLayer (enabled in Task 2 for warm-start
infrastructure); the remaining PPO gate lives in load_policy_from_json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def test_load_policy_from_json_rejects_transformer(tmp_path: Path) -> None:
    from aerocapture.training.model_io import load_policy_from_json

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 4, "activation": "linear"},
            {"type": "transformer", "d_model": 4, "n_heads": 2, "d_ffn": 8, "n_seq": 4},
            {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": [[0.0] * 8] * 4, "b": [0.0] * 4},
            "layer_1": {
                "w_q": [[0.0] * 4] * 4,
                "b_q": [0.0] * 4,
                "w_k": [[0.0] * 4] * 4,
                "b_k": [0.0] * 4,
                "w_v": [[0.0] * 4] * 4,
                "b_v": [0.0] * 4,
                "w_o": [[0.0] * 4] * 4,
                "b_o": [0.0] * 4,
                "w_ffn1": [[0.0] * 4] * 8,
                "b_ffn1": [0.0] * 8,
                "w_ffn2": [[0.0] * 8] * 4,
                "b_ffn2": [0.0] * 4,
                "ln1_gamma": [1.0] * 4,
                "ln1_beta": [0.0] * 4,
                "ln2_gamma": [1.0] * 4,
                "ln2_beta": [0.0] * 4,
            },
            "layer_2": {"w": [[0.0] * 4] * 2, "b": [0.0] * 2},
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model_json))
    with pytest.raises(NotImplementedError, match="Transformer"):
        load_policy_from_json(str(path), device=torch.device("cpu"))
