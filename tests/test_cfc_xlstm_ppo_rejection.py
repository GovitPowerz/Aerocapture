"""cfc/slstm/mlstm are PSO-only: build_layer + load_policy_from_json must raise."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import CfcSpec, MlstmSpec, SlstmSpec

SPECS = [
    CfcSpec(type="cfc", input_size=8, hidden_size=4, backbone_units=4),
    SlstmSpec(type="slstm", input_size=8, hidden_size=4),
    MlstmSpec(type="mlstm", input_size=8, hidden_size=4),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.type)
def test_build_layer_rejects_probe_layers(spec: CfcSpec | SlstmSpec | MlstmSpec) -> None:
    with pytest.raises(NotImplementedError, match="PSO-only"):
        build_layer(spec)


def test_load_policy_from_json_with_cfc_raises() -> None:
    from aerocapture.training.model_io import load_policy_from_json

    minimal_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "linear"},
            {"type": "cfc", "input_size": 4, "hidden_size": 4, "backbone_units": 4},
        ],
        "weights": {
            "layer_0": {"w": [[0.0] * 4] * 4, "b": [0.0] * 4},
            "layer_1": {
                "w_bb": [[0.0] * 8] * 4,
                "b_bb": [0.0] * 4,
                "w_ff1": [[0.0] * 4] * 4,
                "b_ff1": [0.0] * 4,
                "w_ff2": [[0.0] * 4] * 4,
                "b_ff2": [0.0] * 4,
                "w_ta": [[0.0] * 4] * 4,
                "b_ta": [0.0] * 4,
                "w_tb": [[0.0] * 4] * 4,
                "b_tb": [0.0] * 4,
            },
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.json"
        p.write_text(json.dumps(minimal_json))
        with pytest.raises(NotImplementedError):
            load_policy_from_json(str(p))
