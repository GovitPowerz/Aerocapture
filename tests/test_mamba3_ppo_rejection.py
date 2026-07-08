"""Mamba3 is PSO-only: build_layer + load_policy_from_json must raise."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import Mamba3Spec


def test_mamba3_spec_validates_and_resolves_dt_rank() -> None:
    spec = Mamba3Spec(type="mamba3", input_size=32, d_state=16)
    assert spec.dt_rank == 2  # max(1, 32 // 16)
    assert spec.discretization == "euler"
    assert spec.state_mode == "real"


def test_build_layer_rejects_mamba3() -> None:
    spec = Mamba3Spec(type="mamba3", input_size=8, d_state=4, dt_rank=1, discretization="trapezoidal", state_mode="complex")
    with pytest.raises(NotImplementedError, match="PSO-only"):
        build_layer(spec)


def test_load_policy_from_json_with_mamba3_raises() -> None:
    from aerocapture.training.model_io import load_policy_from_json

    minimal_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 8, "activation": "linear"},
            {"type": "mamba3", "input_size": 8, "d_state": 4, "dt_rank": 2, "discretization": "euler", "state_mode": "real"},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": [[0.0] * 8] * 8, "b": [0.0] * 8},
            "layer_1": {
                "x_proj_w": [[0.0] * 8] * 8,
                "dt_proj_w": [[0.0] * 2] * 8,
                "dt_proj_b": [0.0] * 8,
                "a_log": [[0.0] * 4] * 8,
                "d_skip": [0.0] * 8,
            },
            "layer_2": {"w": [[0.0] * 8] * 2, "b": [0.0] * 2},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.json"
        p.write_text(json.dumps(minimal_json))
        with pytest.raises(NotImplementedError, match="Mamba3"):
            load_policy_from_json(str(p))
