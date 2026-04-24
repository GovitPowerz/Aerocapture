"""Phase 4a gate: Mamba layer must reject PPO usage at build_layer / load_policy_from_json.

PSO training bypasses build_layer entirely (Rust direct). PPO path does call
build_layer via V2Policy construction, so this rejection is load-bearing for
the "PSO-only" Phase 4a scope.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import MambaSpec


def test_mamba_spec_validates():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2)
    assert spec.input_size == 8
    assert spec.d_state == 4
    assert spec.dt_rank == 2


def test_mamba_spec_auto_resolves_dt_rank():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16)
    assert spec.dt_rank == 2  # max(1, 32 // 16)


def test_mamba_spec_auto_resolves_dt_rank_small_input():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4)
    assert spec.dt_rank == 1  # max(1, 8 // 16)


def test_mamba_spec_rejects_dt_rank_larger_than_input():
    with pytest.raises(ValueError, match="dt_rank"):
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=16)


def test_build_layer_mamba_raises_not_implemented():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2)
    with pytest.raises(NotImplementedError, match="Mamba is PSO-only in Phase 4a"):
        build_layer(spec)


def test_load_policy_from_json_with_mamba_raises():
    from aerocapture.training.model_io import load_policy_from_json

    minimal_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 8, "activation": "linear"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
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
        with pytest.raises(NotImplementedError, match="Mamba"):
            load_policy_from_json(str(p))
