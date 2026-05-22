"""Phase 4a gate: Mamba spec validation + load_policy_from_json PPO-rejection.

build_layer now constructs MambaLayer (enabled in Task 2 for warm-start), so
the remaining PPO gate lives in load_policy_from_json (V2Policy cannot host
Mamba in Phase 4a) and in rl/train.py::_derive_hidden_shapes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.rl.schemas import MambaSpec


def test_mamba_spec_validates() -> None:
    spec = MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2)
    assert spec.input_size == 8
    assert spec.d_state == 4
    assert spec.dt_rank == 2


def test_mamba_spec_auto_resolves_dt_rank() -> None:
    spec = MambaSpec(type="mamba", input_size=32, d_state=16)
    assert spec.dt_rank == 2  # max(1, 32 // 16)


def test_mamba_spec_auto_resolves_dt_rank_small_input() -> None:
    spec = MambaSpec(type="mamba", input_size=8, d_state=4)
    assert spec.dt_rank == 1  # max(1, 8 // 16)


def test_mamba_spec_rejects_dt_rank_larger_than_input() -> None:
    with pytest.raises(ValueError, match="dt_rank"):
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=16)


def test_load_policy_from_json_with_mamba_raises() -> None:
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
