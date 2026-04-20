"""Phase 2b export_v2_policy_to_json + load_policy_from_json Window guards.

export_v2_policy_to_json obs-norm guard rejects Window as layer 0 (the same
"Dense embedding required" invariant as GRU/LSTM).

load_policy_from_json short-circuits with NotImplementedError on any v2 JSON
containing a Window layer -- V2Policy cannot be built with Window (PPO path
is explicitly unsupported in Phase 2b).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.rl.export import export_v2_policy_to_json
from aerocapture.training.rl.layers import WindowLayer
from aerocapture.training.rl.schemas import DenseSpec, WindowSpec


def test_export_obs_norm_rejects_window_as_layer_0(tmp_path: Path) -> None:
    # The obs-norm guard lives inside the per-layer export loop; it fires when
    # iteration reaches a WindowLayer at index 0 with obs_normalizer != None.
    # V2Policy would never contain a WindowLayer (build_layer raises), so we
    # construct a stand-in with a real WindowLayer in policy.layers[0].
    policy = MagicMock()
    policy.architecture = [
        WindowSpec(type="window", input_size=4, n_steps=3),
        DenseSpec(type="dense", input_size=12, output_size=2, activation="linear"),
    ]
    # policy.layers must be iterable and contain a real WindowLayer at index 0.
    policy.layers = [WindowLayer(input_size=4, n_steps=3)]
    obs_normalizer = MagicMock()

    with pytest.raises(NotImplementedError) as exc_info:
        export_v2_policy_to_json(policy, str(tmp_path / "out.json"), obs_normalizer=obs_normalizer)
    assert "Window" in str(exc_info.value)


def test_load_policy_from_json_rejects_window_architecture(tmp_path: Path) -> None:
    arch_json = {
        "format_version": 2,
        "architecture": [
            {"type": "window", "input_size": 4, "n_steps": 3},
            {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_1": {
                "w": [[0.01] * 12, [0.01] * 12],
                "b": [0.0, 0.0],
            }
        },
        "output_interpretation": "atan2",
        "input_mask": None,
        "ablated_input": None,
    }
    json_path = tmp_path / "window.json"
    json_path.write_text(json.dumps(arch_json))

    with pytest.raises(NotImplementedError) as exc_info:
        load_policy_from_json(str(json_path), device=torch.device("cpu"))
    msg = str(exc_info.value)
    assert "Window" in msg
    assert "PSO-only" in msg
