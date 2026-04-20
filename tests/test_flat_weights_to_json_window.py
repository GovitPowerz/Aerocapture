"""End-to-end verification that aerocapture_rs.flat_weights_to_json handles
architectures containing Window layers (zero-weight spec-only entries).

The PyO3 helper delegates to NeuralNetModel::from_flat_weights_v2 + save_json,
which were extended for Window in the Phase 2b Rust commit. This test confirms
the PyO3 path carries the Window layer through to a valid JSON v2 file and
that the output skips the weights dict entry for the zero-param layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs


def test_flat_weights_to_json_window_roundtrip(tmp_path: Path) -> None:
    architecture = [
        {"type": "window", "input_size": 4, "n_steps": 3},
        {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
    ]
    # Window has zero params; Dense(12, 2) has 12*2 + 2 = 26.
    flat = (np.arange(26, dtype=np.float64) * 0.01).tolist()

    out_path = tmp_path / "window_model.json"
    aerocapture_rs.flat_weights_to_json(
        flat,
        json.dumps(architecture),
        str(out_path),
        "atan2",
        None,  # input_mask
    )
    assert out_path.exists()

    loaded = json.loads(out_path.read_text())
    assert loaded["format_version"] == 2
    assert len(loaded["architecture"]) == 2

    # Window entry is spec-only -- no weights dict under weights["layer_0"].
    window_entry = loaded["architecture"][0]
    assert window_entry == {"type": "window", "input_size": 4, "n_steps": 3}
    assert "layer_0" not in loaded.get("weights", {})

    # Dense entry has standard w/b keys at layer_1.
    dense_entry = loaded["architecture"][1]
    assert dense_entry["type"] == "dense"
    assert dense_entry["input_size"] == 12
    assert dense_entry["output_size"] == 2
    assert dense_entry["activation"] == "linear"
    assert "layer_1" in loaded["weights"]
    assert "w" in loaded["weights"]["layer_1"]
    assert "b" in loaded["weights"]["layer_1"]


def test_flat_weights_to_json_window_rejects_zero_fields(tmp_path: Path) -> None:
    # Window layer with zero n_steps should error at Rust validation time.
    architecture = [{"type": "window", "input_size": 4, "n_steps": 0}]
    out_path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        aerocapture_rs.flat_weights_to_json(
            [],
            json.dumps(architecture),
            str(out_path),
            "direct",
            None,
        )
