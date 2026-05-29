import json
import os
import tempfile

import aerocapture_rs


def test_flat_weights_to_json_delta_roundtrip():
    arch = json.dumps([{"type": "dense", "input_size": 3, "output_size": 1, "activation": "tanh"}])
    flat = [0.0, 0.0, 0.0, 0.0]  # 3*1 + 1
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    aerocapture_rs.flat_weights_to_json(flat, arch, path, None, "delta", None, 0.5)
    doc = json.load(open(path))
    assert doc["output_param"] == "delta"
    assert abs(doc["delta_max"] - 0.5) < 1e-12
    os.remove(path)


def test_flat_weights_to_json_scaled_pi_roundtrip():
    arch = json.dumps([{"type": "dense", "input_size": 3, "output_size": 1, "activation": "tanh"}])
    flat = [0.0, 0.0, 0.0, 0.0]
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    aerocapture_rs.flat_weights_to_json(flat, arch, path, None, "scaled_pi", 2.0, None)
    doc = json.load(open(path))
    assert doc["output_param"] == "scaled_pi"
    assert abs(doc["scaled_pi_n"] - 2.0) < 1e-12
    os.remove(path)
