import json
import math
from pathlib import Path

import numpy as np
from aerocapture.training.calibrate_inputs import (
    derive_affine,
    derive_asinh_scale,
    invert_transform,
)


def test_invert_asinh_roundtrip() -> None:
    raw = np.array([-500.0, 0.0, 880.0, 3000.0])
    s = 880.0
    norm = np.arcsinh(raw / s)
    back = invert_transform(norm, {"transform": "asinh", "scale": s, "center": 0.0})
    assert np.allclose(back, raw, atol=1e-9)


def test_invert_asinh_with_center_roundtrip() -> None:
    raw = np.array([10.0, 100.0, 1000.0])
    s, center = 50.0, 5.0
    norm = np.arcsinh((raw - center) / s)
    back = invert_transform(norm, {"transform": "asinh", "scale": s, "center": center})
    assert np.allclose(back, raw, atol=1e-9)


def test_invert_none_affine_roundtrip() -> None:
    raw = np.array([0.0, 0.5, 1.0])
    center, half = 0.45, 0.4
    norm = (raw - center) / half
    back = invert_transform(norm, {"transform": "none", "scale": half, "center": center})
    assert np.allclose(back, raw, atol=1e-9)


def test_invert_tanh_roundtrip() -> None:
    raw = np.array([-30.0, 0.0, 30.0])
    s, center = 20.0, 0.0
    norm = np.tanh((raw - center) / s)
    back = invert_transform(norm, {"transform": "tanh", "scale": s, "center": center})
    assert np.allclose(back, raw, atol=1e-6)


def test_derive_asinh_scale_puts_p99_at_one() -> None:
    s = derive_asinh_scale(p1=-200.0, p99=180.0)
    assert math.isclose(math.asinh(200.0 / s), 1.0, rel_tol=1e-9)


def test_derive_affine_maps_p1_p99_to_pm1() -> None:
    center, half = derive_affine(p1=10.0, p99=50.0)
    assert math.isclose((10.0 - center) / half, -1.0, rel_tol=1e-9)
    assert math.isclose((50.0 - center) / half, 1.0, rel_tol=1e-9)


def test_derive_affine_floors_degenerate_halfwidth() -> None:
    center, half = derive_affine(p1=5.0, p99=5.0)
    assert half >= 1e-6


def test_write_model_normalization_roundtrip(tmp_path: Path) -> None:
    from aerocapture.training.calibrate_inputs import _write_model_normalization

    model = tmp_path / "model.json"
    doc = {"format_version": 2, "architecture": [], "weights": {}, "input_mask": [0, 1]}
    model.write_text(json.dumps(doc))

    proposed = [{"transform": "asinh", "scale": float(i + 1), "center": 0.0} for i in range(35)]
    _write_model_normalization(str(model), proposed)

    reloaded = json.loads(model.read_text())
    assert reloaded["normalization"] == proposed
    # Existing fields preserved.
    assert reloaded["format_version"] == 2
    assert reloaded["input_mask"] == [0, 1]


def test_format_toml_normalization_roundtrips() -> None:
    import tomllib

    from aerocapture.training.calibrate_inputs import format_toml_normalization

    entries: list[dict] = [
        {"transform": "none", "scale": 0.8754754, "center": 0.9125593},
        {"transform": "asinh", "scale": 1919.853, "center": 0.0},
        {"transform": "tanh", "scale": 30.0, "center": -1.5},
    ]
    names = ["ecc_excess", "predicted_dv1", "time_since_flip"]
    snippet = format_toml_normalization(entries, names)
    # The snippet must be a `normalization = [...]` assignment that parses under [network].
    parsed = tomllib.loads("[network]\n" + snippet)
    got: list[dict] = parsed["network"]["normalization"]
    assert len(got) == 3
    for g, e in zip(got, entries, strict=True):
        assert g["transform"] == e["transform"]
        assert abs(float(g["scale"]) - float(e["scale"])) <= 1e-9 * max(1.0, abs(float(e["scale"])))
        assert abs(float(g["center"]) - float(e["center"])) <= 1e-9 * max(1.0, abs(float(e["center"])))
    # Readability: each entry annotated with its index+name as a trailing comment.
    assert "# 1 predicted_dv1" in snippet
