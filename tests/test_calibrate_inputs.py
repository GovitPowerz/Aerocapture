import math

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
    back = invert_transform(norm, ("asinh", s))
    assert np.allclose(back, raw, atol=1e-9)


def test_invert_affine_roundtrip() -> None:
    raw = np.array([0.0, 25.0, 50.0])
    norm = raw / 50.0 - 1.0
    back = invert_transform(norm, ("affine", 1.0 / 50.0, -1.0))
    assert np.allclose(back, raw, atol=1e-9)


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
