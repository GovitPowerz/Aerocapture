"""Smoke test for the collect_supervised PyO3 helper."""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_collect_supervised_returns_finite_traces():
    import aerocapture_rs
    import numpy as np

    X, y = aerocapture_rs.collect_supervised(
        toml_path="configs/training/msr_aller_ftc_train.toml",
        seeds=[42],
        scheme="ftc",
    )
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.ndim == 2 and X.shape[1] == 21, X.shape
    assert y.ndim == 1 and y.shape[0] == X.shape[0], (X.shape, y.shape)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert (y >= 0.0).all() and (y <= np.pi + 1e-9).all()
