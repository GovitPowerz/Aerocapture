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


@pytest.mark.slow
def test_collect_supervised_overrides_neural_network_guidance_type(tmp_path):
    """Regression for commit 2539901: collect_supervised must push guidance.type
    into seed_overrides BEFORE load_and_override constructs SimData.

    SimData::from_toml gates the [data] neural_network model file load on
    config.guidance_type == NeuralNetwork — evaluated at construction time.
    So a TOML configured with [guidance] type = "neural_network" pointing
    [data] neural_network at a not-yet-trained best_model.json would error
    out at SimData construction, before collect_supervised ever ran the
    actual FTC simulation.

    This test exercises the override path: a TOML with the joint config
    (guidance.type = neural_network) plus an overrides dict pointing
    [data] neural_network at a non-existent file. With the fix, the call
    succeeds because guidance.type=ftc is applied before the NN-file gate
    runs. Without the fix, it fails with "Cannot read <path>".
    """
    import aerocapture_rs
    import numpy as np

    X, y = aerocapture_rs.collect_supervised(
        toml_path="configs/training/msr_aller_nn_joint_train.toml",
        seeds=[42],
        overrides={"data.neural_network": str(tmp_path / "does_not_exist.json")},
        scheme="ftc",
    )
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.shape[0] > 0, "collect_supervised should succeed when scheme=ftc despite missing NN file"
    assert X.shape[1] == 21
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


@pytest.mark.slow
def test_collect_supervised_rejects_nn_scheme():
    """scheme='neural_network' is not allowed — collect_supervised is for non-NN
    schemes that produce unsigned magnitude bank commands.
    """
    import aerocapture_rs

    with pytest.raises(Exception, match="(?i)neural|scheme"):
        aerocapture_rs.collect_supervised(
            toml_path="configs/training/msr_aller_ftc_train.toml",
            seeds=[42],
            scheme="neural_network",
        )
