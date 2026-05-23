"""Smoke test for the collect_supervised PyO3 helper."""

from __future__ import annotations

from pathlib import Path

import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_collect_supervised_returns_finite_traces() -> None:
    import numpy as np

    results = aerocapture_rs.collect_supervised(
        toml_path="configs/training/msr_aller_ftc_train.toml",
        seeds=[42],
        scheme="ftc",
    )
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert set(r.keys()) == {"seed", "X", "y_signed", "dv", "captured"}
    X = np.asarray(r["X"])
    y = np.asarray(r["y_signed"])
    assert X.ndim == 2 and X.shape[1] == 21, X.shape
    assert y.ndim == 1 and y.shape[0] == X.shape[0], (X.shape, y.shape)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    # Supervised target is the post-lateral, pre-shaper signed bank command:
    # in [-pi, pi] (lateral guidance has applied the sign; shaping has NOT
    # run yet, so the value is unaffected by rate limits).
    assert (y >= -np.pi).all() and (y <= np.pi).all()


@pytest.mark.slow
def test_collect_supervised_overrides_neural_network_guidance_type(tmp_path: Path) -> None:
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
    import numpy as np

    results = aerocapture_rs.collect_supervised(
        toml_path="configs/training/msr_aller_nn_joint_train.toml",
        seeds=[42],
        overrides={"data.neural_network": str(tmp_path / "does_not_exist.json")},
        scheme="ftc",
    )
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    X = np.asarray(r["X"])
    y = np.asarray(r["y_signed"])
    assert X.shape[0] > 0, "collect_supervised should succeed when scheme=ftc despite missing NN file"
    assert X.shape[1] == 21
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


@pytest.mark.slow
def test_collect_supervised_rejects_nn_scheme() -> None:
    """scheme='neural_network' is not allowed — collect_supervised is for non-NN
    schemes that produce unsigned magnitude bank commands.
    """
    with pytest.raises(Exception, match="(?i)neural|scheme"):
        aerocapture_rs.collect_supervised(
            toml_path="configs/training/msr_aller_ftc_train.toml",
            seeds=[42],
            scheme="neural_network",
        )
