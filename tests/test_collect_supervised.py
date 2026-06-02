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
    assert set(r.keys()) == {"seed", "X", "y_signed", "prev_realized", "dv", "captured"}
    X = np.asarray(r["X"])
    y = np.asarray(r["y_signed"])
    assert X.ndim == 2 and X.shape[1] == 35, X.shape
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
    assert X.shape[1] == 35
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


@pytest.mark.slow
def test_collect_supervised_honors_normalization_override(tmp_path: Path) -> None:
    """collect_supervised runs a TEACHER scheme, so SimData.neural_net is None.
    The recorded trace's per-input normalization must still honor a config
    [network.normalization] override -- otherwise warm-start BPTT trains on
    DEFAULT-scaled inputs while the deployed NN uses the override scales, a
    silent train/inference mismatch. Regression for that latent bug.
    """
    import numpy as np
    from aerocapture.training.calibrate_inputs import _current_transforms, invert_transform

    ftc = (Path("configs/training/msr_aller_ftc_train.toml")).resolve()
    # Identity override: norm == raw for every input. Differs from DEFAULT on
    # every asinh/affine column, so a trace that ignores it (uses DEFAULT) is
    # detectably wrong.
    ident = "\n".join('  { transform = "none", scale = 1.0, center = 0.0 },' for _ in range(35))
    cfg = tmp_path / "ftc_ident_norm.toml"
    cfg.write_text(f'base = ["{ftc}"]\n\n[network]\nnormalization = [\n{ident}\n]\n')

    seeds = [42]
    x_def = np.asarray(aerocapture_rs.collect_supervised(toml_path=str(ftc), seeds=seeds, scheme="ftc")[0]["X"])
    x_id = np.asarray(aerocapture_rs.collect_supervised(toml_path=str(cfg), seeds=seeds, scheme="ftc")[0]["X"])
    assert x_def.shape == x_id.shape and x_def.shape[1] == 35

    # FTC control is identical between runs (it never reads NN inputs), so the
    # raw per-tick values match. Under the identity override, x_id IS the raw,
    # which must equal the raw recovered by inverting the DEFAULT-normalized run.
    # Skip tanh columns: tanh saturates, so the DEFAULT-normalized trace pins to
    # ~+-1 at the tail and atanh cannot recover the true raw -- a lossy-inverse
    # artifact of the cross-check, not of the override path under test.
    default = _current_transforms()
    for i in range(35):
        if default[i]["transform"] == "tanh":
            continue
        raw_i = invert_transform(x_def[:, i], default[i])
        np.testing.assert_allclose(x_id[:, i], raw_i, rtol=1e-6, atol=1e-6, err_msg=f"input {i} ignored the override")
    # Sanity: the override actually changed the trace (DEFAULT != identity).
    assert not np.allclose(x_id, x_def)
