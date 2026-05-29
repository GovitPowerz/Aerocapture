import json

import aerocapture_rs
import numpy as np
import pytest
from aerocapture.training.toml_utils import load_toml_with_bases


def _mint_zero_model(tmp_path):
    """Mint a loadable zero-weight NN matching the delta config's arch."""
    cfg = load_toml_with_bases("configs/training/msr_aller_nn_delta_train.toml")
    arch = cfg["network"]["architecture"]
    mask = cfg["network"]["input_mask"]

    def n_params(layer):  # dense only in this arch
        return layer["input_size"] * layer["output_size"] + layer["output_size"]

    flat = [0.0] * sum(n_params(l) for l in arch)
    path = str(tmp_path / "zero_model.json")
    aerocapture_rs.flat_weights_to_json(
        flat, json.dumps(arch), path, mask,
        cfg["guidance"]["neural_network"]["output_parameterization"],
        None, cfg["guidance"]["neural_network"]["delta_max"],
    )
    return path


@pytest.mark.slow
def test_collect_nn_inputs_runs_nn_and_returns_shapes(tmp_path):
    model = _mint_zero_model(tmp_path)
    out = aerocapture_rs.collect_nn_inputs(
        "configs/training/msr_aller_nn_delta_train.toml",
        [4_000_000],
        overrides={"data.neural_network": model},
    )
    assert len(out) == 1
    r = out[0]
    assert set(r.keys()) == {"seed", "X", "time", "energy", "dv", "captured"}
    X, t, e = r["X"], r["time"], r["energy"]
    assert X.ndim == 2 and X.shape[1] == 31
    assert t.shape == (X.shape[0],) and e.shape == (X.shape[0],)
    assert np.all(np.diff(t) >= 0)
    assert np.isfinite(X).all() and np.isfinite(t).all() and np.isfinite(e).all()
    assert isinstance(r["captured"], bool)


def test_collect_nn_inputs_rejects_non_nn_config():
    with pytest.raises((ValueError, RuntimeError), match="(?i)neural_network"):
        aerocapture_rs.collect_nn_inputs("configs/training/msr_aller_ftc_train.toml", [1])
