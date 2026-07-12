from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.config import NetworkConfig

_DENSE2 = [
    {"type": "dense", "input_size": 3, "output_size": 4, "activation": "linear"},
    {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
]


def test_network_config_qat_defaults_off() -> None:
    net = NetworkConfig(architecture=list(_DENSE2))
    assert net.qat_bits is None
    assert net.qat_granularity == "per_channel"


def test_network_config_qat_valid() -> None:
    net = NetworkConfig(architecture=list(_DENSE2), qat_bits=4, qat_granularity="per_tensor")
    assert net.qat_bits == 4
    assert net.qat_granularity == "per_tensor"


def test_network_config_qat_bits_below_two_raises() -> None:
    with pytest.raises(ValueError, match="qat_bits"):
        NetworkConfig(architecture=list(_DENSE2), qat_bits=1)


def test_network_config_qat_bad_granularity_raises() -> None:
    with pytest.raises(ValueError, match="qat_granularity"):
        NetworkConfig(architecture=list(_DENSE2), qat_bits=4, qat_granularity="per_row")


def test_network_config_qat_non_dense_raises() -> None:
    arch = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        {"type": "gru", "input_size": 4, "hidden_size": 2},
    ]
    with pytest.raises(ValueError, match="dense\\+mamba"):
        NetworkConfig(architecture=arch, qat_bits=4)


def test_qat_eval_hook_quantizes_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training import problem as problem_mod
    from aerocapture.training.param_spaces import ParamSpec
    from aerocapture.training.parquet_output import FINAL_RECORD_LEN
    from aerocapture.training.problem import AerocaptureProblem
    from aerocapture.training.quantize import quantize_flat_weights_batch

    n_w = 3 * 2 + 2 + 2 * 2 + 2  # 14
    arch = [
        {"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"},
        {"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"},
    ]
    specs = [ParamSpec(f"w{i}", -1.0, 1.0, 0.0) for i in range(n_w)]
    nn_cfg = NetworkConfig(architecture=arch, qat_bits=4, qat_granularity="per_channel")
    prob = AerocaptureProblem(
        param_specs=specs,
        toml_path="dummy.toml",
        seeds=[1],
        cost_kwargs={},
        scheme="neural_network",
        nn_config=nn_cfg,
    )

    captured: dict[str, npt.NDArray[np.float64]] = {}
    rng = np.random.default_rng(0)
    X = rng.random((5, n_w))

    def fake_run_grid(*args: object, **kw: object) -> object:
        captured["weights"] = np.asarray(kw["weights"], dtype=np.float64)
        return np.zeros((X.shape[0], 1, FINAL_RECORD_LEN))

    monkeypatch.setattr(problem_mod._aero_rs, "run_grid", fake_run_grid)
    prob._run_grid_records(X, [1])

    decoded = -1.0 + X * 2.0  # specs are [-1, 1], so decode = 2X - 1
    expected = quantize_flat_weights_batch(decoded, arch, 4, "per_channel", "all")
    np.testing.assert_allclose(captured["weights"], expected, rtol=0, atol=1e-12)


def test_qat_deploy_hook_quantizes_written_weights(tmp_path: Path) -> None:
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.evaluate import write_nn_json
    from aerocapture.training.quantize import quantize_flat_weights_batch

    arch = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "linear"},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    n_w = 3 * 4 + 4 + 4 * 2 + 2  # 26
    net = NetworkConfig(architecture=arch, qat_bits=4, qat_granularity="per_channel")
    rng = np.random.default_rng(1)
    weights = rng.standard_normal(n_w)
    out = tmp_path / "qat_model.json"
    write_nn_json(weights, net, out)

    written = json.loads(out.read_text())
    flat_written: list[float] = []
    for i in range(len(arch)):
        lw = written["weights"][f"layer_{i}"]
        flat_written.extend(np.asarray(lw["w"], dtype=np.float64).ravel().tolist())
        flat_written.extend([float(x) for x in lw["b"]])
    expected = quantize_flat_weights_batch(weights.reshape(1, -1), arch, 4, "per_channel", "all")[0]
    np.testing.assert_allclose(flat_written, expected, rtol=0, atol=1e-10)


def test_network_config_qat_mamba_accepted() -> None:
    arch = [
        {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
        {"type": "mamba", "input_size": 16, "d_state": 12},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
    ]
    net = NetworkConfig(architecture=arch, qat_bits=4, qat_tensor_policy="proj_only")
    assert net.qat_bits == 4
    assert net.qat_tensor_policy == "proj_only"


def test_network_config_qat_mamba3_rejected() -> None:
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "mamba3", "input_size": 4, "d_state": 2, "dt_rank": 1, "discretization": "euler", "state_mode": "real"},
    ]
    with pytest.raises(ValueError, match="dense\\+mamba"):
        NetworkConfig(architecture=arch, qat_bits=4)


def test_network_config_qat_bad_policy_raises() -> None:
    with pytest.raises(ValueError, match="qat_tensor_policy"):
        NetworkConfig(architecture=list(_DENSE2), qat_bits=4, qat_tensor_policy="matrices")


def test_network_config_qat_policy_ignored_when_off() -> None:
    net = NetworkConfig(architecture=list(_DENSE2), qat_tensor_policy="nonsense")  # qat_bits None -> no validation
    assert net.qat_bits is None


@pytest.mark.slow
def test_qat_mamba_end_to_end_deploy_idempotent(tmp_path: Path) -> None:
    """2 real GA gens with qat_bits=8 on a tiny dense->mamba->dense arch; the
    deployed best_model.json must be idempotent under re-quantization (i.e. the
    deploy writer actually rounded the weights)."""
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.quantize import quantize_model_weights

    repo = Path(__file__).resolve().parents[1]
    base = (repo / "configs/training/msr_aller_nn_atan2_train.toml").resolve()
    # train.py hard-requires [data] neural_network's parent to start with the
    # literal "training_output/" (src/python/aerocapture/training/train.py,
    # ~line 2584) so save_dir/checkpoints land next to the deploy JSON. A
    # tmp_path-absolute deploy path trips that check with SystemExit(1). Mirror
    # the established fix already used by test_mamba_pso_end_to_end.py: a
    # dedicated training_output/ subdir, cleaned up before and after the run.
    training_output_dir = repo / "training_output" / "__pytest_qat_e2e__"
    nn_deploy_path = training_output_dir.relative_to(repo) / "best_model.json"
    toml_text = f"""
base = ["{base}"]

[data]
neural_network = "{nn_deploy_path.as_posix()}"
results_suffix = ".qat_smoke"

[network]
qat_bits = 8
qat_granularity = "per_channel"
qat_tensor_policy = "all"

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 4
activation = "swish"

[[network.architecture]]
type = "mamba"
input_size = 4
d_state = 2

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "asinh"

[optimizer]
algorithm = "ga"
n_pop = 4
n_gen = 2
training_n_sims = 2
validation_n_sims = 2
seed_strategy = "fixed"
"""
    cfg = tmp_path / "qat_smoke.toml"
    cfg.write_text(toml_text)
    if training_output_dir.exists():
        shutil.rmtree(training_output_dir)
    try:
        subprocess.run(
            [sys.executable, "-m", "aerocapture.training.train", str(cfg), "--no-tui", "--skip-report", "--from-scratch"],
            check=True,
            cwd=repo,
            timeout=600,
        )
        model = json.loads((training_output_dir / "best_model.json").read_text())
        assert [e["type"] for e in model["architecture"]] == ["dense", "mamba", "dense"]
        q = quantize_model_weights(model, 8, "per_channel", "all")
        for i in range(3):
            for field in model["weights"][f"layer_{i}"]:
                np.testing.assert_allclose(
                    np.asarray(model["weights"][f"layer_{i}"][field], dtype=np.float64),
                    np.asarray(q["weights"][f"layer_{i}"][field], dtype=np.float64),
                    rtol=0,
                    atol=1e-12,
                    err_msg=f"layer_{i}.{field} not on the 8-bit grid",
                )
    finally:
        if training_output_dir.exists():
            shutil.rmtree(training_output_dir)
