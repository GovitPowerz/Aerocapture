from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from aerocapture.training.quantize import _quantize_matrix, quantize_model_weights


def _dense_model(w: npt.NDArray[np.float64]) -> dict:
    return {
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": w.shape[1], "output_size": w.shape[0], "activation": "linear"}],
        "weights": {"layer_0": {"w": w.tolist(), "b": [0.0] * w.shape[0]}},
    }


def test_per_channel_error_le_per_tensor() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 16))
    w[0] *= 50.0  # one large row dominates the per-tensor scale
    err_c = np.abs(w - _quantize_matrix(w, 4, "per_channel")).mean()
    err_t = np.abs(w - _quantize_matrix(w, 4, "per_tensor")).mean()
    assert err_c <= err_t


def test_more_bits_less_error() -> None:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 4))
    e8 = np.abs(w - _quantize_matrix(w, 8, "per_tensor")).mean()
    e4 = np.abs(w - _quantize_matrix(w, 4, "per_tensor")).mean()
    assert e8 < e4


def test_zero_stays_zero() -> None:
    w = np.array([[0.0, 1.0], [-2.0, 0.0]])
    q = _quantize_matrix(w, 8, "per_channel")
    assert q[0, 0] == 0.0
    assert q[1, 1] == 0.0


def test_all_zero_row_no_nan() -> None:
    w = np.array([[0.0, 0.0], [1.0, -1.0]])
    q = _quantize_matrix(w, 4, "per_channel")
    assert np.all(np.isfinite(q))
    assert np.all(q[0] == 0.0)


def test_per_tensor_grid_membership() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((5, 7))
    qmax = 2 ** (4 - 1) - 1
    scale = np.max(np.abs(w)) / qmax
    levels = _quantize_matrix(w, 4, "per_tensor") / scale
    assert np.allclose(levels, np.round(levels))
    assert np.all(np.abs(levels) <= qmax + 1e-9)


def test_bits_below_two_raises() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        quantize_model_weights(_dense_model(np.ones((2, 2))), 1, "per_channel")


def test_unknown_granularity_raises() -> None:
    with pytest.raises(ValueError, match="granularity"):
        quantize_model_weights(_dense_model(np.ones((2, 2))), 8, "per_row")


def test_non_dense_raises() -> None:
    model = {
        "format_version": 2,
        "architecture": [{"type": "gru", "input_size": 4, "hidden_size": 4}],
        "weights": {"layer_0": {}},
    }
    with pytest.raises(ValueError, match="dense\\+mamba"):
        quantize_model_weights(model, 8, "per_channel")


def test_preserves_shape_biases_and_input() -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((3, 5))
    model = _dense_model(w)
    model["weights"]["layer_0"]["b"] = [0.1, 0.2, 0.3]
    out = quantize_model_weights(model, 8, "per_channel")
    wq = np.asarray(out["weights"]["layer_0"]["w"])
    assert wq.shape == w.shape
    assert np.all(np.isfinite(wq))
    assert out["weights"]["layer_0"]["b"] == [0.1, 0.2, 0.3]  # biases untouched
    assert model["weights"]["layer_0"]["w"] == w.tolist()  # input not mutated


def test_variant_metrics_capture_and_dv() -> None:
    from aerocapture.training.quantize import _variant_metrics

    fr = np.zeros((3, 52))
    # record 0: captured (ifinal=3, ecc<1), dv=100
    fr[0, 31], fr[0, 9], fr[0, 41] = 3, 0.5, 100.0
    # record 1: captured, dv=200
    fr[1, 31], fr[1, 9], fr[1, 41] = 3, 0.9, 200.0
    # record 2: not captured (pending crash, hyperbolic)
    fr[2, 31], fr[2, 9], fr[2, 41] = 4, 2.0, 5000.0

    m = _variant_metrics(fr, {})
    assert m["capture_rate"] == pytest.approx(2.0 / 3.0)
    assert m["dv_p50"] == pytest.approx(150.0)
    assert m["dv_p95"] == pytest.approx(195.0)
    assert np.isfinite(m["mean_cost"])


def test_variant_metrics_no_captures_dv_none() -> None:
    from aerocapture.training.quantize import _variant_metrics

    fr = np.zeros((2, 52))
    fr[:, 31] = 4  # all pending-crash -> no captures
    fr[:, 9] = 2.0
    m = _variant_metrics(fr, {})
    assert m["capture_rate"] == 0.0
    assert m["dv_p50"] is None
    assert m["dv_p95"] is None
    assert np.isfinite(m["mean_cost"])


def _mamba_model(rng: np.random.Generator) -> dict:
    """Dense(3->4) -> Mamba(4, d_state=2, dt_rank=1) -> Dense(4->2), random weights."""
    return {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
            {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 1},
            {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": rng.standard_normal((4, 3)).tolist(), "b": rng.standard_normal(4).tolist()},
            "layer_1": {
                "x_proj_w": rng.standard_normal((5, 4)).tolist(),  # (dt_rank + 2*d_state, input) = (5, 4)
                "dt_proj_w": rng.standard_normal((4, 1)).tolist(),
                "dt_proj_b": rng.standard_normal(4).tolist(),
                "a_log": rng.standard_normal((4, 2)).tolist(),
                "d_skip": rng.standard_normal(4).tolist(),
            },
            "layer_2": {"w": rng.standard_normal((2, 4)).tolist(), "b": rng.standard_normal(2).tolist()},
        },
    }


def _arrays_equal(a: object, b: object) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


def test_mamba_all_policy_quantizes_projections_and_dynamics() -> None:
    m = _mamba_model(np.random.default_rng(3))
    q = quantize_model_weights(m, 4, "per_tensor", "all")
    l1, ql1 = m["weights"]["layer_1"], q["weights"]["layer_1"]
    for field in ("x_proj_w", "dt_proj_w", "a_log", "d_skip"):
        assert not _arrays_equal(l1[field], ql1[field]), f"{field} should be rounded at 4 bits"
    assert _arrays_equal(l1["dt_proj_b"], ql1["dt_proj_b"]), "dt_proj_b is a bias: never quantized"
    assert _arrays_equal(m["weights"]["layer_0"]["b"], q["weights"]["layer_0"]["b"])
    assert not _arrays_equal(m["weights"]["layer_0"]["w"], q["weights"]["layer_0"]["w"])


def test_mamba_proj_only_policy_keeps_dynamics_fp() -> None:
    m = _mamba_model(np.random.default_rng(4))
    q = quantize_model_weights(m, 4, "per_channel", "proj_only")
    l1, ql1 = m["weights"]["layer_1"], q["weights"]["layer_1"]
    assert _arrays_equal(l1["a_log"], ql1["a_log"])
    assert _arrays_equal(l1["d_skip"], ql1["d_skip"])
    assert not _arrays_equal(l1["x_proj_w"], ql1["x_proj_w"])


def test_only_tensor_isolates_one_group() -> None:
    m = _mamba_model(np.random.default_rng(5))
    q = quantize_model_weights(m, 4, "per_channel", "all", only_tensor="layer_1.a_log")
    assert not _arrays_equal(m["weights"]["layer_1"]["a_log"], q["weights"]["layer_1"]["a_log"])
    for i, fields in ((0, ("w", "b")), (2, ("w", "b"))):
        for f in fields:
            assert _arrays_equal(m["weights"][f"layer_{i}"][f], q["weights"][f"layer_{i}"][f])
    for f in ("x_proj_w", "dt_proj_w", "dt_proj_b", "d_skip"):
        assert _arrays_equal(m["weights"]["layer_1"][f], q["weights"]["layer_1"][f])


def test_only_tensor_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="only_tensor"):
        quantize_model_weights(_mamba_model(np.random.default_rng(6)), 4, "per_channel", "all", only_tensor="layer_1.dt_proj_b")


def test_d_skip_identical_under_both_granularities() -> None:
    m = _mamba_model(np.random.default_rng(7))
    qc = quantize_model_weights(m, 4, "per_channel", "all")
    qt = quantize_model_weights(m, 4, "per_tensor", "all")
    assert _arrays_equal(qc["weights"]["layer_1"]["d_skip"], qt["weights"]["layer_1"]["d_skip"])


def test_bad_tensor_policy_raises() -> None:
    with pytest.raises(ValueError, match="tensor_policy"):
        quantize_model_weights(_mamba_model(np.random.default_rng(8)), 4, "per_channel", "matrices")


def test_chart_quant_sweep_writes_svg(tmp_path: Path) -> None:
    from aerocapture.training.charts_quant import chart_quant_sweep

    results = {
        "baseline": {"capture_rate": 0.90, "mean_cost": 100.0, "dv_p50": 50.0, "dv_p95": 80.0},
        "variants": [
            {"granularity": "per_channel", "bits": 8, "capture_rate": 0.90, "mean_cost": 101.0},
            {"granularity": "per_channel", "bits": 4, "capture_rate": 0.85, "mean_cost": 120.0},
            {"granularity": "per_tensor", "bits": 8, "capture_rate": 0.88, "mean_cost": 105.0},
            {"granularity": "per_tensor", "bits": 4, "capture_rate": 0.60, "mean_cost": 200.0},
        ],
        "n_sims": 10,
        "bits": [8, 4],
        "granularities": ["per_channel", "per_tensor"],
        "model_path": "x",
    }
    out = tmp_path / "sweep.svg"
    chart_quant_sweep(results, str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
def test_quant_sweep_smoke() -> None:
    pytest.importorskip("aerocapture_rs")

    from aerocapture.training.ablation import _resolve_nn_path
    from aerocapture.training.quantize import run_quant_sweep

    toml = "configs/training/msr_aller_nn_atan2_train.toml"
    if not _resolve_nn_path(toml).exists():
        pytest.skip("deployed dense model not present (gitignored training output)")

    try:
        results = run_quant_sweep(toml, bits=(8, 4), granularities=("per_channel", "per_tensor"), n_sims=2)
    except RuntimeError as e:
        if "input_mask length" in str(e):
            pytest.skip(f"config/model input_mask drift: {e}")
        raise

    assert set(results["baseline"]) >= {"capture_rate", "mean_cost", "dv_p50", "dv_p95"}
    assert len(results["variants"]) == 4
    for v in results["variants"]:
        assert 0.0 <= v["capture_rate"] <= 1.0
        assert np.isfinite(v["mean_cost"])
        assert {"granularity", "bits", "delta_capture_rate", "delta_mean_cost"} <= set(v)


def test_qat_batch_matches_quantize_matrix() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(7)
    n_out, n_in = 4, 6
    w = rng.standard_normal((n_out, n_in))
    b = rng.standard_normal(n_out)
    flat = np.concatenate([w.ravel(), b])[None, :]  # (1, n_out*n_in + n_out)
    arch = [{"type": "dense", "input_size": n_in, "output_size": n_out, "activation": "linear"}]
    out = quantize_flat_weights_batch(flat, arch, 4, "per_channel")
    expected_w = _quantize_matrix(w, 4, "per_channel")
    np.testing.assert_allclose(out[0, : n_out * n_in].reshape(n_out, n_in), expected_w)
    np.testing.assert_allclose(out[0, n_out * n_in :], b)  # biases untouched


def test_qat_batch_per_tensor_multilayer_biases_untouched() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(8)
    arch = [
        {"type": "dense", "input_size": 5, "output_size": 3, "activation": "linear"},
        {"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"},
    ]
    n_w = 5 * 3 + 3 + 3 * 2 + 2  # 26
    flat = rng.standard_normal((4, n_w))
    out = quantize_flat_weights_batch(flat, arch, 8, "per_tensor")
    assert out.shape == flat.shape
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, 15:18], flat[:, 15:18])  # layer-0 biases (idx 15..17) untouched


def test_qat_batch_non_dense_raises() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    arch = [{"type": "gru", "input_size": 4, "hidden_size": 4}]
    with pytest.raises(ValueError, match="dense\\+mamba"):
        quantize_flat_weights_batch(np.zeros((2, 10)), arch, 4, "per_channel")


def test_qat_batch_width_mismatch_raises() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    arch = [{"type": "dense", "input_size": 5, "output_size": 3, "activation": "linear"}]  # expects 18 cols
    with pytest.raises(ValueError, match="flat width"):
        quantize_flat_weights_batch(np.zeros((2, 20)), arch, 4, "per_channel")


def test_qat_batch_bits_below_two_raises() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    arch = [{"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"}]
    with pytest.raises(ValueError, match="n_bits"):
        quantize_flat_weights_batch(np.zeros((1, 6)), arch, 1, "per_channel")


_MAMBA_ARCH = [
    {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
    {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 1},
    {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
]
# flat widths: dense0 = 12w + 4b; mamba = 20 x_proj + 4 dt_proj_w + 4 dt_proj_b + 8 a_log + 4 d_skip; dense2 = 8w + 2b
_N_FLAT = 16 + 40 + 10  # 66


def _flat_to_model(flat: npt.NDArray[np.float64]) -> dict:
    """Slice a flat chromosome into the JSON weights layout (canonical Rust to_flat order)."""
    f = flat
    return {
        "format_version": 2,
        "architecture": [dict(e) for e in _MAMBA_ARCH],
        "weights": {
            "layer_0": {"w": f[0:12].reshape(4, 3).tolist(), "b": f[12:16].tolist()},
            "layer_1": {
                "x_proj_w": f[16:36].reshape(5, 4).tolist(),
                "dt_proj_w": f[36:40].reshape(4, 1).tolist(),
                "dt_proj_b": f[40:44].tolist(),
                "a_log": f[44:52].reshape(4, 2).tolist(),
                "d_skip": f[52:56].tolist(),
            },
            "layer_2": {"w": f[56:64].reshape(2, 4).tolist(), "b": f[64:66].tolist()},
        },
    }


@pytest.mark.parametrize("granularity", ["per_channel", "per_tensor"])
@pytest.mark.parametrize("tensor_policy", ["all", "proj_only"])
def test_flat_and_json_paths_agree(granularity: str, tensor_policy: str) -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch, quantize_model_weights

    rng = np.random.default_rng(11)
    flat = rng.standard_normal((3, _N_FLAT))
    q_flat = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 4, granularity, tensor_policy)
    for row in range(3):
        q_json = quantize_model_weights(_flat_to_model(flat[row]), 4, granularity, tensor_policy)
        np.testing.assert_allclose(q_flat[row], _model_to_flat(q_json), rtol=0, atol=0)


def _model_to_flat(model: dict) -> npt.NDArray[np.float64]:
    w = model["weights"]
    parts = [
        np.asarray(w["layer_0"]["w"]).ravel(),
        np.asarray(w["layer_0"]["b"]).ravel(),
        np.asarray(w["layer_1"]["x_proj_w"]).ravel(),
        np.asarray(w["layer_1"]["dt_proj_w"]).ravel(),
        np.asarray(w["layer_1"]["dt_proj_b"]).ravel(),
        np.asarray(w["layer_1"]["a_log"]).ravel(),
        np.asarray(w["layer_1"]["d_skip"]).ravel(),
        np.asarray(w["layer_2"]["w"]).ravel(),
        np.asarray(w["layer_2"]["b"]).ravel(),
    ]
    return np.concatenate([p.astype(np.float64) for p in parts])


def test_flat_biases_and_dt_proj_b_pass_through() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(12)
    flat = rng.standard_normal((2, _N_FLAT))
    q = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 3, "per_channel", "all")
    np.testing.assert_array_equal(q[:, 12:16], flat[:, 12:16])  # dense0 bias
    np.testing.assert_array_equal(q[:, 40:44], flat[:, 40:44])  # dt_proj_b
    np.testing.assert_array_equal(q[:, 64:66], flat[:, 64:66])  # dense2 bias


def test_flat_proj_only_keeps_dynamics_slabs() -> None:
    from aerocapture.training.quantize import quantize_flat_weights_batch

    rng = np.random.default_rng(13)
    flat = rng.standard_normal((2, _N_FLAT))
    q = quantize_flat_weights_batch(flat, _MAMBA_ARCH, 3, "per_channel", "proj_only")
    np.testing.assert_array_equal(q[:, 44:52], flat[:, 44:52])  # a_log
    np.testing.assert_array_equal(q[:, 52:56], flat[:, 52:56])  # d_skip


def test_flat_scaffolding_tail_width_raises() -> None:
    """A 962+3 chromosome (live scaffolding) must never reach this function."""
    from aerocapture.training.quantize import quantize_flat_weights_batch

    flat = np.zeros((1, _N_FLAT + 3))
    with pytest.raises(ValueError, match="flat width"):
        quantize_flat_weights_batch(flat, _MAMBA_ARCH, 4, "per_channel", "all")
