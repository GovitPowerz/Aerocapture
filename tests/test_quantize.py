from __future__ import annotations

import json
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

    def _v(bits: int, gran: str, policy: str, capture: float, cvar: float | None) -> dict:
        return {"bits": bits, "granularity": gran, "tensor_policy": policy, "capture_rate": capture, "dv_cvar95": cvar}

    results = {
        "baseline": {"capture_rate": 0.90, "dv_cvar95": 90.0},
        "variants": [
            _v(8, "per_channel", "all", 0.90, 91.0),
            _v(4, "per_channel", "all", 0.85, 120.0),
            _v(8, "per_tensor", "all", 0.88, 105.0),
            _v(4, "per_tensor", "all", 0.60, 200.0),
        ],
    }
    out = tmp_path / "sweep.svg"
    chart_quant_sweep(results, str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
def test_quant_sweep_smoke(tmp_path: Path) -> None:
    """Reduced end-to-end sweep on a synthetic champion-arch model: real sims, tiny grid."""
    pytest.importorskip("aerocapture_rs")
    import aerocapture_rs
    from aerocapture.training.quantize import main as quant_main

    arch = [
        {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
        {"type": "mamba", "input_size": 16, "d_state": 12, "dt_rank": 1},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
    ]
    rng = np.random.default_rng(0)
    model_path = tmp_path / "synthetic_962.json"
    aerocapture_rs.flat_weights_to_json(
        flat=rng.uniform(-0.5, 0.5, 962).tolist(),
        architecture_json=json.dumps(arch),
        path=str(model_path),
        input_mask=[0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30, 32, 33, 34],
        output_param="atan2_signed",
    )
    out_dir = tmp_path / "sweep_out"
    quant_main(
        [
            str(out_dir),
            "--toml",
            "configs/training/sweep/mamba_p962.toml",
            "--model",
            str(model_path),
            "--n-sims",
            "3",
            "--bits",
            "8",
            "4",
            "--granularity",
            "per_tensor",
            "--policies",
            "all",
            "--loo-bits",
            "4",
            "--sim-timeout",
            "60",
        ]
    )
    results = json.loads((out_dir / "quantization_results.json").read_text())
    assert len(results["variants"]) == 2
    assert len(results["loo"]) == 6  # layer_0.w, x_proj_w, dt_proj_w, a_log, d_skip, layer_2.w
    assert results["verdict"]["bits"] == 4
    assert (out_dir / "quantization_sweep.svg").exists()
    assert (out_dir / "quantization_loo.svg").exists()


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


_CHAMPION_ARCH = [
    {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"},
    {"type": "mamba", "input_size": 16, "d_state": 12, "dt_rank": 1},
    {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"},
]


def test_memory_footprint_champion_int8_per_tensor_all() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_tensor", "all")
    assert m["quant_params"] == 928 and m["fp_params"] == 34
    assert m["n_scales"] == 6  # dense0.w, x_proj_w, dt_proj_w, a_log, d_skip, dense2.w
    assert m["quant_bytes"] == 928 and m["scale_bytes"] == 24 and m["fp_bytes"] == 136
    assert m["total_bytes"] == 1088
    assert m["f64_baseline_bytes"] == 962 * 8  # 7696


def test_memory_footprint_champion_int4_per_tensor_all() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 4, "per_tensor", "all")
    assert m["quant_bytes"] == 464  # ceil(928 * 4 / 8)
    assert m["total_bytes"] == 624


def test_memory_footprint_champion_per_channel_scale_count() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_channel", "all")
    # dense0 16 rows + x_proj 25 + dt_proj_w 16 + a_log 16 + d_skip 1 (per-tensor rule) + dense2 2
    assert m["n_scales"] == 76


def test_memory_footprint_proj_only_moves_dynamics_to_fp() -> None:
    from aerocapture.training.quantize import memory_footprint

    m = memory_footprint(_CHAMPION_ARCH, 8, "per_tensor", "proj_only")
    assert m["quant_params"] == 720 and m["fp_params"] == 242


def _variant(bits: int, gran: str, policy: str, capture: float, cvar: float) -> dict:
    return {"bits": bits, "granularity": gran, "tensor_policy": policy, "capture_rate": capture, "dv_cvar95": cvar}


def test_pick_verdict_prefers_capture_then_cvar() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_tensor", "all", 0.99, 110.0),
        _variant(4, "per_channel", "proj_only", 1.0, 118.0),
        _variant(4, "per_channel", "all", 1.0, 116.0),
        _variant(8, "per_channel", "all", 1.0, 100.0),  # wrong bits: ignored
    ]
    v = _pick_verdict(variants, 4)
    assert (v["granularity"], v["tensor_policy"]) == ("per_channel", "all")


def test_pick_verdict_tie_breaks_toward_per_channel_all() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_tensor", "proj_only", 1.0, 115.0),
        _variant(4, "per_channel", "all", 1.0, 115.0),
    ]
    v = _pick_verdict(variants, 4)
    assert (v["granularity"], v["tensor_policy"]) == ("per_channel", "all")


def test_pick_verdict_nan_cvar_ranks_last() -> None:
    from aerocapture.training.quantize import _pick_verdict

    variants = [
        _variant(4, "per_channel", "all", 1.0, float("nan")),
        _variant(4, "per_tensor", "all", 1.0, 120.0),
    ]
    assert _pick_verdict(variants, 4)["granularity"] == "per_tensor"


def test_max_viol_pct_skips_unconfigured_limits() -> None:
    from aerocapture.training.quantize import _max_viol_pct

    constraints = {
        "heat_flux": {"viol_pct": 1.5},
        "g_load": {"viol_pct": None},  # limit absent from TOML (e.g. earth.toml has no max_heat_load)
        "heat_load": {"viol_pct": 0.2},
    }
    assert _max_viol_pct(constraints) == 1.5


def test_max_viol_pct_all_none_is_zero() -> None:
    from aerocapture.training.quantize import _max_viol_pct

    assert _max_viol_pct({"heat_flux": {"viol_pct": None}}) == 0.0


def test_run_quant_sweep_rejects_loo_bits_outside_grid() -> None:
    from aerocapture.training.quantize import run_quant_sweep

    with pytest.raises(ValueError, match="loo_bits"):
        run_quant_sweep("unused.toml", "unused.json", bits=(8, 6), loo_bits=4)


def test_scaffolding_overrides_require_raises_on_empty(tmp_path: Path) -> None:
    from aerocapture.training.quantize import _scaffolding_overrides

    with pytest.raises(ValueError, match="scaffolding"):
        _scaffolding_overrides(tmp_path, require=True)  # empty dir: no best_params.json


def test_scaffolding_overrides_none_params_dir_is_empty() -> None:
    from aerocapture.training.quantize import _scaffolding_overrides

    assert _scaffolding_overrides(None, require=True) == {}
