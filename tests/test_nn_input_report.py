import json as _json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from aerocapture.training.charts_nn_inputs import binned_band, chart_nn_input_panel
from aerocapture.training.nn_input_report import _resolve_mask, classify_by_dv, input_summary


def test_resolve_mask_prefers_model_json(tmp_path: Path) -> None:
    model = tmp_path / "m.json"
    model.write_text(_json.dumps({"input_mask": [1, 2, 3]}))
    # model_path's embedded mask wins over the TOML config's mask.
    assert _resolve_mask("configs/training/msr_aller_nn_delta_train.toml", str(model)) == {1, 2, 3}


def test_resolve_mask_falls_back_to_toml(tmp_path: Path) -> None:
    # A non-existent model path falls back to the TOML's input_mask.
    # Delta config now uses the shared 17-input mask (incl. predicted_dv 32-34, excl. periapsis 31).
    missing = str(tmp_path / "nope.json")
    out = _resolve_mask("configs/training/msr_aller_nn_delta_train.toml", missing)
    assert len(out) == 17
    assert 32 in out and 33 in out and 34 in out  # new DV inputs
    assert 31 not in out  # periapsis dropped in the pruned mask


def test_classify_by_dv_threshold() -> None:
    dv = np.array([100.0, 1000.0, 1500.0])
    klass = classify_by_dv(dv, threshold=1000.0)  # 0=blue(low), 1=red(high)
    assert list(klass) == [0, 1, 1]


def test_input_summary_saturation_and_separation() -> None:
    X = [
        np.array([[0.0, -2.0], [0.5, -2.0], [2.0, -2.0]]),  # blue traj
        np.array([[0.0, 2.0], [0.5, 2.0], [2.0, 2.0]]),  # red traj
    ]
    klass = np.array([0, 1])
    rows = input_summary(X, klass, names=["a", "b"], in_mask={0, 1})
    by = {r["name"]: r for r in rows}
    assert abs(cast(float, by["a"]["frac_out_of_range"]) - 2 / 6) < 1e-9  # |2.0|>1 on 2/6 samples
    assert cast(float, by["b"]["separation"]) > cast(float, by["a"]["separation"])  # b separates classes
    assert by["a"]["in_mask"] is True


def test_binned_band_shapes_and_values() -> None:
    x = np.linspace(0, 10, 100)
    y = x.copy()
    centers, lo, hi = binned_band(x, y, n_bins=5, lo_pct=5, hi_pct=95)
    assert centers.shape == lo.shape == hi.shape == (5,)
    assert np.all(hi >= lo)
    assert centers[0] < centers[-1]


def test_chart_nn_input_panel_writes_svg(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X_list = [rng.uniform(-1.5, 1.5, size=(20, 32)) for _ in range(6)]
    time_list = [np.arange(20.0) for _ in range(6)]
    klass = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    out = tmp_path / "panel.svg"
    chart_nn_input_panel(X_list, time_list, klass, input_index=5, name="accel_magnitude", in_mask=True, output=out)
    assert out.exists() and out.stat().st_size > 0


def _mint_zero_model_for_report(tmp_path: Path) -> str:
    """Mint a loadable zero-weight NN matching the delta config's arch."""
    import aerocapture_rs
    from aerocapture.training.toml_utils import load_toml_with_bases

    cfg = load_toml_with_bases(Path("configs/training/msr_aller_nn_delta_train.toml"))
    arch = cfg["network"]["architecture"]
    mask = cfg["network"]["input_mask"]
    flat = [0.0] * sum(ly["input_size"] * ly["output_size"] + ly["output_size"] for ly in arch)
    path = str(tmp_path / "zero_model.json")
    aerocapture_rs.flat_weights_to_json(
        flat,
        _json.dumps(arch),
        path,
        mask,
        cfg["guidance"]["neural_network"]["output_parameterization"],
        None,
        cfg["guidance"]["neural_network"]["delta_max"],
    )
    return path


@pytest.mark.slow
def test_run_report_smoke(tmp_path: Path) -> None:
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.nn_input_report import run_report

    model = _mint_zero_model_for_report(tmp_path)
    out_dir = tmp_path / "rep"
    run_report(
        toml_path="configs/training/msr_aller_nn_delta_train.toml",
        n_sims=4,
        output_dir=out_dir,
        overrides={"data.neural_network": model},
    )
    assert (out_dir / "summary.json").exists()
    summary = _json.loads((out_dir / "summary.json").read_text())
    assert len(summary["inputs"]) == 35
    assert list(out_dir.glob("nn_input_*_time.svg"))
    assert list(out_dir.glob("nn_input_*_energy.svg"))


def test_compile_pdf_from_minimal_report(tmp_path: Path) -> None:
    import shutil

    if shutil.which("typst") is None:
        pytest.skip("typst not installed")
    from aerocapture.training.nn_input_report import _compile_pdf

    rd = tmp_path / "rep"
    rd.mkdir()
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"></svg>'
    (rd / "nn_input_00_a_time.svg").write_text(svg)
    (rd / "nn_input_00_a_energy.svg").write_text(svg)
    (rd / "summary.json").write_text(
        _json.dumps(
            {
                "scheme": "test_scheme",
                "dv_threshold": 200.0,
                "n_sims": 2,
                "n_blue": 1,
                "n_red": 1,
                "inputs": [
                    {
                        "index": 0,
                        "name": "a",
                        "p1": -1.0,
                        "p50": 0.0,
                        "p99": 1.0,
                        "frac_out_of_range": 0.1,
                        "separation": 0.5,
                        "in_mask": True,
                        "time_svg": "nn_input_00_a_time.svg",
                        "energy_svg": "nn_input_00_a_energy.svg",
                    }
                ],
            }
        )
    )
    pdf = _compile_pdf(rd)
    assert pdf is not None and pdf.exists() and pdf.stat().st_size > 0
