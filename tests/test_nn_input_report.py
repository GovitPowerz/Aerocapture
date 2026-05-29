from pathlib import Path

import numpy as np

from aerocapture.training.charts_nn_inputs import binned_band, chart_nn_input_panel
from aerocapture.training.nn_input_report import classify_by_dv, input_summary


def test_classify_by_dv_threshold() -> None:
    dv = np.array([100.0, 1000.0, 1500.0])
    klass = classify_by_dv(dv, threshold=1000.0)  # 0=blue(low), 1=red(high)
    assert list(klass) == [0, 1, 1]


def test_input_summary_saturation_and_separation() -> None:
    X = [
        np.array([[0.0, -2.0], [0.5, -2.0], [2.0, -2.0]]),  # blue traj
        np.array([[0.0, 2.0], [0.5, 2.0], [2.0, 2.0]]),     # red traj
    ]
    klass = np.array([0, 1])
    rows = input_summary(X, klass, names=["a", "b"], in_mask={0, 1})
    by = {r["name"]: r for r in rows}
    assert abs(by["a"]["frac_out_of_range"] - 2 / 6) < 1e-9  # |2.0|>1 on 2/6 samples
    assert by["b"]["separation"] > by["a"]["separation"]     # b separates classes, a doesn't
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
    X_list = [rng.uniform(-1.5, 1.5, size=(20, 31)) for _ in range(6)]
    time_list = [np.arange(20.0) for _ in range(6)]
    klass = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    out = tmp_path / "panel.svg"
    chart_nn_input_panel(X_list, time_list, klass, input_index=5,
                         name="accel_magnitude", in_mask=True, output=out)
    assert out.exists() and out.stat().st_size > 0
