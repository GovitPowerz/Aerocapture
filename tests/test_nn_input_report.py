import numpy as np
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
