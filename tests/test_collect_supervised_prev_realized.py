import aerocapture_rs
import numpy as np


def test_collect_supervised_emits_prev_realized() -> None:
    toml = "configs/training/msr_aller_ftc_train.toml"
    out = aerocapture_rs.collect_supervised(toml, [12345])
    assert len(out) >= 1
    rec = out[0]
    assert "prev_realized" in rec
    assert rec["prev_realized"].shape == rec["y_signed"].shape  # (T,)
    assert rec["X"].shape[1] == 31  # full 31-wide candidate vector
    assert np.all(np.isfinite(rec["prev_realized"]))
