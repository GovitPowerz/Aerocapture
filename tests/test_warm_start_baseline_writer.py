"""Gen-0 validation baseline writer for warm-started chromosomes."""

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training._warm_start_baseline import write_gen0_baseline


def test_writes_baseline_file_with_cost_stats(tmp_path: Path) -> None:
    save_dir = tmp_path
    n_sims = 8
    # Mix of capture-class costs (~100) and one large virtual-DV value (~3000)
    # to exercise the p95 / worst summary fields.
    fake_costs = np.array([100.0, 200.0, 150.0, 50.0, 75.0, 125.0, 175.0, 3000.0])
    fake_capture_rate = 7.0 / 8.0  # 7 captures, 1 crash

    path = write_gen0_baseline(
        save_dir=save_dir,
        costs=fake_costs,
        capture_rate=fake_capture_rate,
        n_sims=n_sims,
    )

    assert path == save_dir / "warm_start_baseline.json"
    data = json.loads(path.read_text())
    assert data["n_sims"] == n_sims
    assert data["n_returned"] == n_sims
    assert data["capture_rate"] == pytest.approx(fake_capture_rate)
    assert data["rms_cost"] == pytest.approx(float(np.sqrt(np.mean(fake_costs**2))))
    assert data["mean_cost"] == pytest.approx(float(np.mean(fake_costs)))
    assert data["median_cost"] == pytest.approx(float(np.median(fake_costs)))
    assert data["p95_cost"] == pytest.approx(float(np.percentile(fake_costs, 95)))
    assert data["worst_cost"] == pytest.approx(float(np.max(fake_costs)))


def test_rejects_shape_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match n_sims"):
        write_gen0_baseline(
            save_dir=tmp_path,
            costs=np.array([1.0, 2.0, 3.0]),
            capture_rate=1.0,
            n_sims=10,  # mismatch
        )
