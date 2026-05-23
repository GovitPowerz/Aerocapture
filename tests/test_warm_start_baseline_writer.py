"""Gen-0 validation MC writer for warm-started chromosomes."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from aerocapture.training._warm_start_baseline import write_gen0_baseline


def test_writes_baseline_file_with_mean_rms_n_sims(tmp_path: Path) -> None:
    save_dir = tmp_path
    n_sims = 8
    fake_dv = np.array([100.0, 200.0, 150.0, 50.0, 75.0, 125.0, 175.0, 225.0])

    class _Result:
        final_records = np.zeros((n_sims, 52))

    _Result.final_records[:, 41] = fake_dv  # dv_total_m_s in raw RunOutput layout

    def _fake_run_mc(toml_path: str, overrides: dict, include_trajectories: bool = False, sim_timeout_secs: float | None = None) -> _Result:
        return _Result()

    with patch("aerocapture.training._warm_start_baseline._aero_rs.run_mc", side_effect=_fake_run_mc):
        path = write_gen0_baseline(
            save_dir=save_dir,
            toml_path="dummy.toml",
            overrides={},
            n_sims=n_sims,
        )

    assert path == save_dir / "warm_start_baseline.json"
    data = json.loads(path.read_text())
    assert data["n_sims"] == n_sims
    assert data["n_returned"] == n_sims
    assert data["mean"] == pytest.approx(float(np.mean(fake_dv)))
    assert data["rms"] == pytest.approx(float(np.sqrt(np.mean(fake_dv**2))))
