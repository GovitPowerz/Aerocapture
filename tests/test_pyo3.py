"""Integration tests for the PyO3 aerocapture_rs module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

aero = pytest.importorskip("aerocapture_rs")

GOLDEN_TOML = "configs/test/test_ref_orig.toml"


class TestSingleRun:
    def test_run_returns_result(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert hasattr(result, "trajectory")
        assert hasattr(result, "final_record")
        assert hasattr(result, "captured")

    def test_final_record_shape(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.final_record.shape == (52,)
        assert result.final_record.dtype == np.float64

    def test_trajectory_is_numpy_array(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.trajectory.ndim == 2
        # Trajectory recording is not yet populated in Rust, so expect (0, 0).
        # When populated, columns should be 8.

    def test_convenience_accessors_match_final_record(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.energy == result.final_record[7]
        assert result.ecc == result.final_record[9]
        assert result.periapsis_alt == result.final_record[14]
        assert result.apoapsis_alt == result.final_record[15]
        assert result.delta_v == result.final_record[41]
        assert result.peri_err == result.final_record[29]
        assert result.apo_err == result.final_record[30]

    def test_captured_flag_consistent_with_orbital_elements(self) -> None:
        result = aero.run(GOLDEN_TOML)
        expected = result.ecc < 1.0 and result.energy < 0.0
        assert result.captured == expected


class TestOverrides:
    def test_override_changes_result(self) -> None:
        r1 = aero.run(GOLDEN_TOML)
        r2 = aero.run(GOLDEN_TOML, overrides={"guidance.reference_bank_angle": 30.0})
        assert not np.array_equal(r1.final_record, r2.final_record)

    def test_invalid_override_type_raises(self) -> None:
        with pytest.raises(TypeError):
            aero.run(GOLDEN_TOML, overrides={"guidance.reference_bank_angle": [1, 2, 3]})


class TestBatchRun:
    def test_batch_returns_correct_count(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(5)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        assert results.final_records.shape == (5, 52)
        assert results.captured.shape == (5,)

    def test_batch_trajectories_off_by_default(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(3)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        # Trajectories list is always present but each entry is empty when off.
        assert len(results.trajectories) == 3
        for traj in results.trajectories:
            assert traj.shape == (0, 0)

    def test_batch_trajectories_on(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(3)]
        results = aero.run_batch(GOLDEN_TOML, overrides, include_trajectories=True)
        assert len(results.trajectories) == 3
        for traj in results.trajectories:
            assert traj.ndim == 2
            # Trajectory recording not yet populated; just verify numpy array.

    def test_batch_len(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(4)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        assert len(results) == 4


class TestCostCompat:
    def test_pyo3_final_records_work_with_compute_cost(self) -> None:
        from aerocapture.training.evaluate import compute_cost

        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(5)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        cost = compute_cost(results.final_records)
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestBitIdenticalRegression:
    def test_pyo3_matches_subprocess(self, rust_binary: Path) -> None:
        from aerocapture.training.config import SimConfig, TrainingConfig
        from aerocapture.training.evaluate import _run_via_subprocess

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
        )
        sub_result = _run_via_subprocess(config)
        assert sub_result is not None, "Subprocess path failed"

        pyo3_result = aero.run(GOLDEN_TOML)
        pyo3_array = pyo3_result.final_record.reshape(1, 52)

        # Subprocess path round-trips through CSV text, losing ~10 significant
        # digits.  PyO3 returns full f64 precision.  Use allclose with tight
        # tolerances that still accommodate the CSV formatting loss.
        np.testing.assert_allclose(
            sub_result,
            pyo3_array,
            rtol=1e-9,
            atol=1e-9,
            err_msg="PyO3 and subprocess paths diverge beyond CSV round-trip tolerance",
        )


class TestLoadConfig:
    def test_load_config_returns_dict(self) -> None:
        config = aero.load_config(GOLDEN_TOML)
        assert isinstance(config, dict)
        assert "mission" in config
        assert "guidance" in config

    def test_load_config_nonexistent_raises(self) -> None:
        with pytest.raises(OSError):
            aero.load_config("nonexistent.toml")


class TestFallback:
    def test_subprocess_fallback_works(self, rust_binary: Path) -> None:
        from aerocapture.training.config import SimConfig, TrainingConfig
        from aerocapture.training.evaluate import _run_via_subprocess

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
        )
        result = _run_via_subprocess(config)
        assert result is not None, "Subprocess path failed"
        assert result.shape[1] == 52
