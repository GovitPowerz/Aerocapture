"""Integration tests for the PyO3 aerocapture_rs module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from aerocapture.training.config import NetworkConfig

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
        # Empty entries carry the correct 17-column width (0, 17), not (0, 0),
        # so downstream column indexing on an empty batch stays valid.
        assert len(results.trajectories) == 3
        for traj in results.trajectories:
            assert traj.shape == (0, 17)

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

    def test_run_batch_rejects_multi_sim(self) -> None:
        with pytest.raises(ValueError, match="n_sims"):
            aero.run_batch(GOLDEN_TOML, [{"simulation.n_sims": 3}])


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
        from aerocapture.training.optimizer import OptimizerConfig

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
            optimizer=OptimizerConfig(seed_strategy="adaptive"),
        )
        sub_result = _run_via_subprocess(config)
        assert sub_result is not None, "Subprocess path failed"

        pyo3_result = aero.run(GOLDEN_TOML)
        pyo3_array = pyo3_result.final_record.reshape(1, 52)

        # Subprocess path round-trips through CSV text, losing ~10 significant
        # digits.  PyO3 returns full f64 precision.  Use allclose with tight
        # tolerances that still accommodate the CSV formatting loss.
        # Column 46 (inclination error) is populated in-memory but not written
        # to the CSV output, so the subprocess path has 0 there — skip it.
        cols = list(range(52))
        cols.remove(46)
        np.testing.assert_allclose(
            sub_result[:, cols],
            pyo3_array[:, cols],
            rtol=1e-9,
            atol=1e-9,
            err_msg="PyO3 and subprocess paths diverge beyond CSV round-trip tolerance",
        )


class TestAdaptiveIntegration:
    """Test adaptive DOPRI45 integration via PyO3 overrides."""

    def test_adaptive_override_produces_valid_result(self) -> None:
        """Setting integration.mode = 'adaptive' via overrides should work."""
        result = aero.run(
            GOLDEN_TOML,
            overrides={"integration.mode": "adaptive", "integration.rtol": 1e-6},
        )
        assert result.captured, "Adaptive mode should produce a captured trajectory"
        assert result.final_record.shape == (52,)

    def test_adaptive_agrees_with_fixed(self) -> None:
        """Adaptive and fixed modes should produce similar results on the same config."""
        r_fixed = aero.run(GOLDEN_TOML)
        r_adaptive = aero.run(
            GOLDEN_TOML,
            overrides={"integration.mode": "adaptive"},
        )
        assert r_fixed.captured
        assert r_adaptive.captured
        # Energy agreement within 1%
        energy_err = abs(r_fixed.energy - r_adaptive.energy) / abs(r_fixed.energy)
        assert energy_err < 0.01, f"Energy mismatch: {energy_err:.4f}"


class TestDefaultNormalization:
    def test_returns_35_entries(self) -> None:
        norm = aero.default_normalization()
        assert len(norm) == 35
        assert norm[0]["transform"] == "none"
        assert norm[11]["transform"] == "asinh"
        assert norm[32]["transform"] == "asinh"
        for entry in norm:
            assert set(entry) == {"transform", "scale", "center"}
            assert entry["transform"] in ("none", "asinh", "tanh")


class TestFlatWeightsNormalization:
    def _write(self, tmp_path: Path, normalization_json: str | None) -> dict:
        import json

        arch = [{"type": "dense", "input_size": 17, "output_size": 2, "activation": "linear"}]
        flat = np.zeros(17 * 2 + 2, dtype=np.float64).tolist()
        out = tmp_path / "model.json"
        aero.flat_weights_to_json(
            flat,
            json.dumps(arch),
            str(out),
            list(range(17)),
            "atan2_signed",
            None,
            None,
            normalization_json,
        )
        with open(out) as fp:
            result: dict = json.load(fp)
        return result

    def test_custom_normalization_is_embedded(self, tmp_path: Path) -> None:
        import json

        custom = [{"transform": "none", "scale": 2.0, "center": 1.0}] * 35
        d = self._write(tmp_path, json.dumps(custom))
        assert d["normalization"] == custom

    def test_none_normalization_uses_default(self, tmp_path: Path) -> None:
        d = self._write(tmp_path, None)
        assert d["normalization"] == aero.default_normalization()

    def test_wrong_length_raises(self, tmp_path: Path) -> None:
        import json

        bad = [{"transform": "none", "scale": 1.0, "center": 0.0}] * 10
        with pytest.raises(ValueError):
            self._write(tmp_path, json.dumps(bad))


class TestWriteNnJsonNormalization:
    def _network(self) -> NetworkConfig:
        from aerocapture.training.config import NetworkConfig

        return NetworkConfig(layer_sizes=[17, 2], activations=["linear"])

    def test_custom_normalization_threaded(self, tmp_path: Path) -> None:
        import json

        from aerocapture.training.evaluate import write_nn_json

        net = self._network()
        custom = [{"transform": "none", "scale": 3.0, "center": 0.5}] * 35
        out = tmp_path / "model.json"
        write_nn_json(
            np.zeros(17 * 2 + 2, dtype=np.float64),
            net,
            out,
            input_mask=list(range(17)),
            normalization=custom,
        )
        with open(out) as fp:
            d = json.load(fp)
        assert d["normalization"] == custom

    def test_none_normalization_uses_default(self, tmp_path: Path) -> None:
        import json

        from aerocapture.training.evaluate import write_nn_json

        net = self._network()
        out = tmp_path / "model.json"
        write_nn_json(
            np.zeros(17 * 2 + 2, dtype=np.float64),
            net,
            out,
            input_mask=list(range(17)),
        )
        with open(out) as fp:
            d = json.load(fp)
        assert d["normalization"] == aero.default_normalization()


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
        from aerocapture.training.optimizer import OptimizerConfig

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
            optimizer=OptimizerConfig(seed_strategy="adaptive"),
        )
        result = _run_via_subprocess(config)
        assert result is not None, "Subprocess path failed"
        assert result.shape[1] == 52


class TestRunWithDraws:
    def test_run_with_draws_returns_batch_results(self) -> None:
        draws = np.zeros((5, 26), dtype=np.float64)
        draws[:, 24] = 1.0  # wind_scale = 1.0
        result = aero.run_with_draws(GOLDEN_TOML, draws)
        assert len(result) == 5
        assert result.final_records.shape == (5, 52)

    def test_run_with_draws_wrong_columns(self) -> None:
        draws = np.zeros((5, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="26 columns"):
            aero.run_with_draws(GOLDEN_TOML, draws)

    def test_run_with_draws_dispersions_roundtrip(self) -> None:
        draws = np.zeros((3, 26), dtype=np.float64)
        draws[:, 24] = 1.0
        draws[0, 3] = 5.0  # velocity offset
        draws[1, 6] = 0.1  # density bias
        result = aero.run_with_draws(GOLDEN_TOML, draws)
        np.testing.assert_allclose(result.dispersions, draws, atol=1e-12)
