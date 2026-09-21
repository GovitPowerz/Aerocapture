"""Integration tests for the PyO3 `aerocapture_rs` seam, by tier (the module doc in
`src/rust/aerocapture-py/src/lib.rs`, ADR-0007): evaluate (`run_batch` / `run_mc` /
`run_with_draws`), config (`validate_config` / `load_config`), contract
(`candidate_inputs` + the width constants) and nn (`flat_weights_to_json`).

One run is `run_batch(toml, [{}])` row 0: the deploy path every reported number takes,
and what the CLI bit-identity gate compares against."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from aerocapture.training.config import candidate_input_normalization

if TYPE_CHECKING:
    from aerocapture.training.config import NetworkConfig

aero = pytest.importorskip("aerocapture_rs")

GOLDEN_TOML = "configs/test/test_ref_orig.toml"
REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted(REPO.joinpath("configs").rglob("*.toml"))
assert CONFIGS, "no committed configs found; the base-resolution parity oracle would be dormant"


def _single(overrides: dict | None = None, **kwargs: Any) -> Any:
    """The deploy path for one run: `run_batch` with one override dict."""
    return aero.run_batch(GOLDEN_TOML, [overrides or {}], **kwargs)


class TestSingleRun:
    def test_final_record_shape(self) -> None:
        records = _single().final_records
        assert records.shape == (1, 52)
        assert records.dtype == np.float64

    def test_trajectory_columns(self) -> None:
        traj = _single(include_trajectories=True).trajectories[0]
        assert traj.ndim == 2
        assert traj.shape[1] == 17
        assert traj.shape[0] > 0

    def test_captured_flag_consistent_with_orbital_elements(self) -> None:
        batch = _single()
        idx = aero.final_record_indices()
        fr = batch.final_records[0]
        expected = fr[idx["ecc"]] < 1.0 and fr[idx["energy_mjkg"]] < 0.0
        assert bool(batch.captured[0]) == expected

    def test_dispersions_row(self) -> None:
        assert _single().dispersions.shape == (1, 26)


class TestOverrides:
    def test_override_changes_result(self) -> None:
        r1 = _single().final_records[0]
        r2 = _single({"guidance.reference_bank_angle": 30.0}).final_records[0]
        assert not np.array_equal(r1, r2)

    def test_invalid_override_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _single({"guidance.reference_bank_angle": [1, 2, 3]})


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
            assert traj.shape[1] == 17
            assert traj.shape[0] > 0

    def test_batch_len(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(4)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        assert len(results) == 4

    def test_run_batch_rejects_multi_sim(self) -> None:
        # Contract violation: n_sims > 1 must raise ValueError (not RuntimeError).
        with pytest.raises(ValueError, match="n_sims"):
            aero.run_batch(GOLDEN_TOML, [{"simulation.n_sims": 3}])

    def test_run_batch_bad_config_raises_runtimeerror(self) -> None:
        # Runtime failure: nonexistent TOML path must raise RuntimeError (not ValueError).
        with pytest.raises(RuntimeError):
            aero.run_batch("configs/test/does_not_exist_xyz.toml", [{}])


class TestValidateConfig:
    def test_valid_config_passes(self) -> None:
        assert aero.validate_config(GOLDEN_TOML) is None

    def test_override_violation_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="navigation.mode"):
            aero.validate_config(GOLDEN_TOML, overrides={"navigation.mode": "EKF"})

    def test_unknown_key_in_rust_section_raises_value_error(self) -> None:
        # Every Rust-owned section struct is deny_unknown_fields: a typo'd dot
        # path (`mde` for `mode`) fails at parse with serde's message instead
        # of silently creating a key nothing reads.
        with pytest.raises(ValueError, match=r"unknown field `mde`"):
            aero.validate_config(GOLDEN_TOML, overrides={"navigation.mde": "ekf"})

    def test_does_not_read_data_tables(self) -> None:
        # Retargeting the atmosphere table at a nonexistent file must still pass:
        # the pass reads no table. The same override makes a run fail.
        overrides = {"data.atmosphere": "/nonexistent/atm.dat"}
        aero.validate_config(GOLDEN_TOML, overrides=overrides)
        with pytest.raises(RuntimeError):
            _single(overrides)

    def test_missing_toml_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            aero.validate_config("configs/test/does_not_exist_xyz.toml")


class TestCostCompat:
    def test_pyo3_final_records_work_with_compute_cost(self) -> None:
        from aerocapture.training.cost import compute_cost

        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(5)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        cost = compute_cost(results.final_records)
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestBitIdenticalRegression:
    def test_pyo3_matches_subprocess(self, rust_binary: Path) -> None:
        """The CLI (CSV round trip) against the deploy path (`run_batch`, preloaded
        shared tables): the path every deploy-side number takes."""
        from aerocapture.training.config import SimConfig, TrainingConfig
        from aerocapture.training.optimizer import OptimizerConfig

        from tests.fixtures.subprocess_oracle import run_via_subprocess

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
            optimizer=OptimizerConfig(seed_strategy="adaptive"),
        )
        sub_result = run_via_subprocess(config)
        assert sub_result is not None, "Subprocess path failed"

        pyo3_array = aero.run_batch(GOLDEN_TOML, [{}]).final_records
        assert pyo3_array.shape == (1, 52)

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
        batch = _single({"integration.mode": "adaptive", "integration.rtol": 1e-6})
        assert batch.captured[0], "Adaptive mode should produce a captured trajectory"
        assert batch.final_records.shape == (1, 52)

    def test_adaptive_agrees_with_fixed(self) -> None:
        """Adaptive and fixed modes should produce similar results on the same config."""
        idx = aero.final_record_indices()
        fixed = _single()
        adaptive = _single({"integration.mode": "adaptive"})
        assert fixed.captured[0]
        assert adaptive.captured[0]
        e_fixed = fixed.final_records[0][idx["energy_mjkg"]]
        e_adaptive = adaptive.final_records[0][idx["energy_mjkg"]]
        # Energy agreement within 1%
        energy_err = abs(e_fixed - e_adaptive) / abs(e_fixed)
        assert energy_err < 0.01, f"Energy mismatch: {energy_err:.4f}"


class TestCandidateInputs:
    def test_one_schema_35_entries(self) -> None:
        schema = aero.candidate_inputs()
        assert len(schema) == aero.NN_FULL_INPUT_SIZE == 35
        assert [e["index"] for e in schema] == list(range(35))
        for entry in schema:
            assert set(entry) == {"index", "name", "transform", "scale", "center"}
            assert entry["transform"] in ("none", "asinh", "tanh")
        assert schema[0]["transform"] == "none"
        assert schema[11]["transform"] == "asinh"
        assert schema[32]["transform"] == "asinh"


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
        assert d["normalization"] == candidate_input_normalization()

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
        assert d["normalization"] == candidate_input_normalization()


class TestLoadConfig:
    def test_load_config_returns_dict(self) -> None:
        config = aero.load_config(GOLDEN_TOML)
        assert isinstance(config, dict)
        assert "mission" in config
        assert "guidance" in config

    def test_load_config_nonexistent_raises(self) -> None:
        with pytest.raises(OSError):
            aero.load_config("nonexistent.toml")

    @pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
    def test_base_resolution_parity(self, path: Path) -> None:
        """Rust `resolve_toml_bases` and Python `load_toml_with_bases` implement the same
        `base` deep-merge; `load_config` is the oracle that keeps that claim tested over
        every committed config (Rust renders datetimes as strings, so equality also proves
        the configs carry none)."""
        from aerocapture.training.toml_utils import load_toml_with_bases

        assert aero.load_config(str(path)) == load_toml_with_bases(path)


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


class TestRunWithDrawsStrided:
    def test_non_contiguous_draws_match_contiguous(self) -> None:
        rng = np.random.default_rng(7)
        draws = rng.normal(size=(4, 26)) * 0.1

        r_contig = aero.run_with_draws(GOLDEN_TOML, np.ascontiguousarray(draws))
        # A non-C-contiguous f64 view with the SAME values.
        non_contig = np.asarray(draws.T.T, order="F")
        assert not non_contig.flags["C_CONTIGUOUS"]
        r_strided = aero.run_with_draws(GOLDEN_TOML, non_contig)

        np.testing.assert_array_equal(np.asarray(r_contig.dispersions), np.asarray(r_strided.dispersions))
        np.testing.assert_array_equal(np.asarray(r_contig.final_records), np.asarray(r_strided.final_records))
