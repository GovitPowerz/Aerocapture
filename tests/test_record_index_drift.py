"""Drift tests: Python list lengths must match Rust const widths exposed via PyO3.

If these fail, either a Python list grew without a matching Rust const update,
or a Rust const changed without updating the Python list.
"""

from __future__ import annotations

import pytest

aero = pytest.importorskip("aerocapture_rs")


class TestFinalRecordIndexDrift:
    def test_final_record_len_matches_rust(self) -> None:
        from aerocapture.training.parquet_output import FINAL_RECORD_LEN

        assert FINAL_RECORD_LEN == aero.FINAL_RECORD_LEN == 52

    def test_raw_indices_match_rust_map(self) -> None:
        from aerocapture.training import parquet_output as pq

        idx = aero.final_record_indices()
        assert idx["dv_total_ms"] == pq.DV_TOTAL_RAW_INDEX
        assert idx["heat_flux_kw_m2"] == pq.HEAT_FLUX_RAW_INDEX
        assert idx["g_load"] == pq.G_LOAD_RAW_INDEX
        assert idx["heat_load_mjm2"] == pq.HEAT_LOAD_RAW_INDEX

    def test_index_map_is_within_record(self) -> None:
        idx = aero.final_record_indices()
        assert idx, "final_record_indices() returned empty"
        assert all(0 <= v < aero.FINAL_RECORD_LEN for v in idx.values())


class TestWidthDrift:
    def test_nn_input_names_len_matches_rust(self) -> None:
        from aerocapture.training.ablation import NN_INPUT_NAMES

        assert len(NN_INPUT_NAMES) == aero.NN_FULL_INPUT_SIZE, (
            f"NN_INPUT_NAMES has {len(NN_INPUT_NAMES)} entries but aerocapture_rs.NN_FULL_INPUT_SIZE == {aero.NN_FULL_INPUT_SIZE}"
        )

    def test_dispersion_columns_len_matches_rust(self) -> None:
        from aerocapture.training.sensitivity import DISPERSION_COLUMNS

        assert len(DISPERSION_COLUMNS) == aero.DISPERSION_DRAW_LEN, (
            f"DISPERSION_COLUMNS has {len(DISPERSION_COLUMNS)} entries but aerocapture_rs.DISPERSION_DRAW_LEN == {aero.DISPERSION_DRAW_LEN}"
        )

    def test_module_exposes_nn_full_input_size(self) -> None:
        assert isinstance(aero.NN_FULL_INPUT_SIZE, int)
        assert aero.NN_FULL_INPUT_SIZE == 35

    def test_module_exposes_dispersion_draw_len(self) -> None:
        assert isinstance(aero.DISPERSION_DRAW_LEN, int)
        assert aero.DISPERSION_DRAW_LEN == 26
