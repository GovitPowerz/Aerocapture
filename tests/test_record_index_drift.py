"""Drift tests: Python list lengths must match Rust const widths exposed via PyO3.

If these fail, either a Python list grew without a matching Rust const update,
or a Rust const changed without updating the Python list.
"""

from __future__ import annotations

import pytest

aero = pytest.importorskip("aerocapture_rs")


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
