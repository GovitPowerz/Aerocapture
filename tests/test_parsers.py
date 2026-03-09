"""Tests for output file parsers (CSV and legacy Fortran text format)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from aerocapture.io.parse_final import FINAL_CSV_COLUMNS, parse_final
from aerocapture.io.parse_photo import PHOTO_COLUMNS, PHOTO_CSV_COLUMNS, parse_photo


class TestParsePhoto:
    """Tests for parse_photo auto-detection and column naming."""

    def test_csv_auto_detection(self, tmp_path: Path) -> None:
        """CSV files are detected by comma in first line."""
        csv_file = tmp_path / "photo.csv"
        csv_file.write_text("time_s,altitude_km,velocity_m_s\n1.0,100.0,5000.0\n2.0,99.0,4900.0\n")
        df = parse_photo(csv_file)
        assert len(df) == 2
        # CSV columns should be renamed to legacy names
        assert "time" in df.columns or "time_s" in df.columns

    def test_csv_column_normalization(self, tmp_path: Path) -> None:
        """CSV column names are normalized to legacy names for backward compat."""
        header = ",".join(PHOTO_CSV_COLUMNS)
        values = ",".join(["1.0e0"] * len(PHOTO_CSV_COLUMNS))
        csv_file = tmp_path / "photo.csv"
        csv_file.write_text(f"{header}\n{values}\n")
        df = parse_photo(csv_file)
        # Should have legacy names after normalization
        assert "time" in df.columns
        assert "altitude" in df.columns
        assert "energy" in df.columns
        assert "velocity" in df.columns

    def test_fortran_text_detection(self, tmp_path: Path) -> None:
        """Fortran D-notation files are detected by absence of commas."""
        # 24 columns of D12.5 format
        values = " ".join([" 0.12345D+02"] * 24)
        text_file = tmp_path / "photo.test"
        text_file.write_text(f"{values}\n{values}\n")
        df = parse_photo(text_file)
        assert len(df) == 2
        assert len(df.columns) == 24
        assert df.columns[0] == PHOTO_COLUMNS[0]

    def test_csv_has_fewer_columns_than_legacy(self, tmp_path: Path) -> None:
        """CSV format has 21 columns (3 dropped from legacy 24)."""
        assert len(PHOTO_CSV_COLUMNS) == 21
        assert len(PHOTO_COLUMNS) == 24

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty DataFrame."""
        empty = tmp_path / "photo.empty"
        empty.write_text("")
        df = parse_photo(empty)
        assert df.empty


class TestParseFinal:
    """Tests for parse_final auto-detection and column naming."""

    def test_csv_auto_detection(self, tmp_path: Path) -> None:
        """CSV files are detected by comma in first line."""
        header = ",".join(FINAL_CSV_COLUMNS)
        values = "1," + ",".join(["1.0e0"] * (len(FINAL_CSV_COLUMNS) - 1))
        csv_file = tmp_path / "final.csv"
        csv_file.write_text(f"{header}\n{values}\n")
        df = parse_final(csv_file)
        assert len(df) == 1
        assert "sim_number" in df.columns
        assert "energy_mj_kg" in df.columns

    def test_fortran_text_detection(self, tmp_path: Path) -> None:
        """Fortran D-notation files detected by absence of commas."""
        # i5 + 52 D15.7 values
        values = "    1" + " ".join([" 0.1234567D+02"] * 52)
        text_file = tmp_path / "final.test"
        text_file.write_text(f"{values}\n")
        df = parse_final(text_file)
        assert len(df) == 1
        assert "sim_number" in df.columns

    def test_csv_has_40_columns(self) -> None:
        """CSV format has exactly 40 named columns."""
        assert len(FINAL_CSV_COLUMNS) == 40

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty DataFrame."""
        empty = tmp_path / "final.empty"
        empty.write_text("")
        df = parse_final(empty)
        assert df.empty


class TestLegacyArrayMapping:
    """Tests for CSV→legacy index mapping in evaluate.py."""

    def test_csv_to_legacy_mapping(self, tmp_path: Path) -> None:
        """CSV final file maps back to correct legacy 53-column positions."""
        from aerocapture.training.evaluate import _parse_final_to_legacy_array

        # Create a CSV file with known values
        header = ",".join(FINAL_CSV_COLUMNS)
        # Set distinct values for key columns
        values = {col: 0.0 for col in FINAL_CSV_COLUMNS}
        values["sim_number"] = 1.0
        values["energy_mj_kg"] = -5.0  # legacy index 8 (after sim_number offset)
        values["eccentricity"] = 0.5  # legacy index 10
        values["periapsis_err_km"] = 42.0  # legacy index 30
        values["apoapsis_err_km"] = 100.0  # legacy index 31
        values["dv_total_m_s"] = 500.0  # legacy index 42

        row = ",".join(str(values[col]) for col in FINAL_CSV_COLUMNS)
        csv_file = tmp_path / "final.csv"
        csv_file.write_text(f"{header}\n{row}\n")

        result = _parse_final_to_legacy_array(csv_file)
        assert result is not None
        assert result.shape == (1, 53)

        # Check key columns at their legacy positions
        # CSV_TO_LEGACY_INDEX maps col name → xsauve index, then result[:, idx+1]
        # because result[:, 0] = sim_number
        assert result[0, 0] == 1.0  # sim_number
        assert result[0, 8] == -5.0  # energy: CSV_TO_LEGACY_INDEX=7, array pos=8
        assert result[0, 10] == 0.5  # eccentricity: legacy_idx=9, pos=10
        assert result[0, 30] == 42.0  # peri_err: legacy_idx=29, pos=30
        assert result[0, 31] == 100.0  # apo_err: legacy_idx=30, pos=31
        assert result[0, 42] == 500.0  # dv_total: legacy_idx=41, pos=42


class TestComputeCost:
    """Tests for compute_cost with both format inputs."""

    def test_compute_cost_captured(self) -> None:
        """Captured trajectory cost is based on orbit errors + delta-V."""
        from aerocapture.training.evaluate import compute_cost

        # Simulate a captured trajectory (energy < 0, ecc < 1)
        final = np.zeros((1, 53))
        final[0, 8] = -5.0  # energy (MJ/kg), negative = captured
        final[0, 10] = 0.5  # eccentricity < 1
        final[0, 28] = 500.0  # sim_time
        final[0, 30] = 10.0  # peri_err (km)
        final[0, 31] = 20.0  # apo_err (km)
        final[0, 42] = 100.0  # dv_total (m/s)

        cost = compute_cost(final)
        assert cost > 0
        assert cost < 1e6  # Should be much less than hyperbolic penalty

    def test_compute_cost_hyperbolic(self) -> None:
        """Hyperbolic trajectory gets high penalty."""
        from aerocapture.training.evaluate import compute_cost

        final = np.zeros((1, 53))
        final[0, 8] = 5.0  # energy > 0 = hyperbolic
        final[0, 10] = 1.5  # eccentricity > 1
        final[0, 28] = 100.0  # sim_time

        cost = compute_cost(final)
        assert cost > 1e6  # Should be penalized heavily
