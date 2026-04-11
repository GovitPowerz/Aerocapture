from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from aerocapture.training.parquet_output import DISPERSION_COLUMNS, FINAL_COLUMNS, FINAL_RECORD_INDICES, read_parquet, write_parquet
from aerocapture.training.sensitivity import DISPERSION_COLUMNS as SENSITIVITY_DISPERSION_COLUMNS

N_SIMS = 10
N_FINAL = 52
N_DISP = 26
N_FINAL_COLS = 39


@pytest.fixture()
def fake_data():
    rng = np.random.default_rng(42)
    final_records = rng.random((N_SIMS, N_FINAL))
    dispersions = rng.random((N_SIMS, N_DISP))
    config = {"guidance": {"type": "equilibrium_glide"}, "monte_carlo": {"n_sims": N_SIMS}}
    return final_records, dispersions, config


def test_final_record_indices_length():
    assert len(FINAL_RECORD_INDICES) == N_FINAL_COLS


def test_final_columns_length():
    assert len(FINAL_COLUMNS) == N_FINAL_COLS


def test_dispersion_columns_match_sensitivity():
    assert DISPERSION_COLUMNS == SENSITIVITY_DISPERSION_COLUMNS


def test_final_record_indices_within_bounds():
    assert all(0 <= i < N_FINAL for i in FINAL_RECORD_INDICES)


def test_final_record_indices_no_duplicates():
    assert len(set(FINAL_RECORD_INDICES)) == len(FINAL_RECORD_INDICES)


def test_write_parquet_creates_file(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    assert out.exists()


def test_write_parquet_creates_parent_dirs(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "nested" / "deep" / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    assert out.exists()


def test_roundtrip_row_count(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    assert table.num_rows == N_SIMS


def test_roundtrip_column_count(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    assert table.num_columns == N_FINAL_COLS + N_DISP


def test_column_names_final(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    cols = table.schema.names
    for name in FINAL_COLUMNS:
        assert name in cols, f"missing final column: {name}"


def test_column_names_dispersion(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    cols = table.schema.names
    for name in DISPERSION_COLUMNS:
        assert f"disp_{name}" in cols, f"missing dispersion column: disp_{name}"


def test_column_order(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    expected = FINAL_COLUMNS + [f"disp_{c}" for c in DISPERSION_COLUMNS]
    assert table.schema.names == expected


def test_metadata_keys_present(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config, toml_path="/some/path.toml")
    meta = pq.read_schema(out).metadata
    assert b"aerocapture.config" in meta
    assert b"aerocapture.toml_path" in meta
    assert b"aerocapture.timestamp" in meta
    assert b"aerocapture.guidance_scheme" in meta
    assert b"aerocapture.n_sims" in meta


def test_metadata_config_is_valid_json(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    meta = pq.read_schema(out).metadata
    parsed = json.loads(meta[b"aerocapture.config"])
    assert parsed == config


def test_metadata_guidance_scheme(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    meta = pq.read_schema(out).metadata
    assert meta[b"aerocapture.guidance_scheme"] == b"equilibrium_glide"


def test_metadata_n_sims(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    meta = pq.read_schema(out).metadata
    assert meta[b"aerocapture.n_sims"] == str(N_SIMS).encode()


def test_metadata_toml_path_none(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config, toml_path=None)
    meta = pq.read_schema(out).metadata
    assert meta[b"aerocapture.toml_path"] == b""


def test_data_integrity_final_columns(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    for col_name, src_idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True):
        col_data = table.column(col_name).to_pylist()
        expected = final_records[:, src_idx].tolist()
        assert col_data == pytest.approx(expected, rel=1e-10), f"mismatch in column {col_name}"


def test_data_integrity_dispersion_columns(fake_data, tmp_path):
    final_records, dispersions, config = fake_data
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    table = pq.read_table(out)
    for i, disp_name in enumerate(DISPERSION_COLUMNS):
        col_data = table.column(f"disp_{disp_name}").to_pylist()
        expected = dispersions[:, i].tolist()
        assert col_data == pytest.approx(expected, rel=1e-10), f"mismatch in disp_{disp_name}"


def test_guidance_scheme_fallback_when_missing(tmp_path):
    rng = np.random.default_rng(0)
    final_records = rng.random((3, N_FINAL))
    dispersions = rng.random((3, N_DISP))
    config = {"monte_carlo": {"n_sims": 3}}  # no guidance key
    out = tmp_path / "output.parquet"
    write_parquet(out, final_records, dispersions, config)
    meta = pq.read_schema(out).metadata
    assert meta[b"aerocapture.guidance_scheme"] == b"unknown"


class TestReadParquet:
    def test_roundtrip(self, fake_data, tmp_path: Path) -> None:
        """write_parquet -> read_parquet returns same data and metadata."""
        import pandas as pd

        final_records, dispersions, config = fake_data
        out = tmp_path / "output.parquet"
        write_parquet(out, final_records, dispersions, config, toml_path="/configs/foo.toml")

        df, meta = read_parquet(out)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N_SIMS, N_FINAL_COLS + N_DISP)
        assert meta["n_sims"] == str(N_SIMS)
        assert meta["toml_path"] == "/configs/foo.toml"
        assert meta["guidance_scheme"] == "equilibrium_glide"
        assert "timestamp" in meta
        assert isinstance(meta["config"], dict)

    def test_metadata_config_deserialized(self, fake_data, tmp_path: Path) -> None:
        """Config metadata is deserialized back to a dict."""
        final_records, dispersions, config = fake_data
        out = tmp_path / "output.parquet"
        write_parquet(out, final_records, dispersions, config)

        _, meta = read_parquet(out)

        assert meta["config"] == config
        assert meta["config"]["guidance"]["type"] == "equilibrium_glide"
