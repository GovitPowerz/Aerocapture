from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from aerocapture.training.sensitivity import DISPERSION_COLUMNS

# Indices into the 52-element final_record array, matching extract_final_csv_values() in runner.rs
# fmt: skip
FINAL_RECORD_INDICES: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    37,
    38,
    39,
    40,
    41,
    45,
    48,
]

FINAL_COLUMNS: list[str] = [
    "altitude_km",
    "longitude_deg",
    "latitude_deg",
    "velocity_m_s",
    "flight_path_deg",
    "azimuth_deg",
    "radial_velocity_m_s",
    "energy_mj_kg",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_periapsis_deg",
    "true_anomaly_deg",
    "periapsis_alt_km",
    "apoapsis_alt_km",
    "max_heat_flux_kw_m2",
    "max_load_factor_g",
    "max_dyn_pressure_kpa",
    "alt_max_flux_km",
    "alt_max_load_km",
    "alt_max_pdyn_km",
    "time_max_flux_s",
    "time_max_load_s",
    "time_max_pdyn_s",
    "bounce_alt_km",
    "bounce_time_s",
    "sim_time_s",
    "integrated_flux_mj_m2",
    "periapsis_err_km",
    "apoapsis_err_km",
    "ifinal",
    "dv1_m_s",
    "dv2_m_s",
    "dv3_m_s",
    "dv12_m_s",
    "dv_total_m_s",
    "cumulative_bank_change_deg",
    "n_roll_reversals",
]


def write_parquet(
    path: str | Path,
    final_records: npt.NDArray[np.float64],
    dispersions: npt.NDArray[np.float64],
    config: dict[str, Any],
    toml_path: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the 39 selected columns from the 52-element final_record rows
    selected = final_records[:, FINAL_RECORD_INDICES]

    arrays: list[pa.Array] = [pa.array(selected[:, i], type=pa.float64()) for i in range(len(FINAL_COLUMNS))]
    arrays += [pa.array(dispersions[:, i], type=pa.float64()) for i in range(len(DISPERSION_COLUMNS))]

    field_names = FINAL_COLUMNS + [f"disp_{c}" for c in DISPERSION_COLUMNS]
    table = pa.table({name: arr for name, arr in zip(field_names, arrays, strict=True)})

    guidance = config.get("guidance")
    guidance_scheme = (guidance.get("type") or guidance.get("scheme") or "unknown") if isinstance(guidance, dict) else "unknown"
    n_sims = len(final_records)

    metadata = {
        "aerocapture.config": json.dumps(config),
        "aerocapture.toml_path": str(toml_path) if toml_path is not None else "",
        "aerocapture.timestamp": datetime.now(UTC).isoformat(),
        "aerocapture.guidance_scheme": guidance_scheme,
        "aerocapture.n_sims": str(n_sims),
    }

    schema = table.schema.with_metadata(metadata)
    table = table.cast(schema)

    pq.write_table(table, path)


def read_parquet(path: str | Path) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()

    raw_meta: dict[bytes, bytes] = table.schema.metadata or {}
    meta: dict[str, Any] = {
        "config": json.loads(raw_meta.get(b"aerocapture.config", b"{}")),
        "toml_path": raw_meta.get(b"aerocapture.toml_path", b"").decode(),
        "timestamp": raw_meta.get(b"aerocapture.timestamp", b"").decode(),
        "guidance_scheme": raw_meta.get(b"aerocapture.guidance_scheme", b"").decode(),
        "n_sims": raw_meta.get(b"aerocapture.n_sims", b"").decode(),
    }

    return df, meta
