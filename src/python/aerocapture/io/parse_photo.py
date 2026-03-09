"""Parse trajectory snapshot files (photo.*) into DataFrames.

Supports both CSV format (with headers) and legacy Fortran D-notation format.
Auto-detects the format from the first line of the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aerocapture.io._fortran import parse_fortran_line

# CSV column names (21 columns — matches Rust PHOTO_CSV_COLUMNS)
PHOTO_CSV_COLUMNS = [
    "time_s",
    "altitude_km",
    "longitude_deg",
    "latitude_deg",
    "velocity_m_s",
    "flight_path_deg",
    "azimuth_deg",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "periapsis_alt_km",
    "apoapsis_alt_km",
    "phase",
    "bank_angle_deg",
    "radial_velocity_m_s",
    "aoa_deg",
    "cumulative_bank_change_deg",
    "energy_j_kg",
    "dynamic_pressure_pa",
    "dynamic_pressure_onboard_kpa",
]

# Legacy Fortran column names (24 columns — kept for backward compatibility)
PHOTO_COLUMNS = [
    "time",
    "altitude",
    "longitude",
    "latitude",
    "velocity",
    "flight_path_angle",
    "azimuth",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "periapsis_alt",
    "apoapsis_alt",
    "phase",
    "bank_angle",
    "radial_velocity",
    "aoa",
    "bank_rate",
    "energy",
    "dynamic_pressure",
    "radial_velocity_2",
    "dynamic_pressure_rho",
    "sim_number",
    "reserved",
]

# Map CSV column names to legacy column names for backward compatibility.
# This ensures plotting modules that access by column name work with both formats.
_CSV_TO_LEGACY_NAMES: dict[str, str] = {
    "time_s": "time",
    "altitude_km": "altitude",
    "longitude_deg": "longitude",
    "latitude_deg": "latitude",
    "velocity_m_s": "velocity",
    "flight_path_deg": "flight_path_angle",
    "azimuth_deg": "azimuth",
    "semi_major_axis_km": "semi_major_axis",
    "eccentricity": "eccentricity",
    "inclination_deg": "inclination",
    "raan_deg": "raan",
    "periapsis_alt_km": "periapsis_alt",
    "apoapsis_alt_km": "apoapsis_alt",
    "phase": "phase",
    "bank_angle_deg": "bank_angle",
    "radial_velocity_m_s": "radial_velocity",
    "aoa_deg": "aoa",
    "cumulative_bank_change_deg": "bank_rate",
    "energy_j_kg": "energy",
    "dynamic_pressure_pa": "dynamic_pressure",
    "dynamic_pressure_onboard_kpa": "dynamic_pressure_rho",
}


def parse_photo(filepath: str | Path) -> pd.DataFrame:
    """Parse a photo trajectory snapshot file into a DataFrame.

    Auto-detects CSV (with headers) vs legacy Fortran D-notation format.

    Args:
        filepath: Path to the photo file (.csv or legacy text).

    Returns:
        DataFrame with named columns.
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        first_line = f.readline()

    # Auto-detect: CSV has commas, Fortran D-notation does not
    if "," in first_line:
        df = pd.read_csv(filepath)
        # Normalize CSV column names to legacy names for backward compatibility
        df = df.rename(columns=_CSV_TO_LEGACY_NAMES)
        return df

    # Legacy Fortran format
    rows = []
    with open(filepath) as f:
        for line in f:
            values = parse_fortran_line(line)
            if values:
                rows.append(values)

    if not rows:
        return pd.DataFrame()

    data = np.array(rows)
    if data.shape[1] == len(PHOTO_COLUMNS):
        return pd.DataFrame(data, columns=PHOTO_COLUMNS)
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
