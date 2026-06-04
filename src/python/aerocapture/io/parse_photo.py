"""Parse trajectory snapshot files (photo.*) into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    """Parse a photo trajectory snapshot CSV file into a DataFrame.

    Args:
        filepath: Path to the photo CSV file.

    Returns:
        DataFrame with named columns (CSV names normalized to legacy names).
    """
    filepath = Path(filepath)

    if not filepath.exists() or filepath.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    # Normalize CSV column names to legacy names for backward compatibility
    df = df.rename(columns=_CSV_TO_LEGACY_NAMES)
    return df
