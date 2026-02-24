"""Parse Fortran photo.* trajectory snapshot files into DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aerocapture.io._fortran import parse_fortran_line

PHOTO_COLUMNS = [
    "time",  # Simulation time (s)
    "altitude",  # Altitude (km)
    "longitude",  # Longitude (deg)
    "latitude",  # Latitude (deg)
    "velocity",  # Relative velocity magnitude (m/s)
    "flight_path_angle",  # Velocity slope (deg)
    "azimuth",  # Velocity azimuth (deg)
    "semi_major_axis",  # Semi-major axis (km)
    "eccentricity",  # Eccentricity
    "inclination",  # Inclination (deg)
    "raan",  # Longitude of ascending node (deg)
    "periapsis_alt",  # Periapsis altitude (km)
    "apoapsis_alt",  # Apoapsis altitude (km)
    "phase",  # Flight phase (1=capture, 2=rebound, 3=exit)
    "bank_angle",  # Bank/roll angle (deg)
    "radial_velocity",  # Radial velocity (m/s)
    "aoa",  # Angle of attack (deg)
    "bank_rate",  # Bank rate (deg)
    "energy",  # Total energy (J/kg)
    "dynamic_pressure",  # Dynamic pressure (Pa)
    "radial_velocity_2",  # Radial velocity (m/s) duplicate
    "dynamic_pressure_rho",  # Dynamic pressure from density (kPa)
    "sim_number",  # Simulation number
    "reserved",  # Reserved (0.0)
]


def parse_photo(filepath: str | Path) -> pd.DataFrame:
    """Parse a photo.* trajectory snapshot file into a DataFrame.

    Args:
        filepath: Path to the photo.* file.

    Returns:
        DataFrame with named columns for the 24-column photo format.
    """
    filepath = Path(filepath)
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
