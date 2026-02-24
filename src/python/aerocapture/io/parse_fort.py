"""Parse Fortran fort.201-204 output files into DataFrames."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from aerocapture.io._fortran import parse_fortran_line

# Column definitions for each fort.* file
FORT201_COLUMNS = [
    "time",  # Simulation time (s)
    "altitude",  # Altitude (km)
    "longitude",  # Longitude (deg)
    "latitude",  # Latitude (deg)
    "velocity",  # Velocity magnitude (m/s)
    "flight_path_angle",  # Velocity slope (deg)
    "azimuth",  # Velocity azimuth (deg)
    "radial_velocity",  # Radial velocity (m/s)
    "energy",  # Total energy (MJ/kg)
    "semi_major_axis",  # Semi-major axis (km)
    "eccentricity",  # Eccentricity
    "inclination",  # Inclination (deg)
    "raan",  # Longitude of ascending node (deg)
    "arg_periapsis",  # Argument of periapsis (deg)
    "periapsis_alt",  # Periapsis altitude (km)
    "apoapsis_alt",  # Apoapsis altitude (km)
    "saturation",  # Saturation indicator
    "density_real",  # Real air density (kg/m3)
    "density_estimated",  # Estimated air density (kg/m3)
    "density_exit",  # Predicted final air density (kg/m3)
    "radius_error",  # Radius error (km)
    "longitude_error",  # Longitude error (deg)
    "latitude_error",  # Latitude error (deg)
    "velocity_error",  # Velocity magnitude error (m/s)
    "slope_error",  # Slope error (deg)
    "azimuth_error",  # Azimuth error (deg)
    "aoa_actual",  # Actual angle of attack (deg)
    "aoa_commanded",  # Commanded angle of attack (deg)
]

FORT202_COLUMNS = [
    "time",
    "heat_flux",  # Current heat flux (kW/m2)
    "g_load",  # Current g-load
    "dynamic_pressure",  # Dynamic pressure (kPa)
    "accel_real_x",  # Real x-acceleration (g)
    "accel_real_y",  # Real y-acceleration (g)
    "bank_commanded",  # Commanded bank angle (deg)
    "bank_actual",  # Actual bank angle (deg)
    "bank_rate",  # Bank rate (deg/s)
    "mach",  # Mach number
    "accel_est_x",  # Estimated x-acceleration (g)
    "accel_est_y",  # Estimated y-acceleration (g)
    "density_real",  # Real air density (kg/m3)
    "heat_flux_integrated",  # Integrated heat flux (MJ/m2)
    "radial_velocity",  # Radial velocity (m/s)
    "saturation",  # Saturation indicator
    "density_estimated",  # Estimated air density (kg/m3)
    "density_coeff",  # Density estimation coefficient
    "security",  # Security indicator
    "altitude",  # Altitude (km)
    "radial_velocity_2",  # Radial velocity (m/s)
    "energy",  # Total energy (MJ/kg)
    "aoa_commanded",  # Commanded AoA (deg)
    "aoa_actual",  # Actual AoA (deg)
]

FORT203_COLUMNS = [
    "time",
    "pred_alt_error",  # Predicted altitude error (km)
    "pred_ecc_error",  # Predicted eccentricity error
    "pred_inc_error",  # Predicted inclination error (deg)
    "pred_orb_error",  # Predicted other orbital error (deg)
    "curr_alt_error",  # Current altitude error (km)
    "curr_ecc_error",  # Current eccentricity error
    "curr_inc_error",  # Current inclination error (deg)
    "curr_orb_error",  # Current other orbital error (deg)
    "integ_dur_1",  # Max integration duration 1 (s)
    "integ_dur_2",  # Max integration duration 2 (s)
    "integ_dur_3",  # Max integration duration 3 (s)
    "pred_traj_dur",  # Predicted trajectory duration (s)
    "saturation",  # Saturation indicator
    "density_real",  # Real air density (kg/m3)
    "density_model",  # Model air density (kg/m3)
    "altitude",  # Altitude (km)
    "radial_velocity",  # Radial velocity (m/s)
    "energy",  # Total energy (MJ/kg)
    "pdyn_eq",  # Equilibrium dynamic pressure (kPa)
    "pdyn_current",  # Current dynamic pressure (Pa)
    "pdyn_eq_pa",  # Equilibrium dynamic pressure (Pa)
    "roll_index",  # Roll profile index
    "gain_alt",  # Altitude gain (m)
    "gain_pdyn",  # Dynamic pressure gain (Pa)
    "ext_index",  # Extended index
    "vel_ref",  # Reference velocity (m/s)
    "apogee_adj",  # Apogee adjustment (km)
    "lon_guide_flag",  # Longitude guidance active
    "lat_guide_flag",  # Latitude guidance active
    "vel_estimated",  # Estimated velocity magnitude (m/s)
    "inclination",  # Inclination (deg)
    "perigee_adj",  # Perigee adjustment (km)
    "lon_ctrl_index",  # Longitude control index
    "corrected_aoa",  # Corrected AoA (deg)
    "velocity",  # Velocity magnitude (m/s)
]

FORT_COLUMN_MAP = {
    201: FORT201_COLUMNS,
    202: FORT202_COLUMNS,
    203: FORT203_COLUMNS,
    # 204 has variable columns (69), not naming them all
}

def parse_fort(filepath: str | Path, unit: int | None = None) -> pd.DataFrame:
    """Parse a fort.* output file into a DataFrame.

    Args:
        filepath: Path to the fort.* file.
        unit: Fortran unit number (201, 202, 203, 204). If None, inferred from filename.

    Returns:
        DataFrame with named columns (if unit is known) or numbered columns.
    """
    filepath = Path(filepath)

    if unit is None:
        match = re.search(r"fort\.(\d+)", filepath.name)
        if match:
            unit = int(match.group(1))

    rows = []
    with open(filepath) as f:
        for line in f:
            values = parse_fortran_line(line)
            if values:
                rows.append(values)

    if not rows:
        return pd.DataFrame()

    data = np.array(rows)
    columns = FORT_COLUMN_MAP.get(unit) if unit else None

    if columns and data.shape[1] == len(columns):
        return pd.DataFrame(data, columns=columns)
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
