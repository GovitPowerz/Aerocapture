"""Parse final conditions files (final.*) into DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# CSV column names (40 columns — matches Rust FINAL_CSV_COLUMNS)
FINAL_CSV_COLUMNS = [
    "sim_number",
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

# Mapping from CSV column name to legacy Fortran column index (0-based in xsauve[52]).
# This enables compute_cost and other consumers to access data by name.
CSV_TO_LEGACY_INDEX: dict[str, int] = {
    "altitude_km": 0,
    "longitude_deg": 1,
    "latitude_deg": 2,
    "velocity_m_s": 3,
    "flight_path_deg": 4,
    "azimuth_deg": 5,
    "radial_velocity_m_s": 6,
    "energy_mj_kg": 7,
    "semi_major_axis_km": 8,
    "eccentricity": 9,
    "inclination_deg": 10,
    "raan_deg": 11,
    "arg_periapsis_deg": 12,
    "true_anomaly_deg": 13,
    "periapsis_alt_km": 14,
    "apoapsis_alt_km": 15,
    "max_heat_flux_kw_m2": 16,
    "max_load_factor_g": 17,
    "max_dyn_pressure_kpa": 18,
    "alt_max_flux_km": 19,
    "alt_max_load_km": 20,
    "alt_max_pdyn_km": 21,
    "time_max_flux_s": 22,
    "time_max_load_s": 23,
    "time_max_pdyn_s": 24,
    "bounce_alt_km": 25,
    "bounce_time_s": 26,
    "sim_time_s": 27,
    "integrated_flux_mj_m2": 28,
    "periapsis_err_km": 29,
    "apoapsis_err_km": 30,
    "ifinal": 31,
    "dv1_m_s": 37,
    "dv2_m_s": 38,
    "dv3_m_s": 39,
    "dv12_m_s": 40,
    "dv_total_m_s": 41,
    "cumulative_bank_change_deg": 45,
    "n_roll_reversals": 48,
}


def parse_final(filepath: str | Path) -> pd.DataFrame:
    """Parse a final conditions CSV file into a DataFrame.

    Args:
        filepath: Path to the final CSV file.

    Returns:
        DataFrame with named columns.
    """
    filepath = Path(filepath)

    if not filepath.exists() or filepath.stat().st_size == 0:
        return pd.DataFrame()

    return pd.read_csv(filepath)
