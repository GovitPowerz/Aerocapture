//! Output file writers (CSV format with named column headers).

use std::io::{self, Write};

// ─── CSV column schemas ───

/// Photo CSV column headers (22 columns, down from 24 + 1 new).
/// Dropped: radial_velocity_2 (duplicate), sim_number (metadata), reserved (always 0).
pub const PHOTO_CSV_COLUMNS: &[&str] = &[
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
    "heat_load_kj_m2",
];

/// Final CSV column headers (39 columns, down from 53).
/// Dropped 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
pub const FINAL_CSV_COLUMNS: &[&str] = &[
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
];

// ─── CSV writers ───

/// Write the CSV header for photo output.
pub fn write_photo_csv_header(writer: &mut impl Write) -> io::Result<()> {
    writeln!(writer, "{}", PHOTO_CSV_COLUMNS.join(","))
}

/// Write a photo CSV data line (22 values).
pub fn write_photo_csv_line(writer: &mut impl Write, values: &[f64]) -> io::Result<()> {
    for (i, val) in values.iter().enumerate() {
        if i > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{:.10e}", val)?;
    }
    writeln!(writer)
}

/// Write the CSV header for final output.
pub fn write_final_csv_header(writer: &mut impl Write) -> io::Result<()> {
    writeln!(writer, "{}", FINAL_CSV_COLUMNS.join(","))
}

/// Write a final CSV data line (sim_number as int, then 38 floats).
pub fn write_final_csv_line(
    writer: &mut impl Write,
    sim_num: i32,
    values: &[f64],
) -> io::Result<()> {
    write!(writer, "{}", sim_num)?;
    for val in values {
        write!(writer, ",{:.10e}", val)?;
    }
    writeln!(writer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_has_correct_column_count() {
        let mut buf = Vec::new();
        write_photo_csv_header(&mut buf).unwrap();
        let line = String::from_utf8(buf).unwrap();
        let cols: Vec<&str> = line.trim().split(',').collect();
        assert_eq!(cols.len(), PHOTO_CSV_COLUMNS.len());
    }

    #[test]
    fn csv_line_uses_scientific_notation() {
        let values = vec![1.23, 4.56];
        let mut buf = Vec::new();
        write_photo_csv_line(&mut buf, &values).unwrap();
        let line = String::from_utf8(buf).unwrap();
        assert!(line.contains('e'), "CSV should use e-notation: {line}");
        assert!(!line.contains('D'), "CSV should not use D-notation");
    }
}
