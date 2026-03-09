//! Output file writers.
//!
//! Supports two output formats:
//! - CSV: Named column headers, standard scientific notation (default)
//! - Text: Legacy Fortran D-notation format (for regression tests)

use std::io::{self, Write};

// ─── CSV column schemas ───

/// Photo CSV column headers (21 columns, down from 24).
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

/// Write a photo CSV data line (21 values).
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

// ─── Legacy Fortran text writers ───

/// Fortran-compatible D-notation float formatter.
///
/// Formats a f64 as Fortran D-notation: " 0.12345678901234D+02"
pub fn fortran_float(val: f64, width: usize, decimals: usize) -> String {
    if val == 0.0 {
        let zeros: String = "0".repeat(decimals);
        return format!("{:>width$}", format!("0.{}D+00", zeros), width = width);
    }

    let sign = if val < 0.0 { "-" } else { " " };
    let abs_val = val.abs();
    let exp = abs_val.log10().floor() as i32;
    let mantissa = abs_val / 10.0_f64.powi(exp);

    // Adjust so mantissa is in [0.1, 1.0)
    let (mantissa, exp) = if mantissa >= 1.0 {
        (mantissa / 10.0, exp + 1)
    } else {
        (mantissa, exp)
    };

    let exp_sign = if exp >= 0 { "+" } else { "-" };
    let exp_abs = exp.unsigned_abs();

    let mant_str = format!("{:.prec$}", mantissa, prec = decimals);
    let full = format!("{}{}D{}{:02}", sign, mant_str, exp_sign, exp_abs);
    format!("{:>width$}", full, width = width)
}

/// Write a trajectory snapshot line in legacy Fortran format.
///
/// Format: 24 columns, (24(1x,d12.5))
pub fn write_photo_text_line(writer: &mut impl Write, values: &[f64]) -> io::Result<()> {
    for val in values {
        write!(writer, " {}", fortran_float(*val, 12, 5))?;
    }
    writeln!(writer)?;
    Ok(())
}

/// Write a final conditions line in legacy Fortran format.
///
/// Format: i5 + 52 D15.7 values — matches `format(1x,i5,52(1x,d15.7))`
pub fn write_final_text_line(
    writer: &mut impl Write,
    sim_num: i32,
    values: &[f64; 52],
) -> io::Result<()> {
    write!(writer, " {:5}", sim_num)?;
    for val in values {
        write!(writer, " {}", fortran_float(*val, 15, 7))?;
    }
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // ─── fortran_float D-notation ───

    #[test]
    fn zero_formats_correctly() {
        let s = fortran_float(0.0, 12, 5);
        assert_eq!(s.trim(), "0.00000D+00");
        assert_eq!(s.len(), 12);
    }

    #[rstest]
    #[case(1.0, " 0.10000D+01")]
    #[case(123.456, " 0.12346D+03")]
    #[case(-42.0, "-0.42000D+02")]
    #[case(0.001, " 0.10000D-02")]
    fn d12_5_known_values(#[case] val: f64, #[case] expected: &str) {
        let s = fortran_float(val, 12, 5);
        assert_eq!(s, expected, "fortran_float({val}, 12, 5)");
    }

    #[test]
    fn width_is_respected() {
        let s12 = fortran_float(3.125, 12, 5);
        let s15 = fortran_float(3.125, 15, 7);
        assert_eq!(s12.len(), 12);
        assert_eq!(s15.len(), 15);
    }

    #[rstest]
    #[case(1e-10)]
    #[case(1e10)]
    #[case(-1e-10)]
    #[case(-1e10)]
    fn extreme_values_no_panic(#[case] val: f64) {
        let s = fortran_float(val, 12, 5);
        assert!(s.contains('D'), "should contain D-notation: {s}");
        assert_eq!(s.len(), 12);
    }

    // ─── write_photo_text_line ───

    #[test]
    fn photo_text_line_has_correct_column_count() {
        let values = vec![0.0; 24];
        let mut buf = Vec::new();
        write_photo_text_line(&mut buf, &values).unwrap();
        let line = String::from_utf8(buf).unwrap();
        let trimmed = line.trim_end_matches('\n');
        let d_count = trimmed.matches('D').count();
        assert_eq!(d_count, 24, "should have 24 D-notation columns");
    }

    // ─── CSV writers ───

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
