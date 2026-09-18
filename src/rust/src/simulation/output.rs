//! Record projections and file writers: the named 30-column photo-line layout,
//! the CSV projections of the photo line and the 52-element final record, the
//! 17-column trajectory rows the PyO3 API exposes, and the CSV writers.

use super::final_record::*;
use super::runner::SimResult;
use super::sim_types::SimError;
use crate::config::SimInput;
use std::fs::File;
use std::io::{self, BufWriter, Write};

// ─── Photo-line layout (30 columns; written by `photo.rs`, read by the projections below) ───

pub(crate) const PHOTO_TIME_S: usize = 0;
pub(crate) const PHOTO_ALT_KM: usize = 1;
pub(crate) const PHOTO_LON_DEG: usize = 2;
pub(crate) const PHOTO_LAT_DEG: usize = 3;
pub(crate) const PHOTO_VEL_MS: usize = 4;
pub(crate) const PHOTO_FPA_DEG: usize = 5;
pub(crate) const PHOTO_HDG_DEG: usize = 6;
pub(crate) const PHOTO_SMA_KM: usize = 7;
pub(crate) const PHOTO_ECC: usize = 8;
pub(crate) const PHOTO_INCL_DEG: usize = 9;
pub(crate) const PHOTO_RAAN_DEG: usize = 10;
pub(crate) const PHOTO_PERIAPSIS_ALT_KM: usize = 11;
pub(crate) const PHOTO_APOAPSIS_ALT_KM: usize = 12;
pub(crate) const PHOTO_PHASE: usize = 13;
pub(crate) const PHOTO_BANK_DEG: usize = 14;
pub(crate) const PHOTO_RADIAL_VEL_MS: usize = 15;
pub(crate) const PHOTO_AOA_DEG: usize = 16;
pub(crate) const PHOTO_CUMULATIVE_BANK_DEG: usize = 17;
pub(crate) const PHOTO_ENERGY_J_KG: usize = 18;
pub(crate) const PHOTO_PDYN_PA: usize = 19;
pub(crate) const PHOTO_RADIAL_VEL_DUP: usize = 20; // duplicate of [15]; not exported
pub(crate) const PHOTO_PDYN_ONBOARD_KPA: usize = 21;
pub(crate) const PHOTO_SIM_NUMBER: usize = 22; // metadata; not exported
pub(crate) const PHOTO_RESERVED: usize = 23; // always 0; not exported
pub(crate) const PHOTO_HEAT_FLUX_KW_M2: usize = 24; // trajectory-only
pub(crate) const PHOTO_G_LOAD: usize = 25; // trajectory-only
pub(crate) const PHOTO_NAV_DENSITY_RATIO: usize = 26; // trajectory-only
pub(crate) const PHOTO_TRUTH_DENSITY: usize = 27; // trajectory-only
pub(crate) const PHOTO_HEAT_LOAD_KJ_M2: usize = 28;
pub(crate) const PHOTO_DENSITY_PERTURBATION: usize = 29; // trajectory-only

/// Width of the photo-line array.
pub(crate) const PHOTO_LINE_LEN: usize = 30;

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

/// Final CSV column headers (40 columns: `sim_number` + 39 of the 52 final-record
/// entries, down from 53). Dropped 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
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

/// Write a final CSV data line (sim_number as int, then 39 floats).
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

// ─── Record projections ───

/// Project each 30-element photo line onto the 17-element trajectory row exposed
/// by the PyO3 API. Index mapping and unit scaling (energy J->MJ, pdyn Pa->kPa)
/// are the contract documented on `BatchResults` trajectory columns.
pub(crate) fn project_trajectory(photo_lines: &[[f64; PHOTO_LINE_LEN]]) -> Vec<[f64; 17]> {
    photo_lines
        .iter()
        .map(|p| {
            [
                p[PHOTO_ALT_KM],               // [0]  alt_km
                p[PHOTO_LON_DEG],              // [1]  lon_deg
                p[PHOTO_LAT_DEG],              // [2]  lat_deg
                p[PHOTO_VEL_MS],               // [3]  vel_m_s
                p[PHOTO_FPA_DEG],              // [4]  fpa_deg
                p[PHOTO_HDG_DEG],              // [5]  heading_deg
                p[PHOTO_HEAT_FLUX_KW_M2],      // [6]  heat_flux_kw_m2
                p[PHOTO_TIME_S],               // [7]  time_s
                p[PHOTO_ENERGY_J_KG] / 1e6,    // [8]  energy_mj_kg
                p[PHOTO_PDYN_PA] / 1e3,        // [9]  pdyn_kpa
                p[PHOTO_BANK_DEG],             // [10] bank_angle_deg
                p[PHOTO_INCL_DEG],             // [11] inclination_deg
                p[PHOTO_G_LOAD],               // [12] g_load_g
                p[PHOTO_NAV_DENSITY_RATIO],    // [13] nav_density_ratio
                p[PHOTO_TRUTH_DENSITY],        // [14] truth_density_kg_m3
                p[PHOTO_HEAT_LOAD_KJ_M2],      // [15] heat_load_kj_m2
                p[PHOTO_DENSITY_PERTURBATION], // [16] density_perturbation
            ]
        })
        .collect()
}

/// Extract the 22 CSV values from the 30-element photo line (`PHOTO_CSV_COLUMNS` order).
/// Drops the duplicate radial velocity, sim number, reserved slot and the trajectory-only columns.
pub(crate) fn extract_photo_csv_values(values: &[f64; PHOTO_LINE_LEN]) -> [f64; 22] {
    [
        values[PHOTO_TIME_S],
        values[PHOTO_ALT_KM],
        values[PHOTO_LON_DEG],
        values[PHOTO_LAT_DEG],
        values[PHOTO_VEL_MS],
        values[PHOTO_FPA_DEG],
        values[PHOTO_HDG_DEG],
        values[PHOTO_SMA_KM],
        values[PHOTO_ECC],
        values[PHOTO_INCL_DEG],
        values[PHOTO_RAAN_DEG],
        values[PHOTO_PERIAPSIS_ALT_KM],
        values[PHOTO_APOAPSIS_ALT_KM],
        values[PHOTO_PHASE],
        values[PHOTO_BANK_DEG],
        values[PHOTO_RADIAL_VEL_MS],
        values[PHOTO_AOA_DEG],
        values[PHOTO_CUMULATIVE_BANK_DEG],
        values[PHOTO_ENERGY_J_KG],
        values[PHOTO_PDYN_PA],
        values[PHOTO_PDYN_ONBOARD_KPA],
        values[PHOTO_HEAT_LOAD_KJ_M2],
    ]
}

/// Extract the 39 CSV values from the 52-element final record (`FINAL_CSV_COLUMNS`
/// order after `sim_number`). Drops the 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
pub(crate) fn extract_final_csv_values(values: &[f64; FINAL_RECORD_LEN]) -> [f64; 39] {
    [
        values[FR_ALT_KM],
        values[FR_LON_DEG],
        values[FR_LAT_DEG],
        values[FR_VEL_MS],
        values[FR_FPA_DEG],
        values[FR_HDG_DEG],
        values[FR_RADIAL_VEL_MS],
        values[FR_ENERGY_MJKG],
        values[FR_SMA_KM],
        values[FR_ECC],
        values[FR_INCL_DEG],
        values[FR_RAAN_DEG],
        values[FR_ARG_PERI_DEG],
        values[FR_TRUE_ANOM_DEG],
        values[FR_PERIAPSIS_ALT_KM],
        values[FR_APOAPSIS_ALT_KM],
        values[FR_HEAT_FLUX_KW_M2],
        values[FR_G_LOAD],
        values[FR_DYN_PRESSURE_KPA],
        values[FR_ALT_MAX_FLUX_KM],
        values[FR_ALT_MAX_LOAD_KM],
        values[FR_ALT_MAX_PDYN_KM],
        values[FR_TIME_MAX_FLUX_S],
        values[FR_TIME_MAX_LOAD_S],
        values[FR_TIME_MAX_PDYN_S],
        values[FR_BOUNCE_ALT_KM],
        values[FR_BOUNCE_TIME_S],
        values[FR_SIM_TIME_S],
        values[FR_HEAT_LOAD_MJM2],
        values[FR_PERIAPSIS_ERR_KM],
        values[FR_APOAPSIS_ERR_KM],
        values[FR_IFINAL],
        values[FR_DV1_MS],
        values[FR_DV2_MS],
        values[FR_DV3_MS],
        values[FR_DV_PLANE_MS],
        values[FR_DV_TOTAL_MS],
        values[FR_CUMULATIVE_BANK_DEG],
        values[FR_N_REVERSALS],
    ]
}

/// Write the CLI output: `final.<suffix>.csv` (every sim) and `photo.<suffix>.csv`
/// (the sim at `photo_sim_idx`), with named headers.
pub(crate) fn write_csv_output(
    config: &SimInput,
    results: &[SimResult],
    photo_sim_idx: i32,
) -> Result<(), SimError> {
    let suffix = config.results_suffix.trim_start_matches('.');
    let final_path = config.output_path(&format!("final.{}.csv", suffix));
    let mut final_file = BufWriter::new(
        File::create(&final_path)
            .map_err(|e| SimError(format!("Cannot create {}: {}", final_path, e)))?,
    );

    write_final_csv_header(&mut final_file)
        .map_err(|e| SimError(format!("Final CSV header error: {}", e)))?;

    for result in results {
        let csv_values = extract_final_csv_values(&result.final_line);
        write_final_csv_line(&mut final_file, result.sim_idx + 1, &csv_values)
            .map_err(|e| SimError(format!("Final CSV write error: {}", e)))?;
    }
    final_file
        .flush()
        .map_err(|e| SimError(format!("Final CSV flush error: {}", e)))?;

    // Write photo CSV
    let photo_path = config.output_path(&format!("photo.{}.csv", suffix));
    if let Some(result) = results.iter().find(|r| r.sim_idx == photo_sim_idx) {
        let mut photo_file = BufWriter::new(
            File::create(&photo_path)
                .map_err(|e| SimError(format!("Cannot create {}: {}", photo_path, e)))?,
        );

        write_photo_csv_header(&mut photo_file)
            .map_err(|e| SimError(format!("Photo CSV header error: {}", e)))?;

        for line in &result.photo_lines {
            let csv_values = extract_photo_csv_values(line);
            write_photo_csv_line(&mut photo_file, &csv_values)
                .map_err(|e| SimError(format!("Photo CSV write error: {}", e)))?;
        }
        photo_file
            .flush()
            .map_err(|e| SimError(format!("Photo CSV flush error: {}", e)))?;
    }

    Ok(())
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
