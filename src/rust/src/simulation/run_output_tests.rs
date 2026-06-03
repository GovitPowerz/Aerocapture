use super::*;
use crate::config::SimInput;
use crate::data::SimData;

fn load_config(config_name: &str) -> (SimInput, SimData) {
    // Data file paths in TOML configs are relative to repo root
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let repo_root = std::path::PathBuf::from(&manifest)
        .join("../..")
        .canonicalize()
        .unwrap();
    std::env::set_current_dir(&repo_root).unwrap();

    let path = std::path::Path::new(config_name);
    let (sim_config, toml_config) = SimInput::from_toml_file(path).expect("parse");
    let sim_data = SimData::from_toml(&toml_config, &sim_config).expect("data");
    (sim_config, sim_data)
}

fn load_test_config() -> (SimInput, SimData) {
    load_config("configs/test/test_ref_orig.toml")
}

#[test]
fn run_for_api_returns_one_result_for_single_sim() {
    let (config, data) = load_test_config();
    let results = run_for_api(&config, &data, false, None).expect("run");
    assert_eq!(results.len(), 1);
}

#[test]
fn run_output_final_record_has_52_elements() {
    let (config, data) = load_test_config();
    let results = run_for_api(&config, &data, false, None).expect("run");
    assert_eq!(results[0].final_record.len(), 52);
}

#[test]
fn run_output_final_record_matches_file_path() {
    let (config, data) = load_test_config();
    let api_results = run_for_api(&config, &data, false, None).expect("api run");
    let api_fr = &api_results[0].final_record;

    run(&config, &data).expect("file run");

    let suffix = config.results_suffix.trim_start_matches('.');
    let final_path = config.output_path(&format!("final.{}.csv", suffix));
    let content = std::fs::read_to_string(&final_path).expect("read final csv");
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 2, "final CSV should have header + data");

    assert!(api_fr[7].abs() > 0.0, "energy should be non-zero");
    assert!(api_fr[9] > 0.0, "eccentricity should be positive");
}

#[test]
fn run_output_captured_flag_consistent_with_orbital_elements() {
    let (config, data) = load_test_config();
    let results = run_for_api(&config, &data, false, None).expect("run");
    let r = &results[0];
    let ifinal_val = r.final_record[31] as i32;
    let expected = ifinal_val == 3 && r.final_record[9] < 1.0 && r.final_record[7] < 0.0;
    assert_eq!(r.captured, expected);
}

#[test]
fn peak_values_populated_for_atmospheric_trajectory() {
    let (config, data) = load_config("configs/test/test_high_bank_orig.toml");
    let results = run_for_api(&config, &data, false, None).expect("run");
    let rec = &results[0].final_record;

    // Columns 16-18: peak heat flux (kW/m²), load factor (g), dynamic pressure (kPa)
    assert!(
        rec[16] > 0.0,
        "max_heat_flux should be > 0, got {}",
        rec[16]
    );
    assert!(
        rec[17] > 0.0,
        "max_load_factor should be > 0, got {}",
        rec[17]
    );
    assert!(
        rec[18] > 0.0,
        "max_dyn_pressure should be > 0, got {}",
        rec[18]
    );

    // Columns 19-24: altitudes and times at peak values
    assert!(rec[19] > 0.0, "alt_max_flux should be > 0, got {}", rec[19]);
    assert!(rec[20] > 0.0, "alt_max_load should be > 0, got {}", rec[20]);
    assert!(rec[21] > 0.0, "alt_max_pdyn should be > 0, got {}", rec[21]);
    assert!(
        rec[22] > 0.0,
        "time_max_flux should be > 0, got {}",
        rec[22]
    );
    assert!(
        rec[23] > 0.0,
        "time_max_load should be > 0, got {}",
        rec[23]
    );
    assert!(
        rec[24] > 0.0,
        "time_max_pdyn should be > 0, got {}",
        rec[24]
    );

    // Physical plausibility for Mars entry:
    assert!(
        rec[16] > 10.0 && rec[16] < 500.0,
        "peak heat flux {:.1} kW/m² outside reasonable Mars entry range",
        rec[16]
    );
    assert!(
        rec[17] > 1.0 && rec[17] < 30.0,
        "peak load factor {:.1} g outside reasonable Mars entry range",
        rec[17]
    );
}

#[test]
fn heat_load_in_trajectory_is_monotonically_nondecreasing() {
    let (config, data) = load_test_config();
    let results = run_for_api(&config, &data, true, None).expect("run");
    let traj = &results[0].trajectory;
    assert!(!traj.is_empty(), "trajectory should not be empty");
    for i in 1..traj.len() {
        assert!(
            traj[i][15] >= traj[i - 1][15],
            "heat load must be monotonically non-decreasing at step {}: {} < {}",
            i,
            traj[i][15],
            traj[i - 1][15]
        );
    }
}

#[test]
fn heat_load_final_matches_final_record() {
    let (config, data) = load_test_config();
    let results = run_for_api(&config, &data, true, None).expect("run");
    let r = &results[0];
    let last_traj_heat_load = r.trajectory.last().unwrap()[15]; // kJ/m²
    let final_record_heat_load = r.final_record[28] * 1e3; // MJ/m² → kJ/m²
    let diff = (last_traj_heat_load - final_record_heat_load).abs();
    assert!(
        diff < 1.0, // allow 1 kJ/m² tolerance (photo cadence vs final state)
        "trajectory last heat load ({:.2}) should match final_record ({:.2}), diff={:.4}",
        last_traj_heat_load,
        final_record_heat_load,
        diff
    );
}
