//! Integration tests for adaptive DOPRI45 integration mode.

mod common;

use aerocapture::config::{AdaptiveConfig, IntegrationMode, SimInput};
use aerocapture::data::SimData;
use aerocapture::simulation::runner::run_for_api;

/// Load a config via `SimInput::from_toml_file`, setting the cwd to repo root first
/// so that data file paths in TOML (relative to repo root) resolve correctly.
fn load_config_for_api(config_name: &str) -> (SimInput, SimData) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs").join(config_name);
    let (sim_input, toml_config) = SimInput::from_toml_file(&path)
        .unwrap_or_else(|e| panic!("Failed to load config {}: {}", path.display(), e));
    let sim_data = SimData::from_toml(&toml_config, &sim_input)
        .unwrap_or_else(|e| panic!("Failed to build SimData for {}: {}", path.display(), e));
    (sim_input, sim_data)
}

#[test]
fn adaptive_produces_valid_capture() {
    let (config, data) = load_config_for_api("test/test_ref_adaptive.toml");

    let results = run_for_api(&config, &data, false, None).expect("run simulation");
    assert_eq!(results.len(), 1, "Should produce exactly one result");

    let r = &results[0];
    assert!(
        r.captured,
        "Adaptive mode should produce a captured trajectory"
    );

    let ecc = r.final_record[9];
    assert!(
        ecc < 1.0,
        "Eccentricity should be < 1.0 for capture, got {}",
        ecc
    );

    let final_time = r.final_record[27];
    assert!(final_time > 0.0, "Final time should be positive");
}

#[test]
fn adaptive_agrees_with_fixed_on_reference_trajectory() {
    // Run with fixed Gill (the golden reference config)
    let (config_fixed, data_fixed) = load_config_for_api("test/test_ref_orig.toml");
    let results_fixed = run_for_api(&config_fixed, &data_fixed, false, None).expect("run fixed");

    // Run with adaptive DOPRI45
    let (config_adaptive, data_adaptive) = load_config_for_api("test/test_ref_adaptive.toml");
    let results_adaptive =
        run_for_api(&config_adaptive, &data_adaptive, false, None).expect("run adaptive");

    let rf = &results_fixed[0];
    let ra = &results_adaptive[0];

    // Both should capture
    assert!(rf.captured, "Fixed should capture");
    assert!(ra.captured, "Adaptive should capture");

    // final_record[7] = energy (MJ/kg), final_record[9] = eccentricity
    let energy_fixed = rf.final_record[7];
    let energy_adaptive = ra.final_record[7];

    // Final energy should agree within 1% (both integrate the same physics)
    let energy_rel_err = ((energy_fixed - energy_adaptive) / energy_fixed).abs();
    assert!(
        energy_rel_err < 0.01,
        "Energy mismatch: fixed={:.6}, adaptive={:.6}, rel_err={:.4}",
        energy_fixed,
        energy_adaptive,
        energy_rel_err,
    );

    let ecc_fixed = rf.final_record[9];
    let ecc_adaptive = ra.final_record[9];

    // Eccentricity should agree within 1%
    let ecc_abs_err = (ecc_fixed - ecc_adaptive).abs();
    assert!(
        ecc_abs_err < 0.01,
        "Eccentricity mismatch: fixed={:.6}, adaptive={:.6}",
        ecc_fixed,
        ecc_adaptive,
    );
}

/// Pathologically tight tolerance should hit the 1000-substep safety cap
/// and still terminate (not hang). The simulation may not capture, but it
/// must not panic or loop forever.
#[test]
fn safety_cap_terminates_with_tight_tolerance() {
    let (config, data) = load_config_for_api("test/test_ref_adaptive.toml");

    // Override with pathologically tight tolerance — the integrator will reject
    // almost every step and hit the 1000-step cap.
    let mut data = data;
    data.integration_mode = IntegrationMode::AdaptiveDopri45(AdaptiveConfig {
        rtol: 1e-20, // impossibly tight
        initial_dt: 0.1,
        min_dt: 1e-6,
        max_dt: 1.0,
    });

    // Must terminate without panic — result quality doesn't matter
    let results = run_for_api(&config, &data, false, None).expect("should not panic");
    assert_eq!(results.len(), 1, "Should produce exactly one result");
}
