mod common;

use aerocapture::config::{Planet, SimInput};

#[test]
fn parse_ftc_consolidated_toml() {
    let path = common::config_path("msr_aller_ftc_consolidated.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert_eq!(config.planet, Planet::Mars);
    assert_eq!(config.n_sims, 1);
    assert!(!config.reference_trajectory);
}

#[test]
fn parse_reference_toml() {
    let path = common::config_path("msr_aller_reference.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert!(config.reference_trajectory);
    assert_eq!(config.planet, Planet::Mars);
    assert!((config.reference_bank_angle - 0.1).abs() < 1e-6);
}

#[test]
fn parse_mc_domain_toml() {
    let path = common::config_path("msr_aller_ftc_mc_domain.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert_eq!(config.n_sims, 100);
    assert!(!config.reference_trajectory);
}

#[test]
fn parse_all_available_configs() {
    let configs_dir = common::repo_root().join("configs");
    let mut count = 0;
    for entry in std::fs::read_dir(&configs_dir).expect("read configs dir") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            let content = std::fs::read_to_string(&path).expect("read config");
            let result = SimInput::from_toml(&content);
            assert!(
                result.is_ok(),
                "Failed to parse {}: {:?}",
                path.display(),
                result.err()
            );
            count += 1;
        }
    }
    assert!(count >= 10, "Expected at least 10 configs, found {}", count);
}
