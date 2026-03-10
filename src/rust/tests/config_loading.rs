mod common;

use aerocapture::config::{Planet, SimInput, TomlConfig};

#[test]
fn parse_ftc_consolidated_toml() {
    let path = common::config_path("nominal/msr_aller_ftc_consolidated.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert_eq!(config.planet, Planet::Mars);
    assert_eq!(config.n_sims, 1);
    assert!(!config.reference_trajectory);
}

#[test]
fn parse_reference_toml() {
    let path = common::config_path("nominal/msr_aller_reference.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert!(config.reference_trajectory);
    assert_eq!(config.planet, Planet::Mars);
    assert!((config.reference_bank_angle - 0.1).abs() < 1e-6);
}

#[test]
fn parse_mc_domain_toml() {
    let path = common::config_path("nominal/msr_aller_ftc_mc_domain.toml");
    let content = std::fs::read_to_string(&path).expect("read config");
    let (config, _toml) = SimInput::from_toml(&content).expect("parse config");
    assert_eq!(config.n_sims, 100);
    assert!(!config.reference_trajectory);
}

#[test]
fn parse_all_available_configs() {
    let configs_dir = common::repo_root().join("configs");
    let mut count = 0;
    for subdir in ["nominal", "training", "test"] {
        let dir = configs_dir.join(subdir);
        for entry in std::fs::read_dir(&dir).expect("read configs subdir") {
            let path = entry.unwrap().path();
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
    }
    assert!(count >= 10, "Expected at least 10 configs, found {}", count);
}

#[test]
fn all_configs_are_consolidated() {
    let configs_dir = common::repo_root().join("configs");
    for subdir in ["nominal", "training", "test"] {
        let dir = configs_dir.join(subdir);
        for entry in std::fs::read_dir(&dir).expect("read configs subdir") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "toml") {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("read config");
            let toml_config: TomlConfig =
                toml::from_str(&content).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
            assert!(
                toml_config.vehicle.is_some(),
                "{} is not consolidated (missing [vehicle] section)",
                path.display()
            );
        }
    }
}
