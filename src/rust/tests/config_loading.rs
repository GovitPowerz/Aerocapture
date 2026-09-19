mod common;

use aerocapture::config::SimInput;
use std::path::Path;

/// A fragment is a base-only config: no `[mission]` section and no top-level
/// `base` key. The `base` check is line-anchored so it doesn't false-match
/// config keys that merely end in "base" (e.g. `pressure_coeff_base = ...`).
fn is_fragment(raw: &str) -> bool {
    let has_mission = raw.contains("[mission]");
    let has_base_key = raw
        .lines()
        .any(|l| matches!(l.trim_start().split_once('='), Some((k, _)) if k.trim() == "base"));
    !has_mission && !has_base_key
}

/// Every committed non-fragment config: `configs/**` recursively (paper,
/// sweep, quant, ou_marginal, probe cells included) plus the trainer seam
/// gate configs under `experiments/`. `configs/planets` and `configs/missions`
/// are shared bases (no `[guidance]`), never run as-is, so they are skipped
/// as directories; every other fragment is caught by `is_fragment`. Sorted so
/// failures are reported in a stable order.
fn all_leaf_configs() -> Vec<std::path::PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read configs dir") {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if path
                    .file_name()
                    .is_some_and(|d| d == "planets" || d == "missions")
                {
                    continue;
                }
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "toml") {
                let raw = std::fs::read_to_string(&path).expect("read config");
                if !is_fragment(&raw) {
                    out.push(path);
                }
            }
        }
    }
    let root = common::repo_root();
    let mut out = Vec::new();
    walk(&root.join("configs"), &mut out);
    walk(&root.join("experiments/trainer_seam_gate"), &mut out);
    out.sort();
    out
}

#[test]
fn parse_ftc_consolidated_toml() {
    let path = common::config_path("nominal/msr_aller_ftc_consolidated.toml");
    let (config, _toml) = SimInput::from_toml_file(Path::new(&path)).expect("parse config");
    assert_eq!(config.planet.name, "mars");
    assert_eq!(config.n_sims, 1);
    assert!(!config.reference_trajectory);
    // reference_bank_angle not set in TOML → falls back to entry.initial_bank_angle
    assert!(
        (config.reference_bank_angle - 64.77026).abs() < 1e-6,
        "expected reference_bank_angle ≈ 64.77026 (from entry.initial_bank_angle), got {}",
        config.reference_bank_angle
    );
}

#[test]
fn parse_reference_toml() {
    let path = common::config_path("nominal/msr_aller_reference.toml");
    let (config, _toml) = SimInput::from_toml_file(Path::new(&path)).expect("parse config");
    assert!(config.reference_trajectory);
    assert_eq!(config.planet.name, "mars");
    assert!((config.reference_bank_angle - 0.1).abs() < 1e-6);
}

#[test]
fn parse_mc_domain_toml() {
    let path = common::config_path("nominal/msr_aller_ftc_mc_domain.toml");
    let (config, _toml) = SimInput::from_toml_file(Path::new(&path)).expect("parse config");
    assert_eq!(config.n_sims, 100);
    assert!(!config.reference_trajectory);
}

/// Every Rust-owned section struct is `deny_unknown_fields`, so this walk is
/// the key-reachability gate: a key no `Toml*` struct declares (typo, wrong
/// section, stale knob) fails here with serde's `unknown field` message and
/// the offending file.
#[test]
fn parse_all_available_configs() {
    let configs = all_leaf_configs();
    for path in &configs {
        let result = SimInput::from_toml_file(path);
        assert!(
            result.is_ok(),
            "Failed to parse {}: {:?}",
            path.display(),
            result.err()
        );
    }
    let count = configs.len();
    assert!(
        count >= 150,
        "Expected at least 150 configs, found {}",
        count
    );
}

#[test]
fn all_configs_are_consolidated() {
    for path in all_leaf_configs() {
        // Use from_toml_file to resolve base inheritance before checking
        let (_config, toml_config) = SimInput::from_toml_file(&path)
            .unwrap_or_else(|e| panic!("{}: {:?}", path.display(), e));
        assert!(
            toml_config.vehicle.is_some(),
            "{} is not consolidated (missing [vehicle] section after base resolution)",
            path.display()
        );
        // Serde only proves the keys exist; the no-IO pass also runs the explicit
        // checks of the two flatten sections ([guidance.piecewise_constant] strays,
        // [monte_carlo.<domain>] custom keys) and every enum-string rule.
        aerocapture::config::validate(&toml_config)
            .unwrap_or_else(|e| panic!("{}: {:?}", path.display(), e));
    }
}
