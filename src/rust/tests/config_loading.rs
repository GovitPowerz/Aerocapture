mod common;

use aerocapture::config::{PlanetConfig, SimInput, resolve_toml_bases};
use std::collections::HashSet;
use std::path::Path;

/// A leaf is a config that runs as-is: after base resolution its root carries
/// BOTH `[mission]` and `[guidance]`. Shared bases fail one or the other
/// (`planets/*` neither, `missions/*` no guidance, `training/*_common.toml`
/// and `nn_ftc_scaffolding.toml` no mission), so no directory name is
/// special-cased.
fn is_leaf(path: &Path) -> bool {
    let raw = std::fs::read_to_string(path).expect("read config");
    let root: toml::Value = toml::from_str(&raw)
        .unwrap_or_else(|e| panic!("{}: TOML parse error: {e}", path.display()));
    let resolved = resolve_toml_bases(root, path, &mut HashSet::new())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let table = resolved.as_table().expect("root table");
    ["mission", "guidance"]
        .iter()
        .all(|k| table.get(*k).is_some_and(|v| v.is_table()))
}

/// Every committed leaf config: `configs/**` recursively (paper, sweep, quant,
/// ou_marginal, probe cells included) plus the trainer seam gate configs under
/// `experiments/`. Sorted so failures are reported in a stable order.
fn all_leaf_configs() -> Vec<std::path::PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read configs dir") {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "toml") && is_leaf(&path) {
                out.push(path);
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

/// The root and every Rust-owned section struct are `deny_unknown_fields`, so
/// this walk is the key-reachability gate: a key no `Toml*` struct declares
/// (typo, wrong section, stale knob) fails here with serde's `unknown field`
/// message and the offending file. Serde only proves the keys exist; the no-IO
/// pass also runs the explicit checks of the two flatten sections
/// ([guidance.piecewise_constant] strays, [monte_carlo.<domain>] custom keys)
/// and every enum-string rule.
#[test]
fn every_committed_config_parses_and_validates() {
    let configs = all_leaf_configs();
    for path in &configs {
        let (_config, toml_config) = SimInput::from_toml_file(path)
            .unwrap_or_else(|e| panic!("{}: {:?}", path.display(), e));
        assert!(
            toml_config.vehicle.is_some(),
            "{} is not consolidated (missing [vehicle] section after base resolution)",
            path.display()
        );
        aerocapture::config::validate(&toml_config)
            .unwrap_or_else(|e| panic!("{}: {:?}", path.display(), e));
    }
    // Each config family must contribute at least one leaf, so a classifier
    // regression cannot silently empty the walk.
    let root = common::repo_root();
    for dir in [
        "configs/nominal",
        "configs/test",
        "configs/training",
        "configs/training/paper",
        "configs/training/sweep",
        "experiments/trainer_seam_gate",
    ] {
        let prefix = root.join(dir);
        assert!(
            configs.iter().any(|p| p.starts_with(&prefix)),
            "no leaf config found under {dir}"
        );
    }
}

/// `configs/planets/*` are not leaves (no `[mission]`/`[guidance]`); they are
/// parsed on their own against the runtime `PlanetConfig` so a stray key or a
/// name/file mismatch in a preset is caught too.
#[test]
fn planet_presets_parse() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PlanetFile {
        planet: PlanetConfig,
    }
    let dir = common::repo_root().join("configs/planets");
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read configs/planets") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).expect("read planet preset");
        let file: PlanetFile =
            toml::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        assert_eq!(
            file.planet.name,
            stem,
            "{}: planet.name != file stem",
            path.display()
        );
        names.push(stem);
    }
    names.sort();
    assert_eq!(names, ["earth", "jupiter", "mars", "moon"]);
}
