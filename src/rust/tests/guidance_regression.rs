mod common;

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use rstest::rstest;

static BUILD_ONCE: Once = Once::new();

fn ensure_release_build() {
    BUILD_ONCE.call_once(|| {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let status = Command::new("cargo")
            .args(["build", "--release", "--quiet"])
            .current_dir(&manifest)
            .status()
            .expect("cargo build failed to start");
        assert!(status.success(), "cargo build --release failed");
    });
}

fn aerocapture_binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest).join("target/release/aerocapture")
}

fn run_sim(config_name: &str) -> std::process::Output {
    ensure_release_build();
    let config = common::config_path(config_name);
    let repo = common::repo_root();
    Command::new(aerocapture_binary())
        .arg(&config)
        .current_dir(&repo)
        .output()
        .expect("failed to execute aerocapture")
}

/// Run a guidance scheme's training config and compare final output against golden reference.
///
/// Since all configs use the same MC seed (42) and deterministic dispersions,
/// the output must be byte-identical to the golden reference.
#[rstest]
#[case("msr_aller_eqglide_train.toml", "eqglide", "EquilibriumGlide")]
#[case("msr_aller_energy_controller_train.toml", "energy_ctrl", "EnergyController")]
#[case("msr_aller_pred_guid_train.toml", "pred_guid", "PredGuid")]
#[case("msr_aller_fnpag_train.toml", "fnpag", "FNPAG")]
#[case("msr_aller_ftc_train.toml", "ftc_train", "FTC")]
#[case("msr_aller_nn_train_consolidated.toml", "neural", "NeuralNetwork")]
fn guidance_regression(
    #[case] config_name: &str,
    #[case] golden_dir: &str,
    #[case] scheme_label: &str,
) {
    let output = run_sim(config_name);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{scheme_label}: sim failed.\nstderr: {stderr}",
    );

    let repo = common::repo_root();

    // Read actual output
    let actual_path = repo.join("old_codebase/sorties/final.train_nn_temp");
    let actual = std::fs::read_to_string(&actual_path).unwrap_or_else(|e| {
        panic!(
            "{scheme_label}: cannot read final output at {}: {e}",
            actual_path.display()
        )
    });

    // Read golden reference
    let golden_path = repo
        .join("tests/reference_data/rust_golden")
        .join(golden_dir)
        .join("final.train_nn_temp");
    let golden = std::fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        panic!(
            "{scheme_label}: cannot read golden reference at {}: {e}",
            golden_path.display()
        )
    });

    // Byte-level comparison: same seed must produce identical output
    assert_eq!(
        actual, golden,
        "{scheme_label}: final output differs from golden reference.\n\
         Golden: {}\n\
         Actual: {}",
        golden_path.display(),
        actual_path.display(),
    );
}
