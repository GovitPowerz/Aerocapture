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

/// Run a guidance scheme's dedicated test config and compare final output
/// against golden reference (byte-level -- same seed must produce identical output).
///
/// Each scheme has its own config with a unique `results_suffix`, so tests
/// can safely run in parallel without overwriting each other's output files.
#[rstest]
#[case(
    "test_eqglide_golden.toml",
    "eqglide",
    ".golden_eqglide",
    "EquilibriumGlide"
)]
#[case(
    "test_energy_ctrl_golden.toml",
    "energy_ctrl",
    ".golden_energy_ctrl",
    "EnergyController"
)]
#[case(
    "test_pred_guid_golden.toml",
    "pred_guid",
    ".golden_pred_guid",
    "PredGuid"
)]
#[case("test_fnpag_golden.toml", "fnpag", ".golden_fnpag", "FNPAG")]
#[case("test_ftc_golden.toml", "ftc", ".golden_ftc", "FTC")]
#[case("test_neural_golden.toml", "neural", ".golden_neural", "NeuralNetwork")]
fn guidance_regression(
    #[case] config_name: &str,
    #[case] golden_dir: &str,
    #[case] suffix: &str,
    #[case] scheme_label: &str,
) {
    let output = run_sim(config_name);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{scheme_label}: sim failed.\nstderr: {stderr}",
    );

    let repo = common::repo_root();

    // Read actual output (Rust simulator appends .csv to the suffix)
    let actual_filename = format!("final{suffix}.csv");
    let actual_path = repo.join("output").join(&actual_filename);
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
        .join(&actual_filename);
    let golden = std::fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        panic!(
            "{scheme_label}: cannot read golden reference at {}: {e}",
            golden_path.display()
        )
    });

    // Byte-level comparison: same seed must produce identical output
    if actual != golden {
        // Show first differing line for diagnostics
        let actual_lines: Vec<&str> = actual.lines().collect();
        let golden_lines: Vec<&str> = golden.lines().collect();
        let mut diff_msg = String::new();
        for (i, (a, g)) in actual_lines.iter().zip(golden_lines.iter()).enumerate() {
            if a != g {
                diff_msg = format!(
                    "First difference at line {} (0-indexed):\n  golden: {}\n  actual: {}",
                    i, g, a
                );
                break;
            }
        }
        if diff_msg.is_empty() && actual_lines.len() != golden_lines.len() {
            diff_msg = format!(
                "Line count differs: golden={}, actual={}",
                golden_lines.len(),
                actual_lines.len()
            );
        }
        panic!(
            "{scheme_label}: final output differs from golden reference.\n\
             Golden: {}\n\
             Actual: {}\n\
             {diff_msg}",
            golden_path.display(),
            actual_path.display(),
        );
    }
}
