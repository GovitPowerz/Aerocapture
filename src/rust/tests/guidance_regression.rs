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

/// Parse a CSV value, returning None for non-numeric entries (header row).
fn parse_csv_value(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

/// Compare two CSV files with approximate floating-point tolerance.
///
/// Allows relative error up to `rel_tol` (default 1e-9) to accommodate
/// cross-platform floating-point differences (macOS ARM vs Linux x86_64).
fn compare_csv_approx(actual: &str, golden: &str, scheme_label: &str, rel_tol: f64) {
    let actual_lines: Vec<&str> = actual.lines().collect();
    let golden_lines: Vec<&str> = golden.lines().collect();

    assert_eq!(
        actual_lines.len(),
        golden_lines.len(),
        "{scheme_label}: line count differs: golden={}, actual={}",
        golden_lines.len(),
        actual_lines.len(),
    );

    for (i, (a_line, g_line)) in actual_lines.iter().zip(golden_lines.iter()).enumerate() {
        // Header line: exact match
        if i == 0 {
            assert_eq!(a_line, g_line, "{scheme_label}: header line differs");
            continue;
        }

        let a_vals: Vec<&str> = a_line.split(',').collect();
        let g_vals: Vec<&str> = g_line.split(',').collect();

        assert_eq!(
            a_vals.len(),
            g_vals.len(),
            "{scheme_label}: column count differs at line {i}"
        );

        for (j, (a_str, g_str)) in a_vals.iter().zip(g_vals.iter()).enumerate() {
            match (parse_csv_value(a_str), parse_csv_value(g_str)) {
                (Some(a_val), Some(g_val)) => {
                    let abs_diff = (a_val - g_val).abs();
                    let max_mag = a_val.abs().max(g_val.abs());
                    // Use absolute tolerance for near-zero values, relative otherwise
                    let tol = if max_mag < 1e-15 {
                        1e-15
                    } else {
                        rel_tol * max_mag
                    };
                    assert!(
                        abs_diff <= tol,
                        "{scheme_label}: line {i}, col {j} differs beyond tolerance.\n\
                         golden: {g_str}\n\
                         actual: {a_str}\n\
                         abs_diff: {abs_diff:.3e}, rel_tol: {rel_tol:.0e}",
                    );
                }
                _ => {
                    // Non-numeric: exact match
                    assert_eq!(
                        a_str, g_str,
                        "{scheme_label}: non-numeric value differs at line {i}, col {j}"
                    );
                }
            }
        }
    }
}

/// Run a guidance scheme's dedicated test config and compare final output
/// against golden reference with approximate floating-point tolerance.
///
/// Each scheme has its own config with a unique `results_suffix`, so tests
/// can safely run in parallel without overwriting each other's output files.
#[rstest]
#[case(
    "test/test_eqglide_golden.toml",
    "eqglide",
    ".golden_eqglide",
    "EquilibriumGlide"
)]
#[case(
    "test/test_energy_ctrl_golden.toml",
    "energy_ctrl",
    ".golden_energy_ctrl",
    "EnergyController"
)]
#[case(
    "test/test_pred_guid_golden.toml",
    "pred_guid",
    ".golden_pred_guid",
    "PredGuid"
)]
#[case("test/test_fnpag_golden.toml", "fnpag", ".golden_fnpag", "FNPAG")]
#[case("test/test_ftc_golden.toml", "ftc", ".golden_ftc", "FTC")]
#[case(
    "test/test_neural_golden.toml",
    "neural",
    ".golden_neural",
    "NeuralNetwork"
)]
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

    // Read actual output
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

    // Approximate comparison: rel_tol=1e-9 accommodates cross-platform FP differences
    compare_csv_approx(&actual, &golden, scheme_label, 1e-9);
}
