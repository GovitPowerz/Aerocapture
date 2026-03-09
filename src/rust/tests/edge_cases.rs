mod common;

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

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

// ─── Single sim, no Monte Carlo ───

#[test]
fn single_sim_no_mc_produces_output() {
    let output = run_sim("msr_aller_ftc_consolidated.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Single sim (consolidated) failed.\nstderr: {}",
        stderr
    );

    // Verify output file exists and is non-empty
    // The consolidated config uses results_suffix = ".test_consolidated"
    let repo = common::repo_root();
    let final_path = find_final_output(&repo, ".test_consolidated");
    assert!(
        final_path.exists(),
        "Final output file missing at {}",
        final_path.display()
    );
    let content = std::fs::read_to_string(&final_path)
        .unwrap_or_else(|e| panic!("Cannot read final output: {}", e));
    assert!(
        !content.trim().is_empty(),
        "Final output file is empty at {}",
        final_path.display()
    );
}

// ─── ESR variant (Earth aerocapture) ───

#[test]
fn esr_aller_completes() {
    // ESR uses old_codebase/donnees/ data files (suffix-based config).
    // TODO: Will break after old_codebase removal — needs consolidated ESR config (Task 4).
    // This test verifies multi-planet coverage (Earth vs Mars).
    let output = run_sim("esr_aller_ftc_nominal.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "ESR aller (Earth) sim failed.\nstderr: {}",
        stderr
    );
}

#[test]
fn esr_retour_completes() {
    let output = run_sim("esr_retour_ftc_nominal.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "ESR retour (Earth) sim failed.\nstderr: {}",
        stderr
    );
}

// ─── Helper ───

/// Find the final output file, trying both .csv and plain suffix variants.
// TODO: Update path once configs use new output_dir (Task 4).
fn find_final_output(repo: &std::path::Path, suffix: &str) -> PathBuf {
    let csv = repo
        .join("old_codebase/sorties")
        .join(format!("final{suffix}.csv"));
    if csv.exists() {
        return csv;
    }
    let plain = repo
        .join("old_codebase/sorties")
        .join(format!("final{suffix}"));
    if plain.exists() {
        return plain;
    }
    // Return CSV path as default (will fail the exists() check with a clear message)
    csv
}
