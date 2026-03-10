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

// ─── Reference trajectory ───

#[test]
fn reference_trajectory_completes() {
    let output = run_sim("nominal/msr_aller_reference.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Reference trajectory failed.\nstderr: {}",
        stderr
    );
    assert!(
        !stderr.to_lowercase().contains("error"),
        "stderr contains 'error': {}",
        stderr
    );
}

// ─── FTC guided ───

#[test]
fn ftc_guided_completes() {
    let output = run_sim("nominal/msr_aller_ftc_consolidated.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "FTC guided sim failed.\nstderr: {}",
        stderr
    );
}

// ─── Each guidance scheme ───

#[test]
fn guidance_eqglide_completes() {
    run_guidance_config("training/msr_aller_eqglide_train.toml");
}

#[test]
fn guidance_energy_controller_completes() {
    run_guidance_config("training/msr_aller_energy_controller_train.toml");
}

#[test]
fn guidance_pred_guid_completes() {
    run_guidance_config("training/msr_aller_pred_guid_train.toml");
}

#[test]
fn guidance_fnpag_completes() {
    run_guidance_config("training/msr_aller_fnpag_train.toml");
}

#[test]
fn guidance_ftc_train_completes() {
    run_guidance_config("training/msr_aller_ftc_train.toml");
}

fn run_guidance_config(config_name: &str) {
    let config_file = common::repo_root().join("configs").join(config_name);
    assert!(
        config_file.exists(),
        "Config file missing: configs/{}",
        config_name
    );
    let output = run_sim(config_name);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{} failed.\nstderr: {}",
        config_name,
        stderr
    );
}

// ─── Monte Carlo domain ───

#[test]
fn mc_domain_completes() {
    let output = run_sim("nominal/msr_aller_ftc_mc_domain.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "MC domain sim failed.\nstderr: {}",
        stderr
    );
}

// ─── Monte Carlo deterministic (same seed → same output) ───

#[test]
fn mc_deterministic_same_seed() {
    let repo = common::repo_root();
    let final_path = repo.join("output/final.mc100_domain");

    // Run 1
    let output1 = run_sim("nominal/msr_aller_ftc_mc_domain.toml");
    assert!(
        output1.status.success(),
        "MC run 1 failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    let content1 = std::fs::read_to_string(&final_path).unwrap_or_else(|e| {
        // Try CSV variant
        let csv_path = repo.join("output/final.mc100_domain.csv");
        std::fs::read_to_string(&csv_path)
            .unwrap_or_else(|_| panic!("Cannot read final output after run 1: {}", e))
    });

    // Run 2
    let output2 = run_sim("nominal/msr_aller_ftc_mc_domain.toml");
    assert!(
        output2.status.success(),
        "MC run 2 failed: {}",
        String::from_utf8_lossy(&output2.stderr)
    );
    let content2 = std::fs::read_to_string(&final_path).unwrap_or_else(|_| {
        let csv_path = repo.join("output/final.mc100_domain.csv");
        std::fs::read_to_string(&csv_path).expect("Cannot read final output after run 2")
    });

    assert_eq!(
        content1, content2,
        "MC outputs differ between two runs with the same seed — determinism broken"
    );
}
