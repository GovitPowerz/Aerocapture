#[allow(dead_code)]
pub mod assertions;
#[allow(dead_code)]
pub mod fixtures;

use std::path::PathBuf;

/// Get absolute path to repo root (2 levels up from src/rust/).
#[allow(dead_code)]
pub fn repo_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest)
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Get path to a TOML config in configs/<subdir>/.
#[allow(dead_code)]
pub fn config_path(name: &str) -> String {
    repo_root()
        .join("configs")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}
