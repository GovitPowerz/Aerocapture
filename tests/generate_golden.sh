#!/usr/bin/env bash
# Regenerate Rust golden reference outputs for regression tests.
#
# Run from the repository root:
#   ./tests/generate_golden.sh
#
# This builds the Rust simulator and runs each test config, then copies
# the photo.* and final.* outputs into tests/reference_data/rust_golden/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BINARY="src/rust/target/release/aerocapture"
GOLDEN_DIR="tests/reference_data/rust_golden"
OUTPUT_DIR="output"

echo "==> Building Rust simulator (release)..."
(cd src/rust && cargo build --release)

run_and_copy() {
    local config="$1"
    local subdir="$2"
    local suffix="$3"
    local dest="$GOLDEN_DIR/$subdir"

    echo "==> Running $config -> $dest"
    mkdir -p "$dest"
    ./"$BINARY" "$config"

    cp "$OUTPUT_DIR/photo.$suffix" "$dest/"
    cp "$OUTPUT_DIR/final.$suffix" "$dest/"
    echo "    Copied photo.$suffix and final.$suffix"
}

run_and_copy "configs/test/test_ref_orig.toml"       "ref"       "test_ref_orig"
run_and_copy "configs/test/test_high_bank_orig.toml"  "high_bank"  "test_high_bank_orig"
run_and_copy "configs/test/test_guided_orig.toml"     "guided"     "test_guided_orig"

echo "==> Golden reference outputs regenerated."
