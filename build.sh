#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="src/rust"
PYO3_DIR="src/rust/aerocapture-py"
CLEAN=false

usage() {
    echo "Usage: $0 [-c]"
    echo "  -c  Clean build artifacts after copying release binaries"
    exit 1
}

while getopts "ch" opt; do
    case $opt in
        c) CLEAN=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "=== Building Rust simulator ==="
cd "$RUST_DIR"
cargo build --release --quiet
cd - > /dev/null

echo ""
echo "=== Building PyO3 bindings ==="
uv run maturin develop --release --quiet --manifest-path "$PYO3_DIR/Cargo.toml"

if $CLEAN; then
    echo ""
    echo "=== Cleaning build artifacts ==="
    # Preserve release binaries
    cp "$RUST_DIR/target/release/aerocapture" /tmp/
    cp "$RUST_DIR/target/release/libaerocapture_rs.dylib" /tmp/ 2>/dev/null || true

    cd "$RUST_DIR"
    cargo clean
    cd - > /dev/null

    # Restore release binaries
    mkdir -p "$RUST_DIR/target/release"
    cp /tmp/aerocapture "$RUST_DIR/target/release/"
    cp /tmp/libaerocapture_rs.dylib "$RUST_DIR/target/release/" 2>/dev/null || true
    echo "  Cleaned. Kept aerocapture binary and PyO3 dylib."
fi

echo ""
echo "=== Done ==="
echo "  Binary: $RUST_DIR/target/release/aerocapture"
echo "  PyO3:   aerocapture_rs installed in .venv"
