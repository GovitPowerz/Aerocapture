#!/usr/bin/env bash
set -uo pipefail

cd src/rust
echo "Running all checks..."

echo "Running tests..."
cargo test --quiet
test_status=$?

echo "Checking formatting..."
cargo fmt --all -- --check
fmt_status=$?

echo "Running clippy..."
cargo clippy --all-targets --all-features --quiet -- -D warnings
clippy_status=$?

cd ../..

# Rebuild release artifacts and clean intermediate files
./build.sh -c
build_status=$?

# Recap
echo ""
echo "=== Results ==="
[ $test_status -eq 0 ]   && echo "  Tests:      ✅ passed" || echo "  Tests:      ❌ FAILED"
[ $fmt_status -eq 0 ]    && echo "  Formatting: ✅ passed" || echo "  Formatting: ❌ FAILED"
[ $clippy_status -eq 0 ] && echo "  Clippy:     ✅ passed" || echo "  Clippy:     ❌ FAILED"
[ $build_status -eq 0 ]  && echo "  Build:      ✅ passed" || echo "  Build:      ❌ FAILED"
echo ""

if [ $test_status -ne 0 ] || [ $fmt_status -ne 0 ] || [ $clippy_status -ne 0 ] || [ $build_status -ne 0 ]; then
    echo "Some checks failed!"
    exit 1
fi

echo "All checks passed!"
