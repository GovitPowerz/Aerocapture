cd src/rust
echo "Running all checks..."
cargo clean

echo "Running tests..."
cargo test --quiet
test_status=$?

echo "Checking formatting..."
cargo fmt --all -- --check
fmt_status=$?

echo "Running clippy..."
cargo clippy --all-targets --all-features --quiet -- -D warnings
clippy_status=$?

# Copy binary out, clean everything, put it back
cp target/release/aerocapture /tmp/ && \
cargo clean && \
mkdir -p target/release && \
cp /tmp/aerocapture target/release/ && \
cd ../..

# Recap
echo ""
echo "=== Results ==="
[ $test_status -eq 0 ]   && echo "  Tests:      ✅ passed" || echo "  Tests:      ❌ FAILED"
[ $fmt_status -eq 0 ]    && echo "  Formatting: ✅ passed" || echo "  Formatting: ❌ FAILED"
[ $clippy_status -eq 0 ] && echo "  Clippy:     ✅ passed" || echo "  Clippy:     ❌ FAILED"
echo ""

if [ $test_status -ne 0 ] || [ $fmt_status -ne 0 ] || [ $clippy_status -ne 0 ]; then
    echo "Some checks failed!"
    exit 1
else
    echo "All checks passed!"
fi
