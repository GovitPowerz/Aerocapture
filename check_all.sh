cd src/rust
cargo clean
cargo test --quiet
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --quiet -- -D warnings
# Copy binary out, clean everything, put it back
cp target/release/aerocapture /tmp/ && \
cargo clean && \
mkdir -p target/release && \
cp /tmp/aerocapture target/release/ && \
cd ../..
