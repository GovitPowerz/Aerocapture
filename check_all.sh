cd src/rust
cargo clean
cargo test --quiet
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --quiet -- -D warnings
cargo clean
cd ../..
