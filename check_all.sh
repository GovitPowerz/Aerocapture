cd src/rust
cargo clean
cargo test --lib --quiet
cargo test --doc --quiet
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --quiet -- -D warnings
cargo clean
cd ../..
