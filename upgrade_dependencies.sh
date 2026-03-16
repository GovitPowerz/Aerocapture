#!/bin/bash
# Update dependencies
echo "🔄 Updating dependencies..."
uv sync --upgrade

# Rebuild PyO3 bindings (uv sync uninstalls locally-built aerocapture-rs)
echo "🔨 Rebuilding PyO3 bindings..."
./build.sh
