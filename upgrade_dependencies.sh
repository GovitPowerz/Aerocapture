#!/bin/bash
rm -rf test.db

# Update dependencies
echo "🔄 Updating dependencies..."
uv sync --upgrade
