#!/bin/bash

# Run linters
echo "Running linters..."

echo ""
echo ""
echo "ruff: Sorting imports..."
uv run ruff check --select I --fix --config=pyproject.toml src tests

echo ""
echo ""
echo "ruff: Formatting code..."
uv run ruff format --config=pyproject.toml src tests

echo ""
echo ""
echo "ruff: Checking code quality and style..."
uv run ruff check --config=pyproject.toml src tests

echo ""
echo ""
echo "mypy: Checking type annotations..."
uv run mypy --config-file pyproject.toml --install-types --non-interactive src tests

echo ""
echo ""
echo "All linters completed successfully!"
