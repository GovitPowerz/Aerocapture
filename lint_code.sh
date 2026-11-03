#!/bin/bash
# Python lint (ruff imports, format, lint; mypy) plus a syntax pass over every tracked shell
# script. Exits non-zero on the first failing tool. Fixes imports and formatting in place; CI runs
# the check-only equivalents (ruff format --check, no --fix) as separate jobs.
set -euo pipefail

echo "ruff: Sorting imports..."
uv run ruff check --select I --fix --config=pyproject.toml src tests experiments articles/paper/scripts

echo ""
echo "ruff: Formatting code..."
uv run ruff format --config=pyproject.toml src tests experiments articles/paper/scripts

echo ""
echo "ruff: Checking code quality and style..."
uv run ruff check --config=pyproject.toml src tests experiments articles/paper/scripts

echo ""
echo "mypy: Checking type annotations..."
uv run mypy --config-file pyproject.toml --install-types --non-interactive src tests experiments

echo ""
echo "bash -n: Checking shell script syntax..."
git ls-files -z '*.sh' | xargs -0 -n1 bash -n

echo ""
echo "All linters passed."
