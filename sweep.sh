#!/usr/bin/env bash
set -euo pipefail

uv run python -m aerocapture.training.train configs/training/sweep/window_p609.toml --n-gen 2000 --n-pop 100 --algorithm islands --from-scratch
uv run python -m aerocapture.training.train configs/training/sweep/window_p1027.toml --n-gen 2000 --n-pop 100 --algorithm islands --from-scratch
uv run python -m aerocapture.training.train configs/training/sweep/window_p2036.toml --n-gen 2000 --n-pop 100 --algorithm islands --from-scratch
uv run python -m aerocapture.training.train configs/training/sweep/window_p4025.toml --n-gen 2000 --n-pop 100 --algorithm islands --from-scratch
