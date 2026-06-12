#!/usr/bin/env bash
set -euo pipefail
# Architecture Pareto sweep -- all 6 families (dense/GRU/LSTM/Mamba/Transformer/
# Window) x ~4 parameter budgets, re-run under GA + the post-fix defaults so the
# Pareto figure shares the campaign regime (the old committed sweep was islands
# + log + pre-fix guidance).
# Uses param_sweep's own orchestration: configs + manifest already exist in
# configs/training/sweep/; deploys to canonical training_output/sweep_<arch>_p<N>.
# --from-scratch wipes stale pre-fix checkpoints per point. ~24 runs, the
# longest script in the campaign after 02.

uv run python -m aerocapture.training.param_sweep --train \
    --n-gen 2000 --n-pop 300 --sim-timeout 5 --from-scratch

# Re-score every point on the shared sweep-eval pool + render the Pareto SVGs.
uv run python -m aerocapture.training.param_sweep --eval --plot
uv run python -m aerocapture.training.param_sweep --eval --plot --out-tag floor --metric capture_rate
