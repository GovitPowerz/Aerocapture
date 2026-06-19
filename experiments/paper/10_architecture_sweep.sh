#!/usr/bin/env bash
set -euo pipefail
# Architecture Pareto sweep -- all 6 families (dense/GRU/LSTM/Mamba/Transformer/
# Window) x ~4 parameter budgets, re-run under GA + the post-fix defaults so the
# Pareto figure shares the campaign regime (the old committed sweep was islands
# + log + pre-fix guidance). adaptive / bucket=max / cost_transform=cubed are
# inherited from the sweep TOMLs; this script only sets the allocation.
# Uses param_sweep's own orchestration: configs + manifest already exist in
# configs/training/sweep/; deploys to canonical training_output/sweep_<arch>_p<N>.
# --from-scratch wipes stale pre-fix checkpoints per point. ~24 runs, the
# longest script in the campaign after 02.
#
# Allocation (the campaign learning): n_sims=5 is the efficiency sweet spot
# (n_sims=2 is the BEST allocation but ~2x the wall-time -- reserved for the
# deployed headline, prohibitive across 24 cells). n_gen=8000 plateaus the
# ~500-2000 cells; the ~4000-param cells need more and will be slightly
# under-trained -- footnote it (consistent with "bigger nets are harder for GA",
# the 515-vs-972 result). n_pop=300 holds for all cells: Study A showed GA@60
# COLLAPSES at ~4000 params, so the big cells need the wide population.

uv run python -m aerocapture.training.param_sweep --train \
    --training-n-sims 5 --n-gen 8000 --n-pop 300 --sim-timeout 5 --from-scratch

# Re-score every point on the shared sweep-eval pool + render the Pareto SVGs.
uv run python -m aerocapture.training.param_sweep --eval --plot
uv run python -m aerocapture.training.param_sweep --eval --plot --out-tag floor --metric capture_rate
