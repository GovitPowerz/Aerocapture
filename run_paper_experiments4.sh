#!/usr/bin/env bash
set -euo pipefail

# Batch 4 -- QPSO column for the optimizer comparison
# (spec: docs/superpowers/specs/2026-06-10-qpso-optimizer-design.md).
# Canonical mbest QPSO (Sun/Feng/Xu 2004), alpha annealed 1.0 -> 0.5.
# Mirrors the batch-2/3 grid: small net @300; big net @60/@150/@300.
# @60 uses the default output dir (matches batch 2); @150/@300 use
# --output-dir (matches batch 3).

# ── Study A (small net, dense_p515): compute-matched n_pop=300 ──
uv run python -m aerocapture.training.train configs/training/paper/opt_qpso.toml    --n-gen 2000 --n-pop 300 --from-scratch

# ── Study A (big net, dense_p3998): budget scaling @60/@150/@300 ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 60  --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_qpso150 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_qpso.toml --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_qpso300 --from-scratch
