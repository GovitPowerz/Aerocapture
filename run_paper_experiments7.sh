#!/usr/bin/env bash
set -euo pipefail

# Curation-trim sub-study (within Study C) -- GA on the big net (dense_p3998),
# adaptive seeds, @300/gen, n_gen=2000.
# Tests whether trimming the non-discriminative extremes from the adaptive
# CDF-curation improves convergence:
#   easiest seeds -> no between-individual signal; hardest seeds -> un-improvable
#   dispersion outliers that destabilize the moving objective.
# trim=0.0 (full range, current behavior) = reuse training_output/paper_optbig_ga300.

uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_trim10.toml --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_trim20.toml --n-gen 2000 --n-pop 300 --from-scratch
