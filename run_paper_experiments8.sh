#!/usr/bin/env bash
set -euo pipefail

# Bucket-representative sub-study (within Study C) -- GA on the big net
# (dense_p3998), adaptive seeds, @300/gen, n_gen=2000.
# The adaptive CDF-curation picks ONE representative per cost-quantile bin.
# This sweeps WHICH representative (holding seed count + quantile coverage fixed):
#   min    -> easiest seed of each bin
#   middle -> median-cost seed of each bin
#   max    -> hardest seed of each bin  ("does the worst case per quantile help?")
#   random -> current default = reuse training_output/paper_optbig_ga300 (do NOT re-run)
# Cleaner than the trim sweep: isolates difficulty-within-bin, not coverage.

uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_bucket_min.toml    --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_bucket_middle.toml --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_bucket_max.toml    --n-gen 2000 --n-pop 300 --from-scratch
