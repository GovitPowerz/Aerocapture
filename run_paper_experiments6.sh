#!/usr/bin/env bash
set -euo pipefail

# cost_transform sweep -- GA on the big net (dense_p3998) @300/gen, n_gen=2000.
# The aggregate fitness is sqrt(mean_seeds(transform(cost)^2)), so cost_transform
# selects which MOMENT of the per-seed cost distribution the optimizer minimizes:
#   log   -> bulk/median (tail compressed)
#   sqrt  -> softened mean
#   linear-> mean (L2)
#   squared -> upper tail (~4th moment)
#   cubed -> extreme tail / worst seeds (~6th moment)
# Hypothesis: cubed wins deployed p95/max (the robustness metric the paper leads on),
# possibly at a small cost on mean. 'log' is the pipeline default ->
# reuse training_output/paper_optbig_ga300 as the log point (do NOT re-run).

uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_log.toml  --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_linear.toml  --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_sqrt.toml    --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_squared.toml --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga_cubed.toml   --n-gen 2000 --n-pop 300 --from-scratch
