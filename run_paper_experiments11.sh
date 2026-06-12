#!/usr/bin/env bash
set -euo pipefail

# Study F -- training_n_sims (sims per individual per generation) sweep {2,5,10,20,100}.
# GA on dense_p3998, post-fix defaults (cost_transform=cubed, bucket=max, adaptive).
# Sample-efficiency companion to Study C: because adaptive/rotating seeds diversify
# scenarios OVER generations, few sims/gen suffice -- but there is a noise floor below
# which the per-gen fitness estimate is too noisy for GA's rank-based selection.
# Two views:
#   (A) FIXED n_gen=2000  -> the selection-noise floor (the user's "5 too low, 10 enough").
#       Compute differs up to 50x; n_sims=100 is the slow cell.
#   (B) FIXED total compute (n_sims * n_gen = 20000, anchored at n_sims=10/n_gen=2000)
#       -> the optimal ALLOCATION; gives low-n_sims cells proportionally more generations,
#       testing whether they are genuinely too noisy or merely under-fed.
# n_sims=10 is the shared cell (A and B coincide at n_gen=2000) -- run once (paper_nsimG_10).
# validation_n_sims (1000) is unchanged across cells, so the deployed comparison is fair.

GA="configs/training/paper/optbig_ga.toml"
run() { uv run python -m aerocapture.training.train "$GA" --n-pop 300 --training-n-sims "$1" --n-gen "$2" --output-dir "training_output/$3" --from-scratch; }

# ── (A) fixed n_gen = 2000 ──
run 2   2000 paper_nsimG_2
run 5   2000 paper_nsimG_5
run 10  2000 paper_nsimG_10      # shared cell (== fixed-compute n_sims=10)
run 20  2000 paper_nsimG_20
run 100 2000 paper_nsimG_100     # slow: 50x the n_sims=2 compute

# ── (B) fixed total compute (n_sims * n_gen = 20000); n_sims=10 reuses paper_nsimG_10 ──
run 2   10000 paper_nsimC_2
run 5   4000  paper_nsimC_5
run 20  1000  paper_nsimC_20
run 100 200   paper_nsimC_100
