#!/usr/bin/env bash
set -euo pipefail
# Study D -- cost_transform sweep (objective shaping for tail robustness).
# GA @300x2000 on dense_p3998. The aggregate fitness is
# sqrt(mean_seeds(transform(cost)^2)), so the transform picks which moment of
# the per-seed cost distribution the optimizer minimizes. Deployed DV is raw,
# so the cross-transform comparison is clean.
# The CUBED cell is the project default = 02's ga_300
# (training_output/paper/optimizer_budget/ga_300) -- run 02 first.
# Reporting rule: quote p99 + CVaR95 as the tail metrics, NOT the sample max.

run() {  # $1=transform
  if [ -f "training_output/paper/cost_transform/$1/final_eval.parquet" ]; then echo "skip $1 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/dense_p3998_ga_transform_$1.toml" \
      --n-gen 2000 --n-pop 300 --output-dir "training_output/paper/cost_transform/$1" --sim-timeout 5 --from-scratch
}

run linear
run sqrt
run log
run squared
