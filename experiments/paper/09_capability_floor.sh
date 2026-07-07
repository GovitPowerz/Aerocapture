#!/usr/bin/env bash
set -euo pipefail
# Capability-collapse sweep: dense nets BELOW 500 params on the same 17-input
# atan2 pipeline, to locate where the NN stops being able to guide.
# GA @300x2000, post-fix defaults -- extends the post-fix Pareto axis anchored
# by 03's dense_p515_ga and 02's ga_300 (3998 params). Watch capture rate
# first, DV second: collapse shows as capture < 100% and/or a p99 blow-up.
# Output dirs are param_sweep's canonical sweep_dense_p<N> so
# `param_sweep --eval/--plot --out-tag floor` (manifest_floor.json) works.

run() {  # $1=config-stem
  if [ -f "training_output/sweep_$1/final_eval.parquet" ]; then echo "skip $1 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/sweep/$1.toml" \
      --n-gen 2000 --n-pop 300 --output-dir "training_output/sweep_$1" --sim-timeout 5 --from-scratch
}

run dense_p102   # 17->4->4->2
run dense_p201   # 17->8->5->2
run dense_p298   # 17->11->7->2
run dense_p416   # 17->15->8->2
