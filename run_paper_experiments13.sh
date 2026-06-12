#!/usr/bin/env bash
set -euo pipefail

# Capability-collapse sweep: dense nets BELOW 500 params on the same 17-input
# atan2 pipeline (sweep family), to locate where the NN stops being able to
# guide. GA @300x2000, post-fix defaults -- directly extends the post-fix Pareto
# axis anchored by paper_pf_small_ga (515, exp12) and paper_pf_ga_300 (3998, exp10).
# Watch capture rate first, DV second: collapse should show as capture < 100%
# and/or a p95/max blow-up.
#
# NOTE: training_output/sweep_dense_p190 (trained 2026-06-12) is MISNAMED --
# its architecture (17->16->8->2) is 442 params, not 190 (the config has since
# been deleted). Do not let a name-derived param count (Pareto figure regex)
# plot that dir at x=190.
#
# The same points are also wired into param_sweep's floor manifest
# (configs/training/sweep/manifest_floor.json, incl. the 515/3998 anchors at
# paper_pf_small_ga / paper_pf_ga_300), so --eval/--plot --out-tag floor can
# score and render the full capability curve once the cells exist.

run() {  # $1=config $2=dir
  if [ -f "training_output/$2/final_eval.parquet" ]; then echo "skip $2 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/sweep/$1.toml" \
      --n-gen 2000 --n-pop 300 --output-dir "training_output/$2" --sim-timeout 5 --from-scratch
}

run dense_p416 sweep_dense_p416   # 17->15->8->2
run dense_p298 sweep_dense_p298   # 17->11->7->2
run dense_p201 sweep_dense_p201   # 17->8->5->2
run dense_p102 sweep_dense_p102   # 17->4->4->2
