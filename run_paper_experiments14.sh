#!/usr/bin/env bash
set -euo pipefail

# Training-run variance: 3x seed-repeats of the two headline cells (GA@300 and
# islands@300 on dense_p3998, post-fix defaults). Repeat #1 of each is exp10's
# paper_pf_ga_300 / paper_pf_islands_300 (--seed 1 default) -- run exp10 first.
# --seed varies the trainer RNG (initial population, curator/rotation draws);
# the pymoo operator stream is unseeded, so even same-seed repeats differ, but
# varying --seed avoids the correlated-repeat underestimate of sigma_run.
# Purpose: the paper's 1-3 m/s cross-optimizer gaps (GA vs islands, GA@150 vs
# islands@300) are quotable only with a training-run error bar; quote headline
# cells as mean +/- range over the 3 repeats.

run() {  # $1=config $2=n_pop $3=seed $4=dir
  if [ -f "training_output/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" --n-gen 2000 --n-pop "$2" \
      --seed "$3" --output-dir "training_output/$4" --sim-timeout 5 --from-scratch
}

run optbig_ga      300 2 paper_pf_ga_300_s2
run optbig_ga      300 3 paper_pf_ga_300_s3
run optbig_islands 100 2 paper_pf_islands_300_s2
run optbig_islands 100 3 paper_pf_islands_300_s3
