#!/usr/bin/env bash
set -euo pipefail
# Study C-sub -- curated-seed shaping under GA + adaptive @300x2000 (dense_p3998).
# Two knobs on the SeedCurator:
#   bucket_selection: which difficulty WITHIN each cost-CDF bin represents it
#     (max = hardest = project default = 02's ga_300 cell -- run 02 first).
#   trim_fraction: drop the extreme cost deciles before binning (the refuted
#     pre-fix hypothesis, re-run here under the post-fix regime so the negative
#     result is quotable without a regime footnote).

run() {  # $1=config-stem $2=cell
  if [ -f "training_output/paper/curation_shaping/$2/final_eval.parquet" ]; then echo "skip $2 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" \
      --n-gen 2000 --n-pop 300 --output-dir "training_output/paper/curation_shaping/$2" --sim-timeout 5 --from-scratch
}

run dense_p3998_ga_bucket_min    bucket_min
run dense_p3998_ga_bucket_middle bucket_middle
run dense_p3998_ga_bucket_random bucket_random
run dense_p3998_ga_trim10        trim_10
run dense_p3998_ga_trim20        trim_20
