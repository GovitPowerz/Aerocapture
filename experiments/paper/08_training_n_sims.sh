#!/usr/bin/env bash
set -euo pipefail
# Study F -- training_n_sims (sims per individual per generation) sweep.
# GA @300 on dense_p3998, two DECOUPLED views:
#  (A) NOISE FLOOR -- seed_strategy=ROTATING, fixed n_gen=2000. Rotating draws
#      n_sims fresh iid seeds per gen, so n_sims is a PURE estimate-width knob.
#      (Under adaptive, n_bins = n_sims + bucket=max also reshape seed-list
#      difficulty -- a confound this view deliberately avoids.)
#  (B) ALLOCATION under the production ADAPTIVE pipeline, n_sims*n_gen = 20000.
#      NOT strictly compute-matched: curation + validation overhead scales with
#      n_gen, not n_sims -- report ACTUAL total sims per cell from the JSONL.
#      The adaptive n_sims=10 anchor is 02's ga_300 -- run 02 first.

run() {  # $1=n_sims $2=n_gen $3=strategy $4=cell
  if [ -f "training_output/paper/training_n_sims/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train configs/training/paper/dense_p3998_ga.toml \
      --n-pop 300 --training-n-sims "$1" --n-gen "$2" --seed-strategy "$3" \
      --output-dir "training_output/paper/training_n_sims/$4" --sim-timeout 5 --from-scratch
}

# ── (A) noise floor: rotating, fixed n_gen = 2000 ──
run 2   2000 rotating rotating_2
run 5   2000 rotating rotating_5
run 10  2000 rotating rotating_10
run 20  2000 rotating rotating_20
run 100 2000 rotating rotating_100   # slow cell: 50x the n_sims=2 training sims

# ── (B) allocation: adaptive, n_sims * n_gen = 20000 (anchor = 02's ga_300) ──
run 2   10000 adaptive adaptive_2
run 5   4000  adaptive adaptive_5
run 20  1000  adaptive adaptive_20
run 100 200   adaptive adaptive_100
