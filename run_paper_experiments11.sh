#!/usr/bin/env bash
set -euo pipefail

# Study F -- training_n_sims (sims per individual per generation) sweep {2,5,10,20,100}.
# GA @300 on dense_p3998, post-fix defaults (cubed + max bucket). Two DECOUPLED
# views (the original A/B design conflated the seed curator with the noise floor
# and mislabeled view B as compute-matched):
#
#  (A) NOISE FLOOR -- seed_strategy=ROTATING, fixed n_gen=2000.
#      Rotating draws training_n_sims fresh iid seeds each generation, so n_sims
#      is a PURE fitness-estimate-width knob. Under ADAPTIVE this is confounded:
#      n_bins = training_n_sims and bucket=max reshape the difficulty composition
#      of the seed list as n_sims changes (at n_sims=2 the list is ~{median-bin
#      max, global max} of a 1000-seed probe -- a tail-only curriculum, not just
#      a noisier estimate).
#
#  (B) ALLOCATION under the production ADAPTIVE pipeline, n_sims * n_gen = 20000.
#      NOT strictly compute-matched: curation (~1000 sims/event, every <=2 gens),
#      the parent re-eval on seed change (n_pop x n_sims), and the validation
#      gate (~0.6 x 1000 sims/gen, measured) scale with n_gen, NOT with n_sims --
#      the low-n_sims/long-n_gen cells get up to ~2x more actual sims and ~50x
#      more validation-promotion attempts. Report ACTUAL total sims per cell
#      (from the JSONL: training = n_pop*n_sims*n_gen; validations = records
#      with a "validation" key x 1000; curations = distinct last_curation_gen
#      values x top_k x 1000) alongside the nominal budget.
#      The n_sims=10 anchor cell IS paper_pf_ga_300 (exp10) -- run exp10 first.

GA="configs/training/paper/optbig_ga.toml"
run() {  # $1=n_sims $2=n_gen $3=strategy $4=dir
  if [ -f "training_output/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "$GA" --n-pop 300 --training-n-sims "$1" --n-gen "$2" \
      --seed-strategy "$3" --output-dir "training_output/$4" --sim-timeout 5 --from-scratch
}

# ── (A) noise floor: rotating, fixed n_gen = 2000 ──
run 2   2000 rotating paper_nsimR_2
run 5   2000 rotating paper_nsimR_5
run 10  2000 rotating paper_nsimR_10
run 20  2000 rotating paper_nsimR_20
run 100 2000 rotating paper_nsimR_100    # slow cell: 50x the n_sims=2 training sims

# ── (B) allocation: adaptive, n_sims * n_gen = 20000 ──
# n_sims=10 anchor = paper_pf_ga_300 (exp10); do not duplicate it here.
run 2   10000 adaptive paper_nsimC_2
run 5   4000  adaptive paper_nsimC_5
run 20  1000  adaptive paper_nsimC_20
run 100 200   adaptive paper_nsimC_100
