#!/usr/bin/env bash
set -euo pipefail

# Study C -- optimizer x seed-strategy on the big net (dense_p3998), @150/gen, n_gen=2000.
# Thesis: GA's advantage GROWS with training-environment non-stationarity
# (fixed -> rotating -> adaptive). If optimizers tie under FIXED (stationary) seeds
# but GA pulls ahead under rotating/adaptive, "GA is robust to the moving objective"
# is demonstrated, not asserted.
#
# The ADAPTIVE column comes from exp10's POST-FIX rebuild:
#   paper_pf_{ga,islands,pso,cmaes}_150    <- run ./run_paper_experiments10.sh FIRST.
# The old paper_optbig_{ga,islands,pso}150 dirs are PRE-FIX (log + random bucket +
# old guidance) and must NOT be mixed into this matrix.
#
# Compute note for the paper: "@150/gen" counts selection-driving evals only.
# Rotating re-evals the parent population every gen and validates ~every gen;
# fixed almost never does either -- actual sims/gen differ ~2.4x across the
# strategy axis. Report actual sims per cell (from the JSONL) next to the matrix
# and phrase the headline result as the WITHIN-strategy optimizer ranking.

run() {  # $1=config $2=n_pop $3=strategy $4=dir
  if [ -f "training_output/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" --n-gen 2000 --n-pop "$2" \
      --seed-strategy "$3" --output-dir "training_output/$4" --sim-timeout 5 --from-scratch
}

# ── FIXED seeds (stationary environment) ──
run optbig_ga      150 fixed paper_seedC_ga_fixed
run optbig_islands 50  fixed paper_seedC_islands_fixed
run optbig_cmaes   150 fixed paper_seedC_cmaes_fixed
run optbig_pso     150 fixed paper_seedC_pso_fixed

# ── ROTATING seeds (fresh random each gen -- maximally non-stationary) ──
run optbig_ga      150 rotating paper_seedC_ga_rotating
run optbig_islands 50  rotating paper_seedC_islands_rotating
run optbig_cmaes   150 rotating paper_seedC_cmaes_rotating
run optbig_pso     150 rotating paper_seedC_pso_rotating
