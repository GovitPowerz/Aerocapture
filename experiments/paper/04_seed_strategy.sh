#!/usr/bin/env bash
set -euo pipefail
# Study C -- optimizer x seed-strategy on dense_p3998 @150/gen, n_gen=2000.
# Thesis: GA's advantage GROWS with training-environment non-stationarity
# (fixed -> rotating -> adaptive). The ADAPTIVE column is 02's @150 row
# (training_output/paper/optimizer_budget/{ga,islands,pso,cmaes}_150) -- run
# 02 first; this script adds the FIXED and ROTATING cells.
# Compute note: "@150/gen" counts selection-driving evals only -- rotating
# re-evals parents + validates ~every gen, fixed almost never (actual sims/gen
# differ ~2.4x across the strategy axis). Report actual sims per cell from the
# JSONL and phrase the headline as the WITHIN-strategy optimizer ranking.

run() {  # $1=config-stem $2=n_pop $3=strategy $4=cell
  if [ -f "training_output/paper/seed_strategy/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" --n-gen 2000 --n-pop "$2" \
      --seed-strategy "$3" --output-dir "training_output/paper/seed_strategy/$4" --sim-timeout 5 --from-scratch
}

# ── FIXED seeds (stationary environment) ──
run dense_p3998_ga      150 fixed ga_fixed
run dense_p3998_islands 50  fixed islands_fixed
run dense_p3998_cmaes   150 fixed cmaes_fixed
run dense_p3998_pso     150 fixed pso_fixed

# ── ROTATING seeds (fresh random each gen -- maximally non-stationary) ──
run dense_p3998_ga      150 rotating ga_rotating
run dense_p3998_islands 50  rotating islands_rotating
run dense_p3998_cmaes   150 rotating cmaes_rotating
run dense_p3998_pso     150 rotating pso_rotating
