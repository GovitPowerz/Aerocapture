#!/usr/bin/env bash
set -euo pipefail
# Optimizer x DIMENSIONALITY + Study B (run from the repo root; FTC GA cell
# reuses training_output/ftc from 01_classical_baselines -- run 01 first).
# Repeats the optimizer benchmark at ~26 params (FTC classical chromosome) and
# 515 params (dense_p515); the ~4000-param end is 02's @300 column. Tests
# whether the optimizer ranking flips with chromosome width (CMA-ES sweet spot
# at low dim; the GA-wins result lives at high dim). All cells @300 evals/gen,
# n_gen=2000, post-fix defaults -- matching 02's @300 column exactly.
# Study B (output parameterization) rides along: scaledpi/delta vs the atan2
# control dense_p515_ga.
# CMA-ES note for the paper: popsize is evals/gen-matched (300), not the
# CMA-canonical 4+3ln(n); footnote this in the methods section.

run_ftc() {  # $1=algorithm $2=n_pop $3=cell
  if [ -f "training_output/paper/optimizer_dimensionality/$3/final_eval.parquet" ]; then echo "skip $3 (done)"; return 0; fi
  uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml \
      --algorithm "$1" --n-gen 2000 --n-pop "$2" \
      --output-dir "training_output/paper/optimizer_dimensionality/$3" --sim-timeout 5 --from-scratch
}
run_cfg() {  # $1=config-stem $2=n_pop $3=study $4=cell
  if [ -f "training_output/paper/$3/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" \
      --n-gen 2000 --n-pop "$2" --output-dir "training_output/paper/$3/$4" --sim-timeout 5 --from-scratch
}

# ── FTC, 26-param chromosome (GA cell = canonical training_output/ftc from 01) ──
run_ftc pso     300 ftc_pso
run_ftc de      300 ftc_de
run_ftc cma_es  300 ftc_cmaes
run_ftc qpso    300 ftc_qpso
run_ftc islands 100 ftc_islands

# ── dense_p515, 515 params ──
run_cfg dense_p515_ga      300 optimizer_dimensionality dense_p515_ga
run_cfg dense_p515_pso     300 optimizer_dimensionality dense_p515_pso
run_cfg dense_p515_de      300 optimizer_dimensionality dense_p515_de
run_cfg dense_p515_cmaes   300 optimizer_dimensionality dense_p515_cmaes
run_cfg dense_p515_qpso    300 optimizer_dimensionality dense_p515_qpso
run_cfg dense_p515_islands 100 optimizer_dimensionality dense_p515_islands

# ── Study B: output parameterization (control = dense_p515_ga above) ──
run_cfg outparam_scaledpi 300 output_param scaledpi
run_cfg outparam_delta    300 output_param delta
