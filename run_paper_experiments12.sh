#!/usr/bin/env bash
set -euo pipefail

# Optimizer x DIMENSIONALITY: the optimizer benchmark repeated at ~26 params
# (FTC classical chromosome: gains + nav/lateral/exit/thermal/shaping) and 515
# params (dense_p515), completing the axis whose ~4000-param end is exp10's @300
# column. Tests whether the optimizer ranking flips with chromosome width
# (CMA-ES's sweet spot is low-dim; the GA-wins result lives at high-dim).
# All cells POST-FIX defaults (cubed + max bucket + adaptive), @300 evals/gen,
# n_gen=2000 -- matching exp10's @300 column exactly.
# GA reuse cells (do NOT duplicate): FTC GA = training_output/ftc (exp2 post-fix
# classical retrain, same GA @300x2000 budget -- run/verify exp2 FIRST);
# dense_p3998 column = exp10's paper_pf_*_300.
# Also reruns Study B (output parameterization) under GA post-fix: control =
# paper_pf_small_ga (same net, optimizer, defaults).

run_ftc() {  # $1=algorithm $2=n_pop $3=dir
  if [ -f "training_output/$3/final_eval.parquet" ]; then echo "skip $3 (done)"; return 0; fi
  uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml \
      --algorithm "$1" --n-gen 2000 --n-pop "$2" --output-dir "training_output/$3" \
      --sim-timeout 5 --from-scratch
}
run_paper() {  # $1=config $2=n_pop $3=dir
  if [ -f "training_output/$3/final_eval.parquet" ]; then echo "skip $3 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" \
      --n-gen 2000 --n-pop "$2" --output-dir "training_output/$3" --sim-timeout 5 --from-scratch
}

# ── FTC, 26-param chromosome (GA cell = training_output/ftc from exp2) ──
run_ftc pso     300 paper_ftcopt_pso
run_ftc de      300 paper_ftcopt_de
run_ftc cma_es  300 paper_ftcopt_cmaes
run_ftc qpso    300 paper_ftcopt_qpso
run_ftc islands 100 paper_ftcopt_islands

# ── dense_p515, 515 params (POST-FIX rerun; supersedes the stale paper_opt_* table) ──
run_paper opt_ga      300 paper_pf_small_ga
run_paper opt_pso     300 paper_pf_small_pso
run_paper opt_de      300 paper_pf_small_de
run_paper opt_cmaes   300 paper_pf_small_cmaes
run_paper opt_qpso    300 paper_pf_small_qpso
run_paper opt_islands 100 paper_pf_small_islands

# ── Study B rerun: output parameterization under GA post-fix (atan2 control = paper_pf_small_ga) ──
run_paper out_scaledpi 300 paper_pf_out_scaledpi
run_paper out_delta    300 paper_pf_out_delta
