#!/usr/bin/env bash
set -euo pipefail

# N=3 seed-repeats for the DECISIVE optimizer x dimensionality cells (companion to
# exp12 + exp14), so the "best optimizer flips with chromosome width" claim carries
# a training-run error bar exactly where the gaps are tight enough for sigma_run to
# matter: GA / CMA-ES / islands at FTC (26 params) and the small dense net (515).
#
# Only these three optimizers are repeated -- PSO/DE/QPSO sit clearly mid-pack
# (gap >> sigma_run) and stay N=1 in exp12. The big-net (3998) GA/islands repeats
# live in exp14; CMA-ES@3998 is worst AND O(n^2)-slow, so it stays N=1.
#
# Seed 1 of each cell already exists from exp12/exp2:
#   FTC   GA -> training_output/ftc           (exp2, GA @300x2000)
#   FTC   CMA-ES/islands -> paper_ftcopt_{cmaes,islands}   (exp12)
#   small GA -> paper_pf_small_ga             (exp12)
#   small CMA-ES/islands -> paper_pf_small_{cmaes,islands} (exp12)
# --seed varies ONLY the trainer RNG (init pop + curation/rotation draws);
# monte_carlo.seed is unchanged, so the validation/final-eval pool is IDENTICAL
# across the three repeats -> deployed mean +/- range is a clean sigma_run estimate.
# Quote each repeated cell as mean +/- range over {s1, s2, s3}.

run_ftc() {  # $1=alg $2=n_pop $3=seed $4=dir
  if [ -f "training_output/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml \
      --algorithm "$1" --n-gen 2000 --n-pop "$2" --seed "$3" --output-dir "training_output/$4" --sim-timeout 5 --from-scratch
}
run_small() {  # $1=config $2=n_pop $3=seed $4=dir
  if [ -f "training_output/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" \
      --n-gen 2000 --n-pop "$2" --seed "$3" --output-dir "training_output/$4" --sim-timeout 5 --from-scratch
}

for S in 2 3; do
  # ── FTC (26 params) ── s1: GA=training_output/ftc, CMA-ES/islands=exp12 ──
  run_ftc ga      300 "$S" "paper_ftcopt_ga_s$S"
  run_ftc cma_es  300 "$S" "paper_ftcopt_cmaes_s$S"
  run_ftc islands 100 "$S" "paper_ftcopt_islands_s$S"

  # ── small dense (515 params) ── s1: GA=paper_pf_small_ga, CMA-ES/islands=exp12 ──
  run_small opt_ga      300 "$S" "paper_pf_small_ga_s$S"
  run_small opt_cmaes   300 "$S" "paper_pf_small_cmaes_s$S"
  run_small opt_islands 100 "$S" "paper_pf_small_islands_s$S"
done
