#!/usr/bin/env bash
set -euo pipefail
# Training-run variance (sigma_run): 3x seed-repeats of the decisive cells.
# Repeat #1 of each cell comes from 01/02/03 (--seed 1 default) -- run those
# first. --seed varies the trainer RNG (init population, curator/rotation
# draws); the eval pool is identical across repeats, so deployed spread is a
# clean sigma_run estimate. Quote repeated cells as mean +/- range over
# {s1,s2,s3}, and POOL the per-dimension ranges into the campaign-wide
# sigma_run that calibrates every N=1 cell comparison.
# Only GA / CMA-ES / islands are repeated at 26p/515p (the tight-gap cells);
# PSO/DE/QPSO are mid-pack (gap >> sigma_run). At 3998p, GA + islands repeat
# (CMA-ES is ~5 sigma worst there AND O(n^2)-slow; stays N=1).

P="training_output/paper/seed_repeats"
run_cfg() {  # $1=config-stem $2=n_pop $3=seed $4=cell
  if [ -f "$P/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/paper/$1.toml" --n-gen 2000 --n-pop "$2" \
      --seed "$3" --output-dir "$P/$4" --sim-timeout 5 --from-scratch
}
run_ftc() {  # $1=algorithm $2=n_pop $3=seed $4=cell
  if [ -f "$P/$4/final_eval.parquet" ]; then echo "skip $4 (done)"; return 0; fi
  uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml \
      --algorithm "$1" --n-gen 2000 --n-pop "$2" --seed "$3" \
      --output-dir "$P/$4" --sim-timeout 5 --from-scratch
}

for S in 2 3; do
  # ── dense_p3998 @300 (s1: 02's ga_300 / islands_300) ──
  run_cfg dense_p3998_ga      300 "$S" "ga_300_s$S"
  run_cfg dense_p3998_islands 100 "$S" "islands_300_s$S"

  # ── FTC 26 params (s1: training_output/ftc + 03's ftc_cmaes / ftc_islands) ──
  run_ftc ga      300 "$S" "ftc_ga_s$S"
  run_ftc cma_es  300 "$S" "ftc_cmaes_s$S"
  run_ftc islands 100 "$S" "ftc_islands_s$S"

  # ── dense_p515 (s1: 03's dense_p515_{ga,cmaes,islands}) ──
  run_cfg dense_p515_ga      300 "$S" "small_ga_s$S"
  run_cfg dense_p515_cmaes   300 "$S" "small_cmaes_s$S"
  run_cfg dense_p515_islands 100 "$S" "small_islands_s$S"
done
