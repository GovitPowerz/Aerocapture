#!/usr/bin/env bash
set -euo pipefail

# Study C -- optimizer x seed-strategy on the big net (dense_p3998), @150/gen, n_gen=2000.
# Thesis: GA's advantage GROWS with training-environment non-stationarity
# (fixed -> rotating -> adaptive). If optimizers tie under FIXED (stationary) seeds
# but GA pulls ahead under rotating/adaptive, "GA is robust to the moving objective"
# is demonstrated, not asserted.
# Compute-matched @150/gen: singles n_pop=150, islands n_pop=50 (x3).
# Adaptive@150 already exists for GA/islands/PSO (paper_optbig_{ga,islands,pso}150);
# only CMA-ES adaptive@150 is added below to complete that column.

# ── CMA-ES adaptive@150 (completes the adaptive column; slow: O(n^2) covariance) ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_cmaes.toml --n-gen 2000 --n-pop 150 --seed-strategy adaptive --output-dir training_output/paper_seedC_cmaes_adaptive --from-scratch

# ── FIXED seeds (stationary environment) ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga.toml      --n-gen 2000 --n-pop 150 --seed-strategy fixed --output-dir training_output/paper_seedC_ga_fixed      --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_islands.toml --n-gen 2000 --n-pop 50  --seed-strategy fixed --output-dir training_output/paper_seedC_islands_fixed --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_cmaes.toml   --n-gen 2000 --n-pop 150 --seed-strategy fixed --output-dir training_output/paper_seedC_cmaes_fixed   --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_pso.toml     --n-gen 2000 --n-pop 150 --seed-strategy fixed --output-dir training_output/paper_seedC_pso_fixed     --from-scratch

# ── ROTATING seeds (fresh random each gen -- maximally non-stationary) ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga.toml      --n-gen 2000 --n-pop 150 --seed-strategy rotating --output-dir training_output/paper_seedC_ga_rotating      --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_islands.toml --n-gen 2000 --n-pop 50  --seed-strategy rotating --output-dir training_output/paper_seedC_islands_rotating --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_cmaes.toml   --n-gen 2000 --n-pop 150 --seed-strategy rotating --output-dir training_output/paper_seedC_cmaes_rotating   --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_pso.toml     --n-gen 2000 --n-pop 150 --seed-strategy rotating --output-dir training_output/paper_seedC_pso_rotating     --from-scratch
