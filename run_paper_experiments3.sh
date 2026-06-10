#!/usr/bin/env bash
set -euo pipefail

# Batch 3 -- optimizer BUDGET-SCALING on the big net (dense_p3998), n_gen=2000.
# Question: does islands close the gap at higher budgets, or does a single-pop
# GA/PSO/DE match/beat it at EVERY budget? (At @60/gen from batch 2, GA beat islands.)
# Compute-matched per budget (islands n_pop is PER ISLAND x3):
#   @150/gen: singles n_pop=150, islands n_pop=50
#   @300/gen: singles n_pop=300, islands n_pop=100  -- islands@300 already exists
#             as training_output/sweep_dense_p3998 (reuse, do NOT re-run).
# Configs are reused via --n-pop + --output-dir (no new config files needed).

# ── @150 evals/gen ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga.toml      --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_ga150      --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_de.toml      --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_de150      --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_pso.toml     --n-gen 2000 --n-pop 150 --output-dir training_output/paper_optbig_pso150     --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_islands.toml --n-gen 2000 --n-pop 50  --output-dir training_output/paper_optbig_islands150 --from-scratch

# ── @300 evals/gen (islands@300 = sweep_dense_p3998, already done) ──
uv run python -m aerocapture.training.train configs/training/paper/optbig_ga.toml  --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_ga300  --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_de.toml  --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_de300  --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/optbig_pso.toml --n-gen 2000 --n-pop 300 --output-dir training_output/paper_optbig_pso300 --from-scratch
