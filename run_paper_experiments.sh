#!/usr/bin/env bash
set -euo pipefail

# Controlled experiments for the aerocapture NN paper.
# Plan: docs/superpowers/plans/2026-06-08-aerocapture-nn-article.md
# Compute-fairness: single optimizers run n_pop=300 to match islands (100 x 3).
# If wall-clock is too long, scale --n-gen DOWN UNIFORMLY across the five pymoo
# lines (keep them equal) -- the comparison stays fair.

# ── Study A: optimizer comparison on dense_p515 ──
# uv run python -m aerocapture.training.train configs/training/paper/opt_pso.toml       --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_ga.toml        --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_de.toml        --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_cmaes.toml     --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_warmstart.toml --n-gen 2000 --n-pop 100 --from-scratch

# ── Study A: RL (PPO) on the dense architecture ──
# uv run python -m aerocapture.training.rl.train configs/training/paper/opt_rl.toml --algorithm ppo --total-steps 5000000

# ── Study B: output parameterization on dense + islands ──
uv run python -m aerocapture.training.train configs/training/paper/out_scaledpi.toml  --n-gen 2000 --n-pop 100 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/out_delta.toml     --n-gen 2000 --n-pop 100 --from-scratch

# ── EqGlide deploy/eval to populate the classical table ──
# (best_params.json exists from a short 5-gen run; retrain first if you want a fairer EqGlide number.)
# uv run python -m aerocapture.training.report training_output/equilibrium_glide/ --toml configs/training/msr_aller_eqglide_train.toml
