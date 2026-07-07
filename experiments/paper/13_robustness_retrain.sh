#!/usr/bin/env bash
set -euo pipefail
# exp-13 -- robustness RETRAIN (NOT part of the numbered campaign reproduction).
#
# Question: the paper reports that the MEDIUM-trained Mamba_962 loses off-nominal
# robustness to the analytic joint-FTC (capture drop 9.9% vs 5.5%, CVaR95 inflation
# +402 vs +197 on the 9M stress pool), and names "widening the NN training regime"
# as future work. This experiment runs that future work: it trains BOTH the joint-FTC
# classical AND the Mamba_962 headline architecture ON the high-dispersion regime
# (atmosphere / density_perturbation / navigation / nav_filter = high -- the SAME
# domains robustness_stress.py bumps), then evaluates every policy (medium- AND
# high-trained) on the reserved 9M stress pool. If the high-trained NN closes the
# gap, the caveat is "just retrain"; if it still loses, the gap is structural.
#
# Budgets are DIRECTIONAL by default (this is a "small experiment"). The Mamba
# plateaus only after ~10-15k generations (fig_plateau), so a 3000-gen run UNDER-
# TRAINS it -- treat a NN loss at the default budget as INCONCLUSIVE and rerun with
# NGEN_MAMBA scaled up. FTC-joint (~27 genes incl. the ref_bank) converges fast, so
# its default is the campaign budget (2000 x 300).
#
#   # directional smoke (default):
#   ./experiments/paper/13_robustness_retrain.sh
#   # conclusive (headline depth for the Mamba):
#   NGEN_MAMBA=15000 NPOP_MAMBA=512 ./experiments/paper/13_robustness_retrain.sh
#
# Idempotent (skip-if-final_eval.parquet per cell; delete a cell dir to rerun).
# The Mamba is an NN and FTC-joint generates per-individual references into its own
# output dir, so neither touches training_output/mars/ -- no ref-tracking conflict.

NGEN_MAMBA=${NGEN_MAMBA:-3000}
NPOP_MAMBA=${NPOP_MAMBA:-256}
NGEN_FTC=${NGEN_FTC:-2000}
NPOP_FTC=${NPOP_FTC:-300}
NSIMS_EVAL=${NSIMS_EVAL:-1000}

train() {  # $1=config-stem  $2=cell  $3=n_gen  $4=n_pop
  if [ -f "training_output/paper/robustness_retrain/$2/final_eval.parquet" ]; then
    echo "skip $2 (done)"; return 0
  fi
  uv run python -m aerocapture.training.train \
      "configs/training/paper/robustness_retrain/$1.toml" \
      --n-gen "$3" --n-pop "$4" \
      --output-dir "training_output/paper/robustness_retrain/$2" \
      --sim-timeout 5 --from-scratch
}

# FTC-joint first (fast); then the Mamba (the long pole).
train ftc_joint_medium_wind   ftc_joint_medium_wind   "$NGEN_FTC"   "$NPOP_FTC"
train mamba_p962_medium_wind  mamba_p962_medium_wind  "$NGEN_MAMBA" "$NPOP_MAMBA"

# Evaluate medium- AND high-trained policies on the reserved 9M stress pool.
uv run python articles/paper/scripts/robustness_retrain_eval.py --n-sims "$NSIMS_EVAL"
