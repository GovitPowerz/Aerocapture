#!/usr/bin/env bash
set -euo pipefail

# Joint-reference study: every table-READING scheme (JOINT_REF_BANK_SCHEMES =
# ftc / energy_controller / pred_guid) retrained with the `ref_bank` gene
# ([reference] joint_bank = true): each individual is evaluated against a
# constant-bank reference generated from ITS OWN gene via the per-individual
# data.reference_trajectory injection in run_grid (fixed 2026-06-12 -- the gene
# was a silent no-op before; SharedTables ignored the per-individual override).
# fnpag is deliberately absent: it is in REQUIRES_REF_TRAJECTORY but never reads
# the table (numerical predictor), so training hard-errors on the gene.
#
# Budgets MATCH the post-fix classical baselines from run_paper_experiments2.sh
# (GA, --n-gen 2000 --n-pop 300, final eval n=1000) -- NOT train_all.sh -- so each
# joint run is directly comparable to its shared-reference baseline in
# training_output/<scheme>. Run/verify exp2 first. A budget mismatch here
# (the previous 2500x50/60 + final 2000) would hand the baselines a 4-5x compute
# advantage and bias Study E toward a null result.
# Dedicated --output-dir keeps the baselines (and training_output/mars/) intact.
# Do NOT regenerate training_output/mars/ref_trajectory.dat between exp2 and this
# run -- the joint-vs-baseline comparison assumes the baselines' shared reference
# is frozen.

run() {  # $1=config $2=dir
  if [ -f "training_output/$2/final_eval.parquet" ]; then echo "skip $2 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/$1" \
      --output-dir "training_output/$2" --n-gen 2000 --n-pop 300 --sim-timeout 5 --from-scratch
}

run msr_aller_ftc_joint_ref_train.toml               ftc_joint_ref
run msr_aller_energy_controller_joint_ref_train.toml energy_controller_joint_ref
run msr_aller_pred_guid_joint_ref_train.toml         pred_guid_joint_ref
