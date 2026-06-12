#!/usr/bin/env bash
set -euo pipefail
# Study E -- joint reference optimization for the table-READING schemes
# (ftc / energy_controller / pred_guid; fnpag never reads the table and the
# trainer hard-errors on the gene). [reference] joint_bank = true appends a
# ref_bank gene; each individual is evaluated against a constant-bank reference
# generated from ITS OWN gene (per-individual data.reference_trajectory
# injection in run_grid, fixed 2026-06-12 in 6bb3f27).
# Budgets MATCH the classical baselines from 01 (GA @2000x300, final n=1000) so
# each joint run is directly comparable to training_output/<scheme>. Run 01
# first, and do NOT regenerate training_output/mars/ in between.

run() {  # $1=config-stem $2=cell
  if [ -f "training_output/paper/joint_reference/$2/final_eval.parquet" ]; then echo "skip $2 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/$1.toml" \
      --n-gen 2000 --n-pop 300 --output-dir "training_output/paper/joint_reference/$2" --sim-timeout 5 --from-scratch
}

run msr_aller_ftc_joint_ref_train               ftc
run msr_aller_energy_controller_joint_ref_train energy_controller
run msr_aller_pred_guid_joint_ref_train         pred_guid
