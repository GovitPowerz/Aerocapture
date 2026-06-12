#!/usr/bin/env bash
set -euo pipefail

# Joint-reference study: every table-READING scheme (JOINT_REF_BANK_SCHEMES =
# ftc / energy_controller / pred_guid) retrained with the `ref_bank` gene
# ([reference] joint_bank = true): each individual is evaluated against a
# constant-bank reference generated from ITS OWN gene via the per-individual
# data.reference_trajectory injection in run_grid (fixed 2026-06-12 -- the gene
# was a silent no-op before; SharedTables ignored the per-individual override).
# fnpag is deliberately absent: it is in REQUIRES_REF_TRAJECTORY but never reads
# the table (numerical predictor), so the config loader hard-errors on the gene.
#
# Budgets mirror the per-scheme train_all.sh baselines so each joint run is
# directly comparable to its shared-reference baseline in training_output/<scheme>;
# dedicated --output-dir keeps those baselines (and training_output/mars/) intact.

uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_joint_ref_train.toml \
    --output-dir training_output/ftc_joint_ref --n-gen 2500 --n-pop 50 --final-n-sims 2000 --sim-timeout 1 --from-scratch

uv run python -m aerocapture.training.train configs/training/msr_aller_energy_controller_joint_ref_train.toml \
    --output-dir training_output/energy_controller_joint_ref --n-gen 2500 --n-pop 60 --final-n-sims 2000 --sim-timeout 1 --from-scratch

uv run python -m aerocapture.training.train configs/training/msr_aller_pred_guid_joint_ref_train.toml \
    --output-dir training_output/pred_guid_joint_ref --n-gen 2500 --n-pop 60 --final-n-sims 2000 --sim-timeout 1 --from-scratch
