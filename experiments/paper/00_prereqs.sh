#!/usr/bin/env bash
set -euo pipefail
# Step 0 -- mission prerequisites + the piecewise_constant classical row
# (run from the repo root). Produces:
#   training_output/piecewise_constant/        (classical table row, GA @2000x300)
#   training_output/mars/corridor_boundaries.npz  (accumulated during that run)
#   training_output/mars/ref_trajectory.dat       (constant-bank energy-matched ref)
# ref_trajectory.dat is committed to git, so 01..11 work without this script;
# run it for a true from-scratch reproduction or after wiping training_output.
# NEVER (re)run this while any ref-tracking scheme (ftc / energy_controller /
# pred_guid) is training -- SimData re-reads the reference file every generation.

# Piecewise-constant baseline: classical table row + corridor accumulation.
if [ -f training_output/piecewise_constant/final_eval.parquet ]; then
  echo "skip piecewise_constant (done)"
else
  uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml \
      --algorithm ga --n-gen 2000 --n-pop 300 --sim-timeout 5 --from-scratch
fi

# Mission reference: constant-bank nominal bisected to the target orbit energy.
if [ -f training_output/mars/ref_trajectory.dat ]; then
  echo "skip make_reference (training_output/mars/ref_trajectory.dat exists; delete to force)"
else
  uv run python -m aerocapture.training.make_reference --toml configs/training/msr_aller_pc_ref_train.toml
fi
