#!/usr/bin/env bash
set -euo pipefail
# Classical baselines -- GA @2000x300 (run from the repo root; needs 00_prereqs).
# Deploys to the CANONICAL training_output/<scheme> dirs: compare_guidance,
# Study E (07_joint_reference) and the optimizer-dimensionality FTC GA cell
# (03) all reference these names.
# Do NOT regenerate training_output/mars/ between this script and 07.

run() {  # $1=config-stem $2=scheme-dir
  if [ -f "training_output/$2/final_eval.parquet" ]; then echo "skip $2 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "configs/training/$1.toml" \
      --algorithm ga --n-gen 2000 --n-pop 300 --sim-timeout 5 --from-scratch
}

run msr_aller_ftc_train               ftc
run msr_aller_eqglide_train           equilibrium_glide
run msr_aller_energy_controller_train energy_controller
run msr_aller_pred_guid_train         pred_guid
run msr_aller_fnpag_train             fnpag   # ~50x slower per sim than FTC; the slow cell
