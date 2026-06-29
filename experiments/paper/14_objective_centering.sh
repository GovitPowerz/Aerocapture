#!/usr/bin/env bash
set -euo pipefail
# exp(objcenter) -- objective-centering under the high/adversarial regime (NOT part
# of the numbered campaign reproduction). Spec:
# docs/superpowers/specs/2026-06-29-objective-centering-regime-matched-design.md
#
# Five dense_515 cells, all UNDER the high regime, flipping one objective lever at
# a time from the medium-regime-winning stack (cubed x max-bucket x n_sims=2) to
# centered (linear x middle x n_sims=16). Iso-compute matched on total training
# sims B = n_pop*n_sims*n_gen (n_pop=256, B~=8.19e6): n_sims=2 -> 16000 gens,
# n_sims=16 -> 2000 gens. Then eval all deployed cells on the reserved 9M pool.
#
# Idempotent (skip-if-final_eval.parquet per cell). Dense vehicle is fast; this is
# the methodology comparison. Override the budget knob with NPOP / BUDGET env vars.
NPOP=${NPOP:-256}
NSIMS_EVAL=${NSIMS_EVAL:-1000}
# gen counts per n_sims to hold B = NPOP * n_sims * n_gen ~= 8.19e6 fixed:
GEN_N2=${GEN_N2:-16000}
GEN_N16=${GEN_N16:-2000}

train() {  # $1=config-stem  $2=cell  $3=n_sims  $4=n_gen
  if [ -f "training_output/paper/objective_centering/$2/final_eval.parquet" ]; then
    echo "skip $2 (done)"; return 0
  fi
  uv run python -m aerocapture.training.train \
      "configs/training/paper/objective_centering/$1.toml" \
      --training-n-sims "$3" --n-gen "$4" --n-pop "$NPOP" \
      --output-dir "training_output/paper/objective_centering/$2" \
      --sim-timeout 5 --from-scratch
}

train dense_stacked_high         dense_stacked         2  "$GEN_N2"
train dense_plus_sims_high       dense_plus_sims       16 "$GEN_N16"
train dense_plus_bucket_high     dense_plus_bucket     2  "$GEN_N2"
train dense_plus_transform_high  dense_plus_transform  2  "$GEN_N2"
train dense_centered_high        dense_centered        16 "$GEN_N16"

uv run python articles/paper/scripts/objective_centering_eval.py --n-sims "$NSIMS_EVAL"

# ---- Phase 2 (gated): confirm the winning centered recipe on Mamba_962 ----
# Run only when RUN_MAMBA=1 (the long pole). Update mamba_centered_high.toml to
# the Phase-1 winning lever first if a single lever dominated. Mamba plateaus
# ~10-15k gens; default GEN_MAMBA is directional -- scale up for a final number.
if [ "${RUN_MAMBA:-0}" = "1" ]; then
  GEN_MAMBA=${GEN_MAMBA:-4000}
  if [ ! -f "training_output/paper/objective_centering/mamba_centered/final_eval.parquet" ]; then
    uv run python -m aerocapture.training.train \
        configs/training/paper/objective_centering/mamba_centered_high.toml \
        --training-n-sims 16 --n-gen "$GEN_MAMBA" --n-pop "$NPOP" \
        --output-dir training_output/paper/objective_centering/mamba_centered \
        --sim-timeout 5 --from-scratch
  else
    echo "skip mamba_centered (done)"
  fi
fi
