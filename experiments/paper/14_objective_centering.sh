#!/usr/bin/env bash
set -euo pipefail
# Objective centering: the regime-matched objective-shaping study of Section 7.3.
#
#   14_objective_centering.sh v4          the centered recipe under per-scenario noise (issue #177, default)
#   14_objective_centering.sh v4-dense    the optional dense lever cells under per-scenario noise (#177)
#   14_objective_centering.sh shared      the arxiv-v3 shared-path study (training_output/paper/objective_centering)
#
# ===== v4 (issue #177): the centered recipe under per-scenario noise =====
# Section 7.3's centered cells trained on the shared noise path; under per-scenario noise the
# retrained joint-FTC out-captures every seed. v4 trains the recipe under per_draw and reads one table:
#   1. baseline  classical_campaign.sh classical: classical/ftc_joint_high, the high-regime joint-FTC
#                retuned under per_draw at the #172 allocation (~3.5 h, like ftc_joint; the nine #172
#                cells are skipped as trained). The medium baseline is #172's classical/ftc_joint.
#   2. seeds     campaign.sh jobs_centering.txt: centered/mamba_centered_s1 / _s2 / _s3 from scratch at
#                the #156 seeds' allocation and budget (GA 256 x 16, 4000 gens, ~4 h each).
#   3. depth     centered_depth_eval.py --v4 -> articles/paper/data/centered_depth_v4.json: the 9M stress
#                pool at n = 10000, per_draw only, bootstrap CIs and paired seed-versus-baseline deltas
#                (the arxiv-v3 centered_depth.json stays). Refuses to run short of a cell.
# Every step is resumable (Ctrl-C, shutdown: rerun the command, finished cells are skipped). Run from
# the Terminal panel under `caffeinate -i`; about 16 h.
#
# PRE-REGISTERED READING (issue #177, before any number exists): capture first (a difference within
# half a point is parity), conditional CVaR95 second, paired intervals excluding zero, no tail win
# across a larger capture deficit (Section 7.2's rule). Outcome A: every centered seed beats both
# joint-FTC baselines: Section 7.3 stays with one table and this regime, and `v4-dense` runs for the
# figure. Outcome B: any seed loses on capture to either baseline: Section 7.3 folds into 7.2 as one
# paragraph and the figure is dropped. The outcome is stated on the issue before #181 writes the section.
#
# ===== v4-dense: the lever cells, under outcome A =====
# campaign.sh jobs_centering_dense.txt: the five dense_515 lever cells under per_draw at the shared
# study's iso-compute budget (B = 8.19 M sims: n = 2 -> 16000 gens, n = 16 -> 2000 gens, ~30 min each),
# then objective_centering_eval.py --v4 -> articles/paper/data/objective_centering_v4.json (n = 1000,
# per_draw; the arxiv-v3 objective_centering.json stays).
#
# ===== shared (arxiv-v3): exp(objcenter), off the numbered campaign reproduction =====
# Spec: docs/design/2026-06-29-objective-centering-regime-matched-design.md
# Five dense_515 cells, all UNDER the high regime, flipping one objective lever at a time from the
# medium-regime-winning stack (cubed x max-bucket x n_sims=2) to centered (linear x middle x
# n_sims=16). Iso-compute matched on total training sims B = n_pop*n_sims*n_gen (n_pop=256,
# B~=8.19e6): n_sims=2 -> 16000 gens, n_sims=16 -> 2000 gens. Then eval all deployed cells on the
# reserved 9M pool. Idempotent (skip-if-final_eval.parquet per cell). Dense vehicle is fast; this is
# the methodology comparison. Override the budget knobs with NPOP / GEN_N2 / GEN_N16 env vars.
# Phase 2 (RUN_MAMBA=1): the centered recipe on Mamba_962, the long pole (Mamba plateaus ~10-15k gens;
# the default GEN_MAMBA is directional, scale up for a final number); its s2 / s3 repeats are
# 16_sigma_extras.sh's, scored with it by `make -C articles/paper mc-centered-depth` (#156).

cd "$(dirname "$0")/../.."
MODE="${1:-v4}"
trap 'echo; echo "Ctrl-C -- stopping (v4 / v4-dense: re-run to resume; shared restarts the interrupted cell from scratch)"; exit 130' INT

v4() {
  echo "=== 1. baseline: classical/ftc_joint_high under per_draw (classical_campaign.sh classical) ==="
  experiments/ou_marginal/classical_campaign.sh classical
  echo "=== 2. seeds: centered/mamba_centered_s1 / _s2 / _s3 under per_draw (campaign.sh) ==="
  experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_centering.txt
  echo "=== 3. depth: centered_depth_v4.json (9M stress pool, n = 10000, per_draw) ==="
  uv run python -u articles/paper/scripts/centered_depth_eval.py --v4 --n-sims 10000
}

v4_dense() {
  echo "=== dense lever cells under per_draw (campaign.sh), then objective_centering_v4.json ==="
  experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_centering_dense.txt
  uv run python -u articles/paper/scripts/objective_centering_eval.py --v4 --n-sims 1000
}

shared() {
  local NPOP=${NPOP:-256}
  local NSIMS_EVAL=${NSIMS_EVAL:-1000}
  # gen counts per n_sims to hold B = NPOP * n_sims * n_gen ~= 8.19e6 fixed:
  local GEN_N2=${GEN_N2:-16000}
  local GEN_N16=${GEN_N16:-2000}

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
  if [ "${RUN_MAMBA:-0}" = "1" ]; then
    local GEN_MAMBA=${GEN_MAMBA:-4000}
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
}

case "$MODE" in
  v4) v4 ;;
  v4-dense) v4_dense ;;
  shared) shared ;;
  *) echo "usage: $0 [v4|v4-dense|shared]"; exit 2 ;;
esac
