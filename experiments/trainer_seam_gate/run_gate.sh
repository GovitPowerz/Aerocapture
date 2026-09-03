#!/bin/bash
# Trainer-seam bit-equivalence gate: train the three gate configs into
# .scratch/trainer_seam_gate/<tag>/ (gitignored). Run once at the pre-refactor
# commit (tag "baseline"), once at the post-refactor commit (tag "post"), then
# `uv run python experiments/trainer_seam_gate/diff_gate.py baseline post`.
#
# Usage: experiments/trainer_seam_gate/run_gate.sh <tag> [gate ...]
#   gates default to: a_eqglide_fixed b_mamba_adaptive c_islands
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG="$1"; shift
CFG="experiments/trainer_seam_gate"
OUT=".scratch/trainer_seam_gate"
GATES="${@:-a_eqglide_fixed b_mamba_adaptive c_islands}"
mkdir -p "$OUT/$TAG"
for gate in $GATES; do
  out="$OUT/$TAG/run_${gate%%_*}"
  rm -rf "$out"
  uv run python -m aerocapture.training.train "$CFG/gate_${gate}.toml" \
    --no-tui --skip-report --final-n-sims 20 --from-scratch --output-dir "$out" \
    > "$OUT/$TAG/run_${gate%%_*}.log" 2>&1
  echo "gate $gate done -> $out"
done
