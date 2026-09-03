#!/bin/bash
# OU-marginal retrain campaign: the five paper NN headline cells retrained
# FROM SCRATCH under [monte_carlo] noise_seeding = "per_draw", for the deployed
# cells' generation count (20000 gens) at the sweep-config allocation (GA n_pop 60,
# training_n_sims 10, adaptive seeds) -- NOT the headline's CLI allocation
# (n_pop 512, training_n_sims 2 in experiments/paper/10b_arch_long_challengers.sh).
#
# STOPPABLE / RESUMABLE by design: run this script; stop it any time
# (Ctrl+C, laptop shutdown -- train.py checkpoints every 10 gens with atomic
# writes and saves on SIGINT); run the script again and it continues each
# cell from its latest checkpoint, computing the remaining generations so
# the 20000-gen target is exact. Finished cells are skipped.
#
# Order: mamba first (the headline), then the co-leader and the rest.
# Rough cost at pilot rates: ~16 h per cell, ~3.5 days total laptop time.
#
# After all cells finish: quote the marginal-regime numbers with
# experiments/ou_marginal/quote_marginal.py (frozen-vs-marginal appendix data).
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET_GEN=20000
CELLS="mamba_p962 lstm_p1082 gru_p1014 dense_p972 dense_p515"

for cell in $CELLS; do
  out="training_output/ou_marginal/${cell}"
  mkdir -p "$out"
  # Latest single-algo checkpoint label (0 when starting fresh).
  # `|| true`: grep exits 1 on a fresh dir with no checkpoints, and under
  # `set -e` that would kill the script inside the substitution.
  last=$( (ls "$out" 2>/dev/null | grep -o 'checkpoint_g[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1) || true)
  last=${last:-0}
  # Force base-10: checkpoint labels are zero-padded (g00010) and $((...))
  # would otherwise parse them as octal.
  last=$((10#$last))
  remaining=$((TARGET_GEN - last))
  if [ "$remaining" -le 0 ]; then
    echo "== ${cell}: done (g${last} >= ${TARGET_GEN}), skipping"
    continue
  fi
  echo "== ${cell}: at g${last}, training ${remaining} more gens (target ${TARGET_GEN})"
  # --n-gen means "N additional" on resume, so remaining-to-target is exact.
  uv run python -m aerocapture.training.train \
    "configs/training/ou_marginal/${cell}.toml" \
    --n-gen "$remaining" --no-tui --skip-report \
    --output-dir "$out" \
    2>&1 | tee -a "$out/campaign.log"
  # train.py exits 0 on Ctrl+C (clean checkpoint save). If the target was not
  # reached, the session was interrupted: stop the whole campaign instead of
  # charging into the next cell while the user is shutting the laptop down.
  now=$( (ls "$out" | grep -o 'checkpoint_g[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1) || true)
  now=$((10#${now:-0}))
  if [ "$now" -lt "$TARGET_GEN" ]; then
    echo "== ${cell}: stopped at g${now} before target; exiting campaign (rerun this script to continue)."
    exit 0
  fi
  echo "== ${cell}: reached g${now}"
done
echo "Campaign pass complete."
