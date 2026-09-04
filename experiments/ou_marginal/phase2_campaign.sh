#!/bin/bash
# OU-marginal campaign, phase 2 -- same stoppable/resumable contract as
# retrain_campaign.sh: run; stop any time (Ctrl+C / laptop shutdown); rerun
# and every job continues from its latest checkpoint with exact
# remaining-gens math; finished jobs are skipped; an interrupted job stops
# the whole campaign.
#
# Jobs, in value order:
#   1. ft_<cell>   : fine-tune recipe -- the frozen champion's g20000
#                    checkpoint + 2000 per_draw gens (target 22000). The
#                    pilot showed this BEATS scratch retraining (CVaR95
#                    138.3 vs 146.9 on the same pool). ~25 min each.
#   2. <cell>_s2/3 : scratch per_draw repeats with --seed 2/3 (trainer RNG
#                    only; eval pools unchanged -- the 10c convention).
#                    sigma_run for the marginal architecture ranking.
#                    ~3.7 h each at phase-1 rates.
#
# Afterwards: uv run python experiments/ou_marginal/quote_marginal.py
set -euo pipefail
cd "$(dirname "$0")/../.."

# name|target_gen|seed|checkpoint_source_dir("" = scratch)
JOBS="
ft_mamba_p962|22000|1|training_output/mamba_p962_long
ft_mamba_p962_s2|22000|2|training_output/mamba_p962_long
ft_mamba_p962_s3|22000|3|training_output/mamba_p962_long
ft_lstm_p1082|22000|1|training_output/lstm_p1082_long
ft_gru_p1014|22000|1|training_output/gru_p1014_long
ft_dense_p972|22000|1|training_output/dense_p972_ga_paper_best
ft_dense_p515|22000|1|training_output/dense_p515_ga_paper_best
mamba_p962_s2|20000|2|
lstm_p1082_s2|20000|2|
gru_p1014_s2|20000|2|
dense_p972_s2|20000|2|
dense_p515_s2|20000|2|
mamba_p962_s3|20000|3|
lstm_p1082_s3|20000|3|
gru_p1014_s3|20000|3|
dense_p972_s3|20000|3|
dense_p515_s3|20000|3|
"

for job in $JOBS; do
  name="${job%%|*}"; rest="${job#*|}"
  target="${rest%%|*}"; rest="${rest#*|}"
  seed="${rest%%|*}"; src="${rest#*|}"
  out="training_output/ou_marginal/${name}"
  mkdir -p "$out"

  last=$( (ls "$out" 2>/dev/null | grep -o 'checkpoint_g[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1) || true)
  last=$((10#${last:-0}))

  # Fine-tune jobs start from the frozen champion's checkpoint.
  if [ -n "$src" ] && [ "$last" -eq 0 ]; then
    cp "$src/checkpoint_g20000.json" "$src/checkpoint_g20000.npz" "$out/"
    # Seed repeats (--seed != 1): strip the checkpointed RNG state, or the
    # resume path restores it and silently overrides --seed -- all "repeats"
    # would replay the identical training trajectory (verified: bit-identical
    # deployed weights before this fix).
    if [ "$seed" != "1" ]; then
      python3 -c "import json,sys; p='$out/checkpoint_g20000.json'; d=json.load(open(p)); d['rng_state']=None; json.dump(d, open(p,'w'))"
      echo "== ${name}: stripped rng_state (trainer seed ${seed} takes effect)"
    fi
    last=20000
    echo "== ${name}: seeded champion checkpoint from ${src}"
  fi

  remaining=$((target - last))
  if [ "$remaining" -le 0 ]; then
    echo "== ${name}: done (g${last} >= ${target}), skipping"
    continue
  fi
  echo "== ${name}: at g${last}, training ${remaining} more gens (target ${target}, seed ${seed})"
  uv run python -m aerocapture.training.train \
    "configs/training/ou_marginal/${name}.toml" \
    --n-gen "$remaining" --seed "$seed" --no-tui --skip-report \
    --output-dir "$out" \
    2>&1 | tee -a "$out/campaign.log"

  now=$( (ls "$out" | grep -o 'checkpoint_g[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1) || true)
  now=$((10#${now:-0}))
  if [ "$now" -lt "$target" ]; then
    echo "== ${name}: stopped at g${now} before target; exiting campaign (rerun this script to continue)."
    exit 0
  fi
  echo "== ${name}: reached g${now}"
done
echo "Phase-2 campaign pass complete."
