#!/bin/bash
# Paper Section 5 RL baseline (issue #101): PPO cells protocol-matched to the
# per-scenario population champions they are compared to (ou_marginal/ft_dense_p515,
# ou_marginal/ft_gru_p1014): same architecture, input mask, normalization, decoder,
# co-trained scaffolding, per-scenario noise regime (ADR-0006), the reserved 1M
# validation pool for promotion and the 2M final-eval pool (n = 1000) for the quote.
#
# Four cells: {dense_p515, gru_p1014} x {scratch, warm-started from the champion}.
# The two cells of a pair run concurrently (PPO's serial Python/torch phase leaves
# cores idle between Rayon env bursts); the pairs run one after the other.
#
# RESUMABLE: a cell with final_eval.parquet is done and skipped; a cell with a
# checkpoint.pt resumes (plain invocation, no --from-scratch / --data-neural-network,
# which would wipe the checkpoint); anything else starts fresh. Each cell's
# stdout/stderr goes to training_output/paper/rl/<cell>/campaign.log.
#
# Afterwards: the population champions need their own 2M-pool per-draw parquet
# (uv run python -m aerocapture.training.report training_output/ou_marginal/<cell> --toml
# configs/training/ou_marginal/<cell>.toml), then 12_collect_results.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

run_cell() {
  local cell="$1" warm_from="$2"
  local out="training_output/paper/rl/${cell}"
  local toml="configs/training/paper/rl/${cell}.toml"
  mkdir -p "$out"
  if [ -f "$out/final_eval.parquet" ]; then
    echo "== ${cell}: done (final_eval.parquet present), skipping"
    return 0
  fi
  local flags=()
  if [ -f "$out/checkpoint.pt" ]; then
    echo "== ${cell}: resuming from checkpoint.pt"
  elif [ -n "$warm_from" ]; then
    echo "== ${cell}: fresh, warm-started from ${warm_from}"
    flags=(--data-neural-network "$warm_from")
  else
    echo "== ${cell}: fresh, from scratch"
    flags=(--from-scratch)
  fi
  uv run python -m aerocapture.training.rl.train "$toml" --no-tui ${flags[@]+"${flags[@]}"} \
    >> "$out/campaign.log" 2>&1
  echo "== ${cell}: trainer exited $?"
}

run_cell dense_p515_ppo_scratch "" &
run_cell dense_p515_ppo_warm training_output/ou_marginal/ft_dense_p515/best_model.json &
wait
run_cell gru_p1014_ppo_scratch "" &
run_cell gru_p1014_ppo_warm training_output/ou_marginal/ft_gru_p1014/best_model.json &
wait
echo "Campaign pass complete."
