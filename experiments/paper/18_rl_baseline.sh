#!/bin/bash
# Paper Section 5 RL baseline (issue #101): PPO cells protocol-matched to the
# per-scenario population champions they are compared to (ou_marginal/ft_dense_p515,
# ou_marginal/ft_gru_p1014): same architecture, input mask, normalization, decoder,
# co-trained scaffolding, per-scenario noise regime (ADR-0006), the reserved 1M
# validation pool for promotion and the 2M final-eval pool (n = 1000) for the quote.
#
# Four cells: {dense_p515, gru_p1014} x {scratch, warm-started from the champion}.
# The two cells of a pair run concurrently (PPO's serial Python/torch phase leaves
# cores idle between Rayon env bursts); the pairs run one after the other. A failed
# cell is reported and the script exits non-zero after the other pair has run.
#
# Three cell sets, chosen by the first argument:
#   ft (default)  the arxiv-v3 cells, matched to the fine-tunes ou_marginal/ft_dense_p515 / ft_gru_p1014;
#   hl            the v4 cells (issue #175), hl_<cell>_ppo_*, matched to the #173 scratch cells
#                 ou_marginal/hl_dense_p515 / hl_gru_p1014 at the 512 x 2 headline allocation
#                 (the #173 rule retired the fine-tunes from the main body);
#   hl_repeats    the PPO-scratch seed repeats of the hl set, hl_<cell>_ppo_scratch_s2 / _s3
#                 ([rl] torch_seed 2 / 3, same protocol): the s2/s3 of a cell run as one pair.
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
  # `set -e` would end this background subshell on a trainer failure before any
  # status line prints, and a bare `wait` returns 0 regardless: test the status here.
  if uv run python -m aerocapture.training.rl.train "$toml" --no-tui ${flags[@]+"${flags[@]}"} \
      >> "$out/campaign.log" 2>&1; then
    echo "== ${cell}: trainer exited 0"
  else
    echo "== ${cell}: trainer FAILED (exit $?), see $out/campaign.log" >&2
    return 1
  fi
}

failed=0
run_pair() {
  local pids=()
  run_cell "$1" "$2" & pids+=("$!")
  run_cell "$3" "$4" & pids+=("$!")
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
}

mode="${1:-ft}"
case "$mode" in
  ft) cell_prefix="" champion_prefix="ft_" ;;
  hl) cell_prefix="hl_" champion_prefix="hl_" ;;
  hl_repeats) ;;
  *) echo "usage: $0 [ft|hl|hl_repeats]" >&2; exit 2 ;;
esac
if [ "$mode" = hl_repeats ]; then
  run_pair hl_dense_p515_ppo_scratch_s2 "" hl_dense_p515_ppo_scratch_s3 ""
  run_pair hl_gru_p1014_ppo_scratch_s2 "" hl_gru_p1014_ppo_scratch_s3 ""
else
  run_pair "${cell_prefix}dense_p515_ppo_scratch" "" "${cell_prefix}dense_p515_ppo_warm" "training_output/ou_marginal/${champion_prefix}dense_p515/best_model.json"
  run_pair "${cell_prefix}gru_p1014_ppo_scratch" "" "${cell_prefix}gru_p1014_ppo_warm" "training_output/ou_marginal/${champion_prefix}gru_p1014/best_model.json"
fi
if [ "$failed" -ne 0 ]; then
  echo "Campaign pass INCOMPLETE: at least one cell failed (rerun to resume it)." >&2
  exit 1
fi
echo "Campaign pass complete."
