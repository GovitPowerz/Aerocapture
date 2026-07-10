#!/usr/bin/env bash
set -euo pipefail
# State-ablation controls for the R4/R5 revision (reviewer R1 major 5) -- the two
# retrained controls that, with the reset-state eval flag, decide whether the
# Mamba's sizing-tail win is INTERNAL STATE or something else:
#   window_ctrl_p970  -- explicit 5-tick observation history, no learned state
#                        (Window(17x5) -> Dense stack, 970 params ~ Mamba's 962)
#   mamba_p962_nodv   -- deployed Mamba arch retrained WITHOUT the 3 predicted
#                        correction-DV inputs (mask minus 32-34, 914 params)
# 3 seeds each at the HEADLINE allocation (n_sims=2 / n_pop=512 / 20000 gens),
# mirroring 10c_tail_sigma_repeats.sh, so the results are directly comparable to
# the deployed Mamba_962 / dense_515 / lstm_1082 sigma_run triplets.
#
# Resumable: skip-if-final_selection.json, auto-resume from checkpoint on crash
# (--n-gen is "additional" on resume, so a crash-resume trains past 20000 --
# harmless). Ctrl-C stops cleanly; re-run to resume. Each cell is multi-day.
# Do NOT run concurrently with 16_sigma_extras.sh unless cores are plentiful,
# and never two cells of the same config TOML at once.
#
# AFTER training: the revision plan's Task 19 evaluates all 6 cells on the
# far-tail (n=10000) and confirmatory (10x100k) pools with pre-registered
# interpretation rules (docs/superpowers/plans/2026-07-10-reviewer-4-5-revision.md).

P="training_output/paper/state_controls"
trap 'echo; echo "Ctrl-C -- stopping (re-run to resume from the last checkpoint)"; exit 130' INT

run() {  # $1=config  $2=seed  $3=cell
  local out="$P/$3"
  if [ -f "$out/final_selection.json" ]; then
    echo "skip $3 (final_selection.json present -- already trained)"
    return 0
  fi
  echo "=== $3 (seed $2) -> $out ==="
  uv run python -m aerocapture.training.train "$1" \
      --training-n-sims 2 --n-gen 20000 --n-pop 512 --seed "$2" \
      --output-dir "$out" --sim-timeout 5 \
    || echo "WARNING: $3 exited non-zero -- continuing (re-run to retry)"
}

for S in 1 2 3; do
  run configs/training/paper/window_ctrl_p970.toml "$S" "window_s$S"
  run configs/training/paper/mamba_p962_nodv.toml  "$S" "nodv_s$S"
done
