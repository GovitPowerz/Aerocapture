#!/usr/bin/env bash
set -euo pipefail
# sigma_run on the SIZING TAIL at the headline allocation -- tests whether 10b's
# result (the recurrent nets beat the dense_515 headline on the tank-sizing tail:
# Mamba_962 CVaR99.9 122.0 / max 124.4 and LSTM_1082 123.2 / 126.0 vs dense_515
# 128.1 / 146.7, far-tail n=10000) survives run-to-run variance, or is single-run
# luck. The 10b/headline runs are s1; this adds s2/s3 = FRESH 20000-gen runs with
# --seed varying the trainer RNG (init population + curator draws) while the eval
# pool stays fixed, so the deployed CVaR99.9 spread is a clean sigma_run.
#
# Mirrors 11_seed_repeats.sh's methodology, but at the n_sims=2 / n_pop=512 /
# 20000-gen HEADLINE allocation (not 11's n=10/2000) and keyed on the TAIL
# (CVaR99.9 / 3sigma), not the mean. Resumable (skip-if-final_selection,
# auto-resume on crash -- NO --from-scratch, so a multi-day run survives a crash;
# --n-gen is "additional" on resume, so a crash-resume trains PAST 20000, harmless).
# Ctrl-C stops cleanly. Each run is multi-day; 4 runs (2 cells x s2/s3).
#
# Provenance note: dense515_s1 = the deployed headline (fresh 20000-gen run);
# mamba962_s1 = 10b's mamba_p962_long (sweep-5000 + resume-15000). Both plateaued,
# so the seeding washes out -- but if you want a fully-fresh mamba s1, also run
# `run configs/training/sweep/mamba_p962.toml 1 mamba962_s1`.
#
# AFTER training: far-tail each repeat and quote CVaR99.9 as mean +/- range over
# {s1,s2,s3}; if mamba's tail edge over dense exceeds sigma_run, it is real and
# the deployed headline should be reconsidered (Mamba_962 for the sizing tail,
# dense_515 for efficiency). far-tail command:
#   uv run python articles/paper/scripts/far_tail_eval.py --n-sims 10000 --cells \
#     paper/tail_repeats/dense515_s2:configs/training/sweep/dense_p515.toml \
#     paper/tail_repeats/mamba962_s2:configs/training/sweep/mamba_p962.toml  ...

P="training_output/paper/tail_repeats"
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

for S in 2 3; do
  run configs/training/msr_aller_nn_atan2_best_paper.toml "$S" "dense515_s$S"   # dense_515 headline arch
  run configs/training/sweep/mamba_p962.toml              "$S" "mamba962_s$S"   # the tail-leading challenger
  # run configs/training/sweep/lstm_p1082.toml             "$S" "lstm1082_s$S"   # uncomment to also repeat the LSTM co-leader
done
