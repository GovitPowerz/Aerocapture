#!/usr/bin/env bash
set -euo pipefail
# sigma_run extras for the R4/R5 revision (reviewer R1 statistical comment 1 +
# R1-P6): seed-repeats for the DECISIVE Study C optimizer cells and for the
# centered-Mamba stress retrain of section 7.3, so those claims carry measured
# run-to-run scatter instead of single-run hedges.
#
#   Study C repeats (s2/s3; s1 = the existing campaign cells under
#   training_output/paper/seed_strategy/ and .../optimizer_budget/ga_150):
#     ga_fixed / ga_rotating / cmaes_fixed / cmaes_rotating (dense_p3998 @150,
#     2000 gens, n_sims 10 -- 04_seed_strategy.sh regime) and ga_adaptive
#     (= 02's ga_150 cell regime, adaptive is the TOML default).
#   Stress retrain repeats (s2/s3; s1 = objective_centering/mamba_centered):
#     mamba_centered_high at the exact s1 allocation (n_sims 16, 4000 gens,
#     n_pop 256 -- verified from the s1 checkpoint/JSONL).
#
# Ordered cheapest-first so partial completion is still useful: the optimizer
# cells are hours each; the two mamba_centered repeats are the long poles
# (~16M sims each). Resumable: skip-if-final_eval.parquet is NOT used here
# (repeats must not skip on the s1 sentinel); each cell skips itself when its
# own final_selection.json exists, and a crashed cell auto-resumes on re-run.
# Ctrl-C stops cleanly. Do NOT run concurrently with 15_state_controls.sh
# unless cores are plentiful; never two cells of the same config TOML at once
# (this script is strictly serial for exactly that reason).
#
# AFTER training: the revision plan's Task 19 evaluates the optimizer repeats
# on the n=1000 final-eval pool (Study C's original metric) and the
# mamba_centered repeats on the 9M stress pool
# (docs/superpowers/plans/2026-07-10-reviewer-4-5-revision.md).

P="training_output/paper/sigma_extras"
trap 'echo; echo "Ctrl-C -- stopping (re-run to resume from the last checkpoint)"; exit 130' INT

run() {  # $1=config  $2=seed-strategy-or-"default"  $3=seed  $4=cell  $5=n_sims  $6=n_gen  $7=n_pop
  local out="$P/$4"
  if [ -f "$out/final_selection.json" ]; then
    echo "skip $4 (final_selection.json present -- already trained)"
    return 0
  fi
  local strat=()
  if [ "$2" != "default" ]; then strat=(--seed-strategy "$2"); fi
  echo "=== $4 (seed $3) -> $out ==="
  uv run python -m aerocapture.training.train "$1" \
      --training-n-sims "$5" --n-gen "$6" --n-pop "$7" --seed "$3" "${strat[@]}" \
      --output-dir "$out" --sim-timeout 5 \
    || echo "WARNING: $4 exited non-zero -- continuing (re-run to retry)"
}

GA=configs/training/paper/dense_p3998_ga.toml
CM=configs/training/paper/dense_p3998_cmaes.toml
MC=configs/training/paper/objective_centering/mamba_centered_high.toml

# ── Study C optimizer repeats (hours each; 04/02 regime: n_sims 10, 2000 gens, pop 150) ──
for S in 2 3; do
  run "$GA" fixed    "$S" "ga_fixed_s$S"       10 2000 150
  run "$GA" rotating "$S" "ga_rotating_s$S"    10 2000 150
  run "$GA" default  "$S" "ga_adaptive_s$S"    10 2000 150   # adaptive = TOML default (02's ga_150)
  run "$CM" fixed    "$S" "cmaes_fixed_s$S"    10 2000 150   # may self-terminate early -- expected, that IS the result
  run "$CM" rotating "$S" "cmaes_rotating_s$S" 10 2000 150
done

# ── Section 7.3 stress-retrain repeats (long poles; s1 allocation: 16 sims, 4000 gens, pop 256) ──
for S in 2 3; do
  run "$MC" default "$S" "mamba_centered_s$S" 16 4000 256
done
