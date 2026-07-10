#!/usr/bin/env bash
set -euo pipefail
# State-ablation controls for the R4/R5 revision (reviewer R1 major 5) -- the two
# retrained controls that, with the reset-state eval flag, decide whether the
# Mamba's sizing-tail win is INTERNAL STATE or something else:
#   window_ctrl_p970  -- explicit 5-tick observation history, no learned state
#                        (Window(17x5) -> Dense stack, 970 params ~ Mamba's 962)
#   mamba_p962_nodv   -- deployed Mamba arch retrained WITHOUT the 3 predicted
#                        correction-DV inputs (mask minus 32-34, 914 params)
# Budgets (NOT 5000 gens: the paper itself shows that budget cannot resolve the
# tail -- section 6.1 -- and Appendix B measures 4-6 m/s pure-budget artifacts;
# an under-trained control is a rebuttable control):
#   window: 20000 gens -- the dense-family head keeps improving late (fig-plateau)
#   nodv:   15000 gens -- matches the deployed mamba_p962_long's ACTUAL budget
#           (15001 gens) exactly; mamba plateaus 10-15k
# at the headline allocation otherwise (n_sims=2 / n_pop=512), mirroring
# 10c_tail_sigma_repeats.sh so results compare to the sigma_run triplets.
#
# SEQUENTIAL-SEED PROTOCOL (pre-registered 2026-07-10, before any control ran):
# the loop runs window_s1 + nodv_s1 first (~11 h); Ctrl-C after nodv_s1 and
# evaluate. Decision rule: if a control's far-tail CVaR99.9 lands OUTSIDE the
# intact Mamba's confirmatory seed range (122.2-131.0, i.e. >= ~135, dense
# territory), the single run + single-run caveat suffices (the GRU precedent of
# section 6.2) and s2/s3 are optional polish; only if s1 lands INSIDE that
# range are s2/s3 required to separate the hypotheses. The reset-state control
# (CVaR99.9 123 -> 414 at capture parity) already carries the causal claim;
# these retrains corroborate.
#
# Resumable: skip-if-final_selection.json, auto-resume from checkpoint on crash
# (--n-gen is "additional" on resume, so a crash-resume trains past the target --
# harmless). Ctrl-C stops cleanly; re-run to resume. Each cell is ~4.5-6 h.
# Do NOT run concurrently with 16_sigma_extras.sh unless cores are plentiful,
# and never two cells of the same config TOML at once.
#
# AFTER training: the revision plan's Task 19 evaluates all 6 cells on the
# far-tail (n=10000) and confirmatory (10x100k) pools with pre-registered
# interpretation rules (docs/superpowers/plans/2026-07-10-reviewer-4-5-revision.md).

P="training_output/paper/state_controls"
trap 'echo; echo "Ctrl-C -- stopping (re-run to resume from the last checkpoint)"; exit 130' INT

run() {  # $1=config  $2=seed  $3=cell  $4=n_gen
  local out="$P/$3"
  if [ -f "$out/final_selection.json" ]; then
    echo "skip $3 (final_selection.json present -- already trained)"
    return 0
  fi
  echo "=== $3 (seed $2, $4 gens) -> $out ==="
  uv run python -m aerocapture.training.train "$1" \
      --training-n-sims 2 --n-gen "$4" --n-pop 512 --seed "$2" \
      --output-dir "$out" --sim-timeout 5 \
    || echo "WARNING: $3 exited non-zero -- continuing (re-run to retry)"
}

for S in 1 2 3; do
  run configs/training/paper/window_ctrl_p970.toml "$S" "window_s$S" 20000
  run configs/training/paper/mamba_p962_nodv.toml  "$S" "nodv_s$S"   15000
done
