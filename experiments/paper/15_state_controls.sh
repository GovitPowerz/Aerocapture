#!/usr/bin/env bash
set -euo pipefail
# State-ablation controls: the retrained controls that, with the reset-state eval flag, decide
# whether the Mamba's sizing-tail win is INTERNAL STATE or something else:
#   window_ctrl_p970  -- explicit 5-tick observation history, no learned state
#                        (Window(17x5) -> Dense stack, 970 params ~ Mamba's 962)
#   mamba_p962_nodv   -- deployed Mamba arch retrained WITHOUT the 3 predicted
#                        correction-DV inputs (mask minus 32-34, 914 params)
#
#   15_state_controls.sh v4       the v4 champion (issue #176, default): everything below
#   15_state_controls.sh shared   the arxiv-v3 shared-path study (training_output/paper/state_controls)
#
# ===== v4 (issue #176): mechanism and deployment controls on the v4 champion =====
# The champion is seed 1 of #173's rule: the scratch cell training_output/ou_marginal/hl_mamba_p962.
# Every step flies under per-scenario noise (per_draw, ADR-0006) and is skipped once its output exists
# and states that flight (an output stating another regime or transform stops the runner), so the
# command resumes after a stop (Ctrl-C, shutdown) at the step it was in:
#   1. reset state   confirmatory_marginal.py on the champion with reset_state_every_tick = true
#                    (label ou_marginal/v4_reset_state, 10^6 pool; claim: capture parity, collapsed tail).
#   3. ablation      aerocapture.training.ablation -> <champion>/ablation_results.json (n = 1000,
#                    cost transform log: the paper's ablation figure is in log-transform units).
#   5. fresh pool    fresh_pool_requote.py --noise-seeding per_draw -> <champion>/fresh_pool_requote.json
#                    (8M pool, n = 1000; the selection-optimism check next to the 2M-pool numbers).
#   4. stress depth  stress_depth_eval.py -> articles/paper/data/stress_depth.json (9M pool, n = 10000,
#                    the champion + the #172 retunes, ~30 min, FNPAG-dominated).
#   2. controls      campaign.sh experiments/ou_marginal/jobs_controls.txt: ctrl_window_p970 and
#                    ctrl_mamba_p962_nodv from scratch at the champion's allocation and budget
#                    (GA 512 x 2, 20000 gens, 4-7 h each), then their seed 2 / 3 repeats; every
#                    finished control on the 10^6 pool (an unfinished one is skipped until a rerun).
#   6. compute       regime-independent: compute_benchmark.json is kept unless the release binary changed
#                    (rustc / crate version); the script prints the recorded and current rustc.
# Quick steps first (minutes) so their records post while the retrains drain.
#
# PRE-REGISTERED DECISION RULE (before either v4 control ran, 2026-10-06): the controls are single seeds.
# A control whose pooled CVaR99.9 lands OUTSIDE the intact champion's three-seed range
# (hl_mamba_p962 / _s2 / _s3 at 10^6: 142.3-173.9 m/s, #174) needs no repeats; one INSIDE it gets
# seeds 2 and 3 before the paper reads it (confirmatory_marginal.py --table prints the rule's outcome
# as `#176 controls rule`). The md5 and allocation checks are campaign.sh's.
# OUTCOME (2026-10-06): both seed-1 controls landed inside (window 158.3, no-predicted-DV 145.6), so
# jobs_controls.txt carries seeds 2 and 3 of both.
#
# ===== shared (arxiv-v3): the same two controls on the shared-path champion =====
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
# interpretation rules (git show ee1518a^:docs/superpowers/plans/2026-07-10-reviewer-4-5-revision.md).

cd "$(dirname "$0")/../.."
MODE="${1:-v4}"
trap 'echo; echo "Ctrl-C -- stopping (re-run to resume)"; exit 130' INT

# $1 = a step's output, then the '"key": "value"' lines it must hold: false when absent, true (skip)
# when it states this step's flight, and a stop when it states another (existence proves nothing).
done_as() {
  local f="$1" kv
  shift
  [ -f "$f" ] || return 1
  for kv in "$@"; do
    grep -qF "$kv" "$f" || { echo "$f does not state $kv: move it aside and rerun"; exit 1; }
  done
  echo "skip: $f present"
}

v4() {
  local champ=training_output/ou_marginal/hl_mamba_p962
  local toml=configs/training/ou_marginal/hl_mamba_p962.toml
  local cm=experiments/ou_marginal/confirmatory_marginal.py
  local controls="" c s
  for c in ctrl_window_p970 ctrl_mamba_p962_nodv; do
    for s in "" _s2 _s3; do controls="$controls ou_marginal/$c$s:configs/training/ou_marginal/$c$s.toml"; done
  done
  local reset="ou_marginal/v4_reset_state:${toml}:${champ}"
  [ -f "$champ/final_selection.json" ] || { echo "$champ is not trained (#173 first)"; exit 1; }

  echo "=== 1. reset-state eval (10^6 pool, per_draw) ==="
  uv run python -u "$cm" --cells "$reset" --extra-override guidance.neural_network.reset_state_every_tick=true

  echo "=== 3. input ablation (n = 1000, the TOML's regime, log cost transform) ==="
  if ! done_as "$champ/ablation_results.json" '"noise_seeding": "per_draw"' '"cost_transform": "log"'; then
    uv run python -u -m aerocapture.training.ablation "$champ" --toml "$toml" --n-sims 1000 --sim-timeout 5 --cost-transform log
  fi

  echo "=== 5. fresh-pool re-quote (8M pool, n = 1000, per_draw) ==="
  if ! done_as "$champ/fresh_pool_requote.json" '"noise_seeding": "per_draw"'; then
    uv run python -u articles/paper/scripts/fresh_pool_requote.py "$champ" --toml "$toml" --n-sims 1000 --noise-seeding per_draw
  fi

  echo "=== 4. off-nominal stress at depth (9M pool, n = 10000, per_draw) ==="
  if ! done_as articles/paper/data/stress_depth.json '"noise_seeding": "per_draw"'; then
    uv run python -u articles/paper/scripts/stress_depth_eval.py --n-sims 10000
  fi

  echo "=== 2. controls: retrain at the champion's allocation, then 10^6 ==="
  experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_controls.txt
  # shellcheck disable=SC2086  # six label:toml specs; an unfinished run is skipped
  uv run python -u "$cm" --cells $controls
  # shellcheck disable=SC2086
  uv run python -u "$cm" --table --cells $reset $controls

  echo "=== 6. compute benchmark: regime-independent ==="
  echo "recorded: $(uv run python -c 'import json; print(json.load(open("articles/paper/data/compute_benchmark.json"))["meta"]["rustc"])')"
  echo "current:  $(rustc --version)"
  echo "(differ -> make -C articles/paper mc-compute-benchmark on an idle box, all schemes in one session)"
}

shared() {
  local P="training_output/paper/state_controls"
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
}

case "$MODE" in
  v4) v4 ;;
  shared) shared ;;
  *) echo "usage: $0 [v4|shared]"; exit 2 ;;
esac
