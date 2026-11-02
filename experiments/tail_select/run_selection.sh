#!/bin/bash
# #234: re-run the final selection of the six headline runs under the RMS rule and the tail-aware
# criterion (validation pool, then the 5000-seed tail-select pool), every pick into its own
# run-local dir under training_output/tail_select/<run>/<arm>/. The deployed artifacts under
# training_output/ou_marginal/ are never written (--output-dir). Run from the repo root:
#   caffeinate -i bash experiments/tail_select/run_selection.sh 2>&1 | tee training_output/tail_select/selection.log
set -euo pipefail
RUNS=(hl_mamba_p962 hl_mamba_p962_s2 hl_mamba_p962_s3 hl_dense_p515 hl_dense_p515_s2 hl_dense_p515_s3)
# Stop (Ctrl-C) and rerun at will: an arm whose final_selection.json exists is skipped.
run_arm() { # <run> <arm> <cli flags...>
    local run="$1" arm="$2"; shift 2
    local out="training_output/tail_select/$run/$arm"
    if [ -f "$out/final_selection.json" ]; then echo "== $run $arm: done, skipping"; return; fi
    echo "== $run $arm"
    uv run python -u -m aerocapture.training.final_select "training_output/ou_marginal/$run" --toml "configs/training/ou_marginal/$run.toml" --sim-timeout 5 --output-dir "$out" "$@"
}
for run in "${RUNS[@]}"; do
    run_arm "$run" rms_validation
    run_arm "$run" cvar95_validation --criterion cvar95 --pool validation
    run_arm "$run" cvar95_tail_select --criterion cvar95 --pool tail_select --pool-n 5000
done
echo "== done"
