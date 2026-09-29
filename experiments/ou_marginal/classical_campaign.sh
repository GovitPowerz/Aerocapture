#!/bin/bash
# Classical retune under per-scenario noise (#172): the nine classical cells of
# experiments/paper/01_classical_baselines.sh and 07_joint_reference.sh re-tuned
# by the same GA allocation (2000 gens x 300 individuals x 10 scenarios, adaptive
# seeds; FNPAG 300 gens) under [monte_carlo] noise_seeding = "per_draw", into
# training_output/ou_marginal/classical/<cell>/. The canonical
# training_output/<scheme>/ dirs are the shared-path bundle's inputs and are
# never written.
#
# Same stoppable/resumable contract as retrain_campaign.sh: run it; stop it any
# time (Ctrl+C, laptop shutdown: train.py checkpoints every 10 gens with atomic
# writes and saves on SIGINT); rerun and each cell continues from its latest
# checkpoint with exact remaining-gens math; finished cells are skipped; an
# interrupted job stops the whole campaign. A cell's target, population and
# scenario count come from its TOML ([optimizer] n_gen, n_pop, training_n_sims)
# and are checked against its latest checkpoint before and after every pass.
# A cell is trained once final_selection.json is newer than its latest
# checkpoint: a Ctrl+C during the final selection re-saves the checkpoint and
# writes no sidecar, and the next pass finishes it with a zero-generation
# resume (the same selection an uninterrupted run makes).
#
# After a cell reaches its target, report.py writes the n = 1000 per-scenario
# final_eval.parquet (and report.pdf) when it is missing or older than the
# cell's optimized_<scheme>.toml.
#
# Run from the Terminal panel:
#   caffeinate -i experiments/ou_marginal/classical_campaign.sh
# Cost: about 18 h on the M4 Pro (2026-09-29 run), FNPAG about 11.5 h of it.
# Sanity table afterwards:
#   uv run python experiments/ou_marginal/quote_marginal.py --manifest experiments/ou_marginal/classical_cells.txt
set -euo pipefail
cd "$(dirname "$0")/../.."

# piecewise_constant first: report.py overlays its sibling's corridor_boundaries.npz on every later cell.
CELLS="piecewise_constant ftc energy_controller pred_guid ftc_joint energy_controller_joint pred_guid_joint equilibrium_glide fnpag"
REF=training_output/mars/ref_trajectory.dat

latest_gen() {
  local g
  # The .json is the trainer's resume key (renamed into place after the .npz), so an
  # npz-only label from a crash mid-save is not counted. `|| true`: grep exits 1 on a
  # fresh dir; force base 10 on the zero-padded label.
  g=$( (ls "$1" 2>/dev/null | grep -o 'checkpoint_g[0-9]*\.json' | grep -o '[0-9]*' | sort -n | tail -1) || true)
  echo $((10#${g:-0}))
}

check_alloc() {  # $1=out $2=gen $3=n_pop $4=training_n_sims
  uv run python -c '
import json, sys
import numpy as np
out, gen, n_pop, n_sims = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pop = np.load(f"{out}/checkpoint_g{gen:05d}.npz")["population"].shape[0]
curator = json.load(open(f"{out}/checkpoint_g{gen:05d}.json")).get("seed_curator")
if curator is None:
    sys.exit(f"== {out} g{gen}: no seed_curator in the checkpoint; the classical cells train under adaptive seeds")
bins = curator["n_bins"]
if (pop, bins) != (n_pop, n_sims):
    sys.exit(f"== {out} g{gen}: population {pop} x curator bins {bins}, the TOML says {n_pop} x {n_sims}")
' "$@"
}

selected() {  # $1=out $2=gen: final_selection.json written after checkpoint g$2 (ns mtimes; bash 3.2 -nt is whole seconds)
  uv run python -c '
import os, sys
sidecar, ckpt = f"{sys.argv[1]}/final_selection.json", f"{sys.argv[1]}/checkpoint_g{int(sys.argv[2]):05d}.json"
sys.exit(0 if os.path.exists(sidecar) and os.stat(sidecar).st_mtime_ns > os.stat(ckpt).st_mtime_ns else 1)
' "$@"
}

for cell in $CELLS; do
  toml="configs/training/ou_marginal/classical/${cell}.toml"
  out="training_output/ou_marginal/classical/${cell}"
  mkdir -p "$out"
  # The fixed-reference cells fly the tracked mission reference: never train on a modified one.
  if ! git diff --quiet HEAD -- "$REF"; then
    echo "== ${REF} differs from git: restore it (git checkout -- ${REF}) before rerunning."
    exit 1
  fi
  alloc=$(uv run python -c '
import sys
from aerocapture.training.toml_utils import load_toml_with_bases
t = load_toml_with_bases(sys.argv[1])
o = t["optimizer"]
print(t["guidance"]["type"], o["n_gen"], o["n_pop"], o["training_n_sims"])
' "$toml")
  read -r scheme target n_pop n_sims <<< "$alloc"

  last=$(latest_gen "$out")
  if [ "$last" -gt 0 ]; then
    check_alloc "$out" "$last" "$n_pop" "$n_sims"
  fi
  if [ "$last" -lt "$target" ] || ! selected "$out" "$last"; then
    remaining=$((target > last ? target - last : 0))
    echo "== ${cell}: at g${last}, training ${remaining} more gens (target ${target}, ${n_pop} x ${n_sims})"
    # --n-gen means "N additional" on resume, so remaining-to-target is exact;
    # 0 only re-runs the final selection. tee -i: a Ctrl+C must not close the
    # pipe while train.py saves its checkpoint.
    uv run python -u -m aerocapture.training.train "$toml" \
      --n-gen "$remaining" --sim-timeout 5 --no-tui --skip-report \
      --output-dir "$out" \
      2>&1 | tee -ai "$out/campaign.log"
    now=$(latest_gen "$out")
    check_alloc "$out" "$now" "$n_pop" "$n_sims"
    # train.py exits 0 on Ctrl+C (clean checkpoint save): short of the target or
    # of a final selection means the session was interrupted, so stop instead of
    # starting the next cell.
    if [ "$now" -lt "$target" ] || ! selected "$out" "$now"; then
      echo "== ${cell}: stopped at g${now} before target; exiting campaign (rerun this script to continue)."
      exit 0
    fi
    echo "== ${cell}: reached g${now}"
  else
    echo "== ${cell}: trained (g${last} >= ${target})"
  fi

  opt="$out/optimized_${scheme}.toml"
  if [ ! -f "$opt" ]; then
    echo "== ${cell}: no ${opt} after training; stopping."
    exit 1
  fi
  if [ ! -f "$out/final_eval.parquet" ] || [ "$opt" -nt "$out/final_eval.parquet" ]; then
    echo "== ${cell}: n = 1000 final evaluation + report"
    uv run python -u -m aerocapture.training.report "$out" --toml "$toml" --sim-timeout 5 \
      2>&1 | tee -ai "$out/campaign.log"
    if [ ! -f "$out/final_eval.parquet" ]; then
      echo "== ${cell}: report wrote no final_eval.parquet; stopping."
      exit 1
    fi
  fi
done
echo "Classical campaign pass complete."
