#!/bin/bash
# OU-marginal NN campaign runner: trains every job of a jobs file under per_draw.
#
#   caffeinate -i experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_headline.txt
#
# Jobs file: one `name|target_gen|seed|checkpoint_source_dir` per line, value
# order, '#' comments. Job <name> trains configs/training/ou_marginal/<name>.toml
# into training_output/ou_marginal/<name>/ with trainer --seed <seed> (eval pools
# unchanged). An empty checkpoint_source_dir trains from scratch; otherwise the
# job copies <source>/checkpoint_g20000.{json,npz} once and continues from it
# (a fine-tune), stripping the copied rng_state when seed != 1: the resume path
# restores a checkpointed RNG and would silently override --seed (the first
# phase-2 repeats were bit-identical replays).
#
# Stoppable / resumable: stop any time (Ctrl+C, laptop shutdown: train.py
# checkpoints every 10 gens with atomic writes and saves on SIGINT); rerun and
# each job continues from its latest checkpoint with exact remaining-gens math;
# finished jobs are skipped; an interrupted job stops the whole campaign. A job
# is trained once final_selection.json is newer than its latest checkpoint (an
# interrupted final selection is finished by a zero-generation resume).
#
# Checks, each stopping the campaign: the resolved TOML's [data] neural_network
# is the run-local <out>/best_model.json; the job's latest checkpoint, before and
# after each pass, holds the TOML's population and a seed curator with
# training_n_sims bins; after a job, its best_model.json differs from every
# sibling seed's (<cell>, <cell>_sN). A pass checks its allocation only when it
# ends: to check a fresh job at its first checkpoint, Ctrl+C once
# checkpoint_g00010 exists and rerun (the resume checks it).
# After a job reaches its target, report.py writes the n = 1000 per-scenario
# final_eval.parquet when it is missing or older than best_model.json.
set -euo pipefail

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <jobs file>"
  exit 2
fi
JOBS=$(grep -v '^[[:space:]]*#' "$1" | grep -v '^[[:space:]]*$' || true)
for job in $JOBS; do
  if [ "$(echo "$job" | tr -cd '|')" != "|||" ]; then
    echo "== $1: '${job}' is not name|target_gen|seed|checkpoint_source_dir"
    exit 2
  fi
done
cd "$(dirname "$0")/../.."
SEED_GEN=20000

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
    sys.exit(f"== {out} g{gen}: no seed_curator in the checkpoint; the campaign trains under adaptive seeds")
bins = curator["n_bins"]
if (pop, bins) != (n_pop, n_sims):
    sys.exit(f"== {out} g{gen}: population {pop} x curator bins {bins}, the TOML says {n_pop} x {n_sims}")
print(f"== {out} g{gen}: population {pop} x curator bins {bins}")
' "$@"
}

selected() {  # $1=out $2=gen: final_selection.json written after checkpoint g$2 (ns mtimes; bash 3.2 -nt is whole seconds)
  uv run python -c '
import os, sys
sidecar, ckpt = f"{sys.argv[1]}/final_selection.json", f"{sys.argv[1]}/checkpoint_g{int(sys.argv[2]):05d}.json"
sys.exit(0 if os.path.exists(sidecar) and os.stat(sidecar).st_mtime_ns > os.stat(ckpt).st_mtime_ns else 1)
' "$@"
}

for job in $JOBS; do
  name="${job%%|*}"; rest="${job#*|}"
  target="${rest%%|*}"; rest="${rest#*|}"
  seed="${rest%%|*}"; src="${rest#*|}"
  toml="configs/training/ou_marginal/${name}.toml"
  out="training_output/ou_marginal/${name}"
  mkdir -p "$out"

  alloc=$(uv run python -c '
import sys
from aerocapture.training.toml_utils import load_toml_with_bases
t = load_toml_with_bases(sys.argv[1])
print(t["optimizer"]["n_pop"], t["optimizer"]["training_n_sims"], t["data"]["neural_network"])
' "$toml")
  read -r n_pop n_sims deploy <<< "$alloc"
  if [ "$deploy" != "$out/best_model.json" ]; then
    echo "== ${name}: [data] neural_network ${deploy} is not run-local to ${out}; stopping."
    exit 1
  fi

  last=$(latest_gen "$out")
  if [ -n "$src" ] && [ "$last" -eq 0 ]; then
    # The .json goes in last and whole: latest_gen keys on it, so a crash mid-seed
    # reseeds on rerun instead of resuming a copy that still holds the source RNG.
    ckpt="$out/checkpoint_g${SEED_GEN}.json"
    cp "$src/checkpoint_g${SEED_GEN}.npz" "$out/"
    cp "$src/checkpoint_g${SEED_GEN}.json" "$ckpt.tmp"
    if [ "$seed" != "1" ]; then
      uv run python -c 'import json, sys; p = sys.argv[1]; d = json.load(open(p)); d["rng_state"] = None; json.dump(d, open(p, "w"))' \
        "$ckpt.tmp"
      echo "== ${name}: stripped rng_state (trainer seed ${seed} takes effect)"
    fi
    mv "$ckpt.tmp" "$ckpt"
    last=$SEED_GEN
    echo "== ${name}: seeded checkpoint g${SEED_GEN} from ${src}"
  fi
  # A copied source checkpoint holds the source's allocation; the resume re-allocates it.
  if [ "$last" -gt 0 ] && { [ -z "$src" ] || [ "$last" -gt "$SEED_GEN" ]; }; then
    check_alloc "$out" "$last" "$n_pop" "$n_sims"
  fi

  if [ "$last" -lt "$target" ] || ! selected "$out" "$last"; then
    remaining=$((target > last ? target - last : 0))
    echo "== ${name}: at g${last}, training ${remaining} more gens (target ${target}, seed ${seed}, ${n_pop} x ${n_sims})"
    # --n-gen means "N additional" on resume, so remaining-to-target is exact;
    # 0 only re-runs the final selection. tee -i: a Ctrl+C must not close the
    # pipe while train.py saves its checkpoint.
    uv run python -u -m aerocapture.training.train "$toml" \
      --n-gen "$remaining" --seed "$seed" --sim-timeout 5 --no-tui --skip-report \
      --output-dir "$out" \
      2>&1 | tee -ai "$out/campaign.log"
    now=$(latest_gen "$out")
    if [ "$now" -gt "$SEED_GEN" ] || { [ -z "$src" ] && [ "$now" -gt 0 ]; }; then
      check_alloc "$out" "$now" "$n_pop" "$n_sims"
    fi
    # train.py exits 0 on Ctrl+C (clean checkpoint save): short of the target or
    # of a final selection means the session was interrupted, so stop instead of
    # starting the next job.
    if [ "$now" -lt "$target" ] || ! selected "$out" "$now"; then
      echo "== ${name}: stopped at g${now} before target; exiting campaign (rerun this script to continue)."
      exit 0
    fi
    echo "== ${name}: reached g${now}"
  else
    echo "== ${name}: trained (g${last} >= ${target})"
  fi

  cell="${name%_s[0-9]}"
  for sib in "training_output/ou_marginal/${cell}" "training_output/ou_marginal/${cell}"_s[0-9]; do
    if [ "$sib" != "$out" ] && [ -f "$sib/best_model.json" ] && cmp -s "$sib/best_model.json" "$out/best_model.json"; then
      echo "== ${name}: best_model.json is byte-identical to ${sib}'s (a seed repeat replayed its sibling); stopping."
      exit 1
    fi
  done

  if [ ! -f "$out/final_eval.parquet" ] || [ "$out/best_model.json" -nt "$out/final_eval.parquet" ]; then
    echo "== ${name}: n = 1000 final evaluation + report"
    uv run python -u -m aerocapture.training.report "$out" --toml "$toml" --sim-timeout 5 \
      2>&1 | tee -ai "$out/campaign.log"
    if [ ! -f "$out/final_eval.parquet" ]; then
      echo "== ${name}: report wrote no final_eval.parquet; stopping."
      exit 1
    fi
  fi
done
echo "Campaign $1 pass complete."
