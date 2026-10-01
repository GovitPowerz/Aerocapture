#!/bin/bash
# Helpers shared by campaign.sh and classical_campaign.sh (sourced, not run).

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
    sys.exit(f"== {out} g{gen}: no seed_curator in the checkpoint; campaign cells train under adaptive seeds")
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
