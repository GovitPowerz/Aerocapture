#!/usr/bin/env bash
set -euo pipefail
# Architecture sweep FOLLOW-UP: train the genuine recurrent contenders to the
# HEADLINE depth, to answer "does any recurrent arch beat the small dense once
# trained as long as the dense headline?" -- the reviewer's first objection to
# "dense is best", since at the sweep's equal 5000-gen budget the recurrent cells
# are competitive-to-better (lstm_p1082 dv50 111.6, gru_p4082 112.2, mamba ~114
# vs the best dense cell 115.8). The dense advantage so far is the LONG budget
# (dense_p515 headline: dv50 109.7 / val RMS 1.326M @ n=2/20000 gens), which the
# sweep never gave the recurrent archs.
#
# Each cell is the sweep cell EXTENDED to 20000 gens at the EXACT headline
# allocation: n_sims=2, n_pop=512 (verified from the headline checkpoint:
# population (512, 518)), continuous to 20000. We SEED from the sweep cell's
# 5000-gen checkpoint (reuse, don't redo) into a dedicated <arch>_p<N>_long dir
# via --output-dir, so the sweep's Pareto points (training_output/sweep_*) stay
# intact, then resume +15000 gens (= 20000 total). Report ON (no --skip-report)
# so each cell gets final_eval.parquet on the SAME 2M pool as the headline ->
# directly paired-comparable.
#
# RESUMABLE: a cell is skipped once its _long dir has final_selection.json (the
# end-only completion marker). A crash-interrupted cell auto-resumes from its last
# checkpoint -- NB --n-gen is "additional", so a crash-resume trains PAST 20000
# (harmless: more gens only help; watch val RMS in the TUI and Ctrl-C at plateau).
# Ctrl-C stops the whole run cleanly (train.py exits 0 on graceful interrupt).
#
# STRATEGY: cells are ordered strongest-first (lstm > mamba > gru by the 1000-ish
# sweep cells). lstm_p1082 is the single best sweep cell -- run it FIRST; if the
# strongest contender fully trained still loses to the 515-dense, the weaker two
# won't surpass it (early-stop: Ctrl-C after lstm+mamba). gru_p1014 was the
# WEAKEST ~1000 cell (gru peaked at 4082, not 1000) -- least likely to surprise.
# Each cell is ~a full headline run (multi-day); budget accordingly.

trap 'echo; echo "Ctrl-C -- stopping (re-run to resume from the last checkpoint)"; exit 130' INT

run() {  # $1=arch  $2=params  (extends the configs/training/sweep/<arch>_p<N>.toml sweep cell)
  local cfg="configs/training/sweep/$1_p$2.toml"
  local sweep="training_output/sweep_$1_p$2"
  local out="training_output/$1_p$2_long"
  if [ -f "$out/final_selection.json" ]; then
    echo "skip $1 p$2 (long run already complete -- final_selection.json present)"
    return 0
  fi
  # First launch: seed the long run from the sweep cell's 5000-gen state. Copy the
  # checkpoint pair + run log (for the full 0->20000 val-RMS history) but NOT
  # final_selection.json, so train.py RESUMES instead of marking the copy done.
  if [ ! -d "$out" ]; then
    if [ ! -f "$sweep/checkpoint_g05000.npz" ]; then
      echo "ERROR: $sweep has no g05000 checkpoint to extend -- run 10_architecture_sweep.sh first"
      return 1
    fi
    mkdir -p "$out"
    cp "$sweep"/checkpoint_g05000.json "$sweep"/checkpoint_g05000.npz "$out"/
    cp "$sweep"/best_model.json "$sweep"/best_params.json "$out"/ 2>/dev/null || true
    cp "$sweep"/run_*.jsonl "$out"/ 2>/dev/null || true
    echo "seeded $out from $sweep @ gen 5000 (reusing the sweep's first 5000 gens)"
  fi
  echo "=== extend $1 p$2 to 20000 gens (n_sims=2, n_pop=512) -> $out ==="
  uv run python -m aerocapture.training.train "$cfg" \
      --training-n-sims 2 --n-gen 15000 --n-pop 512 --output-dir "$out" --sim-timeout 5 \
    || echo "WARNING: $1 p$2 exited non-zero -- continuing (re-run to retry)"
}

run lstm  1082   # strongest sweep cell (dv50 111.6 @ 5000 gens) -- the contender to beat
run mamba 962    # most CONSISTENT recurrent arch across budgets (~114) -- likely the real edge
run gru   1014   # weakest ~1000 cell (gru peaked at 4082); least likely to surprise -- run last / skip

# Compare each long challenger to the 515-dense headline on the SHARED 2M pool:
#   uv run python -c "import pyarrow.parquet as pq, numpy as np; \
#     a=pq.read_table('training_output/lstm_p1082_long/final_eval.parquet').to_pandas(); \
#     b=pq.read_table('training_output/dense_p515_ga_paper_best/final_eval.parquet').to_pandas(); \
#     cap=lambda d:(d.ifinal==3)&(d.eccentricity<1.0); \
#     print('lstm', a.dv_total_m_s[cap(a)].median(), 'vs dense515', b.dv_total_m_s[cap(b)].median())"
# (the headline plateaued at val RMS 1.326M / dv50 109.7 -- the number to beat).
