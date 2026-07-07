#!/usr/bin/env bash
set -euo pipefail
# Architecture Pareto sweep -- all 6 families (dense/GRU/LSTM/Mamba/Transformer/
# Window) x ~4 parameter budgets, GA + the post-fix campaign defaults (adaptive /
# bucket=max / cost_transform=cubed, inherited from the sweep TOMLs in
# configs/training/sweep/; the old committed sweep was islands + log + pre-fix
# guidance, hence the re-run). Configs + manifest already exist there.
#
# Explicit per-cell calls (like 08_training_n_sims.sh) so any cell can be
# commented out / re-run individually, and RESUMABLE:
#   - a cell is SKIPPED once it has final_selection.json -- the end-only
#     completion marker (best_model.json is rewritten EVERY checkpoint, so it is
#     NOT a valid "done" sentinel: param_sweep's own skip check is unsafe here);
#   - a PARTIAL cell (checkpoints but no final_selection.json) is auto-resumed by
#     train.py (no --resume needed). NB on train.py's resume contract: --n-gen
#     means "N ADDITIONAL gens", so a crash-resumed cell trains PAST 8000 -- only
#     a problem if you need an exactly-equal budget across cells; to cold-start a
#     cell instead, `rm -rf` its training_output/sweep_<arch>_p<N> first.
# A cell that crashes is WARNED and skipped (re-run to retry); Ctrl-C stops the
# whole sweep cleanly (train.py exits 0 on graceful interrupt, so without the
# trap bash would march on to the next cell).
#
# Allocation (the campaign learning): n_sims=5 is the efficiency sweet spot
# (n_sims=2 is the BEST allocation but ~2x the wall-time -- reserved for the
# deployed headline, prohibitive across 24 cells). n_gen=8000 plateaus the
# ~500-2000 cells; the ~4000-param cells need more and will be slightly
# under-trained -- footnote it (consistent with "bigger nets are harder for GA",
# the 515-vs-972 result). n_pop=300 holds for ALL cells: Study A showed GA@60
# COLLAPSES at ~4000 params, so the big cells need the wide population.
# save_dir is derived from each config's [data] neural_network parent
# (training_output/sweep_<arch>_p<N>), distinct per cell -- no --output-dir needed.
# ~24 runs, the longest script in the campaign after 02 (2-4 days).

TRAINING_N_SIMS=2
N_GEN=5000
N_POP=512

trap 'echo; echo "Ctrl-C -- stopping the sweep (re-run to resume from the last completed cell)"; exit 130' INT

run() {  # $1=arch  $2=params (matches the configs/training/sweep/<arch>_p<N>.toml cells)
  local cfg="configs/training/sweep/$1_p$2.toml"
  local out="training_output/sweep_$1_p$2"
  if [ -f "$out/final_selection.json" ]; then
    echo "skip $1 p$2 (final_selection.json present -- already trained)"
    return 0
  fi
  echo "=== train $1 p$2 -> $out ==="
  uv run python -m aerocapture.training.train "$cfg" \
      --training-n-sims "$TRAINING_N_SIMS" --n-gen "$N_GEN" --n-pop "$N_POP" \
      --sim-timeout 5 \
    || echo "WARNING: $1 p$2 exited non-zero -- continuing to next cell (re-run to retry)"
}

# ── dense ──
run dense 515
run dense 972
run dense 1957
run dense 3998
# ── gru ──
run gru 478
run gru 1014
run gru 1954
run gru 4082
# ── lstm ──
run lstm 458
run lstm 1082
run lstm 1962
run lstm 4118
# ── mamba ──
run mamba 482
run mamba 962
run mamba 2027
run mamba 4072
# ── transformer ──
run transformer 762
run transformer 1112
run transformer 2004
run transformer 3822
# ── window ──
run window 609
run window 1027
run window 2036
run window 4025

# Re-score every deployed point on the shared sweep-eval pool (7M) WITH each
# cell's co-trained best_params.json scaffolding, and render the Pareto SVGs.
# Idempotent -- safe to re-run after a partial training pass.
uv run python -m aerocapture.training.param_sweep --eval --plot
# Capability-collapse view (higher-better capture-rate upper envelope) over the
# sub-500 floor manifest (study 09's dense_p102/201/298/416 + the 515/3998 anchors).
uv run python -m aerocapture.training.param_sweep --eval --plot --out-tag floor --metric capture_rate
