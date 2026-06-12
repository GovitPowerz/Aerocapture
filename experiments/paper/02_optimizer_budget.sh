#!/usr/bin/env bash
set -euo pipefail
# Study A -- optimizer x budget on dense_p3998 (~4000 params), n_gen=2000.
# 6 optimizers x {60,150,300} evals/gen; islands n_pop is per-island (x3).
# ga_300 is the HEADLINE NN cell, and is reused as: Study D's cubed cell (05),
# C-sub's bucket-max cell (06), Study F's adaptive n_sims=10 anchor (08), and
# seed-repeat #1 (11). The @150 row doubles as Study C's adaptive column (04).
# Budget axis counts selection-driving evals only -- validation (~0.6x1000
# sims/gen measured) and curation (~1000 sims per <=2 gens) overheads are
# per-generation; report ACTUAL sims per cell from the JSONL.

T="configs/training/paper"
run() {  # $1=config-stem $2=n_pop $3=cell
  if [ -f "training_output/paper/optimizer_budget/$3/final_eval.parquet" ]; then echo "skip $3 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "$T/$1.toml" --n-gen 2000 --n-pop "$2" \
      --output-dir "training_output/paper/optimizer_budget/$3" --sim-timeout 5 --from-scratch
}

# ── @60 evals/gen ──
run dense_p3998_islands 20  islands_60
run dense_p3998_pso     60  pso_60
run dense_p3998_de      60  de_60
run dense_p3998_qpso    60  qpso_60
run dense_p3998_cmaes   60  cmaes_60
run dense_p3998_ga      60  ga_60

# ── @150 evals/gen (Study C adaptive column) ──
run dense_p3998_islands 50  islands_150
run dense_p3998_pso     150 pso_150
run dense_p3998_de      150 de_150
run dense_p3998_qpso    150 qpso_150
run dense_p3998_cmaes   150 cmaes_150
run dense_p3998_ga      150 ga_150

# ── @300 evals/gen ──
run dense_p3998_islands 100 islands_300
run dense_p3998_pso     300 pso_300
run dense_p3998_de      300 de_300
run dense_p3998_qpso    300 qpso_300
run dense_p3998_cmaes   300 cmaes_300   # very slow (O(n^2) covariance at ~4000 params)
run dense_p3998_ga      300 ga_300      # the headline NN cell
