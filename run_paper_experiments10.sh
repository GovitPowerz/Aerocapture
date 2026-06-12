#!/usr/bin/env bash
set -euo pipefail

# Study A REBUILD on the POST-FIX codebase + new defaults (cost_transform=cubed,
# curation_bucket_selection=max). The pre-fix optimizer runs used log + random
# bucket and a since-fixed guidance/training pipeline, so they are NOT comparable
# to the post-fix studies. This rebuilds the whole optimizer x budget matrix on
# dense_p3998 for a clean, internally-consistent comparison.
# n_pop: singles 60/150/300; islands 20/50/100 (x3). New dirs: paper_pf_<opt>_<budget>.
# GA@300 cubed+max already exists -> training_output/paper_optbig_ga_bucket_max (reuse, do NOT re-run).
# CMA-ES is O(n^2) at ~4000 params; @300 is very slow (left optional below).

T="configs/training/paper"
run() { uv run python -m aerocapture.training.train "$T/$1.toml" --n-gen 2000 --n-pop "$2" --output-dir "training_output/$3" --from-scratch; }

# ── @60 evals/gen ──
run optbig_islands 20  paper_pf_islands_60
run optbig_pso     60  paper_pf_pso_60
run optbig_de      60  paper_pf_de_60
run optbig_qpso    60  paper_pf_qpso_60
run optbig_cmaes   60  paper_pf_cmaes_60
run optbig_ga      60  paper_pf_ga_60

# ── @150 evals/gen ──
run optbig_islands 50  paper_pf_islands_150
run optbig_pso     150 paper_pf_pso_150
run optbig_de      150 paper_pf_de_150
run optbig_qpso    150 paper_pf_qpso_150
run optbig_cmaes   150 paper_pf_cmaes_150
run optbig_ga      150 paper_pf_ga_150

# ── @300 evals/gen (GA@300 = paper_optbig_ga_bucket_max, reuse) ──
run optbig_islands 100 paper_pf_islands_300
run optbig_pso     300 paper_pf_pso_300
run optbig_de      300 paper_pf_de_300
run optbig_qpso    300 paper_pf_qpso_300
# run optbig_cmaes 300 paper_pf_cmaes_300   # optional: very slow (O(n^2) covariance, popsize 300)
