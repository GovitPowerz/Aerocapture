#!/usr/bin/env bash
set -euo pipefail

# Study A REBUILD on the POST-FIX codebase + new defaults (cost_transform=cubed,
# curation_bucket_selection=max, seed_pool_interval=2, curation_top_k=1).
# The pre-fix optimizer runs used log + random bucket + a since-fixed
# guidance/training pipeline, so they are NOT comparable to the post-fix studies.
# Full matrix re-run INCLUDING GA@300 and CMA-ES@300, so every cell shares one
# regime: paper_optbig_ga_bucket_max (trained at seed_pool_interval=3 /
# curation_top_k=2, before HEAD's default change) is SUPERSEDED as the headline
# cell by paper_pf_ga_300.
# n_pop: singles 60/150/300; islands 20/50/100 (x3). New dirs: paper_pf_<opt>_<budget>.
# Skip-if-done: re-running after a crash skips completed cells instead of
# wiping them (--from-scratch would rmtree); delete a cell's dir to force a rerun.
# --sim-timeout 5: never fires for healthy sims, caps the documented NaN-hang.

T="configs/training/paper"
run() {
  if [ -f "training_output/$3/final_eval.parquet" ]; then echo "skip $3 (done)"; return 0; fi
  uv run python -m aerocapture.training.train "$T/$1.toml" --n-gen 2000 --n-pop "$2" \
      --output-dir "training_output/$3" --sim-timeout 5 --from-scratch
}

# ── @60 evals/gen ──
run optbig_islands 20  paper_pf_islands_60
run optbig_pso     60  paper_pf_pso_60
run optbig_de      60  paper_pf_de_60
run optbig_qpso    60  paper_pf_qpso_60
run optbig_cmaes   60  paper_pf_cmaes_60
run optbig_ga      60  paper_pf_ga_60

# ── @150 evals/gen (also Study C's adaptive column -- see run_paper_experiments5.sh) ──
run optbig_islands 50  paper_pf_islands_150
run optbig_pso     150 paper_pf_pso_150
run optbig_de      150 paper_pf_de_150
run optbig_qpso    150 paper_pf_qpso_150
run optbig_cmaes   150 paper_pf_cmaes_150
run optbig_ga      150 paper_pf_ga_150

# ── @300 evals/gen ──
run optbig_islands 100 paper_pf_islands_300
run optbig_pso     300 paper_pf_pso_300
run optbig_de      300 paper_pf_de_300
run optbig_qpso    300 paper_pf_qpso_300
run optbig_cmaes   300 paper_pf_cmaes_300   # very slow (O(n^2) covariance at ~4000 params)
run optbig_ga      300 paper_pf_ga_300      # the new headline NN cell
