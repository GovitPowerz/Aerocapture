#!/usr/bin/env bash
set -euo pipefail

# Batch 2 -- methodology fixes (2026-06-09):
#  (a) Optimizer comparison ALSO on the BIG net (dense_p3998, ~4000 params).
#      The small net did NOT separate optimizers (islands ~= CMA-ES ~= GA), so the
#      big net is the discriminating test of the local-minima hypothesis.
#      Compute-matched: single optimizers n_pop=300, islands/warm-start n_pop=100.
#  (b) Retrain ALL classical schemes with islands for a FAIR classical-vs-NN table
#      (the committed classical were GA-trained; the NN sweep was islands-trained).
#      New dirs via --output-dir; the shared reference trajectory
#      (data/reference_trajectory/msr_aller.dat) is a fixed committed file, so there
#      is no cascade and the NN sweep stays valid.

# ── Study A (small net): complete the table with PSO ──
# uv run python -m aerocapture.training.train configs/training/paper/opt_pso.toml --n-gen 2000 --n-pop 300 --from-scratch

# ── Study A (big net, dense_p3998): optimizer comparison ──
# uv run python -m aerocapture.training.train configs/training/paper/optbig_islands.toml --n-gen 2000 --n-pop 20 --from-scratch
# uv run python -m aerocapture.training.train configs/training/paper/optbig_pso.toml       --n-gen 2000 --n-pop 60 --from-scratch
# uv run python -m aerocapture.training.train configs/training/paper/optbig_ga.toml        --n-gen 2000 --n-pop 60 --from-scratch
# uv run python -m aerocapture.training.train configs/training/paper/optbig_de.toml        --n-gen 2000 --n-pop 60 --from-scratch
# uv run python -m aerocapture.training.train configs/training/paper/optbig_cmaes.toml     --n-gen 2000 --n-pop 60 --from-scratch  # slow: O(n^2) covariance at ~4000 params
# big-net islands baseline already exists: training_output/sweep_dense_p3998

# ── Classical schemes retrained with islands (new dirs; fair vs the islands-trained NN) ──
# uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml                --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/msr_aller_fnpag_train.toml              --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/msr_aller_pred_guid_train.toml          --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/msr_aller_energy_controller_train.toml  --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml            --algorithm ga --n-gen 2000 --n-pop 300 --from-scratch
