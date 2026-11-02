#!/bin/bash
# Assemble the #232 committee cells from the deployed headline seeds (run from the repo root).
set -euo pipefail
M="training_output/ou_marginal"
MAMBA=("$M/hl_mamba_p962/best_model.json" "$M/hl_mamba_p962_s2/best_model.json" "$M/hl_mamba_p962_s3/best_model.json")
DENSE=("$M/hl_dense_p515/best_model.json" "$M/hl_dense_p515_s2/best_model.json" "$M/hl_dense_p515_s3/best_model.json")
for rule in circular_mean raw_sum medoid; do
    uv run python -m aerocapture.training.committee --rule "$rule" --out "training_output/committee/hl_mamba3_$rule" "${MAMBA[@]}"
done
uv run python -m aerocapture.training.committee --rule circular_mean --out training_output/committee/hl_mixed6_circular_mean "${MAMBA[@]}" "${DENSE[@]}"
# Post hoc (not pre-registered): the sign_leader pilot arms.
uv run python -m aerocapture.training.committee --rule sign_leader --out training_output/committee/hl_mamba3_sign_leader "${MAMBA[@]}"
uv run python -m aerocapture.training.committee --rule sign_leader --out training_output/committee/hl_mixed6_sign_leader "${MAMBA[@]}" "${DENSE[@]}"
