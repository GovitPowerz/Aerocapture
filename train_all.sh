#!/usr/bin/env bash
set -euo pipefail

COMMON="--n-gen 100 --n-pop 50 --adaptive-seeds -fs --sim-timeout 30"

# Piecewise constant first (produces ref trajectory + corridor for other schemes)
echo "Training piecewise constant scheme..."
uv run python -m aerocapture.training.train configs/training/msr_aller_piecewise_constant_train.toml $COMMON
echo "\n\n"

# Independent schemes (no ref trajectory needed)
echo "Training independent schemes..."
uv run python -m aerocapture.training.train configs/training/msr_aller_nn_train_consolidated.toml $COMMON
echo "\n\n"
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml $COMMON
echo "\n\n"

# Schemes that require ref trajectory
echo "Training schemes that require reference trajectory..."
uv run python -m aerocapture.training.train configs/training/msr_aller_energy_controller_train.toml $COMMON
echo "\n\n"
uv run python -m aerocapture.training.train configs/training/msr_aller_pred_guid_train.toml $COMMON
echo "\n\n"
uv run python -m aerocapture.training.train configs/training/msr_aller_fnpag_train.toml $COMMON
echo "\n\n"
uv run python -m aerocapture.training.train configs/training/msr_aller_ftc_train.toml $COMMON
echo "\n\n"
