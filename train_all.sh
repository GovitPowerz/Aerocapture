#!/usr/bin/env bash
set -euo pipefail

# Train all guidance schemes with optimized pymoo settings (real-valued GA by default).
# Usage:
#   ./train_all.sh                    # train all schemes in order
#   ./train_all.sh eqglide            # train a single scheme
#   ./train_all.sh ftc fnpag          # train specific schemes
#
# Piecewise constant must run first (produces ref trajectory + corridor).
# All others can run in any order.

TRAIN="uv run python -m aerocapture.training.train"

# Suppress adaptive integrator step-limit warnings (expected during GA exploration
# when the optimizer tries degenerate parameter combos that make the ODE stiff).
run_train() {
    $TRAIN "$@" 2> >(grep -v "WARNING: adaptive integrator hit" >&2)
}

train_piecewise_constant() {
    echo "=== piecewise_constant (11 params) ==="
    run_train configs/training/msr_aller_piecewise_constant_train.toml \
        --n-gen 3000 --n-pop 40 \
        --adaptive-seeds --cost-alpha 0.85 --cvar-percentile 5 \
        --seed-pool-cap 120 --stress-interval 15 --stress-probes 200 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_ftc() {
    echo "=== ftc (26 params) ==="
    run_train configs/training/msr_aller_ftc_train.toml \
        --n-gen 2500 --n-pop 50 \
        --adaptive-seeds --cost-alpha 0.65 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_eqglide() {
    echo "=== equilibrium_glide (24 params) ==="
    run_train configs/training/msr_aller_eqglide_train.toml \
        --n-gen 2500 --n-pop 60 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_energy_controller() {
    echo "=== energy_controller (20 params) ==="
    run_train configs/training/msr_aller_energy_controller_train.toml \
        --n-gen 2500 --n-pop 60 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_pred_guid() {
    echo "=== pred_guid (20 params) ==="
    run_train configs/training/msr_aller_pred_guid_train.toml \
        --n-gen 2500 --n-pop 60 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 150 --stress-interval 10 --stress-probes 300 --stress-inject 15 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_fnpag() {
    echo "=== fnpag (22 params) ==="
    run_train configs/training/msr_aller_fnpag_train.toml \
        --n-gen 600 --n-pop 50 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 100 --stress-interval 15 --stress-probes 150 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_neural_network() {
    echo "=== neural_network (458 params) ==="
    run_train configs/training/msr_aller_nn_train_consolidated.toml \
        --n-gen 1500 --n-pop 120 \
        --adaptive-seeds --cost-alpha 0.6 --cvar-percentile 15 \
        --seed-pool-cap 100 --stress-interval 15 --stress-probes 200 --stress-inject 10 \
        --final-n-sims 2000 --from-scratch --sim-timeout 1
}

train_all() {
    train_piecewise_constant
    echo ""
    train_ftc
    echo ""
    train_eqglide
    echo ""
    train_energy_controller
    echo ""
    train_pred_guid
    echo ""
    train_fnpag
    echo ""
    train_neural_network
}

# Dispatch: no args = all, otherwise run named schemes
if [ $# -eq 0 ]; then
    train_all
else
    for scheme in "$@"; do
        case "$scheme" in
            piecewise_constant|piecewise|pc)  train_piecewise_constant ;;
            ftc)                               train_ftc ;;
            eqglide|equilibrium_glide|eq)      train_eqglide ;;
            energy_controller|energy|ec)       train_energy_controller ;;
            pred_guid|predguid|pg)             train_pred_guid ;;
            fnpag)                             train_fnpag ;;
            neural_network|nn)                 train_neural_network ;;
            all)                               train_all ;;
            *)
                echo "Unknown scheme: $scheme"
                echo "Valid: piecewise_constant ftc eqglide energy_controller pred_guid fnpag neural_network all"
                exit 1
                ;;
        esac
    done
fi
