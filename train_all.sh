#!/usr/bin/env bash
set -euo pipefail

# Train all guidance schemes with optimized pymoo settings.
# Usage:
#   ./train_all.sh                              # train all schemes in order
#   ./train_all.sh eqglide                      # train a single scheme
#   ./train_all.sh ftc fnpag                    # train specific schemes
#   ./train_all.sh --algorithm de               # train all with DE optimizer
#   ./train_all.sh eqglide --n-gen 500          # override n-gen for eqglide
#   ./train_all.sh --n-pop 80 --algorithm pso   # all schemes, PSO, pop=80
#
# Optional flags (override per-scheme defaults when provided):
#   --n-gen N           Override generation count
#   --n-pop N           Override population size
#   --final-n-sims N    Override final evaluation sims
#   --sim-timeout N     Override per-sim timeout (seconds)
#   --algorithm ALG     Override optimizer (ga, de, cma_es, pso)
#   --from-scratch      Start fresh (no checkpoint resume)
#
# Piecewise constant must run first (produces ref trajectory + corridor).
# All others can run in any order.

TRAIN="uv run python -m aerocapture.training.train"

# Parse optional flags and scheme names from arguments
SCHEMES=()
OPT_N_GEN=""
OPT_N_POP=""
OPT_FINAL_N_SIMS=""
OPT_SIM_TIMEOUT=""
OPT_ALGORITHM=""
OPT_FROM_SCRATCH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --n-gen)         OPT_N_GEN="$2"; shift 2 ;;
        --n-pop)         OPT_N_POP="$2"; shift 2 ;;
        --final-n-sims)  OPT_FINAL_N_SIMS="$2"; shift 2 ;;
        --sim-timeout)   OPT_SIM_TIMEOUT="$2"; shift 2 ;;
        --algorithm|-alg) OPT_ALGORITHM="$2"; shift 2 ;;
        --from-scratch|-fs) OPT_FROM_SCRATCH="1"; shift ;;
        *)               SCHEMES+=("$1"); shift ;;
    esac
done

# Suppress adaptive integrator step-limit warnings (expected during GA exploration
# when the optimizer tries degenerate parameter combos that make the ODE stiff).
run_train() {
    $TRAIN "$@" 2> >(grep -v "WARNING: adaptive integrator hit" >&2)
}

# Build the extra flags string from overrides
build_extra_flags() {
    local extra=""
    [ -n "$OPT_ALGORITHM" ]     && extra="$extra --algorithm $OPT_ALGORITHM"
    [ -n "$OPT_FROM_SCRATCH" ]  && extra="$extra --from-scratch"
    echo "$extra"
}

# Run a training command with per-scheme defaults, overridable by CLI flags
run_scheme() {
    local toml="$1"
    local default_n_gen="$2"
    local default_n_pop="$3"
    local default_final_n_sims="$4"
    local default_sim_timeout="$5"

    local n_gen="${OPT_N_GEN:-$default_n_gen}"
    local n_pop="${OPT_N_POP:-$default_n_pop}"
    local final_n_sims="${OPT_FINAL_N_SIMS:-$default_final_n_sims}"
    local sim_timeout="${OPT_SIM_TIMEOUT:-$default_sim_timeout}"

    run_train "$toml" \
        --n-gen "$n_gen" --n-pop "$n_pop" \
        --final-n-sims "$final_n_sims" --sim-timeout "$sim_timeout" \
        $(build_extra_flags)
}

train_piecewise_constant() {
    echo "=== piecewise_constant (11 params) ==="
    run_scheme configs/training/msr_aller_piecewise_constant_train.toml 3000 40 2000 1
}

train_ftc() {
    echo "=== ftc (26 params) ==="
    run_scheme configs/training/msr_aller_ftc_train.toml 2500 50 2000 1
}

train_eqglide() {
    echo "=== equilibrium_glide (24 params) ==="
    run_scheme configs/training/msr_aller_eqglide_train.toml 2500 60 2000 1
}

train_energy_controller() {
    echo "=== energy_controller (20 params) ==="
    run_scheme configs/training/msr_aller_energy_controller_train.toml 2500 60 2000 1
}

train_pred_guid() {
    echo "=== pred_guid (20 params) ==="
    run_scheme configs/training/msr_aller_pred_guid_train.toml 2500 60 2000 1
}

train_fnpag() {
    echo "=== fnpag (22 params) ==="
    run_scheme configs/training/msr_aller_fnpag_train.toml 600 50 2000 1
}

train_neural_network() {
    echo "=== neural_network (a lot of params) ==="
    run_scheme configs/training/msr_aller_nn_train_consolidated.toml 1500 120 2000 1
}

train_nn_rl() {
    echo "=== neural_network_rl (RL/PPO) ==="
    uv run python -m aerocapture.training.rl.train \
        configs/training/msr_aller_rl_train.toml \
        --algorithm ppo --total-steps 5000000
}

train_neural_network_gru_ppo() {
    echo "=== neural_network_gru_ppo (Dense -> GRU -> Dense, PPO+BPTT) ==="
    uv run python -m aerocapture.training.rl.train \
        configs/training/msr_aller_gru_ppo_train.toml \
        --algorithm ppo --total-steps 5000000
}

train_neural_network_gru_pso() {
    echo "=== neural_network_gru_pso (Dense -> GRU -> Dense, PSO) ==="
    # TOML carries [optimizer] algorithm/n_pop/n_gen defaults; CLI flags still override.
    run_scheme configs/training/msr_aller_gru_pso_train.toml 5000 64 2000 1
}

train_neural_network_lstm_ppo() {
    echo "=== neural_network_lstm_ppo (Dense -> LSTM -> Dense, PPO+BPTT) ==="
    uv run python -m aerocapture.training.rl.train \
        configs/training/msr_aller_lstm_ppo_train.toml \
        --algorithm ppo --total-steps 5000000
}

train_neural_network_lstm_pso() {
    echo "=== neural_network_lstm_pso (Dense -> LSTM -> Dense, PSO) ==="
    run_scheme configs/training/msr_aller_lstm_pso_train.toml 5000 64 2000 1
}

train_neural_network_window_pso() {
    echo "=== neural_network_window_pso (Window -> Dense trunk, PSO) ==="
    run_scheme configs/training/msr_aller_window_pso_train.toml 500 64 2000 1
}

train_neural_network_transformer_pso() {
    echo "=== neural_network_transformer_pso (Dense -> Transformer -> Dense, PSO) ==="
    run_scheme configs/training/msr_aller_transformer_pso_train.toml 5000 64 2000 1
}

train_all() {
    train_piecewise_constant
    echo ""
    train_nn_rl
    echo ""
    train_neural_network_gru_ppo
    echo ""
    train_neural_network_gru_pso
    echo ""
    train_neural_network_lstm_ppo
    echo ""
    train_neural_network_lstm_pso
    echo ""
    train_neural_network_window_pso
    echo ""
    train_neural_network_transformer_pso
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

# Dispatch: no schemes = all, otherwise run named schemes
if [ ${#SCHEMES[@]} -eq 0 ]; then
    train_all
else
    for scheme in "${SCHEMES[@]}"; do
        case "$scheme" in
            piecewise_constant|piecewise|pc)  train_piecewise_constant ;;
            ftc)                               train_ftc ;;
            eqglide|equilibrium_glide|eq)      train_eqglide ;;
            energy_controller|energy|ec)       train_energy_controller ;;
            pred_guid|predguid|pg)             train_pred_guid ;;
            fnpag)                             train_fnpag ;;
            neural_network|nn)                 train_neural_network ;;
            neural_network_rl|nn_rl|rl)        train_nn_rl ;;
            neural_network_gru_ppo|nn_gru_ppo|gru_ppo)  train_neural_network_gru_ppo ;;
            neural_network_gru_pso|nn_gru_pso|gru_pso|gru)  train_neural_network_gru_pso ;;
            neural_network_lstm_ppo|nn_lstm_ppo|lstm_ppo)  train_neural_network_lstm_ppo ;;
            neural_network_lstm_pso|nn_lstm_pso|lstm_pso|lstm)  train_neural_network_lstm_pso ;;
            neural_network_window_pso|nn_window_pso|window_pso|window)  train_neural_network_window_pso ;;
            neural_network_transformer_pso|nn_transformer_pso|transformer_pso|transformer)  train_neural_network_transformer_pso ;;
            all)                               train_all ;;
            *)
                echo "Unknown scheme: $scheme"
                echo "Valid: piecewise_constant ftc eqglide energy_controller pred_guid fnpag neural_network neural_network_rl neural_network_gru_pso neural_network_gru_ppo neural_network_lstm_pso neural_network_lstm_ppo neural_network_window_pso neural_network_transformer_pso all"
                exit 1
                ;;
        esac
    done
fi
