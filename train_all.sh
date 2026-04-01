#!/usr/bin/env bash
set -euo pipefail

COMMON="--n-gen 100 --n-pop 50 --adaptive-seeds -fs --sim-timeout 2"

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[32m'
CYAN='\033[36m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

SCHEMES=(
    "piecewise_constant|configs/training/msr_aller_piecewise_constant_train.toml"
    "neural_network|configs/training/msr_aller_nn_train_consolidated.toml"
    "equilibrium_glide|configs/training/msr_aller_eqglide_train.toml"
    "energy_controller|configs/training/msr_aller_energy_controller_train.toml"
    "pred_guid|configs/training/msr_aller_pred_guid_train.toml"
    "fnpag|configs/training/msr_aller_fnpag_train.toml"
    "ftc|configs/training/msr_aller_ftc_train.toml"
)
TOTAL=${#SCHEMES[@]}

fmt_duration() {
    local secs=$1
    if (( secs >= 3600 )); then
        printf "%dh %02dm %02ds" $((secs/3600)) $((secs%3600/60)) $((secs%60))
    elif (( secs >= 60 )); then
        printf "%dm %02ds" $((secs/60)) $((secs%60))
    else
        printf "%ds" "$secs"
    fi
}

separator() {
    printf "${DIM}%.0s-${RESET}" {1..60}
    echo
}

echo -e "${BOLD}Aerocapture GA Training Pipeline${RESET}"
echo -e "${DIM}Schemes: ${TOTAL} | Args: ${COMMON}${RESET}"
separator

SCRIPT_START=$SECONDS
RESULTS=()

for i in "${!SCHEMES[@]}"; do
    IFS='|' read -r name toml <<< "${SCHEMES[$i]}"
    step=$((i + 1))

    echo -e "\n${CYAN}[${step}/${TOTAL}]${RESET} ${BOLD}${name}${RESET} ${DIM}(${toml})${RESET}"
    STEP_START=$SECONDS

    if uv run python -m aerocapture.training.train "$toml" $COMMON; then
        elapsed=$((SECONDS - STEP_START))
        echo -e "${GREEN}  done${RESET} ${DIM}$(fmt_duration $elapsed)${RESET}"
        RESULTS+=("${GREEN}  ok${RESET}  $(fmt_duration $elapsed)  ${name}")
    else
        elapsed=$((SECONDS - STEP_START))
        echo -e "${RED}  FAILED${RESET} ${DIM}$(fmt_duration $elapsed)${RESET}"
        RESULTS+=("${RED}FAIL${RESET}  $(fmt_duration $elapsed)  ${name}")
    fi

    separator
done

TOTAL_ELAPSED=$((SECONDS - SCRIPT_START))

echo -e "\n${BOLD}Summary${RESET} ${DIM}(total: $(fmt_duration $TOTAL_ELAPSED))${RESET}\n"
for line in "${RESULTS[@]}"; do
    echo -e "  $line"
done
echo
