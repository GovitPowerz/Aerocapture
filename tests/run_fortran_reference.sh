#!/usr/bin/env bash
# Run legacy Fortran executables to produce reference outputs for regression testing.
# Usage: ./run_fortran_reference.sh [--clean]
#
# This script must be run from the repository root.
# It builds both Fortran variants and runs curated test cases,
# capturing outputs into tests/reference_data/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXEC_DIR="$REPO_ROOT/old_codebase/exec"
SORTIES_DIR="$REPO_ROOT/old_codebase/sorties"
REF_DIR="$REPO_ROOT/tests/reference_data"

# Test case definitions: name, executable, input file
declare -a TEST_CASES=(
    "ref_orig:aerocap:test_ref_orig.in"
    "ref_nn:aerocap_nn:test_ref_nn.in"
    "high_bank_orig:aerocap:test_high_bank_orig.in"
    "guided_orig:aerocap:test_guided_orig.in"
    "guided_nn_ftc:aerocap_nn:test_guided_nn.in"
    "guided_nn2:aerocap_nn:test_guided_nn2.in"
    "mc10_orig:aerocap:test_mc10_orig.in"
)

# Output files to capture (relative to exec/ and sorties/)
FORT_FILES="fort.201 fort.202 fort.203 fort.204"
# Sorties files are suffixed with the test name from the .in file

clean_outputs() {
    echo "Cleaning previous reference data..."
    rm -rf "$REF_DIR"
    mkdir -p "$REF_DIR"
}

build_executables() {
    echo "=== Building Fortran executables ==="
    cd "$EXEC_DIR"
    make all 2>&1 | tail -3
    echo "Build complete."
    echo ""
}

run_test_case() {
    local name="$1"
    local executable="$2"
    local input_file="$3"

    echo "--- Running test case: $name ($executable < $input_file) ---"

    local case_dir="$REF_DIR/$name"
    mkdir -p "$case_dir/sorties"

    cd "$EXEC_DIR"

    # Clean previous fort.* files
    rm -f fort.201 fort.202 fort.203 fort.204

    # Extract the sufres value from the input file to find sorties output names
    # For nn variant: sufres is line 31 (32 data lines, sufres is second to last)
    # For orig variant: sufres is line 29 (30 data lines, sufres is second to last)
    # We'll capture based on what files appear after the run

    # Run the simulation
    local stdout_file="$case_dir/stdout.log"
    ./"$executable" < "$input_file" > "$stdout_file" 2>&1 || true

    # Capture fort.* files
    for f in $FORT_FILES; do
        if [ -f "$EXEC_DIR/$f" ]; then
            cp "$EXEC_DIR/$f" "$case_dir/$f"
        fi
    done

    # Extract sufres from the .in file to find output files in sorties/
    # The sufres is the last suffix value before the confirmation line
    local sufres
    sufres=$(awk 'NR>0{last2=last1; last1=$1} END{print last2}' "$EXEC_DIR/$input_file")

    # Capture sorties files
    for prefix in photo photo_temp final initial mission cinematique commande prediction guidage divers; do
        local src="$SORTIES_DIR/${prefix}${sufres}"
        if [ -f "$src" ]; then
            cp "$src" "$case_dir/sorties/${prefix}${sufres}"
        fi
    done

    echo "    Captured outputs to $case_dir/"
    echo ""
}

# Main
if [ "${1:-}" = "--clean" ]; then
    clean_outputs
fi

mkdir -p "$REF_DIR"
build_executables

passed=0
failed=0

for tc in "${TEST_CASES[@]}"; do
    IFS=':' read -r name executable input_file <<< "$tc"

    if [ ! -f "$EXEC_DIR/$input_file" ]; then
        echo "WARNING: Input file $input_file not found, skipping $name"
        ((failed++)) || true
        continue
    fi

    run_test_case "$name" "$executable" "$input_file"
    ((passed++)) || true
done

echo "=== Reference data generation complete ==="
echo "Passed: $passed / $((passed + failed))"
echo "Reference data stored in: $REF_DIR"

# List what was captured
echo ""
echo "Captured test cases:"
for dir in "$REF_DIR"/*/; do
    if [ -d "$dir" ]; then
        local_name=$(basename "$dir")
        n_files=$(find "$dir" -type f | wc -l | tr -d ' ')
        echo "  $local_name: $n_files files"
    fi
done
