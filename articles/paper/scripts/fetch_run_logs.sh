#!/bin/bash
# Fetch the raw paper training logs (95 x run.jsonl.gz, ~195 MB) into
# articles/paper/data/runs/. They are not tracked in git (see .gitignore);
# the tarball is attached to the arxiv-v2 GitHub Release.
# Needed only to re-run collect_runs.py / aggregate_results.py from scratch;
# the aggregated results.json and all figures are tracked.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
URL="https://github.com/GovitPowerz/Aerocapture/releases/download/arxiv-v2/paper_run_logs.tar"
TARBALL="$(mktemp -d)/paper_run_logs.tar"

echo "Downloading ${URL} ..."
curl -L --fail -o "${TARBALL}" "${URL}"
tar -xf "${TARBALL}" -C "${REPO_ROOT}"
rm -f "${TARBALL}"
echo "Run logs restored under ${REPO_ROOT}/articles/paper/data/runs/"
