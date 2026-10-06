#!/bin/bash
# Fetch the raw paper training logs (run.jsonl.gz per cell, the files of
# data/SHA256SUMS.runlogs) into articles/paper/data/runs/. They are not tracked
# in git (see .gitignore); the tarball is a GitHub Release asset whose URL is
# the `run_logs_asset` of data/provenance.json (write_provenance.py names the
# Release tag; the v4 asset is uploaded by the v4 release, issue #183).
# Needed only to re-run collect_runs.py / aggregate_results.py from scratch;
# the aggregated results.json and all figures are tracked.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
URL="$(sed -n 's/^ *"run_logs_asset": "\(.*\)",*$/\1/p' "${REPO_ROOT}/articles/paper/data/provenance.json")"
[ -n "${URL}" ] || { echo "error: no run_logs_asset in articles/paper/data/provenance.json" >&2; exit 1; }
TARBALL="$(mktemp -d)/paper_run_logs.tar"

echo "Downloading ${URL} ..."
curl -L --fail -o "${TARBALL}" "${URL}" || { echo "error: ${URL} not fetched: the Release asset is uploaded by the release that tags it (see write_provenance.py)" >&2; exit 1; }
tar -xf "${TARBALL}" -C "${REPO_ROOT}"
rm -f "${TARBALL}"
echo "Run logs restored under ${REPO_ROOT}/articles/paper/data/runs/"
