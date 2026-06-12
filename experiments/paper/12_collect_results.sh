#!/usr/bin/env bash
set -euo pipefail
# Collect the quotable artifacts of every completed paper run into the
# committed bundle at articles/paper/data/runs/ (per run: best_model.json,
# best_params.json, final_eval.parquet, final_selection.json, run.jsonl.gz).
# Idempotent; re-run after any study completes. Checkpoints/PDFs stay out.

uv run python articles/paper/scripts/collect_runs.py "$@"
