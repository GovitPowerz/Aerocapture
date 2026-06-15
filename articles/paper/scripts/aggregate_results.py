"""Aggregate the committed bundle into articles/paper/data/results.json.

Per run: capture + DV stats (p99/CVaR95 + bootstrap CIs; max descriptive),
best validation RMS (within-transform only), actual-sims accounting.
Plus: paired comparisons for the named cross-cell tables, sigma_run pooled
from the seed_repeats triplets, and the fresh-pool headline re-quote.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import actual_sims, capture_mask, paired_comparison, run_stats  # noqa: E402

RUNS_DIR = REPO / "articles/paper/data/runs"
OUT = REPO / "articles/paper/data/results.json"

HEADLINE = "optimizer_budget/ga_300"
# Paired tables: (label, run_a, run_b) -- delta = a - b, negative = a better.
PAIRED = [
    ("ga_vs_islands_300", "optimizer_budget/ga_300", "optimizer_budget/islands_300"),
    ("ga150_vs_islands300", "optimizer_budget/ga_150", "optimizer_budget/islands_300"),
    ("cubed_vs_log", "optimizer_budget/ga_300", "cost_transform/log"),
    ("max_vs_middle_bucket", "optimizer_budget/ga_300", "curation_shaping/bucket_middle"),
    ("max_vs_random_bucket", "optimizer_budget/ga_300", "curation_shaping/bucket_random"),
    ("nn_vs_ftc", "optimizer_budget/ga_300", "classical_baselines/ftc"),
    ("nn_vs_fnpag", "optimizer_budget/ga_300", "classical_baselines/fnpag"),
    ("ftc_vs_fnpag", "classical_baselines/ftc", "classical_baselines/fnpag"),
    ("jointftc_vs_fnpag", "joint_reference/ftc", "classical_baselines/fnpag"),
    ("joint_vs_fixed_ftc", "joint_reference/ftc", "classical_baselines/ftc"),
    ("joint_vs_fixed_ec", "joint_reference/energy_controller", "classical_baselines/energy_controller"),
    ("joint_vs_fixed_pg", "joint_reference/pred_guid", "classical_baselines/pred_guid"),
    ("atan2_vs_scaledpi", "optimizer_dimensionality/dense_p515_ga", "output_param/scaledpi"),
    ("atan2_vs_delta", "optimizer_dimensionality/dense_p515_ga", "output_param/delta"),
]
# sigma_run triplets: repeat #1 cell + its _s2/_s3 siblings in seed_repeats/.
REPEAT_GROUPS = {
    "ga_300": ["optimizer_budget/ga_300", "seed_repeats/ga_300_s2", "seed_repeats/ga_300_s3"],
    "islands_300": ["optimizer_budget/islands_300", "seed_repeats/islands_300_s2", "seed_repeats/islands_300_s3"],
    "ftc_ga": ["classical_baselines/ftc", "seed_repeats/ftc_ga_s2", "seed_repeats/ftc_ga_s3"],
    "ftc_cmaes": ["optimizer_dimensionality/ftc_cmaes", "seed_repeats/ftc_cmaes_s2", "seed_repeats/ftc_cmaes_s3"],
    "ftc_islands": ["optimizer_dimensionality/ftc_islands", "seed_repeats/ftc_islands_s2", "seed_repeats/ftc_islands_s3"],
    "small_ga": ["optimizer_dimensionality/dense_p515_ga", "seed_repeats/small_ga_s2", "seed_repeats/small_ga_s3"],
    "small_cmaes": ["optimizer_dimensionality/dense_p515_cmaes", "seed_repeats/small_cmaes_s2", "seed_repeats/small_cmaes_s3"],
    "small_islands": ["optimizer_dimensionality/dense_p515_islands", "seed_repeats/small_islands_s2", "seed_repeats/small_islands_s3"],
}


def _infer_training_n_sims(key: str) -> int:
    # Study F cells encode n_sims in the cell name; everything else trains at 10.
    if key.startswith("training_n_sims/"):
        return int(key.rsplit("_", 1)[1])
    return 10


def _load_parquet(key: str):
    p = RUNS_DIR / key / "final_eval.parquet"
    if not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def _jsonl_records(key: str) -> list[dict]:
    gz = RUNS_DIR / key / "run.jsonl.gz"
    if not gz.exists():
        return []
    with gzip.open(gz, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _best_val_rms(records: list[dict]) -> float | None:
    vals = [r["validation"]["rms_cost"] for r in records if r.get("validation")]
    return min(vals) if vals else None


def _disp_fingerprint_ok(df_a, df_b) -> bool:
    """Paired comparisons require identical scenarios: the parquet has no seed
    column, so assert the dispersion draws match row-by-row."""
    cols = [c for c in df_a.columns if c.startswith("disp_")][:3]
    n = min(len(df_a), len(df_b))
    return all(np.allclose(df_a[c].to_numpy()[:n], df_b[c].to_numpy()[:n]) for c in cols)


def summarize(key: str) -> dict:
    df = _load_parquet(key)
    if df is None:
        return {"key": key, "missing": True}
    records = _jsonl_records(key)
    out: dict = {"key": key, "legacy_prefix_regime": key.startswith("legacy/")}
    out.update(run_stats(df["ifinal"].to_numpy(), df["eccentricity"].to_numpy(), df["dv_total_m_s"].to_numpy()))
    out["heat_flux_p95"] = round(float(np.percentile(df["max_heat_flux_kw_m2"], 95)), 1)
    out["g_load_p95"] = round(float(np.percentile(df["max_load_factor_g"], 95)), 2)
    out["best_val_rms_within_transform_only"] = _best_val_rms(records)
    if records:
        out["actual_sims"] = actual_sims(records, training_n_sims=_infer_training_n_sims(key))
    return out


def main() -> None:
    keys = sorted(str(p.parent.relative_to(RUNS_DIR)) for p in RUNS_DIR.rglob("final_eval.parquet"))
    if not keys:
        sys.exit(f"Empty bundle at {RUNS_DIR}; run experiments/paper/12_collect_results.sh first")
    runs = {k: summarize(k) for k in keys}

    paired = {}
    for label, ka, kb in PAIRED:
        da, db = _load_parquet(ka), _load_parquet(kb)
        if da is None or db is None:
            paired[label] = {"missing": True, "a": ka, "b": kb}
            continue
        n = min(len(da), len(db))  # prefix property: first n rows = same seeds
        da, db = da.head(n), db.head(n)
        assert _disp_fingerprint_ok(da, db), f"dispersion mismatch {ka} vs {kb} -- not the same scenario pool"
        paired[label] = {
            "a": ka,
            "b": kb,
            **paired_comparison(
                da["dv_total_m_s"].to_numpy(),
                capture_mask(da["ifinal"].to_numpy(), da["eccentricity"].to_numpy()),
                db["dv_total_m_s"].to_numpy(),
                capture_mask(db["ifinal"].to_numpy(), db["eccentricity"].to_numpy()),
            ),
        }

    sigma_run = {}
    for label, members in REPEAT_GROUPS.items():
        means = [runs[m]["dv_mean"] for m in members if m in runs and not runs[m].get("missing")]
        if len(means) >= 2:
            sigma_run[label] = {
                "n": len(means),
                "dv_means": means,
                "range": round(max(means) - min(means), 2),
                "std": round(float(np.std(means, ddof=1)), 2),
            }
    pooled = [g["std"] for g in sigma_run.values()]
    sigma_summary = {
        "groups": sigma_run,
        "pooled_std": round(float(np.sqrt(np.mean(np.square(pooled)))), 2) if pooled else None,
    }

    requote_path = RUNS_DIR / HEADLINE / "fresh_pool_requote.json"
    headline_requote = json.loads(requote_path.read_text()) if requote_path.exists() else {"missing": True}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"runs": runs, "paired": paired, "sigma_run": sigma_summary, "headline": HEADLINE, "headline_fresh_pool": headline_requote},
            indent=2,
        )
    )
    n_ok = sum(1 for r in runs.values() if not r.get("missing"))
    n_paired = sum(1 for p in paired.values() if not p.get("missing"))
    print(f"wrote {OUT}: {n_ok}/{len(runs)} runs, {n_paired}/{len(PAIRED)} paired tables")


if __name__ == "__main__":
    main()
