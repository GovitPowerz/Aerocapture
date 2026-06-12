# Aerocapture Neural-Guidance Article — Implementation Plan v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the Typst research paper (follow-up to Gelly & Vernis, AIAA GNC 2009) from the post-fix experiment campaign, with the GA/non-stationarity narrative and the locked statistical reporting rules.

**Architecture:** Three build phases on top of the already-shipped campaign infrastructure. (1) The **user executes** `experiments/paper/00..12` (the campaign IS Phase 1 — this plan adds no experiment code). (2) Aggregate the committed bundle into `results.json` with the five locked reporting rules (p99/CVaR95+bootstrap CIs, paired bootstrap/Wilcoxon, actual-sims accounting, fresh-pool re-quote, σ_run pooling) and render ~10 figures. (3) Author the Typst paper section by section around the GA + non-stationarity quartet. Finish with `smart-commit`.

**Tech Stack:** Python (uv, pyarrow/numpy/scipy/matplotlib), the committed bundle `articles/paper/data/runs/`, Typst (single-column academic, Hayagriva).

**Supersedes:** `2026-06-08-aerocapture-nn-article.md` (pre-flip, pre-reorg — bannered DO NOT EXECUTE).

---

## Phase 1 — the campaign (user-executed, already shipped)

No tasks. The experiments are `experiments/paper/00_prereqs.sh` … `12_collect_results.sh`, run from the repo root in README order (`experiments/paper/README.md` holds the run order, reuse-cell map, and reporting rules). Phase 2/3 tasks below state which campaign scripts gate them. The small infrastructure repairs from the 2026-06-12 plan review (ablation `--model`/scaffolding, collector legacy list + stale-parquet guard, `HEADLINE_REQUOTE_SEED_OFFSET = 8_000_000` + `fresh_pool_requote.py`) are already committed.

**Gating map (task → blocking campaign scripts):**

| Plan task | Blocks on |
|---|---|
| Task 1-2 (stats lib + aggregator) | nothing (test on synthetic data); useful output needs `12` run at least once |
| Task 3 (figures) | per-figure: pareto → 09+10 (+02/03 anchors); optimizer → 02 (+03, 04); output-param → 03; classical-vs-NN → 00+01+02; cost-transform → 05 (+02); curation → 06 (+02); training-n-sims → 08 (+02); joint-ref → 01+07; pruning-quant → nothing (legacy dirs); σ_run error bars → 11 |
| Task 4 (ablation + input report on headline) | 02 |
| Task 5 (fresh-pool re-quote) | 02 |
| Tasks 6-14 (Typst) | scaffolding/template: nothing; problem/testbed/classical/neural/training prose: nothing; every results number: the relevant figures + `12` |
| Task 15 (smart-commit) | all |

---

## File structure

```
src/python/aerocapture/training/paper_stats.py   NEW — pure, unit-tested stats helpers
tests/test_paper_stats.py                         NEW — tests for the above
articles/paper/scripts/
  aggregate_results.py   NEW — bundle → articles/paper/data/results.json
  figlib.py              NEW — shared matplotlib helpers (bars+CI, CDF, pareto)
  fig_pareto.py fig_optimizer.py fig_output_param.py fig_classical_vs_nn.py
  fig_cost_transform.py fig_curation.py fig_seed_strategy.py
  fig_training_n_sims.py fig_joint_reference.py fig_pruning_quant.py   NEW
articles/paper/data/results.json                 GENERATED
articles/paper/figures/*.svg                     GENERATED
articles/paper/{main.typ,template.typ,refs.yml,sections/*.typ}   NEW
```

---

# Phase 2 — statistics, aggregation, figures

### Task 1: Pure stats helpers (`paper_stats.py`) — TDD

**Files:**
- Create: `src/python/aerocapture/training/paper_stats.py`
- Test: `tests/test_paper_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the paper's statistical helpers (locked 2026-06-12 reporting rules)."""

import numpy as np
import pytest

from aerocapture.training.paper_stats import (
    actual_sims,
    bootstrap_ci,
    cvar,
    paired_comparison,
    run_stats,
)


def test_cvar_is_mean_of_worst_tail() -> None:
    x = np.arange(1.0, 101.0)  # 1..100
    assert cvar(x, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_cvar_small_sample_uses_at_least_one() -> None:
    assert cvar(np.array([3.0, 1.0]), 0.95) == 3.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng_x = np.random.default_rng(0).normal(100.0, 5.0, size=1000)
    lo, hi = bootstrap_ci(rng_x, np.mean, n_boot=2000, seed=1)
    assert lo < float(np.mean(rng_x)) < hi
    assert hi - lo < 2.0  # ~2*1.96*5/sqrt(1000)


def test_run_stats_capture_conditional() -> None:
    ifinal = np.array([3.0, 3.0, 3.0, 2.0])
    ecc = np.array([0.5, 0.5, 1.2, 0.5])  # third sim: ifinal 3 but hyperbolic
    dv = np.array([100.0, 200.0, 999.0, 999.0])
    s = run_stats(ifinal, ecc, dv, n_boot=200, seed=0)
    assert s["n"] == 4
    assert s["capture_pct"] == 50.0
    assert s["dv_mean"] == pytest.approx(150.0)
    assert "dv_p99" in s and "dv_cvar95" in s and "dv_mean_ci" in s


def test_paired_comparison_sign_and_win_rate() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(120.0, 10.0, size=500)
    b = a + 2.0  # b uniformly worse by 2 m/s
    cap = np.ones(500, dtype=bool)
    p = paired_comparison(a, cap, b, cap, n_boot=500, seed=0)
    assert p["n_pairs"] == 500
    assert p["delta_mean"] == pytest.approx(-2.0)  # a - b
    assert p["win_rate_a"] == 1.0
    assert p["wilcoxon_p"] < 1e-6
    assert p["delta_mean_ci"][1] < 0  # CI excludes zero


def test_paired_comparison_drops_either_failed() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    p = paired_comparison(a, np.array([True, False, True]), b, np.array([True, True, False]), n_boot=100, seed=0)
    assert p["n_pairs"] == 1


def test_actual_sims_formula() -> None:
    # 3 gens, n_pop=2, n_sims=10; validation fired twice; one curation event;
    # seeds changed once after the curation -> one parent re-eval.
    records = [
        {"all_costs": [1, 2], "pool_metrics": {"last_curation_gen": 0}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 5.0}, "pool_metrics": {"last_curation_gen": 1}},
        {"all_costs": [1, 2], "validation": {"rms_cost": 4.0}, "pool_metrics": {"last_curation_gen": 1}},
    ]
    s = actual_sims(records, training_n_sims=10, validation_n_sims=1000, curation_sample_size=1000, curation_top_k=1)
    assert s["training"] == 3 * 2 * 10
    assert s["validation"] == 2 * 1000
    assert s["curation"] == 2 * 1 * 1000  # two distinct last_curation_gen values
    assert s["reeval"] == 2 * 2 * 10  # parent re-eval on each curation-change gen
    assert s["total"] == s["training"] + s["validation"] + s["curation"] + s["reeval"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paper_stats.py -q`
Expected: FAIL (ModuleNotFoundError: paper_stats)

- [ ] **Step 3: Implement `paper_stats.py`**

```python
"""Pure statistical helpers for the paper (locked 2026-06-12 reporting rules).

Capture = ifinal==3 & eccentricity<1.0. Tail metrics are p99 + CVaR95 (mean of
the worst 5% of captured DV); the sample max is descriptive only. Cross-run
comparisons are PAIRED on the shared final-eval pool (same seeds, same row
order); failures in EITHER run drop the pair. Bootstrap CIs are percentile
CIs over seed resamples and cover eval-sampling noise only, NOT training-run
variance (sigma_run from the seed-repeats study covers that).
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def cvar(x: npt.NDArray[np.float64], level: float = 0.95) -> float:
    """Mean of the worst (1-level) tail (expected shortfall), >= 1 sample."""
    k = max(1, int(round(len(x) * (1.0 - level))))
    return float(np.sort(x)[-k:].mean())


def bootstrap_ci(
    x: npt.NDArray[np.float64],
    stat: Callable[[npt.NDArray[np.float64]], float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def capture_mask(ifinal: npt.NDArray[np.float64], ecc: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return (ifinal == 3) & (ecc < 1.0)


def run_stats(
    ifinal: npt.NDArray[np.float64],
    ecc: npt.NDArray[np.float64],
    dv: npt.NDArray[np.float64],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    cap = capture_mask(ifinal, ecc)
    dvc = dv[cap]
    r2 = lambda v: round(float(v), 2)
    out: dict[str, Any] = {"n": int(len(ifinal)), "capture_pct": r2(100 * cap.mean())}
    if len(dvc) == 0:
        return out
    out.update(
        dv_mean=r2(dvc.mean()),
        dv_p50=r2(np.percentile(dvc, 50)),
        dv_p95=r2(np.percentile(dvc, 95)),
        dv_p99=r2(np.percentile(dvc, 99)),
        dv_cvar95=r2(cvar(dvc, 0.95)),
        dv_max_descriptive=r2(dvc.max()),
        dv_mean_ci=[r2(v) for v in bootstrap_ci(dvc, np.mean, n_boot, seed)],
        dv_p95_ci=[r2(v) for v in bootstrap_ci(dvc, lambda a: float(np.percentile(a, 95)), n_boot, seed)],
        dv_cvar95_ci=[r2(v) for v in bootstrap_ci(dvc, lambda a: cvar(a, 0.95), n_boot, seed)],
    )
    return out


def paired_comparison(
    dv_a: npt.NDArray[np.float64],
    cap_a: npt.NDArray[np.bool_],
    dv_b: npt.NDArray[np.float64],
    cap_b: npt.NDArray[np.bool_],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired a-vs-b on the shared seed pool (row i = same MC scenario).

    delta = a - b on both-captured seeds: negative delta_mean = a better.
    """
    from scipy.stats import wilcoxon

    both = cap_a & cap_b
    d = dv_a[both] - dv_b[both]
    r2 = lambda v: round(float(v), 3)
    out: dict[str, Any] = {"n_pairs": int(both.sum())}
    if len(d) < 10:
        return out
    w = wilcoxon(d)
    out.update(
        delta_mean=r2(d.mean()),
        delta_mean_ci=[r2(v) for v in bootstrap_ci(d, np.mean, n_boot, seed)],
        win_rate_a=r2((d < 0).mean()),
        wilcoxon_p=float(w.pvalue),
    )
    return out


def actual_sims(
    records: list[dict],
    training_n_sims: int,
    validation_n_sims: int = 1000,
    curation_sample_size: int = 1000,
    curation_top_k: int = 1,
) -> dict[str, int]:
    """Reconstruct the run's ACTUAL simulation count from its JSONL records.

    training  = n_gen x n_pop x training_n_sims (selection-driving evals)
    validation = validation-gate fires x validation_n_sims
    curation  = distinct curation events x top_k x sample_size (probe sims)
    reeval    = parent re-evals on seed-change gens (n_pop x n_sims each;
                approximated by the curation-event count -- rotating-strategy
                runs re-eval EVERY gen, handled by the caller passing the
                event count via records' pool_metrics)
    """
    n_gen = len(records)
    n_pop = max((len(r.get("all_costs", [])) for r in records), default=0)
    validations = sum(1 for r in records if r.get("validation"))
    curation_gens = {r["pool_metrics"]["last_curation_gen"] for r in records if r.get("pool_metrics", {}).get("last_curation_gen") is not None}
    n_curations = len(curation_gens)
    out = {
        "n_gen": n_gen,
        "n_pop": n_pop,
        "training": n_gen * n_pop * training_n_sims,
        "validation": validations * validation_n_sims,
        "curation": n_curations * curation_top_k * curation_sample_size,
        "reeval": n_curations * n_pop * training_n_sims,
    }
    out["total"] = out["training"] + out["validation"] + out["curation"] + out["reeval"]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_paper_stats.py -q`
Expected: 7 passed

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/python/aerocapture/training/paper_stats.py tests/test_paper_stats.py && uv run ruff format --check src/python/aerocapture/training/paper_stats.py tests/test_paper_stats.py
git add src/python/aerocapture/training/paper_stats.py tests/test_paper_stats.py
git commit -m "feat(paper): pure stats helpers (p99/CVaR95, bootstrap CIs, paired tests, actual-sims)"
```

---

### Task 2: Bundle-driven aggregator (`aggregate_results.py`)

**Files:**
- Create: `articles/paper/scripts/aggregate_results.py`
- Generate: `articles/paper/data/results.json`

Reads the COMMITTED BUNDLE (`articles/paper/data/runs/`), never `training_output` directly. Keys mirror bundle paths (`optimizer_budget/ga_300`). The JSONL ships gzipped as `run.jsonl.gz`.

- [ ] **Step 1: Write `aggregate_results.py`**

```python
"""Aggregate the committed bundle into articles/paper/data/results.json.

Per run: capture + DV stats (p99/CVaR95 + bootstrap CIs; max descriptive),
best validation RMS (within-transform only), actual-sims accounting.
Plus: paired comparisons for the named cross-cell tables, sigma_run pooled
from the seed_repeats triplets, and the fresh-pool headline re-quote.
"""

import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import actual_sims, paired_comparison, run_stats  # noqa: E402

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
    df = pq.read_table(p).to_pandas()
    return df


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
    """Paired comparisons require identical scenarios: parquet has no seed
    column, so assert the dispersion draws match row-by-row."""
    cols = [c for c in df_a.columns if c.startswith("disp_")][:3]
    n = min(len(df_a), len(df_b))
    return all(np.allclose(df_a[c].to_numpy()[:n], df_b[c].to_numpy()[:n]) for c in cols)


def summarize(key: str) -> dict:
    df = _load_parquet(key)
    if df is None:
        return {"key": key, "missing": True}
    records = _jsonl_records(key)
    out = {"key": key, "legacy_prefix_regime": key.startswith("legacy/")}
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
        from aerocapture.training.paper_stats import capture_mask

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
            sigma_run[label] = {"n": len(means), "dv_means": means, "range": round(max(means) - min(means), 2), "std": round(float(np.std(means, ddof=1)), 2)}
    pooled = [g["std"] for g in sigma_run.values()]
    sigma_summary = {"groups": sigma_run, "pooled_std": round(float(np.sqrt(np.mean(np.square(pooled)))), 2) if pooled else None}

    requote_path = RUNS_DIR / HEADLINE / "fresh_pool_requote.json"
    headline_requote = json.loads(requote_path.read_text()) if requote_path.exists() else {"missing": True}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runs": runs, "paired": paired, "sigma_run": sigma_summary, "headline": HEADLINE, "headline_fresh_pool": headline_requote}, indent=2))
    print(f"wrote {OUT}: {sum(1 for r in runs.values() if not r.get('missing'))}/{len(runs)} runs, {sum(1 for p in paired.values() if not p.get('missing'))}/{len(PAIRED)} paired tables")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (pre-campaign it exits with the empty-bundle message; post-`12` it writes results.json)**

Run: `uv run python articles/paper/scripts/aggregate_results.py`
Expected pre-campaign: `Empty bundle at .../articles/paper/data/runs; run experiments/paper/12_collect_results.sh first` (exit 1). Post-campaign: `wrote .../results.json: N/N runs, 11/11 paired tables`.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check articles/paper/scripts/aggregate_results.py
git add articles/paper/scripts/aggregate_results.py
git commit -m "feat(paper): bundle-driven aggregator with paired stats, actual-sims, sigma_run"
```

---

### Task 3: Figure library + the 10 figure scripts

**Files:**
- Create: `articles/paper/scripts/figlib.py` + `fig_pareto.py`, `fig_optimizer.py`, `fig_output_param.py`, `fig_classical_vs_nn.py`, `fig_cost_transform.py`, `fig_curation.py`, `fig_seed_strategy.py`, `fig_training_n_sims.py`, `fig_joint_reference.py`, `fig_pruning_quant.py`
- Generate: `articles/paper/figures/*.svg`

- [ ] **Step 1: Write `figlib.py`**

```python
"""Shared matplotlib helpers for the paper figures. All read results.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "articles/paper/figures"
SIGMA_NOTE = "error bars: eval bootstrap CI; shaded: pooled sigma_run (training-run variance)"


def load_results() -> dict:
    return json.loads((REPO / "articles/paper/data/results.json").read_text())


def get(res: dict, key: str) -> dict | None:
    r = res["runs"].get(key)
    return None if r is None or r.get("missing") else r


def grouped_bars(ax, labels: list[str], series: dict[str, list[float]], cis: dict[str, list[list[float]]] | None = None, sigma: float | None = None) -> None:
    """series: metric name -> per-label values; cis: metric -> per-label [lo,hi]."""
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        pos = x + (i - (len(series) - 1) / 2) * w
        err = None
        if cis and name in cis:
            arr = np.array(cis[name], dtype=float)  # (n, 2)
            err = np.abs(arr.T - np.array(vals))
        ax.bar(pos, vals, w, label=name, yerr=err, capsize=3)
        if sigma:
            ax.bar(pos, [2 * sigma] * len(vals), w, bottom=np.array(vals) - sigma, color="grey", alpha=0.25, zorder=0)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def dv_cdf(ax, res: dict, keys: dict[str, str]) -> None:
    """Not implementable from results.json (needs per-seed DV) -- reads the
    bundle parquets directly. keys: legend label -> bundle key."""
    import pyarrow.parquet as pq

    for label, key in keys.items():
        p = REPO / "articles/paper/data/runs" / key / "final_eval.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        cap = (df["ifinal"] == 3) & (df["eccentricity"] < 1.0)
        dv = np.sort(df.loc[cap, "dv_total_m_s"].to_numpy())
        ax.plot(dv, np.arange(1, len(dv) + 1) / len(dv), label=f"{label} ({100 * cap.mean():.1f}% capture)")
    ax.set_xlabel("correction DV (m/s, captured)")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)
    print("wrote", FIG / f"{name}.svg")
```

- [ ] **Step 2: Write the 10 figure scripts.** Each is ~20-40 lines on the figlib pattern; the content spec per figure (exact keys, panels, y-metric = `dv_p99`/`dv_cvar95` with `dv_mean` context, never `dv_max`):

  - `fig_pareto.py` — params vs `dv_p99` per architecture family, log-x. Keys `architecture_sweep/sweep_<arch>_p<N>`; param counts from `configs/training/sweep/manifest.json` + `manifest_floor.json` (NEVER name-regex — the floor anchors live under `optimizer_budget/`/`optimizer_dimensionality/`). Second panel: capture rate vs params for the dense floor cells (`sweep_dense_p{102,201,298,416}` + 515/3998 anchors) — the capability-collapse view.
  - `fig_optimizer.py` — panel A: 6 optimizers x 3 budgets (`optimizer_budget/<opt>_<b>`), `dv_mean`+`dv_p99` bars with CIs + pooled `sigma_run` shading; panel B: dimensionality — optimizer ranking at 26 (`optimizer_dimensionality/ftc_*` + `classical_baselines/ftc` for GA), 515 (`optimizer_dimensionality/dense_p515_*`), 3998 (`optimizer_budget/*_300`) params; panel C: convergence — running-min `validation.rms_cost` vs gen from the bundle `run.jsonl.gz` (gzip!), y-axis labeled "validation RMS (cubed-cost space)".
  - `fig_seed_strategy.py` — THE thesis figure: per optimizer (ga/islands/cmaes/pso), bars fixed/rotating (`seed_strategy/<opt>_{fixed,rotating}`) + adaptive (`optimizer_budget/<opt>_150`), annotated with each cell's `actual_sims.total`.
  - `fig_cost_transform.py` — linear/sqrt/log/squared (`cost_transform/*`) + cubed (`optimizer_budget/ga_300`): `dv_mean`/`dv_p99`/`dv_cvar95` bars + the paired `cubed_vs_log` delta CI printed in the caption.
  - `fig_curation.py` — bucket min/middle/random (`curation_shaping/bucket_*`) + max (`ga_300`); trim 0 (`ga_300`)/10/20 (`curation_shaping/trim_*`).
  - `fig_output_param.py` — atan2 control (`optimizer_dimensionality/dense_p515_ga`) vs `output_param/{scaledpi,delta}`.
  - `fig_training_n_sims.py` — two panels: rotating noise floor (`training_n_sims/rotating_*` vs n_sims) and adaptive allocation (`training_n_sims/adaptive_*` + `ga_300` anchor) with x = `actual_sims.total`, not nominal budget.
  - `fig_classical_vs_nn.py` — `dv_cdf` overlay: classical 6 (`classical_baselines/*`) + headline NN (`optimizer_budget/ga_300`).
  - `fig_joint_reference.py` — per scheme (ftc/ec/pg): fixed-ref baseline vs `joint_reference/<scheme>` bars + paired delta CIs from `results.json["paired"]["joint_vs_fixed_*"]`.
  - `fig_pruning_quant.py` — LEGACY (pre-fix regime, caption footnote): `legacy/neural_network_atan2` (full) vs `legacy/neural_network_atan2_qat8`/`_qat4` + the pruned variants.

- [ ] **Step 3: Run all + verify SVGs**

Run: `for f in articles/paper/scripts/fig_*.py; do uv run python "$f"; done && ls articles/paper/figures/`
Expected: one SVG per script (scripts must skip-and-note missing cells, not crash — campaign may be partial).

- [ ] **Step 4: Commit**

```bash
git add articles/paper/scripts/fig*.py articles/paper/scripts/figlib.py articles/paper/figures/
git commit -m "feat(paper): figure library + 10 campaign figures"
```

---

### Task 4: Ablation + NN input report on the HEADLINE model *(gate: campaign 02 complete)*

- [ ] **Step 1: Run ablation on ga_300** (the `--model`/scaffolding flags shipped 2026-06-12)

```bash
uv run python -m aerocapture.training.ablation training_output/paper/optimizer_budget/ga_300 \
    --toml configs/training/paper/dense_p3998_ga.toml --n-sims 500 --cost-transform log
```
Expected: prints `Model: .../ga_300/best_model.json` + scaffolding overrides; writes `ablation_results.json` into the cell dir.

- [ ] **Step 2: Run the input behavior report on the same dir**

```bash
uv run python -m aerocapture.training.nn_input_report training_output/paper/optimizer_budget/ga_300 \
    --toml configs/training/paper/dense_p3998_ga.toml --n-sims 200
```

- [ ] **Step 3: `fig_ablation.py`** — horizontal bar of per-input cost delta from `ga_300/ablation_results.json` (reuse `charts_ablation.chart_ablation_bar`); re-run `12_collect_results.sh` so the JSON lands in the bundle; commit.

---

### Task 5: Fresh-pool headline re-quote *(gate: campaign 02 complete)*

- [ ] **Step 1: Run it** (script shipped 2026-06-12)

```bash
uv run python articles/paper/scripts/fresh_pool_requote.py \
    training_output/paper/optimizer_budget/ga_300 --toml configs/training/paper/dense_p3998_ga.toml
```
Expected: prints + writes `fresh_pool_requote.json` (capture, mean/p50/p95/p99/CVaR95). THIS is the abstract number.

- [ ] **Step 2: Re-run `./experiments/paper/12_collect_results.sh` + `aggregate_results.py`** so `results.json["headline_fresh_pool"]` is populated; commit the bundle delta.

---

# Phase 3 — Typst paper

### Task 6: Template, shell, bibliography skeleton

Same mechanics as the old plan's Task 10 (verified clean): create `articles/paper/template.typ` + `main.typ` + minimal `refs.yml` + empty `sections/00_abstract.typ … 09_conclusion.typ`, compile with `typst compile articles/paper/main.typ`. Use the old plan's `template.typ`/`main.typ` code verbatim (it is narrative-neutral). Commit.

### Task 7: Abstract + Introduction

**Voice:** `articles/markdown/05_authorial_voice_and_style.md`; lineage from `00_synthesis_writing_kit.md`.

- [ ] Abstract spec: aerocapture problem → **stateful NN guidance trained by a GA under a non-stationary MC objective** (adaptive seed curation + tail-weighted cubed cost + worst-case bucket) → benchmarked on identical MC vs FTC + predictor-correctors → headline = `results.json["headline_fresh_pool"]` numbers led by CVaR95/p99 (the propellant-sizing metrics) with mean/p95 + capture % alongside (fresh 8M pool) vs post-fix FTC (`classical_baselines/ftc`), 3998-param net, sub-500 capability floor noted. NO islands, NO warm-start in the method sentence; NO sample max.
- [ ] Introduction: 2009 lineage (quote the closing "next step" line), the 2015-17 speech detour (CG-LSTM, QPSO, divide-and-conquer), contributions list: (1) stateful policy family on a bit-validated simulator; (2) compute-matched optimizer benchmark **across budget AND dimensionality** with actual-sims accounting; (3) the non-stationary-training methodology quartet; (4) first neural-vs-predictor-corrector aerocapture comparison; (5) reproducible campaign (committed bundle). Cite gelly2009/2015/2016/2017.

### Task 8: Problem + Testbed (+ statistical protocol)

Old Task 12 content survives (MSR entry/target, corridor, ΔV metric, bit-validation, EKF/winds/J2-J4, DOPRI45) with two changes: source the 26-dim dispersion list from `sensitivity.DISPERSION_COLUMNS` (not spec §5), and ADD: (a) one paragraph in the DV-metric description stating WHY the tail is the objective — the design-case correction DV sizes the propellant (ergols) budget the spacecraft must carry, and propellant mass directly and considerably drives mission cost; the mean DV is operationally near-irrelevant. CVaR95 approximates exactly this sizing quantity. (b) a **Statistical protocol** subsection carrying the five locked rules verbatim: shared 1000-seed final-eval pool (offset 2M) with paired bootstrap + Wilcoxon; p99 + CVaR95 as tail metrics (sample max descriptive); actual-sims accounting beside any compute-matched claim; fresh-pool (8M) re-quote for the headline; pooled σ_run from the seed-repeats study calibrating every N=1 comparison.

### Task 9: Classical guidance

Old Task 13 CORRECTED: the canonical reference is the **constant-bank target-energy-matched** nominal (`make_reference.py`) — present the reference-design progression as the methodological arc: constant-bank (works; coverage lesson: a reference must reach target energy) → PC-optimized open-loop (tried, under-reaches, refuted) → **jointly-optimized `ref_bank` gene (Study E)** with the `joint_vs_fixed_*` paired deltas. Keep FTC Eq. 10 + roll-reversal as-is. Cite cerimele1985, cherry1964, Lu (FNPAG).

### Task 10: Neural guidance

Old Task 14 with one wording fix: "a **configurable input mask selected via ablation analysis**" (not "learned"). 35-input vector incl. the 3 live correction-DV autoregressive inputs + seam-free (sin,cos) bank-history pairs; architecture family Dense/GRU/LSTM/Window/Transformer/Mamba; decoders atan2/scaled_pi/delta. Cite vaswani2017 + gu2023 for the architectures.

### Task 11: Training & optimization

REWRITTEN from old Task 15: the centerpiece is the **robustness quartet** — (1) GA (population recombination robust to a moving objective; CMA-ES's stationary covariance model suffers most), (2) non-stationary seeds (fixed/rotating/adaptive strategies; adaptive CDF curation), (3) tail-weighted objective (cost_transform moment selection) and (4) worst-case bucket curation — both motivated by the propellant-sizing argument (the tail IS the mission cost function), not as a robustness prior. Optimizer lineage GA(2009) → QPSO(2015 speech) → GA again, vindicated at matched compute. Protocol: 6 optimizers × 3 budgets × 3 dimensionalities (26/515/3998), evals/gen matched, **actual sims reported per cell** (from `results.json["runs"][k]["actual_sims"]`); CMA-ES popsize footnote (evals-matched, not CMA-canonical). Warm-start + RL presented as budget-noted negative results (legacy regime footnote).

### Task 12: Results (~10 subsections)

Every number from `results.json` (no hand-typed values); every cross-cell claim quotes the paired delta + CI; every N=1 comparison cites pooled σ_run.

- [ ] 8.1 Optimizer × budget (fig_optimizer A+C): GA-wins reading + scaling; actual-sims column.
- [ ] 8.2 Optimizer × dimensionality (fig_optimizer B): does the ranking flip at 26 params?
- [ ] 8.3 Seed strategy (fig_seed_strategy — the thesis test): report whatever the data shows; the honest fallback ("GA is simply better here") is pre-authorized.
- [ ] 8.4 Objective shaping: cost_transform (fig_cost_transform) + curation bucket/trim (fig_curation), owning the mean-vs-tail tradeoff with paired CIs.
- [ ] 8.5 Sample efficiency: training_n_sims (fig_training_n_sims), noise floor vs allocation vs ACTUAL sims.
- [ ] 8.6 Architecture Pareto + capability floor (fig_pareto): dense-vs-recurrent + where guiding collapses.
- [ ] 8.7 Output parameterization (fig_output_param).
- [ ] 8.8 Input ablation on the headline model (fig_ablation) + input-report saturation stats.
- [ ] 8.9 Classical vs NN (fig_classical_vs_nn) + joint-reference progression (fig_joint_reference).
- [ ] 8.10 Deployability: pruning/quantization (fig_pruning_quant, legacy-regime footnote) + RL/warm-start negative results.

### Task 13: Discussion + Conclusion

REWRITTEN from old Task 17: robustness via p99/CVaR95 with σ_run honesty; the headline-at-3998 vs floor-at-~200-400 parameter-efficiency tension; why dense + engineered autoregressive inputs (conditional on the GA-regime sweep outcome); cubed/max framed via propellant sizing (the tail sizes the ergols; a ~1-2 m/s mean concession buys the design-case budget down) with the measured paired deltas quoted either way; on-board feasibility. Conclusion: **the 2009 GA endures**; the contribution is training-for-robustness in a moving environment + the dimensionality/floor maps; future work (skip-entry, Earth-return, online adaptation). Echo the 2009 closer, answered.

### Task 14: Bibliography + full compile

Old Task 18 + vaswani2017 (attention), gu2023 (Mamba), and the FNPAG (Lu) / PredGuid refs. `typst compile` with zero unresolved references.

---

# Phase 4 — Finalize

### Task 15: smart-commit

- [ ] Invoke the `smart-commit` skill, instructing it to take the whole `feature/parameter_sweep` branch into account.

---

## Self-review

**Spec coverage:** every campaign study (A, dimensionality, B, C, D, C-sub, E, F, floor, sweep, repeats) has an aggregation key (Task 2), a figure (Task 3), and a results subsection (Task 12); the five reporting rules each have an implementation site (Task 1/2) and a prose home (Task 8). RL/warm-start/QAT covered via the legacy bundle + 8.10. **Placeholder scan:** figure scripts in Task 3 Step 2 are content-spec'd with exact keys/panels/metrics rather than 10x repeated boilerplate — the figlib code in Step 1 is the complete pattern they instantiate. **Type consistency:** `run_stats`/`paired_comparison`/`actual_sims` signatures match between Task 1 tests, Task 1 implementation, and Task 2 imports; bundle keys in Task 2 PAIRED/REPEAT_GROUPS match the `experiments/paper/` output map and `collect_runs.py`'s `legacy/` prefix.
