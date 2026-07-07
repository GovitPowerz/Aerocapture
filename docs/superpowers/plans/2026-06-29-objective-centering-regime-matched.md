# Objective-Centering (Regime-Matched Worst-Case Shaping) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runnable infrastructure (configs + runner + eval + figure) for a controlled experiment showing the GA's worst-case objective-shaping (cubed transform x max-bucket curation x few sims) backfires under the high/adversarial dispersion regime and that "centering" the objective recovers the selection gradient.

**Architecture:** Five base-inheriting cell configs on the fast dense_515 net, all under the high regime, flipping one objective lever at a time from "stacked" to "centered"; an idempotent shell runner that trains them iso-compute-matched on total training sims (gen counts set so `n_pop * n_sims * n_gen` is constant); a Python eval that scores each deployed cell on the reserved 9M stress pool and extracts the per-generation validation capture-rate convergence series; a figure builder. Phase 2 adds one Mamba_962 config carrying the winning centered recipe. The heavy training is run by the user; every task here is verifiable without a full training run.

**Tech Stack:** Rust simulator via PyO3 (`aerocapture_rs`), Python 3.14 + `uv`, the existing `aerocapture.training` package (`train.py`, `evaluate.py`, `report.py`, `paper_stats.py`), TOML base-inheritance, matplotlib/figlib for the figure.

**Spec:** `docs/superpowers/specs/2026-06-29-objective-centering-regime-matched-design.md`

## Global Constraints

- Python tooling via `uv` (`uv run python ...`); never bare `python`.
- All training/eval passes `--sim-timeout 5` / `sim_timeout_secs=5.0` (caps the known NaN-hang failure mode; never fires for healthy sims).
- `cost_transform` is set under `[cost_function]`; `curation_bucket_selection` and `training_n_sims` under `[optimizer]`. `training_n_sims` is also overridable via the `--training-n-sims` CLI flag (used by the runner so the budget math lives in one place).
- The high regime = exactly these four MC domains at `level = "high"`: `atmosphere`, `density_perturbation`, `navigation`, `nav_filter` (everything else inherits the controlled regime). This MUST match `robustness_stress.py` / `robustness_retrain_eval.py` so the 9M pool numbers are comparable.
- Iso-compute budget `B = n_pop * n_sims * n_gen`, held constant per cell. `n_pop = 256`, `B ~= 8.19e6` training sims: `n_sims=2 -> n_gen=16000`, `n_sims=16 -> n_gen=2000`.
- Cross-cell convergence MUST be read on `validation.capture_rate` (transform-independent). `rms_cost` / `*_cost` fields are in each cell's transform space (cubed vs linear) and are comparable ONLY among same-transform cells.
- Reserved seed pools (disjoint, do not change): VALIDATION 1M, FINAL_EVAL 2M, STRESS_EVAL 9M.
- Git: work on the current feature branch (`feature/parameter_sweep`); never commit to `main`; never push; stage only the files a task creates/edits (never `git add -A`). Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Cells are NN schemes with `scaffolding = "live"` (inherited from `msr_aller_nn_atan2_train.toml`), so each writes both `best_model.json` and `best_params.json`; the eval applies the co-trained scaffolding via `report._resolve_eval_toml`.

---

### Task 1: Phase 1 cell configs (5 dense cells)

**Files:**
- Create: `configs/training/paper/objective_centering/dense_stacked_high.toml`
- Create: `configs/training/paper/objective_centering/dense_plus_sims_high.toml`
- Create: `configs/training/paper/objective_centering/dense_plus_bucket_high.toml`
- Create: `configs/training/paper/objective_centering/dense_plus_transform_high.toml`
- Create: `configs/training/paper/objective_centering/dense_centered_high.toml`

**Interfaces:**
- Consumes: `configs/training/sweep/dense_p515.toml` (verified to exist; base-inherits `msr_aller_nn_atan2_train.toml`, defines the 17->18->9->2 atan2 dense arch + scaffolding=live).
- Produces: five resolvable training configs; each deploys to `training_output/paper/objective_centering/<cell>/` (cell = the file stem minus `_high`). `n_sims` is NOT set in these TOMLs — the runner passes it via `--training-n-sims`.

- [ ] **Step 1: Write `dense_stacked_high.toml`**

```toml
# objective-centering Phase 1 -- STACKED control: the medium-regime winning
# objective stack (cubed transform x max-bucket curation), run UNDER the high
# regime. n_sims (=2) is passed by the runner via --training-n-sims. dense_515
# vehicle (fast, memoryless); the objective-shaping effect is arch-independent.
base = ["../../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper/objective_centering/dense_stacked/best_model.json"
results_suffix = ".objcenter_dense_stacked"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "max"

[cost_function]
cost_transform = "cubed"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 2: Write `dense_plus_sims_high.toml`** (same as stacked but the cell name; lever flipped = sims, applied by the runner via `--training-n-sims 16`)

```toml
# objective-centering Phase 1 -- +SIMS: stacked objective (cubed x max) but a
# larger sample budget (n_sims=16 via the runner). Tests whether more sims alone
# recovers the gradient under the high regime (hypothesized dominant lever).
base = ["../../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper/objective_centering/dense_plus_sims/best_model.json"
results_suffix = ".objcenter_dense_plus_sims"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "max"

[cost_function]
cost_transform = "cubed"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 3: Write `dense_plus_bucket_high.toml`** (lever flipped = bucket -> middle; n_sims stays 2)

```toml
# objective-centering Phase 1 -- +BUCKET: stacked objective but a CENTRAL
# curation bucket (middle) instead of max. Tests whether not feeding only the
# hardest seeds recovers the gradient. n_sims=2 via the runner.
base = ["../../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper/objective_centering/dense_plus_bucket/best_model.json"
results_suffix = ".objcenter_dense_plus_bucket"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "middle"

[cost_function]
cost_transform = "cubed"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 4: Write `dense_plus_transform_high.toml`** (lever flipped = transform -> linear; n_sims stays 2)

```toml
# objective-centering Phase 1 -- +TRANSFORM: stacked objective but a MILD
# (linear) cost transform instead of cubed. Tests whether not cubing a
# failure-dominated cost recovers the gradient. n_sims=2 via the runner.
base = ["../../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper/objective_centering/dense_plus_transform/best_model.json"
results_suffix = ".objcenter_dense_plus_transform"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "max"

[cost_function]
cost_transform = "linear"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 5: Write `dense_centered_high.toml`** (all three levers centered: linear + middle + n_sims=16)

```toml
# objective-centering Phase 1 -- CENTERED: all three levers centered (linear
# transform + middle bucket + n_sims=16 via the runner). The full "centered"
# objective for the adversarial regime.
base = ["../../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper/objective_centering/dense_centered/best_model.json"
results_suffix = ".objcenter_dense_centered"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "middle"

[cost_function]
cost_transform = "linear"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 6: Verify all five resolve with the intended knobs**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python - <<'PY'
import sys; sys.path.insert(0,"src/python")
from aerocapture.training.toml_utils import load_toml_with_bases
import glob
for f in sorted(glob.glob("configs/training/paper/objective_centering/dense_*_high.toml")):
    c=load_toml_with_bases(f); mc=c["monte_carlo"]; o=c["optimizer"]; cf=c["cost_function"]
    hi=[d for d in ("atmosphere","density_perturbation","navigation","nav_filter") if mc[d]["level"]=="high"]
    arch=[l["type"] for l in c["network"]["architecture"]]
    print(f.split('/')[-1], "| transform=",cf["cost_transform"], "bucket=",o["curation_bucket_selection"],
          "| high=",len(hi), "arch=",arch, "guidance=",c["guidance"]["type"])
PY
```
Expected: all 5 print `guidance= neural_network`, `arch= ['dense','dense','dense']`, `high= 4`, and the transform/bucket per the table (stacked & plus_sims: cubed/max; plus_bucket: cubed/middle; plus_transform: linear/max; centered: linear/middle).

- [ ] **Step 7: Commit**

```bash
git add configs/training/paper/objective_centering/
git commit -m "$(printf 'exp(objcenter): Phase 1 dense cell configs (high regime, lever attribution)\n\nFive base-inheriting dense_515 cells under the high regime, flipping one\nobjective lever at a time from stacked (cubed/max) to centered (linear/middle).\nn_sims passed by the runner for the iso-compute budget.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: Eval + convergence extractor (`objective_centering_eval.py`)

**Files:**
- Create: `articles/paper/scripts/objective_centering_eval.py`
- Reference (do not modify): `articles/paper/scripts/robustness_retrain_eval.py` (the eval-plumbing template), `src/python/aerocapture/training/paper_stats.py` (`run_stats`).

**Interfaces:**
- Consumes: each cell's deployed `training_output/paper/objective_centering/<cell>/{best_model.json,best_params.json,final_eval.parquet}` and `run_*.jsonl`.
- Produces: `articles/paper/data/objective_centering.json` with `{"stress_overrides", "n_sims_eval", "pool", "cells":[{label, capture_pct, dv_mean, dv_cvar95, dv_cvar99, ...}], "convergence":{label:[[cum_train_sims, capture_rate], ...]}}`. Function `extract_convergence(jsonl_path, n_pop, n_sims) -> list[[int,float]]` is the transform-independent series.

- [ ] **Step 1: Write the convergence-extractor failing test**

```bash
cd /Users/govit/Git/Govit/Aerocapture && cat > /tmp/test_objcenter_extract.py <<'PY'
import sys; sys.path.insert(0,"articles/paper/scripts")
from objective_centering_eval import extract_convergence
# the stopped Mamba run is a real high-regime jsonl with validation records
J="training_output/paper/robustness_retrain/mamba_p962/run_000_20260627T155837.jsonl"
series = extract_convergence(J, n_pop=512, n_sims=2)
assert len(series) > 100, f"expected many validation points, got {len(series)}"
xs=[p[0] for p in series]; caps=[p[1] for p in series]
assert xs == sorted(xs), "cumulative sims must be monotone non-decreasing"
assert all(0.0 <= c <= 1.0 for c in caps), "capture rate in [0,1]"
assert all(x % (512*2) == 0 for x in xs), "x-axis must be gen*n_pop*n_sims (multiple of 1024)"
print("OK", len(series), "points; last", series[-1])
PY
uv run python /tmp/test_objcenter_extract.py
```
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'extract_convergence'` (file not created yet).

- [ ] **Step 2: Write `objective_centering_eval.py`**

```python
"""objective-centering eval: score the five Phase-1 cells on the 9M stress pool
and extract the transform-independent validation capture-rate convergence series.

Same machinery / regime / pool as robustness_retrain_eval.py, so the deployed
off-nominal numbers are directly comparable. Convergence is read on
validation.capture_rate (NOT rms_cost, which is in each cell's transform space).

Usage:
    uv run python articles/paper/scripts/objective_centering_eval.py [--n-sims 1000]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import run_stats  # noqa: E402

# (label, run_dir under training_output/, training TOML). n_sims is the training
# budget used (for the convergence x-axis), set by the runner.
CELLS = [
    ("stacked", "paper/objective_centering/dense_stacked", "configs/training/paper/objective_centering/dense_stacked_high.toml", 2),
    ("plus_sims", "paper/objective_centering/dense_plus_sims", "configs/training/paper/objective_centering/dense_plus_sims_high.toml", 16),
    ("plus_bucket", "paper/objective_centering/dense_plus_bucket", "configs/training/paper/objective_centering/dense_plus_bucket_high.toml", 2),
    ("plus_transform", "paper/objective_centering/dense_plus_transform", "configs/training/paper/objective_centering/dense_plus_transform_high.toml", 2),
    ("centered", "paper/objective_centering/dense_centered", "configs/training/paper/objective_centering/dense_centered_high.toml", 16),
]
N_POP = 256
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/objective_centering.json"


def extract_convergence(jsonl_path: str, n_pop: int, n_sims: int) -> list[list]:
    """Per-validation [cumulative_training_sims, capture_rate]. Transform-independent."""
    series: list[list] = []
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("validation") or {}
            cap = v.get("capture_rate")
            gen = r.get("generation")
            if cap is None or gen is None:
                continue
            series.append([int(gen) * n_pop * n_sims, float(cap)])
    series.sort(key=lambda p: p[0])
    return series


def _eval_one(label: str, run_dir: str, toml: str, n_sims_train: int, n_eval: int) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import STRESS_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(Path(toml), scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, STRESS_EVAL_SEED_OFFSET, n_eval)
    base: dict = {"simulation.n_sims": 1, **STRESS_OVERRIDES, **scaffolding}
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        base["data.neural_network"] = str(local_model.resolve())
    overrides = [{**base, "monte_carlo.seed": s} for s in seeds]
    results = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, sim_timeout_secs=5.0)
    recs = np.asarray(results.final_records)
    col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    stats = {"label": label, **run_stats(col["ifinal"], col["eccentricity"], col["dv_total_m_s"], n_boot=2000)}
    jsonls = sorted(glob.glob(str(scheme_dir / "run_*.jsonl")))
    stats["convergence"] = extract_convergence(jsonls[-1], N_POP, n_sims_train) if jsonls else []
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)
    cells_out, convergence = [], {}
    for label, run_dir, toml, n_sims_train in CELLS:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        s = _eval_one(label, run_dir, toml, n_sims_train, args.n_sims)
        convergence[label] = s.pop("convergence")
        cells_out.append(s)
        print(f"  {label:16s} stress: capture {s['capture_pct']:5.1f}% | mean {s['dv_mean']:7.1f} | CVaR95 {s.get('dv_cvar95'):7.1f} | conv pts {len(convergence[label])}")
    if cells_out:
        OUT.write_text(json.dumps({"stress_overrides": STRESS_OVERRIDES, "n_sims_eval": args.n_sims, "pool": "STRESS_EVAL 9M", "n_pop": N_POP, "cells": cells_out, "convergence": convergence}, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the extractor test to verify it passes**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python /tmp/test_objcenter_extract.py`
Expected: `OK <N> points; last [<sims>, <cap>]` with N in the thousands (the stopped Mamba had 8262 validated gens), capture near ~0.96.

- [ ] **Step 4: Smoke the eval end-to-end (cells absent -> clean skip)**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python articles/paper/scripts/objective_centering_eval.py --n-sims 20`
Expected: five `skip <label> (... not deployed yet)` lines and NO `wrote` line (no cells trained yet). Confirms imports + control flow.

- [ ] **Step 5: Commit**

```bash
git add articles/paper/scripts/objective_centering_eval.py
git commit -m "$(printf 'exp(objcenter): eval + transform-independent convergence extractor\n\nScores the five Phase-1 cells on the 9M stress pool (same machinery/regime as\nrobustness_retrain_eval) and extracts validation capture-rate vs cumulative\ntraining sims (transform-independent -- rms_cost is in per-cell transform space).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
rm -f /tmp/test_objcenter_extract.py
```

---

### Task 3: Phase 1 runner (`14_objective_centering.sh`)

**Files:**
- Create: `experiments/paper/14_objective_centering.sh`
- Reference: `experiments/paper/13_robustness_retrain.sh` (runner template).

**Interfaces:**
- Consumes: the five Task 1 configs; `objective_centering_eval.py` from Task 2.
- Produces: trained cells under `training_output/paper/objective_centering/<cell>/` and `articles/paper/data/objective_centering.json`.

- [ ] **Step 1: Write the runner**

```bash
#!/usr/bin/env bash
set -euo pipefail
# exp(objcenter) -- objective-centering under the high/adversarial regime (NOT part
# of the numbered campaign reproduction). Spec:
# docs/superpowers/specs/2026-06-29-objective-centering-regime-matched-design.md
#
# Five dense_515 cells, all UNDER the high regime, flipping one objective lever at
# a time from the medium-regime-winning stack (cubed x max-bucket x n_sims=2) to
# centered (linear x middle x n_sims=16). Iso-compute matched on total training
# sims B = n_pop*n_sims*n_gen (n_pop=256, B~=8.19e6): n_sims=2 -> 16000 gens,
# n_sims=16 -> 2000 gens. Then eval all deployed cells on the reserved 9M pool.
#
# Idempotent (skip-if-final_eval.parquet per cell). Dense vehicle is fast; this is
# the methodology comparison. Override the budget knob with NPOP / BUDGET env vars.
NPOP=${NPOP:-256}
NSIMS_EVAL=${NSIMS_EVAL:-1000}
# gen counts per n_sims to hold B = NPOP * n_sims * n_gen ~= 8.19e6 fixed:
GEN_N2=${GEN_N2:-16000}
GEN_N16=${GEN_N16:-2000}

train() {  # $1=config-stem  $2=cell  $3=n_sims  $4=n_gen
  if [ -f "training_output/paper/objective_centering/$2/final_eval.parquet" ]; then
    echo "skip $2 (done)"; return 0
  fi
  uv run python -m aerocapture.training.train \
      "configs/training/paper/objective_centering/$1.toml" \
      --training-n-sims "$3" --n-gen "$4" --n-pop "$NPOP" \
      --output-dir "training_output/paper/objective_centering/$2" \
      --sim-timeout 5 --from-scratch
}

train dense_stacked_high         dense_stacked         2  "$GEN_N2"
train dense_plus_sims_high       dense_plus_sims       16 "$GEN_N16"
train dense_plus_bucket_high     dense_plus_bucket     2  "$GEN_N2"
train dense_plus_transform_high  dense_plus_transform  2  "$GEN_N2"
train dense_centered_high        dense_centered        16 "$GEN_N16"

uv run python articles/paper/scripts/objective_centering_eval.py --n-sims "$NSIMS_EVAL"
```

- [ ] **Step 2: Make executable and shell-check**

Run: `cd /Users/govit/Git/Govit/Aerocapture && chmod +x experiments/paper/14_objective_centering.sh && bash -n experiments/paper/14_objective_centering.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 3: Verify the iso-compute budget arithmetic**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && python3 -c "
npop=256
for nsims,ngen in [(2,16000),(16,2000)]:
    print(f'n_sims={nsims:2d} n_gen={ngen:5d} -> B={npop*nsims*ngen:,} training sims')
"
```
Expected: both lines print `B=8,192,000 training sims` (iso-compute confirmed).

- [ ] **Step 4: Commit**

```bash
git add experiments/paper/14_objective_centering.sh
git commit -m "$(printf 'exp(objcenter): Phase 1 runner (iso-compute lever sweep + 9M eval)\n\nTrains the five dense cells under the high regime, iso-compute matched on total\ntraining sims (n_sims=2 -> 16000 gens, n_sims=16 -> 2000 gens; B~=8.19e6), then\nevals on the 9M stress pool. Idempotent; budget overridable via env vars.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Figure builder (`fig_objective_centering.py`)

**Files:**
- Create: `articles/paper/scripts/fig_objective_centering.py`
- Reference: `articles/paper/scripts/figlib.py` (`fl.style()`, `fl.save()`, `fl.C` color map), `articles/paper/scripts/fig_seed_strategy.py` (a two-panel example).

**Interfaces:**
- Consumes: `articles/paper/data/objective_centering.json` (Task 2 output).
- Produces: `articles/paper/figures/fig_objective_centering.svg` (left: validation capture-rate vs cumulative training sims, one line per cell; right: deployed 9M CVaR95 bars per cell). Degrades gracefully if fewer than five cells are present.

- [ ] **Step 1: Write the figure builder**

```python
"""fig_objective_centering -- the regime-matched methodology figure.

Left: validation capture-rate vs cumulative training sims (transform-independent),
one line per cell -- the gradient-recovery evidence (stacked flat ~96%, centered
climbing). Right: deployed off-nominal CVaR95 on the 9M stress pool, per cell.
Reads only articles/paper/data/objective_centering.json; degrades if cells missing.
"""

import json
from pathlib import Path

import figlib as fl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "articles/paper/data/objective_centering.json"
ORDER = ["stacked", "plus_sims", "plus_bucket", "plus_transform", "centered"]


def main():
    fl.style()
    d = json.loads(DATA.read_text())
    conv = d.get("convergence", {})
    cells = {c["label"]: c for c in d.get("cells", [])}
    present = [k for k in ORDER if k in conv or k in cells]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for i, label in enumerate(present):
        series = conv.get(label, [])
        if series:
            xs = [p[0] / 1e6 for p in series]
            ys = [100.0 * p[1] for p in series]
            axL.plot(xs, ys, label=label, linewidth=1.4)
    axL.set_xlabel("cumulative training sims (millions)")
    axL.set_ylabel("validation capture rate (%)")
    axL.set_title("Gradient recovery under the high regime", fontsize=10, loc="left")
    axL.legend(fontsize=7, loc="lower right")

    labels = [k for k in ORDER if k in cells]
    vals = [cells[k].get("dv_cvar95") for k in labels]
    axR.bar(range(len(labels)), vals, color="#4C72B0")
    axR.set_xticks(range(len(labels)))
    axR.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axR.set_ylabel("deployed CVaR$_{95}$ on 9M pool (m/s)")
    axR.set_title("Off-nominal deployment", fontsize=10, loc="left")

    fig.tight_layout()
    fl.save(fig, "fig_objective_centering")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke with a synthetic data file (no trained cells yet)**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && python3 -c "
import json
json.dump({'convergence':{'stacked':[[0,0.54],[2000000,0.95],[8000000,0.96]],'centered':[[0,0.6],[2000000,0.99],[8000000,1.0]]},
           'cells':[{'label':'stacked','dv_cvar95':520.0},{'label':'centered','dv_cvar95':320.0}]},
          open('articles/paper/data/objective_centering.json','w'))
"
cd articles/paper/scripts && uv run python fig_objective_centering.py && ls -la ../figures/fig_objective_centering.svg
```
Expected: `wrote ...fig_objective_centering.svg` (or the figlib save message) and the SVG file exists.

- [ ] **Step 3: Remove the synthetic data file (it is not a real result)**

Run: `cd /Users/govit/Git/Govit/Aerocapture && rm -f articles/paper/data/objective_centering.json && echo removed`
Expected: `removed`.

- [ ] **Step 4: Commit**

```bash
git add articles/paper/scripts/fig_objective_centering.py
git commit -m "$(printf 'exp(objcenter): figure builder (gradient recovery + off-nominal bars)\n\nTwo-panel fig from objective_centering.json: validation capture-rate vs\ncumulative training sims (transform-independent gradient-recovery signal) and\ndeployed 9M CVaR95 bars. Degrades gracefully when cells are missing.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 5: Phase 2 Mamba config + runner section

**Files:**
- Create: `configs/training/paper/objective_centering/mamba_centered_high.toml`
- Modify: `experiments/paper/14_objective_centering.sh` (append a gated Phase 2 block)

**Interfaces:**
- Consumes: `configs/training/sweep/mamba_p962.toml` (the deployed headline arch).
- Produces: `training_output/paper/objective_centering/mamba_centered/` when run with `RUN_MAMBA=1`.

- [ ] **Step 1: Write the Mamba config (winning centered recipe; default = full centered)**

```toml
# objective-centering Phase 2 -- confirm the Phase-1 winning centered recipe on
# the deployed Mamba_962 architecture under the high regime. Defaults to the FULL
# centered recipe (linear + middle, n_sims=16 via the runner); if Phase 1 shows a
# single dominant lever, narrow this to that lever before running.
base = ["../../sweep/mamba_p962.toml"]

[data]
neural_network = "training_output/paper/objective_centering/mamba_centered/best_model.json"
results_suffix = ".objcenter_mamba_centered"

[optimizer]
algorithm = "ga"
curation_bucket_selection = "middle"

[cost_function]
cost_transform = "linear"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.density_perturbation]
level = "high"

[monte_carlo.navigation]
level = "high"

[monte_carlo.nav_filter]
level = "high"
```

- [ ] **Step 2: Verify it resolves (Mamba arch + centered knobs + high regime)**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture && uv run python - <<'PY'
import sys; sys.path.insert(0,"src/python")
from aerocapture.training.toml_utils import load_toml_with_bases
c=load_toml_with_bases("configs/training/paper/objective_centering/mamba_centered_high.toml")
print("arch=",[l["type"] for l in c["network"]["architecture"]],
      "transform=",c["cost_function"]["cost_transform"],
      "bucket=",c["optimizer"]["curation_bucket_selection"],
      "high=",sum(c["monte_carlo"][d]["level"]=="high" for d in ("atmosphere","density_perturbation","navigation","nav_filter")))
PY
```
Expected: `arch= ['dense','mamba','dense'] transform= linear bucket= middle high= 4`.

- [ ] **Step 3: Append the gated Phase 2 block to the runner**

Add to the END of `experiments/paper/14_objective_centering.sh`:

```bash

# ---- Phase 2 (gated): confirm the winning centered recipe on Mamba_962 ----
# Run only when RUN_MAMBA=1 (the long pole). Update mamba_centered_high.toml to
# the Phase-1 winning lever first if a single lever dominated. Mamba plateaus
# ~10-15k gens; default GEN_MAMBA is directional -- scale up for a final number.
if [ "${RUN_MAMBA:-0}" = "1" ]; then
  GEN_MAMBA=${GEN_MAMBA:-4000}
  if [ ! -f "training_output/paper/objective_centering/mamba_centered/final_eval.parquet" ]; then
    uv run python -m aerocapture.training.train \
        configs/training/paper/objective_centering/mamba_centered_high.toml \
        --training-n-sims 16 --n-gen "$GEN_MAMBA" --n-pop "$NPOP" \
        --output-dir training_output/paper/objective_centering/mamba_centered \
        --sim-timeout 5 --from-scratch
  else
    echo "skip mamba_centered (done)"
  fi
fi
```

- [ ] **Step 4: Re-shell-check the runner**

Run: `cd /Users/govit/Git/Govit/Aerocapture && bash -n experiments/paper/14_objective_centering.sh && echo OK`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/training/paper/objective_centering/mamba_centered_high.toml experiments/paper/14_objective_centering.sh
git commit -m "$(printf 'exp(objcenter): Phase 2 Mamba confirmation config + gated runner block\n\nmamba_centered_high.toml carries the centered recipe on the deployed Mamba_962\narch; runner Phase 2 block runs it only under RUN_MAMBA=1 (the long pole).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 6: Wire exp into docs + final smart-commit

**Files:**
- Modify: `paper_resume.md` (campaign table + a results-pending note)
- Modify: `experiments/paper/README.md` (run-order list)

**Interfaces:**
- Consumes: nothing.
- Produces: docs reflecting the new experiment; the whole branch committed.

- [ ] **Step 1: Add the exp row to `paper_resume.md` campaign table**

Add after the exp-13 row (the `| 13 | ... |` line):
```markdown
| 14 | `14_objective_centering.sh` | objective-centering: 5-cell lever attribution (dense_515) under the high regime + Phase 2 Mamba confirm | SET UP, not run (tests regime-matched worst-case shaping; spec 2026-06-29) |
```

- [ ] **Step 2: Add the runner to `experiments/paper/README.md` run-order block**

Add after the `13_robustness_retrain.sh` line inside the run-order code fence:
```
./experiments/paper/14_objective_centering.sh        OPTIONAL, off-campaign: objective-centering lever attribution under the high regime (dense_515; Phase 2 Mamba via RUN_MAMBA=1). Tests that worst-case shaping is regime-matched. Spec 2026-06-29.
```

- [ ] **Step 3: Commit the doc edits**

```bash
git add paper_resume.md experiments/paper/README.md
git commit -m "$(printf 'docs(objcenter): wire exp-14 into the campaign table + README\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

- [ ] **Step 4: Final smart-commit over the whole branch**

Invoke the `smart-commit` skill, instructing it to take the whole git branch into account (it will sync any CLAUDE.md/README.md left stale and commit any remaining intended changes). Do NOT stage the untracked partial `articles/paper/data/robustness_retrain.json` (incomplete exp-13 result).

---

## Notes for the executor

- This plan delivers **runnable infrastructure**; the heavy GA training is launched by the user (`./experiments/paper/14_objective_centering.sh`, then `RUN_MAMBA=1 ...` for Phase 2). Do not run the full training as part of executing the plan — the per-task verifications (config resolution, shell-check, extractor test, eval smoke, figure smoke) are the deliverable gates.
- After the user trains the cells, `objective_centering_eval.py` and `fig_objective_centering.py` produce the real `objective_centering.json` + SVG; only then is the figure paper-ready.
- The starting budget (`GEN_N2=16000`, `GEN_N16=2000`, `NPOP=256`) is a knob; if the `n_sims=16` cells are still descending at 2000 gens, raise `GEN_N16` (and `GEN_N2` to keep B matched) — `train.py` auto-resumes.
