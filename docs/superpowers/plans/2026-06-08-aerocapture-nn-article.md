# Aerocapture Neural-Guidance Article — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a comprehensive Typst research paper (the follow-up to Gelly & Vernis, AIAA GNC 2009) presenting the repo's neural aerocapture guidance, benchmarked against classical/predictor-corrector schemes and across optimizers and NN architectures.

**Architecture:** Three build phases. (1) Generate controlled-experiment configs + a runner; the **user executes** the ~6 new training runs + EqGlide eval. (2) Aggregate all results (committed + fresh) into one JSON and render figures with matplotlib. (3) Author the Typst paper section by section, reading numbers from the aggregated JSON and following the authorial-voice guide. Finish with `smart-commit`.

**Tech Stack:** Python (uv, pyarrow/pandas/matplotlib), the repo's `aerocapture.training` pipeline (pymoo islands/PSO/GA/DE/CMA-ES + RL PPO), Typst (single-column academic template, Hayagriva bibliography).

**Spec:** `docs/superpowers/specs/2026-06-08-aerocapture-nn-article-design.md` — §5 holds the committed reference numbers; cite it for any value not produced by a fresh run.

**Note on prose tasks:** Typst section tasks (12-18) are *content-spec'd*, not pre-written prose. Each lists the exact claims, numbers (from spec §5 or the fresh aggregated JSON), figures, and citations the section must contain, plus the voice file to follow. Prose is written at execution. Code artifacts (configs, scripts, template) carry complete code.

---

## File structure

```
configs/training/paper/                         NEW — controlled-experiment leaf configs
  opt_pso.toml  opt_ga.toml  opt_de.toml  opt_cmaes.toml  opt_warmstart.toml  opt_rl.toml
  out_scaledpi.toml  out_delta.toml
run_paper_experiments.sh                        NEW — runner (mirrors sweep.sh), repo root
articles/paper/
  main.typ            NEW — document shell
  template.typ        NEW — page/heading/abstract/figure helpers
  refs.yml            NEW — Hayagriva bibliography
  sections/00_abstract.typ … 09_conclusion.typ  NEW — one file per section
  data/results.json   GENERATED — aggregated per-run summary stats
  figures/*.svg       GENERATED
  scripts/
    aggregate_results.py   NEW — parquet+JSONL → data/results.json
    fig_pareto.py          NEW — params vs DV-p95 across architectures
    fig_optimizer.py       NEW — convergence + deployed-DV bar
    fig_output_param.py     NEW — atan2/scaled_pi/delta bar
    fig_classical_vs_nn.py  NEW — DV CDF + corridor
    fig_ablation.py         NEW — wraps aerocapture.training.ablation output
    fig_pruning_quant.py    NEW — DV vs bit-width/sparsity
```

Reuse where possible: `src/python/aerocapture/training/charts.py` (corridor/CDF helpers), `aerocapture.training.ablation`, `aerocapture.training.report` (EqGlide final eval).

---

# Phase 1 — Controlled experiments (user runs the training)

### Task 1: Study-A optimizer configs (PSO/GA/DE/CMA-ES/warm-start)

**Files:**
- Create: `configs/training/paper/opt_pso.toml`, `opt_ga.toml`, `opt_de.toml`, `opt_cmaes.toml`, `opt_warmstart.toml`

All five base-inherit `sweep/dense_p515.toml` (so they reuse its 3-layer dense architecture + the atan2 pipeline), override only the deploy path and the optimizer. Compute-fairness: single optimizers run at `n_pop=300` (set on the CLI in Task 4); islands-based warm-start runs at `n_pop=100`.

- [ ] **Step 1: Write `opt_pso.toml`**

```toml
# Study A — PSO on the dense_p515 control architecture (compute-matched n_pop=300).
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_pso/best_model.json"
results_suffix = ".paper_opt_pso"

[optimizer]
algorithm = "pso"
```

- [ ] **Step 2: Write `opt_ga.toml`** (identical but `algorithm = "ga"`, paths `paper_opt_ga`)

```toml
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_ga/best_model.json"
results_suffix = ".paper_opt_ga"

[optimizer]
algorithm = "ga"
```

- [ ] **Step 3: Write `opt_de.toml`** (`algorithm = "de"`, paths `paper_opt_de`)

```toml
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_de/best_model.json"
results_suffix = ".paper_opt_de"

[optimizer]
algorithm = "de"
```

- [ ] **Step 4: Write `opt_cmaes.toml`** (`algorithm = "cma_es"`, paths `paper_opt_cmaes`)

```toml
# CMA-ES runs natively here: 515 params < _CMAES_MAX_PARAMS (20000).
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_cmaes/best_model.json"
results_suffix = ".paper_opt_cmaes"

[optimizer]
algorithm = "cma_es"
```

- [ ] **Step 5: Write `opt_warmstart.toml`** (islands + supervised warm-start)

```toml
# Study A — supervised warm-start then islands (the 2016 divide-and-conquer analogue).
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_opt_warmstart/best_model.json"
results_suffix = ".paper_opt_warmstart"

[optimizer]
algorithm = "islands"

[warm_start]
supervisor_schemes = ["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]
n_warm_seeds = 200
n_epochs = 10
bptt_length = 32
```

- [ ] **Step 6: Verify each config resolves and yields a 515-param dense net**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
for n in ['pso','ga','de','cmaes','warmstart']:
    c=load_toml_with_bases(f'configs/training/paper/opt_{n}.toml')
    arch=c['network']['architecture']
    print(n, c['optimizer']['algorithm'], [l['output_size'] for l in arch])
"
```
Expected: each prints its algorithm and `[18, 9, 2]` (the inherited dense_p515 architecture).

- [ ] **Step 7: Commit**

```bash
git add configs/training/paper/opt_pso.toml configs/training/paper/opt_ga.toml configs/training/paper/opt_de.toml configs/training/paper/opt_cmaes.toml configs/training/paper/opt_warmstart.toml
git commit -m "feat(paper): Study-A optimizer configs (PSO/GA/DE/CMA-ES/warm-start) on dense_p515"
```

---

### Task 2: Study-A RL (PPO) config on the dense architecture

**Files:**
- Create: `configs/training/paper/opt_rl.toml`

The RL path uses `aerocapture.training.rl.train` and `rl_common.toml`. To compare fairly, give it the same dense architecture + the atan2 17-input mask. This is the run most likely to need iteration — flagged in spec §10.

- [ ] **Step 1: Read the atan2 mask + normalization to copy**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
c=load_toml_with_bases('configs/training/sweep/dense_p515.toml')
print('input_mask =', c['network']['input_mask'])
"
```
Record the printed `input_mask` for Step 2.

- [ ] **Step 2: Write `opt_rl.toml`** (paste the `input_mask` from Step 1 in place of `<MASK>`)

```toml
# Study A — PPO on the dense_p515 architecture (RL track; see spec §10 — may need iteration).
base = ["../missions/mars.toml", "common.toml", "rl_common.toml"]

[mission]
mission_type = "msr_aller"

[guidance]
type = "neural_network"

[guidance.neural_network]
mode = "full_neural"
output_parameterization = "atan2_signed"

[network]
input_mask = <MASK>

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 18
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 18
output_size = 9
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 9
output_size = 2
activation = "asinh"

[data]
neural_network       = "training_output/paper_opt_rl/best_model.json"
reference_trajectory = "training_output/mars/ref_trajectory.dat"
```

- [ ] **Step 3: Smoke-check the RL config parses**

Run: `uv run python -m aerocapture.training.rl.train configs/training/paper/opt_rl.toml --total-steps 2000 --no-tui --skip-report`
Expected: starts a short PPO run without a config error (Ctrl-C after it begins stepping). If it errors on the architecture/mask, note the error for follow-up — this is the flagged-risky run.

- [ ] **Step 4: Commit**

```bash
git add configs/training/paper/opt_rl.toml
git commit -m "feat(paper): Study-A RL/PPO config on dense_p515 architecture"
```

---

### Task 3: Study-B output-parameterization configs (scaled_pi, delta)

**Files:**
- Create: `configs/training/paper/out_scaledpi.toml`, `out_delta.toml`

Same dense control + islands, but the last layer becomes 9→1 (tanh) and the decoder changes. `[[network.architecture]]` arrays REPLACE under merge, so respecify all three layers.

- [ ] **Step 1: Write `out_scaledpi.toml`**

```toml
# Study B — 1D scaled_pi head on dense control + islands.
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_out_scaledpi/best_model.json"
results_suffix = ".paper_out_scaledpi"

[optimizer]
algorithm = "islands"

[guidance.neural_network]
output_parameterization = "scaled_pi"
scaled_pi_n = 2.0

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 18
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 18
output_size = 9
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 9
output_size = 1
activation = "tanh"
```

- [ ] **Step 2: Write `out_delta.toml`** (decoder `delta`, `delta_max = 0.35`, paths `paper_out_delta`)

```toml
# Study B — 1D delta head on dense control + islands.
base = ["../sweep/dense_p515.toml"]

[data]
neural_network = "training_output/paper_out_delta/best_model.json"
results_suffix = ".paper_out_delta"

[optimizer]
algorithm = "islands"

[guidance.neural_network]
output_parameterization = "delta"
delta_max = 0.35

[[network.architecture]]
type = "dense"
input_size = 17
output_size = 18
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 18
output_size = 9
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 9
output_size = 1
activation = "tanh"
```

- [ ] **Step 3: Verify both resolve with a 1-output tanh head**

Run:
```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
for n in ['scaledpi','delta']:
    c=load_toml_with_bases(f'configs/training/paper/out_{n}.toml')
    last=c['network']['architecture'][-1]
    print(n, c['guidance']['neural_network']['output_parameterization'], last['output_size'], last['activation'])
"
```
Expected: `scaledpi scaled_pi 1 tanh` and `delta delta 1 tanh`.

- [ ] **Step 4: Commit**

```bash
git add configs/training/paper/out_scaledpi.toml configs/training/paper/out_delta.toml
git commit -m "feat(paper): Study-B output-parameterization configs (scaled_pi, delta)"
```

---

### Task 4: Runner script + EqGlide eval

**Files:**
- Create: `run_paper_experiments.sh` (repo root, mirrors `sweep.sh`)

- [ ] **Step 1: Confirm EqGlide has a deployable best**

Run: `ls training_output/equilibrium_glide/best_params.json`
- If present: the EqGlide line below generates its `final_eval.parquet`.
- If absent: prepend an EqGlide training line (`uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --n-gen 2000 --n-pop 60`).

- [ ] **Step 2: Write `run_paper_experiments.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── Study A: optimizer comparison on dense_p515 (compute-matched) ──
uv run python -m aerocapture.training.train configs/training/paper/opt_pso.toml       --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_ga.toml        --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_de.toml        --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_cmaes.toml     --n-gen 2000 --n-pop 300 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/opt_warmstart.toml --n-gen 2000 --n-pop 100 --from-scratch

# ── Study A: RL (PPO) on the dense architecture ──
uv run python -m aerocapture.training.rl.train configs/training/paper/opt_rl.toml --algorithm ppo --total-steps 5000000

# ── Study B: output parameterization on dense + islands ──
uv run python -m aerocapture.training.train configs/training/paper/out_scaledpi.toml --n-gen 2000 --n-pop 100 --from-scratch
uv run python -m aerocapture.training.train configs/training/paper/out_delta.toml    --n-gen 2000 --n-pop 100 --from-scratch

# ── EqGlide deploy/eval to populate the classical table ──
uv run python -m aerocapture.training.report training_output/equilibrium_glide/ --toml configs/training/msr_aller_eqglide_train.toml
```

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x run_paper_experiments.sh
git add run_paper_experiments.sh
git commit -m "feat(paper): runner for Study-A/B controlled experiments + EqGlide eval"
```

- [ ] **Step 4: GATE — hand off to the user**

Tell the user: *"Configs ready. Run `./run_paper_experiments.sh` (compute budget: scale `--n-gen` down uniformly across the five pymoo lines if needed — keep all five equal). Ping me when the runs finish and I'll aggregate."* Do not proceed to Phase 2 for the fresh-run tables until the runs exist; figures/sections using committed data (spec §5) can proceed in parallel.

---

# Phase 2 — Aggregate results + figures

### Task 5: Results aggregator

**Files:**
- Create: `articles/paper/scripts/aggregate_results.py`
- Generate: `articles/paper/data/results.json`

- [ ] **Step 1: Write `aggregate_results.py`**

```python
"""Aggregate per-run summary stats (committed + fresh) into one JSON for the paper."""
import glob, json, os
import numpy as np
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(REPO, "articles/paper/data/results.json")

# label -> training_output subdir
RUNS = {
    # classical
    "ftc": "ftc", "piecewise_constant": "piecewise_constant", "fnpag": "fnpag",
    "pred_guid": "pred_guid", "energy_controller": "energy_controller",
    "equilibrium_glide": "equilibrium_glide",
    # architecture sweep (islands)
    **{f"sweep_{a}_{p}": f"sweep_{a}_{p}" for a, ps in {
        "dense": ["p515","p972","p1957","p3998"], "gru": ["p478","p1014","p1954","p4082"],
        "lstm": ["p458","p1082","p1962","p4118"], "mamba": ["p482","p962","p2027","p4072"],
        "transformer": ["p762","p1112","p2004","p3822"], "window": ["p609","p1027","p2036"],
    }.items() for p in ps},
    # optimizer study (fresh; islands reuse = sweep_dense_p515)
    "opt_pso": "paper_opt_pso", "opt_ga": "paper_opt_ga", "opt_de": "paper_opt_de",
    "opt_cmaes": "paper_opt_cmaes", "opt_warmstart": "paper_opt_warmstart",
    "opt_islands": "sweep_dense_p515", "opt_rl": "paper_opt_rl",
    # output-param study (fresh; atan2 reuse = sweep_dense_p515)
    "out_atan2": "sweep_dense_p515", "out_scaledpi": "paper_out_scaledpi", "out_delta": "paper_out_delta",
    # pruning / quantization
    "qat8": "neural_network_atan2_qat8", "qat4": "neural_network_atan2_qat4",
}

def best_val_rms(d):
    best = None
    for f in glob.glob(os.path.join(d, "run_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            v = (json.loads(line).get("validation") or {}).get("rms_cost")
            if v is not None and (best is None or v < best):
                best = v
    return best

def summarize(label, sub):
    d = os.path.join(REPO, "training_output", sub)
    p = os.path.join(d, "final_eval.parquet")
    if not os.path.exists(p):
        return {"label": label, "dir": sub, "missing": True}
    df = pq.read_table(p).to_pandas()
    cap = (df["ifinal"] == 3) & (df["eccentricity"] < 1.0)
    dvc = df.loc[cap, "dv_total_m_s"].to_numpy()
    f = lambda a, q: float(np.percentile(a, q)) if len(a) else None
    return {
        "label": label, "dir": sub, "n": int(len(df)), "capture_pct": round(100 * cap.mean(), 2),
        "dv_mean": round(float(dvc.mean()), 2) if len(dvc) else None,
        "dv_p50": f(dvc, 50), "dv_p95": f(dvc, 95), "dv_max": float(dvc.max()) if len(dvc) else None,
        "heat_flux_p95": f(df["max_heat_flux_kw_m2"].to_numpy(), 95),
        "g_load_max": float(df["max_load_factor_g"].max()),
        "bank_consumption_mean": round(float(df["cumulative_bank_change_deg"].mean()), 1),
        "best_val_rms": best_val_rms(d),
    }

out = {label: summarize(label, sub) for label, sub in RUNS.items()}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2)
print(f"wrote {OUT} ({sum(1 for v in out.values() if not v.get('missing'))}/{len(out)} runs present)")
```

- [ ] **Step 2: Run it**

Run: `uv run python articles/paper/scripts/aggregate_results.py`
Expected: prints `wrote .../results.json (N/M runs present)`; fresh runs (`opt_*`, `out_*`) show as `missing` until Phase 1 completes — that is fine.

- [ ] **Step 3: Commit**

```bash
git add articles/paper/scripts/aggregate_results.py articles/paper/data/results.json
git commit -m "feat(paper): results aggregator + initial committed-data snapshot"
```

---

### Task 6: Architecture Pareto figure (params vs DV-p95)

**Files:**
- Create: `articles/paper/scripts/fig_pareto.py`
- Generate: `articles/paper/figures/pareto_arch.svg`

Param counts come from the config names (`p515` → 515). This is the figure that carries the "dense best, Mamba 2nd, strongest at low params" claim.

- [ ] **Step 1: Write `fig_pareto.py`**

```python
"""Params vs DV-p95 Pareto across the six NN architectures (islands sweep)."""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
res = json.load(open(os.path.join(REPO, "articles/paper/data/results.json")))
ARCHS = ["dense", "gru", "lstm", "mamba", "transformer", "window"]

fig, ax = plt.subplots(figsize=(7, 4.5))
for arch in ARCHS:
    pts = []
    for label, v in res.items():
        m = re.match(rf"sweep_{arch}_p(\d+)$", label)
        if m and not v.get("missing"):
            pts.append((int(m.group(1)), v["dv_p95"]))
    pts.sort()
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=arch)
ax.set_xlabel("trainable parameters")
ax.set_ylabel("DV p95 (m/s, captured)")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = os.path.join(REPO, "articles/paper/figures/pareto_arch.svg")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print("wrote", out)
```

- [ ] **Step 2: Run + verify the SVG exists and dense dominates at low params**

Run: `uv run python articles/paper/scripts/fig_pareto.py && ls -la articles/paper/figures/pareto_arch.svg`
Expected: SVG written; the dense curve sits at/below the others at the ~500-param end.

- [ ] **Step 3: Commit**

```bash
git add articles/paper/scripts/fig_pareto.py articles/paper/figures/pareto_arch.svg
git commit -m "feat(paper): architecture params-vs-DV-p95 Pareto figure"
```

---

### Task 7: Optimizer figures (convergence + deployed-DV bar)

**Files:**
- Create: `articles/paper/scripts/fig_optimizer.py`
- Generate: `articles/paper/figures/opt_convergence.svg`, `articles/paper/figures/opt_bar.svg`

- [ ] **Step 1: Write `fig_optimizer.py`** — two panels:
  - **Convergence:** for each of `paper_opt_pso/ga/de/cmaes/warmstart` + `sweep_dense_p515` (islands), read every `run_*.jsonl`, plot the running-min of `validation.rms_cost` (fallback `best_cost`) vs `generation`. Skip missing dirs.
  - **Bar:** grouped bars of `dv_mean`, `dv_p95`, `dv_max` from `results.json` keys `opt_pso/ga/de/cmaes/warmstart/islands/rl`. Skip missing.

  Follow the `fig_pareto.py` structure (Agg backend, read `results.json`, `savefig` SVG). For the convergence panel, read JSONL directly with the running-min pattern from `aggregate_results.py::best_val_rms`.

- [ ] **Step 2: Run + verify** both SVGs exist (panels may be sparse until fresh runs land).

Run: `uv run python articles/paper/scripts/fig_optimizer.py && ls articles/paper/figures/opt_*.svg`

- [ ] **Step 3: Commit**

```bash
git add articles/paper/scripts/fig_optimizer.py articles/paper/figures/opt_convergence.svg articles/paper/figures/opt_bar.svg
git commit -m "feat(paper): optimizer convergence + deployed-DV figures"
```

---

### Task 8: Output-parameterization figure

**Files:**
- Create: `articles/paper/scripts/fig_output_param.py`
- Generate: `articles/paper/figures/output_param.svg`

- [ ] **Step 1: Write `fig_output_param.py`** — grouped bars (`dv_mean`/`dv_p95`/`dv_max`) for `results.json` keys `out_atan2`, `out_scaledpi`, `out_delta`. Title notes "dense control + islands". Same Agg/savefig pattern as `fig_pareto.py`.

- [ ] **Step 2: Run + verify** `articles/paper/figures/output_param.svg` exists.

- [ ] **Step 3: Commit**

```bash
git add articles/paper/scripts/fig_output_param.py articles/paper/figures/output_param.svg
git commit -m "feat(paper): output-parameterization comparison figure"
```

---

### Task 9: Remaining figures (classical-vs-NN, ablation, pruning/quant)

**Files:**
- Create: `articles/paper/scripts/fig_classical_vs_nn.py`, `fig_ablation.py`, `fig_pruning_quant.py`
- Generate: `figures/classical_vs_nn.svg`, `figures/corridor.svg`, `figures/ablation.svg`, `figures/pruning_quant.svg`

- [ ] **Step 1: `fig_classical_vs_nn.py`** — (a) ECDF of `dv_total_m_s` (captured) overlaying FTC, FNPAG, PredGuid, PiecewiseConstant, and the best dense NN (`sweep_dense_p515`), read straight from each `final_eval.parquet`; (b) a corridor panel for best-NN vs FTC reusing `aerocapture.training.charts` corridor helpers if a trajectory parquet is available, else skip with a printed note. Output `classical_vs_nn.svg` (+ `corridor.svg` if produced).

- [ ] **Step 2: `fig_ablation.py`** — run `python -m aerocapture.training.ablation training_output/sweep_dense_p515 --toml configs/training/sweep/dense_p515.toml --n-sims 200`, then render its JSON as a horizontal bar of DV-delta per input (or reuse `charts_ablation.chart_ablation_bar`). Output `ablation.svg`.

- [ ] **Step 3: `fig_pruning_quant.py`** — bars of `dv_p95` for `out_atan2` (full), `qat8`, `qat4` from `results.json`. Output `pruning_quant.svg`.

- [ ] **Step 4: Run all three + verify SVGs exist.**

Run: `uv run python articles/paper/scripts/fig_classical_vs_nn.py && uv run python articles/paper/scripts/fig_ablation.py && uv run python articles/paper/scripts/fig_pruning_quant.py && ls articles/paper/figures/`

- [ ] **Step 5: Commit**

```bash
git add articles/paper/scripts/fig_classical_vs_nn.py articles/paper/scripts/fig_ablation.py articles/paper/scripts/fig_pruning_quant.py articles/paper/figures/
git commit -m "feat(paper): classical-vs-NN, input-ablation, and pruning/quant figures"
```

---

# Phase 3 — Typst paper

### Task 10: Typst template, shell, and bibliography skeleton

**Files:**
- Create: `articles/paper/template.typ`, `articles/paper/main.typ`, `articles/paper/refs.yml`

- [ ] **Step 1: Write `template.typ`**

```typ
#let paper(title: "", authors: (), abstract: [], body) = {
  set document(title: title)
  set page("a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
  set text(font: "New Computer Modern", size: 10.5pt)
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, text(size: 13pt, weight: "bold", it))
  align(center)[#block(text(17pt, weight: "bold", title))]
  align(center)[#text(11pt, authors.join("  "))]
  v(0.4em)
  block(inset: (x: 1.2em), [*Abstract* — #abstract])
  v(0.6em)
  body
}
#let figref(path, caption) = figure(image(path, width: 100%), caption: caption)
```

- [ ] **Step 2: Write `main.typ`**

```typ
#import "template.typ": paper

#show: paper.with(
  title: "Neural-Network Guidance for Aerocapture: Architectures, Optimization, and a Benchmark Against Predictor-Corrector Schemes",
  authors: ("G. Gelly",),
  abstract: include "sections/00_abstract.typ",
)

#include "sections/01_introduction.typ"
#include "sections/02_problem.typ"
#include "sections/03_testbed.typ"
#include "sections/04_classical.typ"
#include "sections/05_neural.typ"
#include "sections/06_training.typ"
#include "sections/07_results.typ"
#include "sections/08_discussion.typ"
#include "sections/09_conclusion.typ"

#bibliography("refs.yml", title: "References", style: "ieee")
```

- [ ] **Step 3: Write a minimal `refs.yml`** with the four self-citations (gelly2009, gelly2015, gelly2016, gelly2017) + cherry1964 + cerimele1985 in Hayagriva YAML (title/author/date/parent). Full list completed in Task 18.

- [ ] **Step 4: Create empty section stubs so the doc compiles**

```bash
mkdir -p articles/paper/sections
for s in 00_abstract 01_introduction 02_problem 03_testbed 04_classical 05_neural 06_training 07_results 08_discussion 09_conclusion; do echo "// $s" > "articles/paper/sections/$s.typ"; done
```

- [ ] **Step 5: Verify it compiles**

Run: `typst compile articles/paper/main.typ articles/paper/main.pdf`
Expected: produces `main.pdf` with title, author, empty body. (If `typst` is absent: `brew install typst`.)

- [ ] **Step 6: Commit**

```bash
git add articles/paper/template.typ articles/paper/main.typ articles/paper/refs.yml articles/paper/sections/
git commit -m "feat(paper): Typst template, document shell, and section stubs"
```

---

### Task 11: Abstract + Introduction

**Files:** `articles/paper/sections/00_abstract.typ`, `01_introduction.typ`
**Voice:** follow `articles/markdown/05_authorial_voice_and_style.md` (§7 has a ready opener); lineage from `articles/markdown/00_synthesis_writing_kit.md`.

- [ ] **Step 1: Write the abstract** — one paragraph: aerocapture problem → stateful NN guidance trained by 3-island PSO/GA/DE with supervised warm-start → benchmarked on identical MC against FTC + predictor-correctors → headline (best dense NN **119.6 / 131.1 / 164.5** mean/p95/max m/s, 100% capture, ~515 params vs FTC 136.2/172.6/275.7).
- [ ] **Step 2: Write the introduction** — 2009 lineage (quote the "next step: predictor-correctors" closer), the 2015-2017 speech-NN detour that built the recurrent-cell + swarm + warm-start machinery, contributions list (stateful policies; compute-matched optimizer benchmark; bit-validated simulator; first neural-vs-predictor-corrector aerocapture comparison). Cite `gelly2009/2015/2016/2017`.
- [ ] **Step 3: Compile + commit**

Run: `typst compile articles/paper/main.typ articles/paper/main.pdf`
```bash
git add articles/paper/sections/00_abstract.typ articles/paper/sections/01_introduction.typ
git commit -m "docs(paper): abstract + introduction"
```

---

### Task 12: Problem formulation + Simulation testbed

**Files:** `sections/02_problem.typ`, `03_testbed.typ`
**Source:** `articles/markdown/01_2009_AIAA_neural_guidance.md` §IV (corridor, MSR conditions, ΔV metric); repo CLAUDE.md (sim fidelity).

- [ ] **Step 1: Problem section** — aerocapture definition; corridor in (energy, pdyn) with restricted ±δZa; MSR entry (120 km, 5687 m/s, −10.24°, 38.04°) + target (apo 500 km, peri 11 km, incl 50°); ΔV correction-cost (apo+peri+incl), 113 m/s floor; EI energy 4.91 → exit −5.87 MJ/kg.
- [ ] **Step 2: Testbed section** — bit-validated Rust sim (725 timesteps, 22/24 photo columns exact); MC dispersions (entry state, density ±50%, winds, Gauss-Markov density OU, mass/aero — reference spec §5 + the 26-dim dispersion list); EKF nav + bias mode; winds; J2/J3/J4; fixed-RK4 vs adaptive DOPRI45. Contrast with 2009's 4-DOF/1 Hz.
- [ ] **Step 3: Compile + commit** (`docs(paper): problem formulation + simulation testbed`).

---

### Task 13: Classical guidance algorithms

**Files:** `sections/04_classical.typ`
**Source:** `articles/markdown/01_2009_AIAA_neural_guidance.md` §IV.F (FTC/Cerimele-Gamble); repo guidance modules.

- [ ] **Step 1: Write the section** — PiecewiseConstant (corridor + reference generator); **FTC with the PC-reference improvement** (FTC enslaves a piecewise-constant-optimized reference trajectory rather than a single constant-bank trajectory; in-plane apoapsis law Eq. 10 cos μ_com = cos μ_ref + G_ḣ(ḣ−ḣ_ref)/q + G_q(q−q_ref)/q + roll-reversal out-of-plane); FNPAG (Lu numerical predictor-corrector, 3-DOF forward predictor); PredGuid (Apollo/Shuttle drag tracking); EqGlide; EnergyController. Cite `cerimele1985`, `cherry1964`, FNPAG/PredGuid refs (add to refs.yml in Task 18).
- [ ] **Step 2: Compile + commit** (`docs(paper): classical guidance algorithms`).

---

### Task 14: Neural guidance (architectures, inputs, output parameterizations)

**Files:** `sections/05_neural.typ`
**Source:** `articles/markdown/02-04` (CG-LSTM lineage), repo CLAUDE.md (architecture family, 35-input vector, decoders).

- [ ] **Step 1: Write the section** — the stateful architecture family (Dense/GRU/LSTM/Window/Transformer/Mamba) as the generalization of the 2009 single-hidden-layer net; the 35-candidate input vector incl. the 3 live correction-DV autoregressive inputs + bank-history (sin,cos) pairs, with a learned input mask; output parameterizations — 2D atan2 (the 2009 Eq. 11 sin/cos decoder), 1D scaled_pi, 1D delta. Cite `gelly2015/2016/2017` for the recurrent-cell lineage.
- [ ] **Step 2: Compile + commit** (`docs(paper): neural guidance architectures and decoders`).

---

### Task 15: Training & optimization

**Files:** `sections/06_training.typ`
**Source:** `articles/markdown/00` (optimizer lineage), `02/03` (QPSO, divide-and-conquer), spec §4 (compute-fairness).

- [ ] **Step 1: Write the section** — optimizer lineage GA(2009)→QPSO(2015)→islands; PSO/GA/DE/CMA-ES/RL(PPO)/warm-start; the 3-island PSO/GA/DE model with migration; supervised warm-start as the 2016 divide-and-conquer analogue; **state the compute-fairness protocol explicitly** (islands = 3×n_pop/gen; single optimizers run at n_pop=300 to match islands@100; RL budgeted in env steps).
- [ ] **Step 2: Compile + commit** (`docs(paper): training and optimization`).

---

### Task 16: Results

**Files:** `sections/07_results.typ`
**Source:** the fresh `articles/paper/data/results.json` (Phase 1/2) + spec §5 for committed values; figures from Phase 2.

- [ ] **Step 1: Write the six results subsections, each with its table + figure:**
  - 8.1 Optimizer comparison (table: best_val_rms + DV mean/p95/max + capture for opt_pso/ga/de/cmaes/islands/warmstart/rl; figure `opt_convergence.svg` + `opt_bar.svg`) → islands best, RL worst.
  - 8.2 Architecture sweep (table from `sweep_*`; figure `pareto_arch.svg`) → dense best, Mamba 2nd, **framed as low-param regime** (note convergence at ~4000 params).
  - 8.3 Output parameterization (table out_atan2/scaledpi/delta; figure `output_param.svg`).
  - 8.4 Input ablation (figure `ablation.svg`) → autoregressive correction-DV inputs explain dense > Mamba.
  - 8.5 Classical vs NN (headline table: FTC/PC/FNPAG/PredGuid/EnergyController/EqGlide vs best dense NN; figure `classical_vs_nn.svg`).
  - 8.6 Pruning & quantization (table full/QAT8/QAT4; figure `pruning_quant.svg`).
- [ ] **Step 2: Re-run `aggregate_results.py`** first so the tables use fresh numbers; pull every value from `results.json` (no hand-typed numbers that a fresh run supersedes).
- [ ] **Step 3: Compile + commit** (`docs(paper): results`).

---

### Task 17: Discussion + Conclusion

**Files:** `sections/08_discussion.typ`, `09_conclusion.typ`

- [ ] **Step 1: Discussion** — robustness (impressively low p95/max), parameter efficiency (best at ~515 params), why dense+autoregressive-inputs beats internal recurrence, on-board feasibility (training is the only heavy cost; deployed policy is tiny and quantizable).
- [ ] **Step 2: Conclusion** — plain dense NN best + incredibly robust with very few parameters; islands improved training over the 2009 GA; future work (skip-entry, Earth-return leg, on-line adaptation). Echo the 2009 closer, now answered.
- [ ] **Step 3: Compile + commit** (`docs(paper): discussion + conclusion`).

---

### Task 18: Complete bibliography + full compile

**Files:** `articles/paper/refs.yml`

- [ ] **Step 1: Complete `refs.yml`** — self-citations (gelly2009/2015/2016/2017, gelly2007 EUCASS, vernis2004 ESA GNC); classical (cherry1964, cerimele1985); methods (hochreiter1997 LSTM, graves2012, werbos1990 BPTT, kennedy1995/clerc2002 PSO, sun2004 QPSO); add FNPAG (Lu) + PredGuid references. Use the bibliography in `articles/markdown/00_synthesis_writing_kit.md` §5 as the source list.
- [ ] **Step 2: Full compile + check every `@cite` resolves and every figure renders**

Run: `typst compile articles/paper/main.typ articles/paper/main.pdf 2>&1 | tail -20`
Expected: no unresolved-reference or missing-image warnings; `main.pdf` present.

- [ ] **Step 3: Commit** (`docs(paper): complete bibliography + full compile`).

---

# Phase 4 — Finalize

### Task 19: smart-commit over the branch

- [ ] **Step 1:** Invoke the `smart-commit` skill, instructing it to take the whole `feature/parameter_sweep` branch into account (syncs CLAUDE.md/README if needed, then commits anything outstanding). Per the user's planning rule, this is the final step.

---

## Self-review

**Spec coverage:** every spec §2 section maps to a Typst task (11-17); §3 experiments → Tasks 1-4; §4 compute-fairness → Tasks 1/4/15; §5 reference numbers → Task 5 aggregator + Task 16; §6 figures → Tasks 6-9; §7 Typst layout → Task 10; §8 sequencing → phase order; §9/§10 constraints → Tasks 1 (CMA-ES native), 2 (RL risk), 4 (EqGlide + budget). No gaps.

**Placeholder scan:** the only intentional fill-in is `<MASK>` in Task 2 Step 2, with Step 1 producing the exact value to paste — not a placeholder. Prose tasks are content-spec'd by design (stated in the header), with exact numbers/figures/citations. Figure Tasks 7/9 reference the complete `fig_pareto.py` pattern rather than repeating boilerplate.

**Type/name consistency:** `results.json` keys (`opt_pso`, `out_atan2`, `sweep_dense_p515`, …) are defined in Task 5 `RUNS` and reused verbatim by Tasks 6-9 and 16. Figure paths (`articles/paper/figures/*.svg`) are consistent across generation (Phase 2) and inclusion (Task 16). Config deploy dirs (`paper_opt_*`, `paper_out_*`) match between Tasks 1-3 and the aggregator `RUNS` map.
