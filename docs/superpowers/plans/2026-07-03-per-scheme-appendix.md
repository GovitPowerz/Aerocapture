# Per-scheme mission-report appendix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append an "Appendix A" to `articles/paper/paper.typ` with a 2-page end-of-training-style mission report for each of 10 guidance schemes.

**Architecture:** One data collector (`collect_appendix.py`) re-runs 1000-sim MC per scheme (pinned to the deployed model + scaffolding, reserved 2M pool) and writes report-style SVGs + a `stats.json` per scheme, reusing the training pipeline's own `charts.py` panels. A Typst module (`appendix.typ`) lays out 2 pages per scheme from those committed files; `paper.typ` includes it after the bibliography. Same collector-vs-figure split as `collect_corridor.py`.

**Tech Stack:** Python (numpy, matplotlib, seaborn), `aerocapture_rs` PyO3 bindings, `aerocapture.training.{charts,report,reference,evaluate}`, Typst.

## Global Constraints

- Collector scripts MAY read `training_output/`; figure/Typst layers read ONLY committed `articles/paper/` files (established pattern, `collect_corridor.py`).
- MC must reproduce Table 3 numbers: reserved `FINAL_EVAL_SEED_OFFSET` (2M) pool, `n_sims = 1000`, pinned run-local `best_model.json` + `best_params.json` scaffolding, `simulation.n_sims = 1` per override.
- Constraint limits: heat flux 200 kW/m², g-load 4 g, heat load 25000 kJ/m² (config is kJ/m²), dynamic pressure 1.08 kPa.
- Trajectory column indices (`charts._TC_*`): ENERGY 8, PDYN 9, BANK 10, INCL 11, GLOAD 12, HEAT_FLUX 6, HEAT_LOAD 15, TIME 7.
- Final-record indices (`charts._FR_*`): DV1 37, DV2 38, DV3 39, DV_TOTAL 41.
- Ten schemes, Table-3 order, exact `(slug, title, run_dir, training_toml, results_key)` in the `SCHEMES` table in Task 1.
- Appendix uses the report's native chart styling (blue/orange/red three-way classification), NOT the paper's green=Mamba figlib palette — uniform cards.
- Never commit `training_output/` or transient trajectories; commit only the per-scheme SVGs + `stats.json`.
- The user runs the heavy MC; do not push. Final task hands off to smart-commit over the whole branch.

---

### Task 1: Collector engine + DV-CDF helper, proven on one scheme

**Files:**
- Create: `articles/paper/scripts/collect_appendix.py`

**Interfaces:**
- Consumes: `aerocapture.training.report.{_resolve_eval_toml, _read_constraint_limits, compute_eval_summary}`; `aerocapture.training.charts.{chart_corridor_pdyn, chart_corridor_inclination, chart_corridor_bank, chart_heat_flux_time, chart_gload_time, chart_heat_load_time, classify_trajectories, is_captured, _FR_DV_TOTAL, _FR_DV1, _FR_DV2, _FR_DV3}`; `aerocapture.training.evaluate.{FINAL_EVAL_SEED_OFFSET, make_reserved_seeds}`; `aerocapture.training.reference._MC_DISPERSION_DOMAINS`; `aerocapture.training.toml_utils.load_toml_with_bases`; `aerocapture_rs.{run_batch, run_mc}`.
- Produces: for each `slug`, `articles/paper/figures/appendix/<slug>/{corridor_pdyn,corridor_inclination,corridor_bank,dv_cdf,heat_flux,g_load,heat_load}.svg` + `stats.json`. `stats.json` schema = `compute_eval_summary(...)` payload (`capture_rate`, `cost`, `captured.{dv,apoapsis,periapsis,inclination,dv1,dv2,dv3}`, `constraints.{heat_flux,g_load,heat_load}`) plus added top-level `dv_p99`, `dv_cvar95`, `title`.

- [ ] **Step 1: Write the collector script**

Create `articles/paper/scripts/collect_appendix.py`:

```python
"""Collect the per-scheme appendix mission-report data (Appendix A).

For each benchmarked guidance scheme, re-run 1000-sim MC on the reserved
FINAL_EVAL (2M) pool with trajectories (pinned to the run-local deployed model
+ co-trained scaffolding, so the numbers reproduce Table 3 / results.json), then
render the report-style corridor + constraint SVGs and a stats.json into
articles/paper/figures/appendix/<slug>/. Collector-vs-figure split: this reads
training_output/; the committed SVGs + stats.json are the durable artifacts.

Usage:
    uv run python articles/paper/scripts/collect_appendix.py [--schemes SLUG ...] [--n-sims 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

FIGROOT = REPO / "articles/paper/figures/appendix"
RESULTS = REPO / "articles/paper/data/results.json"
CORRIDOR_CACHE = REPO / "training_output/mars/corridor_boundaries.npz"
N_TRAJ_SPAGHETTI = 300
POINT_STRIDE = 3

# (slug, title, run_dir under training_output/, training TOML, results.json key)
SCHEMES = [
    ("nn_mamba", "NN -- Mamba (962 params)", "mamba_p962_long",
     "configs/training/sweep/mamba_p962.toml", "headline/mamba_p962"),
    ("nn_lstm", "NN -- LSTM (1082 params)", "lstm_p1082_long",
     "configs/training/sweep/lstm_p1082.toml", "headline/lstm_p1082"),
    ("nn_gru", "NN -- GRU (1014 params)", "gru_p1014_long",
     "configs/training/sweep/gru_p1014.toml", "headline/gru_p1014"),
    ("nn_dense", "NN -- Dense (515 params)", "dense_p515_ga_paper_best",
     "configs/training/msr_aller_nn_atan2_best_paper.toml", "headline/dense_p515"),
    ("ftc", "FTC (joint reference)", "paper/joint_reference/ftc",
     "configs/training/msr_aller_ftc_joint_ref_train.toml", "joint_reference/ftc"),
    ("fnpag", "FNPAG", "fnpag",
     "configs/training/msr_aller_fnpag_train.toml", "classical_baselines/fnpag"),
    ("predguid", "PredGuid (joint reference)", "paper/joint_reference/pred_guid",
     "configs/training/msr_aller_pred_guid_joint_ref_train.toml", "joint_reference/pred_guid"),
    ("energyctl", "Energy controller (joint reference)", "paper/joint_reference/energy_controller",
     "configs/training/msr_aller_energy_controller_joint_ref_train.toml", "joint_reference/energy_controller"),
    ("eqglide", "Equilibrium glide", "equilibrium_glide",
     "configs/training/msr_aller_eqglide_train.toml", "classical_baselines/equilibrium_glide"),
    ("piecewise", "Piecewise constant", "piecewise_constant",
     "configs/training/msr_aller_piecewise_constant_train.toml", "classical_baselines/piecewise_constant"),
]


def chart_dv_cdf_overlay(final_records, output):
    """4-curve ECDF: total correction DV (bold) + the 3 burns, captured only."""
    from aerocapture.training import charts

    cap = charts.is_captured(final_records)
    rec = final_records[cap]
    series = [
        ("total Δv", np.abs(rec[:, charts._FR_DV_TOTAL]), "#111111", 2.0),
        ("dv1 periapsis raise", np.abs(rec[:, charts._FR_DV1]), "#4878cf", 1.2),
        ("dv2 circularization", np.abs(rec[:, charts._FR_DV2]), "#d1701f", 1.2),
        ("dv3 plane change", np.abs(rec[:, charts._FR_DV3]), "#6a51a3", 1.2),
    ]
    sns.set_theme(style="whitegrid", palette="muted", rc={"axes.facecolor": "#f5f5f5"})
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for label, vals, color, lw in series:
        v = np.sort(vals)
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, color=color, lw=lw, label=label)
    ax.set_xlabel("correction Δv (m/s)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Correction Δv -- empirical CDF (total + burns)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small", loc="lower right")
    sns.despine(fig=fig)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def collect_one(slug, title, run_dir, toml, results_key, n_sims):
    import aerocapture_rs
    from aerocapture.training import charts
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
    from aerocapture.training.report import _read_constraint_limits, _resolve_eval_toml, compute_eval_summary
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(REPO / toml, scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    pin = dict(scaffolding)
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        pin["data.neural_network"] = str(local_model.resolve())
    overrides = [{"simulation.n_sims": 1, "monte_carlo.seed": s, **pin} for s in seeds]
    batch = aerocapture_rs.run_batch(
        toml_path=str(eval_toml.resolve()), overrides_list=overrides,
        include_trajectories=True, sim_timeout_secs=5.0,
    )
    recs = np.asarray(batch.final_records)
    trajs = [np.asarray(t) for t in batch.trajectories]

    # drift self-check vs results.json (the far_tail mislabel trap)
    ref = json.loads(RESULTS.read_text())["runs"][results_key]
    cap = charts.is_captured(recs)
    got_cap = 100.0 * float(cap.mean())
    got_mean = float(np.abs(recs[cap, charts._FR_DV_TOTAL]).mean())
    tag = "OK" if (abs(got_cap - ref["capture_pct"]) <= 0.6 and abs(got_mean - ref["dv_mean"]) <= 2.0) else "DRIFT"
    print(f"[{slug}] capture {got_cap:.1f} vs {ref['capture_pct']} | mean {got_mean:.1f} vs {ref['dv_mean']}  [{tag}]")

    hfl, gll, hll = _read_constraint_limits(eval_toml)
    traj_class = charts.classify_trajectories(recs, heat_flux_limit=hfl, g_load_limit=gll, heat_load_limit=hll)

    # spaghetti subsample (stats use all sims; lines use a subset + point stride)
    k = max(1, len(trajs) // N_TRAJ_SPAGHETTI)
    idx = list(range(0, len(trajs), k))[:N_TRAJ_SPAGHETTI]
    sub_trajs = [trajs[i][::POINT_STRIDE] for i in idx]
    sub_class = traj_class[idx]

    zones = dict(np.load(CORRIDOR_CACHE))
    corridor_data = {key: zones[key] for key in (
        "energy_bins", "envelope_crash_pdyn", "envelope_restricted_max_pdyn",
        "envelope_restricted_min_pdyn", "envelope_capture_pdyn")}

    nom_ov = {"simulation.n_sims": 1,
              **{f"monte_carlo.{d}.level": "off" for d in _MC_DISPERSION_DOMAINS}, **pin}
    nom = aerocapture_rs.run_mc(toml_path=str(eval_toml.resolve()), overrides=nom_ov,
                                include_trajectories=True, sim_timeout_secs=5.0)
    undispersed = np.asarray(nom.trajectories[0]) if nom.trajectories else None
    nk = {"undispersed_nominal": undispersed}

    out = FIGROOT / slug
    out.mkdir(parents=True, exist_ok=True)
    charts.chart_corridor_pdyn(sub_trajs, sub_class, out / "corridor_pdyn.svg", corridor_data=corridor_data, **nk)
    charts.chart_corridor_inclination(sub_trajs, sub_class, out / "corridor_inclination.svg", **nk)
    charts.chart_corridor_bank(sub_trajs, sub_class, out / "corridor_bank.svg", **nk)
    chart_dv_cdf_overlay(recs, out / "dv_cdf.svg")
    charts.chart_heat_flux_time(sub_trajs, sub_class, out / "heat_flux.svg", limit_kw_m2=hfl, **nk)
    charts.chart_gload_time(sub_trajs, sub_class, out / "g_load.svg", limit_g=gll, **nk)
    charts.chart_heat_load_time(sub_trajs, sub_class, out / "heat_load.svg", limit_kj_m2=hll, **nk)

    summary = compute_eval_summary(recs, n_sims=len(recs), cost_kwargs=None)
    dvc = np.abs(recs[cap, charts._FR_DV_TOTAL])
    summary["dv_p99"] = float(np.percentile(dvc, 99))
    summary["dv_cvar95"] = float(np.sort(dvc)[-max(1, round(len(dvc) * 0.05)):].mean())
    summary["title"] = title
    (out / "stats.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"  wrote {out.relative_to(REPO)} (7 svg + stats.json)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="*", default=None, help="slugs to collect (default all)")
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args()
    wanted = set(args.schemes) if args.schemes else None
    for row in SCHEMES:
        if wanted is None or row[0] in wanted:
            collect_one(*row, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the collector on the Mamba scheme only**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python articles/paper/scripts/collect_appendix.py --schemes nn_mamba`
Expected: prints `[nn_mamba] capture 100.0 vs 100.0 | mean 109.x vs 109.85  [OK]` then `wrote articles/paper/figures/appendix/nn_mamba (7 svg + stats.json)`.

- [ ] **Step 3: Verify the outputs exist and the stats reproduce Table 3**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
ls articles/paper/figures/appendix/nn_mamba/
uv run python -c "import json; s=json.load(open('articles/paper/figures/appendix/nn_mamba/stats.json')); print('cap', round(s['capture_rate']*100,1), 'dv_p50', round(s['captured']['dv']['p50'],1), 'dv_cvar95', round(s['dv_cvar95'],1), 'hf_viol', s['constraints']['heat_flux']['viol_pct'])"
```
Expected: 7 `.svg` files + `stats.json`; printed `cap 100.0`, `dv_p50 ~109`, `dv_cvar95 ~115`, `hf_viol 0.0` (matches `headline/mamba_p962` in results.json + Table 3).

- [ ] **Step 4: Commit the collector (not the generated figures yet)**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/scripts/collect_appendix.py
git commit -m "paper(appendix): per-scheme MC report collector"
```

---

### Task 2: Generate and commit all 10 scheme reports

**Files:**
- Create (generated): `articles/paper/figures/appendix/<slug>/*.svg` + `stats.json` for all 10 slugs.

**Interfaces:**
- Consumes: `collect_appendix.py` from Task 1.
- Produces: the committed figure tree the Typst appendix reads in Task 3.

- [ ] **Step 1: Run the collector for all 10 schemes**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python articles/paper/scripts/collect_appendix.py 2>&1 | tee /tmp/appendix_collect.log`
Expected: 10 `[slug] ... [OK]` lines. FNPAG is slowest (~90 s). Total ~3-5 min. Every scheme should print `[OK]`, not `[DRIFT]`.

- [ ] **Step 2: Confirm no DRIFT and all trees present**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
grep DRIFT /tmp/appendix_collect.log && echo "!! DRIFT — investigate before committing" || echo "all OK"
for d in nn_mamba nn_lstm nn_gru nn_dense ftc fnpag predguid energyctl eqglide piecewise; do
  n=$(ls articles/paper/figures/appendix/$d/*.svg 2>/dev/null | wc -l | tr -d ' ')
  echo "$d: $n svg $([ -f articles/paper/figures/appendix/$d/stats.json ] && echo '+ stats' || echo 'MISSING stats')"
done
```
Expected: `all OK`; each scheme `7 svg + stats`. If any line reads `DRIFT`, stop: the pin/label for that scheme is wrong — re-check its `run_dir`/`results_key` before proceeding.

- [ ] **Step 3: Spot-check a sub-100%-capture classical renders the failed class**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "import json; s=json.load(open('articles/paper/figures/appendix/piecewise/stats.json')); print('piecewise cap', round(s['capture_rate']*100,1), 'dv_p50', round(s['captured']['dv']['p50'],1))"`
Expected: `piecewise cap 99.8 dv_p50 ~250` (matches `classical_baselines/piecewise_constant`) — confirms the classification path handles failed trajectories.

- [ ] **Step 4: Commit the generated figures**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/figures/appendix/
git commit -m "paper(appendix): generated per-scheme report figures (10 schemes)"
```

---

### Task 3: Typst appendix layout + paper.typ include

**Files:**
- Create: `articles/paper/appendix.typ`
- Modify: `articles/paper/paper.typ` (after `#bibliography("refs.bib")`)

**Interfaces:**
- Consumes: `articles/paper/figures/appendix/<slug>/{*.svg, stats.json}` from Task 2.
- Produces: ~20 appendix pages in `paper.pdf`.

- [ ] **Step 1: Write the Typst appendix module**

Create `articles/paper/appendix.typ`:

```typst
// Appendix A: per-scheme mission reports. Two pages per scheme.
// Data: figures/appendix/<slug>/{*.svg, stats.json} (built by scripts/collect_appendix.py).
// Report-style panels reuse the training pipeline's charts.py output verbatim.

#let apx = "figures/appendix/"

// One right-aligned stats row: label + up to four value cells.
#let srow(label, ..vals) = (
  [#label], ..vals.pos().map(v => align(right)[#v])
)

#let fnum(x, d: 1) = if x == none { "--" } else { calc.round(x * 1.0, digits: d) }

#let scheme_report(slug, title) = {
  let s = json(apx + slug + "/stats.json")
  let cap = s.captured
  let con = s.constraints

  // ---- Page 1: corridor behaviour ----
  heading(level: 2, title)
  image(apx + slug + "/corridor_pdyn.svg", width: 100%)
  image(apx + slug + "/corridor_inclination.svg", width: 100%)
  image(apx + slug + "/corridor_bank.svg", width: 100%)
  pagebreak()

  // ---- Page 2: cost + constraints + stats ----
  heading(level: 2, title + " (continued)")
  image(apx + slug + "/dv_cdf.svg", width: 100%)
  v(4pt)
  grid(columns: 3, gutter: 5pt,
    image(apx + slug + "/heat_flux.svg", width: 100%),
    image(apx + slug + "/g_load.svg", width: 100%),
    image(apx + slug + "/heat_load.svg", width: 100%))
  v(8pt)

  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.5pt + luma(200),
    inset: 4pt,
    table.header([*Statistic*], [*p50*], [*p95*], [*mean/max*], [*note*]),
    srow([Capture rate], fnum(s.capture_rate * 100), [], [], [%]),
    srow([Correction Δv (m/s)], fnum(cap.dv.p50), fnum(cap.dv.p95), fnum(cap.dv.mean), [mean]),
    srow([Δv CVaR95 / p99 / max], fnum(s.dv_cvar95), fnum(s.dv_p99), fnum(cap.dv.max), [tail]),
    srow([dv1 periapsis raise], fnum(cap.dv1.p50), fnum(cap.dv1.p95), fnum(cap.dv1.max), [m/s]),
    srow([dv2 circularization], fnum(cap.dv2.p50), fnum(cap.dv2.p95), fnum(cap.dv2.max), [m/s]),
    srow([dv3 plane change], fnum(cap.dv3.p50), fnum(cap.dv3.p95), fnum(cap.dv3.max), [m/s]),
    srow([Apoapsis error (km)], fnum(cap.apoapsis.p50), fnum(cap.apoapsis.p95), fnum(cap.apoapsis.mean), [mean]),
    srow([Periapsis error (km)], fnum(cap.periapsis.p50), fnum(cap.periapsis.p95), fnum(cap.periapsis.mean), [mean]),
    srow([Inclination error (deg)], fnum(cap.inclination.p50, d: 2), fnum(cap.inclination.p95, d: 2), fnum(cap.inclination.mean, d: 2), [mean]),
    srow([Heat flux (kW/m²)], fnum(con.heat_flux.p50), fnum(con.heat_flux.p95), fnum(con.heat_flux.max), [viol #fnum(con.heat_flux.viol_pct)%]),
    srow([G-load (g)], fnum(con.g_load.p50, d: 2), fnum(con.g_load.p95, d: 2), fnum(con.g_load.max, d: 2), [viol #fnum(con.g_load.viol_pct)%]),
    srow([Heat load (kJ/m²)], fnum(con.heat_load.p50, d: 0), fnum(con.heat_load.p95, d: 0), fnum(con.heat_load.max, d: 0), [viol #fnum(con.heat_load.viol_pct)%]),
  )
  pagebreak()
}

#scheme_report("nn_mamba", "NN -- Mamba (962 params)")
#scheme_report("nn_lstm", "NN -- LSTM (1082 params)")
#scheme_report("nn_gru", "NN -- GRU (1014 params)")
#scheme_report("nn_dense", "NN -- Dense (515 params)")
#scheme_report("ftc", "FTC (joint reference)")
#scheme_report("fnpag", "FNPAG")
#scheme_report("predguid", "PredGuid (joint reference)")
#scheme_report("energyctl", "Energy controller (joint reference)")
#scheme_report("eqglide", "Equilibrium glide")
#scheme_report("piecewise", "Piecewise constant")
```

- [ ] **Step 2: Wire the appendix into paper.typ**

In `articles/paper/paper.typ`, the last content line is:
```typst
#bibliography("refs.bib")
```
Replace it with:
```typst
#bibliography("refs.bib")

#pagebreak()
#set heading(numbering: none)
= Appendix A: per-scheme mission reports

Each scheme below gets a two-page mission-performance card on the final-evaluation
Monte-Carlo pool ($n = 1000$), pinned to its deployed policy so the statistics
reproduce @tbl-perf. The first page shows the corridor behaviour -- the classified
trajectory ensemble in the (energy, dynamic pressure), (energy, inclination), and
(energy, bank) planes, with the undispersed nominal overlaid. The second page shows
the correction-$Delta v$ distribution (total and its three burns), the thermal and
load-constraint margins, and the full statistics. Panels reuse the training
pipeline's own report charts; captured trajectories are blue, constraint violations
orange, failures red.

#include "appendix.typ"
```

- [ ] **Step 3: Compile the paper and verify the appendix renders**

Run: `cd /Users/govit/Git/Govit/Aerocapture && typst compile articles/paper/paper.typ articles/paper/paper.pdf 2>&1 | head && echo COMPILE_OK`
Expected: `COMPILE_OK` with no errors.

- [ ] **Step 4: Verify page count grew by ~20 and spot-check two cards**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
SCR=/private/tmp/claude-501/-Users-govit-Git-Govit-Aerocapture/bde7d896-ec32-488f-9789-2cb52745ca76/scratchpad
typst compile --format png --pages 21 --ppi 90 articles/paper/paper.typ $SCR/apx_p21.png && echo "rendered a mid-appendix page"
uv run python -c "import pypdf; print('pages', len(pypdf.PdfReader('articles/paper/paper.pdf').pages))" 2>/dev/null || echo "(page count check optional)"
```
Expected: total pages ~40 (was ~20). Read `$SCR/apx_p21.png` to confirm a scheme card (corridors or the stats table + constraint row) renders correctly.

- [ ] **Step 5: Commit the Typst appendix**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add articles/paper/appendix.typ articles/paper/paper.typ
git commit -m "paper(appendix): Appendix A layout + wire into paper.typ"
```

---

### Task 4: Whole-branch documentation sync + final commit

**Files:**
- Modify (if warranted): `paper_resume.md` (note the appendix pipeline).

- [ ] **Step 1: Invoke the smart-commit skill over the whole branch**

Use the `smart-commit` skill, telling it to take the whole git branch into account: it should sync any stale doc references (e.g. a `paper_resume.md` line noting `collect_appendix.py` + `appendix.typ` as the per-scheme appendix pipeline, and the ~40pp paper) and make a final commit of anything outstanding related to the appendix. Do NOT stage the user's in-flight exp-13 files (`TODO.md`, `articles/paper/data/robustness_retrain.json`, `experiments/paper/13_robustness_retrain.sh`, `configs/training/paper/robustness_retrain/*.toml`).

---

## Self-Review

**Spec coverage:** 10-scheme scope + Table-3 order → SCHEMES table (Task 1). Joint-ref variants → run_dirs in SCHEMES. Page 1 three corridors → `chart_corridor_*` calls (Task 1 collector, Task 3 layout). Page 2 DV CDF total+3 burns → `chart_dv_cdf_overlay` (Task 1). Three constraint panels reused verbatim → `chart_heat_flux_time/_gload_time/_heat_load_time` (Task 1). Stats table → `compute_eval_summary` + `dv_p99`/`dv_cvar95` (Task 1) rendered in `appendix.typ` (Task 3). Pinned MC reproducing Table 3 + drift check → Task 1 collector + Task 2 gate. Collector-vs-figure split, commit SVGs only → Tasks 1/2. Appended to paper.typ, heading numbering off, captionless cards → Task 3. Window/Transformer excluded → SCHEMES omits them. Final smart-commit → Task 4. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step carries complete code; every run step has an exact command + expected output. No "handle edge cases" hand-waves (the drift check and sub-100%-capture spot-check are concrete).

**Type consistency:** `collect_one(slug, title, run_dir, toml, results_key, n_sims)` matches the SCHEMES 5-tuple unpacked with `n_sims=` kwarg in `main`. `chart_dv_cdf_overlay(final_records, output)` matches its Task 1 call. `stats.json` keys written in Task 1 (`capture_rate`, `captured.dv/dv1/dv2/dv3/apoapsis/periapsis/inclination`, `constraints.heat_flux/g_load/heat_load.{p50,p95,max,viol_pct}`, `dv_p99`, `dv_cvar95`) exactly match the `appendix.typ` reads in Task 3. Constraint chart kwargs (`limit_kw_m2`, `limit_g`, `limit_kj_m2`) match `charts.py` signatures verified against source. `compute_eval_summary(final_records, n_sims, cost_kwargs)` signature matches report.py:204.
