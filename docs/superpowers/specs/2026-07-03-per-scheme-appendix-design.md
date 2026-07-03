# Per-scheme mission-report appendix

**Date:** 2026-07-03
**Status:** design (awaiting review)
**Author:** Grégory Gelly (with Claude)

## Goal

Add an **Appendix A** to `articles/paper/paper.typ` with a compact end-of-training-style
mission report for each benchmarked guidance scheme: **two pages per scheme**, reusing the
chart panels the training pipeline already produces (`report.py` Part 2 / `charts.py`). The
appendix gives the reader the per-scheme corridor behaviour and the full statistics behind
Table 3, which the main body only summarizes.

## Scope

**Ten schemes, best variant each, ordered as in Table 3 (best-first):**

| # | Report title | `run_dir` (under `training_output/`) | training TOML | results.json key |
|---|---|---|---|---|
| 1 | NN — Mamba (962) | `mamba_p962_long` | `configs/training/sweep/mamba_p962.toml` | `headline/mamba_p962` |
| 2 | NN — LSTM (1082) | `lstm_p1082_long` | `configs/training/sweep/lstm_p1082.toml` | `headline/lstm_p1082` |
| 3 | NN — GRU (1014) | `gru_p1014_long` | `configs/training/sweep/gru_p1014.toml` | `headline/gru_p1014` |
| 4 | NN — Dense (515) | `dense_p515_ga_paper_best` | `configs/training/msr_aller_nn_atan2_best_paper.toml` | `headline/dense_p515` |
| 5 | FTC (joint ref.) | `paper/joint_reference/ftc` | `configs/training/msr_aller_ftc_joint_ref_train.toml` | `joint_reference/ftc` |
| 6 | FNPAG | `fnpag` | `configs/training/msr_aller_fnpag_train.toml` | `classical_baselines/fnpag` |
| 7 | PredGuid (joint ref.) | `paper/joint_reference/pred_guid` | `configs/training/msr_aller_pred_guid_joint_ref_train.toml` | `joint_reference/pred_guid` |
| 8 | Energy controller (joint ref.) | `paper/joint_reference/energy_controller` | `configs/training/msr_aller_energy_controller_joint_ref_train.toml` | `joint_reference/energy_controller` |
| 9 | Equilibrium glide | `equilibrium_glide` | `configs/training/msr_aller_eqglide_train.toml` | `classical_baselines/equilibrium_glide` |
| 10 | Piecewise constant | `piecewise_constant` | `configs/training/msr_aller_piecewise_constant_train.toml` | `classical_baselines/piecewise_constant` |

Window and Transformer NN families are excluded: they have no fully-converged (~15-20k-gen)
run, only 5000-gen sweep cells, so a "best per cell type" report would not be comparable to
the four converged NN cells.

Reference-tracking classicals (FTC, EnergyController, PredGuid) use their **joint-optimized
reference** variant (the paper's "best classical"). FNPAG, EqGlide, and Piecewise have no
joint variant and use their `classical_baselines/` result.

## Two-page layout per scheme

**Page 1 — corridor behaviour (three stacked full-width spaghetti panels):**
- energy vs dynamic pressure (`charts.chart_corridor_pdyn`, with the mission 4-layer corridor zone fills)
- energy vs inclination (`charts.chart_corridor_inclination`, captured envelope)
- energy vs bank angle (`charts.chart_corridor_bank`)

All three reuse the training-report chart functions verbatim: three-way classified spaghetti
(captured-OK blue / constraint-violation orange / failed red), the undispersed-nominal overlay,
and the report's native styling. The appendix deliberately uses the **report's** colour scheme
(not the paper's green=Mamba figlib palette) so all ten scheme cards look uniform — this is what
"like the report" means.

**Page 2 — cost, constraints, statistics:**
- **DV CDF panel** (full width, NEW helper `chart_dv_cdf_overlay`): empirical CDF of total
  correction ΔV (bold) with the three burns overlaid — dv1 (periapsis raise), dv2
  (circularization / apoapsis correction), dv3 (plane change). Captured runs only, shared linear
  x-axis (m/s), percentile-free (the CDF shows the whole distribution).
- **Three constraint panels** (one row): peak heat flux, peak g-load, integrated heat load vs
  time — reuse `charts.chart_heat_flux_time`, `chart_gload_time`, `chart_heat_load_time`
  **verbatim** (each already carries its limit line — 200 kW/m², 4 g, 25 MJ/m² — and three-way
  classification). The functions take a fixed figsize; the "small" size is achieved by scaling
  each SVG to ~1/3 text width in the Typst 3-across row, not by a figsize argument.
- **Stats table** (from `stats.json`, produced by `report.compute_eval_summary`): capture %;
  ΔV min / p50 / p95 / mean / max (and RMS-cost); per-burn dv1 / dv2 / dv3 (p50, p95); apoapsis /
  periapsis / inclination error (p50, p95); peak heat flux / g-load / heat load (p50, max, limit,
  violation %). `compute_eval_summary` returns this payload directly; the collector adds p99 and
  CVaR95 of ΔV (not in the base payload) computed inline.

## Architecture — three artifacts

Follows the established **collector-vs-figure split** (cf. `collect_corridor.py`): a collector
script reads `training_output/` and writes committed SVGs + JSON; the Typst layer reads only
the committed `articles/paper/figures/` + `stats.json`.

### 1. `articles/paper/scripts/collect_appendix.py` (data collector)

```
for scheme in SCHEMES:
    eval_toml, scaffolding = report._resolve_eval_toml(training_toml, run_dir)   # pinned model + co-trained scaffolding
    seeds   = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims=1000)
    batch   = aerocapture_rs.run_batch(eval_toml, overrides=[{n_sims=1, seed=s, **pin} ...],
                                       include_trajectories=True, sim_timeout=5)
    recs        = batch.final_records                     # (1000, 52) -> stats on ALL 1000
    trajs, cls  = subsample(batch.trajectories, 300), classify_trajectories(recs, limits)
    nominal     = undispersed run (all dispersion domains off, pinned model)
    # page 1
    charts.chart_corridor_pdyn/_inclination/_bank(trajs, cls, out/<svg>, corridor_data, nominal_kwargs)
    # page 2
    chart_dv_cdf_overlay(recs, out/dv_cdf.svg)
    charts.chart_heat_flux_time/_gload_time/_heat_load_time(trajs, cls, out/<svg>, limit=...)
    write stats.json  <- compute_eval_summary(recs, n_sims) + p99 + CVaR95 of dv
```

- **Pinning:** run-local `best_model.json` + `best_params.json` scaffolding, so the numbers
  reproduce Table 3 / `results.json` (same reserved 2M pool, n=1000). A per-scheme assertion
  checks the recomputed capture% + mean ΔV against `results.json` within tolerance and warns on
  drift (the `far_tail_eval.py` mislabel trap).
- **Subsampling:** stats use all 1000 sims; spaghetti uses ~300 trajectories × every-3rd point
  to keep each corridor SVG light (the corridor SVGs are the repo-size driver).
- **Corridor zones + nominal:** mission zones from `training_output/mars/corridor_boundaries.npz`;
  undispersed nominal via `reference.nominal_flight_overrides` (per scheme), exactly as `report.py`.
- **Output:** `articles/paper/figures/appendix/<scheme_slug>/{corridor_pdyn,corridor_inclination,corridor_bank,dv_cdf,heat_flux,g_load,heat_load}.svg` + `stats.json`.
- **CLI:** `uv run python articles/paper/scripts/collect_appendix.py [--schemes ...] [--n-sims 1000]`
  (default all 10; per-scheme so a single scheme can be re-collected).

### 2. `chart_dv_cdf_overlay` (one new chart helper)

The only new chart, defined **inside `collect_appendix.py`** (single-use, paper-local): a 4-curve
ECDF (total ΔV bold + dv1/dv2/dv3) over the captured subset, shared linear m/s x-axis, report
seaborn theme (`sns.set_theme` matching `charts.py`). Everything else on both pages reuses
`charts.py` verbatim.

### 3. `articles/paper/appendix.typ` (Typst layout) + `paper.typ` edit

- `appendix.typ` defines `scheme_report(slug, title, subtitle)`: page 1 = the three corridor
  SVGs stacked under a scheme heading; `pagebreak()`; page 2 = the DV CDF, the 3-across constraint
  row, and the stats table read from `figures/appendix/<slug>/stats.json` (Typst `json()`).
  Iterates the 10 schemes in Table-3 order.
- `paper.typ`: after `#bibliography("refs.bib")`, disable heading numbering
  (`#set heading(numbering: none)`) and add a top-level `= Appendix A: per-scheme mission reports`,
  then `#include "appendix.typ"`. The appendix panels are **captionless report cards** — each panel
  is self-titled by its chart function and the per-scheme heading names the card — so no `figure()`
  wrapping and the body's figure counter is untouched.

## Data flow

```
collect_appendix.py  (reads training_output/ + configs/, PyO3 run_batch)
        │  writes (committed)
        ▼
articles/paper/figures/appendix/<slug>/*.svg + stats.json
        │  read by
        ▼
articles/paper/appendix.typ  ──included by──▶  paper.typ  ──typst──▶  paper.pdf (~40pp)
```

## Non-goals / YAGNI

- No Part-1 (training-convergence) panels — this is a mission-performance card, not the full report.
- No dispersion-correlation grid, entry/exit-condition panels, or sensitivity section.
- No Window/Transformer NN cards (no converged run).
- No standalone appendix PDF — it compiles into the single `paper.pdf`.
- No re-run on every `typst compile`; the collector runs once, outputs are committed.

## Cost & risks

- **Compute:** ~1000 sims × 10 schemes, one-time. FNPAG dominates (86 ms/sim → ~86 s); NN/FTC
  are ms/sim. Total ~3–5 min.
- **Repo size:** ~80 SVGs; corridor spaghetti is the driver, mitigated by the 300-trajectory /
  point-subsample. Estimate 15–40 MB added. Acceptable (the `runs/` bundle already commits parquets).
- **Reproducibility:** collector reads `training_output/`, which is not committed. If those dirs
  are later wiped the SVGs cannot be regenerated — accepted (same as `collect_corridor.py`); the
  committed SVGs are the durable artifact.
- **Number drift:** the per-scheme assertion vs `results.json` catches a wrong pin/label before
  the SVGs are trusted.

## Testing / verification

- Collector self-check: recomputed capture% + mean ΔV per scheme within tolerance of `results.json`.
- `typst compile articles/paper/paper.typ` succeeds; visual spot-check of one NN card and one
  classical card (page 1 corridors populated, page 2 table numbers match Table 3 for that scheme).
- Piecewise/EqGlide (sub-100% capture) render the failed-trajectory red class and a non-100%
  capture cell — confirms the three-way classification path.
