# Aerocapture NN Paper — Session Resume

> **Purpose:** let a fresh session pick up the paper work without re-reading the whole history.
> **As of:** 2026-06-12. **Phase:** experiments (mostly done); paused before figures + Typst drafting.

## TL;DR

Writing a comprehensive **Typst** research paper — the follow-up to **Gelly & Vernis, AIAA GNC 2009** ("Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture"). It benchmarks the repo's NN aerocapture guidance against classical + predictor-corrector schemes, and presents the training methodology. The optimizer narrative **flipped from "islands" to "GA"** mid-project; the spine is now **training a robust policy in a non-stationary, dispersed environment**.

## Authoritative docs (read first, in order)

1. **Spec (design + all studies + results + rerun plan):** `docs/superpowers/specs/2026-06-08-aerocapture-nn-article-design.md` ← the single source of truth; §5.1 has the live numbers.
2. **Plan (build order: experiments → aggregate → figures → Typst → smart-commit):** `docs/superpowers/plans/2026-06-08-aerocapture-nn-article.md`.
3. **Source extracts (reuse for prose + citations + voice):** `articles/markdown/00_synthesis_writing_kit.md`, `01_2009_AIAA_neural_guidance.md`, `02..04` (speech papers: CG-LSTM, QPSO, divide-and-conquer, custom loss), `05_authorial_voice_and_style.md`.

## The narrative (current)

17-year arc: 2009 feed-forward NN + GA for aerocapture → 2015-17 recurrent NN for speech (CG-LSTM, QPSO, divide-and-conquer, custom losses — the author's own work) → now stateful NN guidance trained for robustness, vs FTC + predictor-correctors on a bit-validated simulator.

**The optimizer/training-for-robustness quartet (the centerpiece):** a robust policy in a moving, dispersed environment needs (1) **GA** — population optimizer, beats islands/PSO/DE/CMA-ES/QPSO/RL at matched compute; (2) **non-stationary seeds** — adaptive curation (the objective moves each gen); (3) **tail-weighted objective** — `cost_transform = cubed`; (4) **worst-case curation** — `bucket_selection = max`. "islands is best" was a 3× compute artifact; RL and warm-start lose badly (parsimony wins).

## Key results so far (deployed DV mean / p95 / max, m/s; capture 100% unless noted)

- **Headline NN:** GA + cubed + max bucket, `dense_p3998` = **118.4 / 128.6 / 153.8** (`training_output/paper_optbig_ga_bucket_max`).
- **Optimizer @300 (PRE-FIX, log):** GA 115.4/126.2/172.6 best; islands 118.4, PSO 122.6, DE 126.3, QPSO 129.0, CMA-ES worst. GA@150 beat islands@300 (half the compute). **Stale — being rebuilt post-fix.**
- **Architecture sweep (islands, log, pre-fix):** dense best, Mamba 2nd, strongest at low params (~500). Relative ranking likely optimizer/transform-invariant.
- **Output-param (B):** atan2 (2D) 119.6 > scaled_pi 145 > delta 142 (pre-fix); rerun post-fix pending.
- **cost_transform (D, post-fix):** log best mean/p95 (117.1/130.5), cubed/squared best max (162/164). Tradeoff. **Default now cubed.**
- **Bucket representative (C-sub, post-fix, all cubed):** min 123.7 (worst), random 119.1, max 118.4/128.6/**153.8**, middle **117.0/127.6**/164.9. middle best mean/p95, max best worst-case; both > random. **Default now max.**
- **Curation trim:** NULL result (refuted — full range > trimmed).
- **RL / warm-start:** RL ~636 (5× worse); warm-start below plain GA. Negative results.
- **Classical (PRE-FIX, GA, log):** FTC 136, EnergyController 174, PC 191, FNPAG 266, PredGuid 392. **Stale — guidance algos changed, rerunning.**

## Defaults changed in `common.toml` (CRITICAL for consistency)

- `cost_transform = "cubed"` (was log) · `curation_bucket_selection = "max"` (was random) · `algorithm = "ga"` (the winner) · `training_n_sims = 10`, `curation_top_k = 2`, `seed_pool_interval = 3`.
- The user's fixes also changed the **EqGlide/FNPAG/PredGuid guidance algorithms** (`*.rs`) and the training pipeline. **The committed reference file `data/reference_trajectory/msr_aller.dat` is UNCHANGED** (static-ref runs unaffected by the reference overhaul).
- ⇒ **Pre-fix runs are NOT comparable to post-fix runs.** GA@300-log shifted 115.4 → 117.1.

## Experiment status & runners

| Runner | Study | Status |
|---|---|---|
| `run_paper_experiments.sh` | A small-net optimizer + B output-param | pre-fix (stale); B to rerun |
| `run_paper_experiments2.sh` | classical retrain (now GA) | **in progress, post-fix** |
| `run_paper_experiments3.sh` | budget scaling @150/@300 | pre-fix (superseded by 10) |
| `run_paper_experiments4.sh` | QPSO column | pre-fix (superseded by 10) |
| `run_paper_experiments5.sh` | **C — seed strategy (fixed/rotating/adaptive)** | **PENDING (never run)** |
| `run_paper_experiments6.sh` | D — cost_transform | **DONE (post-fix)** |
| `run_paper_experiments7.sh` | C-sub — curation trim | DONE (null result) |
| `run_paper_experiments8.sh` | C-sub — bucket representative | **DONE (post-fix)** |
| `run_paper_experiments9.sh` | **E — joint reference (ftc/energy_controller/pred_guid)** | **PENDING** |
| `run_paper_experiments10.sh` | **A REBUILD — optimizer × budget, post-fix cubed+max** | **PENDING (run FIRST)** |
| `run_paper_experiments11.sh` | **F — training_n_sims sweep (noise-floor + allocation)** | **PENDING** |

## What to run next (priority order)

1. **`./run_paper_experiments10.sh`** — optimizer rebuild (islands/PSO/DE/QPSO/CMA-ES/GA @60/150/300, post-fix cubed+max). The headline "GA wins" table; without it the optimizer comparison mixes pre/post-fix. ~16 runs (GA@300 reuses `paper_optbig_ga_bucket_max`; CMA-ES@300 optional/slow).
2. **Classical post-fix** — `run_paper_experiments2.sh` (GA) finishing; needed for the classical-vs-NN table.
3. **`./run_paper_experiments5.sh`** — Study C seed-strategy (the "why GA wins" test), under cubed+max.
4. **Output-param post-fix** — rerun the `out_scaledpi`/`out_delta` lines (now inherit cubed+max).
5. **`./run_paper_experiments9.sh`** — Study E joint-reference (after the optimizer rebuild). Compare each `<scheme>_joint_ref` to its **post-fix** `<scheme>` baseline.
6. **`./run_paper_experiments11.sh`** — Study F training_n_sims sweep (sample efficiency; companion to Study C).
7. *(optional/expensive)* architecture sweep rerun under GA-cubed-max; otherwise keep the islands sweep as a relative comparison and note it.

## After all experiments (Phase 2-4 of the plan)

- **Aggregate:** write `articles/paper/scripts/aggregate_results.py` (plan Task 5) → `articles/paper/data/results.json`. **Update its `RUNS` map** to the post-fix dirs (`paper_pf_*`, `paper_optbig_ga_bucket_max`, `<scheme>` post-fix, `<scheme>_joint_ref`).
- **Figures** (plan Tasks 6-9): param-vs-DV Pareto, optimizer scaling, cost_transform bar, bucket bar, seed-strategy bar, classical-vs-NN CDF, ablation, pruning/quant.
- **Typst paper** (plan Tasks 10-18): `articles/paper/` (standalone arXiv-style, NOT the training-report template), section by section, voice from `articles/markdown/05`.
- **Finish:** `smart-commit` skill over the branch.

## How to extract numbers (reusable)

Capture = `ifinal==3 & eccentricity<1.0`; DV = `dv_total_m_s` over captured; one `final_eval.parquet` per `training_output/<dir>`.

```python
import glob, json, os, numpy as np, pyarrow.parquet as pq
def stats(d):
    df = pq.read_table(f"training_output/{d}/final_eval.parquet").to_pandas()
    cap = (df["ifinal"]==3) & (df["eccentricity"]<1.0)
    dvc = df.loc[cap,"dv_total_m_s"].to_numpy()
    return dict(n=len(df), cap=round(100*cap.mean(),1),
               mean=round(dvc.mean(),1), p95=round(float(np.percentile(dvc,95)),1), max=round(float(dvc.max()),1))
```

(`cost_transform` rescales the *training* best-val, so it is NOT comparable across transforms — use deployed DV. Within one transform, training best-val from `run_*.jsonl` `validation.rms_cost` is comparable.)

## Process notes

- Flow used: `brainstorming` → spec → `writing-plans` → `executing-plans` (inline). Currently **mid-execution, Phase 1 (experiments), paused for the user's training runs.**
- Division of labor: **the user runs the heavy training**; the assistant sets up configs/runners (TDD for code knobs: `--output-dir`, `--seed-strategy`, `curation_trim_fraction`, `curation_bucket_selection`), analyzes `final_eval.parquet`, and keeps the spec current. Branch: `feature/parameter_sweep`. Never push.
- New experiment knobs added this project (all in `train.py`/`OptimizerConfig`/`SeedCurator`): `--output-dir`, `--seed-strategy`, `[optimizer] curation_trim_fraction` (null result), `[optimizer] curation_bucket_selection` (max adopted).
