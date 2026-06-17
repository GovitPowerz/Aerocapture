# Aerocapture NN Paper — Session Resume

> **Purpose:** let a fresh session pick up the paper work without re-reading the whole history.
> **As of:** 2026-06-14. **Phase:** campaign in progress — 00-04 DONE + analysed (05 running, 06-11 pending), then figures + Typst drafting.

## TL;DR

Writing a comprehensive **Typst** research paper — the follow-up to **Gelly & Vernis, AIAA GNC 2009** ("Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture"). It benchmarks the repo's NN aerocapture guidance against classical + predictor-corrector schemes, and presents the training methodology. The optimizer narrative **flipped from "islands" to "GA"** mid-project; the spine is now **training a robust policy in a non-stationary, dispersed environment**. On 2026-06-12 the whole campaign was **reorganized for reproducibility and reset to one regime**: study-named runners in `experiments/paper/`, cell-named configs in `configs/training/paper/`, all prior training outputs wiped (except footnoted legacy dirs), everything re-runs under the post-fix defaults.

## Authoritative docs (read first, in order)

1. **Spec (design + all studies + prior results + rerun plan):** `docs/superpowers/specs/2026-06-08-aerocapture-nn-article-design.md` ← study definitions; its §5 numbers are now HISTORICAL (pre-wipe) and will be superseded by the campaign re-runs.
2. **Reorg spec (layout + wipe policy + bundle):** `docs/superpowers/specs/2026-06-12-paper-experiments-reorg-design.md`.
3. **Campaign guide:** `experiments/paper/README.md` ← run order, reuse cells, reporting rules, preserved legacy dirs.
4. **Plan v2 (aggregate → figures → Typst → smart-commit; the 06-08 plan is bannered DO NOT EXECUTE):** `docs/superpowers/plans/2026-06-12-aerocapture-nn-article-v2.md`.
5. **Source extracts (prose + citations + voice):** `articles/markdown/00..05`.

## The narrative (current)

17-year arc: 2009 feed-forward NN + GA for aerocapture → 2015-17 recurrent NN for speech (CG-LSTM, QPSO, divide-and-conquer, custom losses — the author's own work) → now stateful NN guidance trained for robustness, vs FTC + predictor-correctors on a bit-validated simulator.

**The centerpiece (REFRAMED 2026-06-14 by Study C): the adaptive-seed methodology is the load-bearing contribution — the quartet is a matched SYSTEM, not four independent wins.** The quartet is (1) **GA**, (2) **non-stationary seeds** (adaptive curation), (3) **tail-weighted objective** (`cost_transform = cubed`), (4) **worst-case curation** (`bucket_selection = max`). Study C (exp04) shows the mechanism: GA does NOT win because it is robust to a moving objective — under FIXED seeds GA is the WORST optimizer (160.3, overfits the 10 repeated scenarios); non-stationary seeds RESCUE it (fixed→rotating→adaptive = 160.3→120.0→118.0, −42 m/s, the campaign's biggest single effect), while CMA-ES is flat (~127). So **GA *needs* the moving environment**; the adaptive-seed strategy converts GA from worst optimizer to best. Iso-compute clincher: GA rotating-vs-fixed is +40 m/s at 1.14× compute, CMA-ES is +0 at exact iso-compute — the effect is the seed strategy, not compute. Plus the **dimensionality axis** (26/515/3998 params; GA separates only at HIGH dim — at 26p all optimizers tie ~170) and the **capability floor**. Framing (Grégory, 2026-06-12): tail optimization is the MISSION-CORRECT objective — the design-case DV sizes the ergols, and propellant mass directly drives mission cost; the mean is operationally near-irrelevant. So cubed/max optimize the actual cost function. SIZING METRIC CORRECTED 2026-06-14: tanks are sized at the FAR tail (3σ≈p99.87 / CVaR99.9 / worst-case), not p95/CVaR95 -- estimated on n=10000 (far_tail_eval.py), where Study D shows cubed WINS the far tail (CVaR99.9 153.0, max 160.1, vs 156-181) by compressing the extreme, VINDICATING the cubed default; at the shallow tail (CVaR95) mild transforms edge it, but that is the wrong sizing depth. The ~1-2 m/s mean cost is a footnote. The rigor rules still apply: tanks are sized off an ESTIMATE of the tail, hence CIs + σ_run, never the noisy sample max. Dual-role note (2026-06-12): the transform also sets the fitness-estimate VARIANCE at n_sims=10 (all optimizers are rank-based, so cubed acts via noisy rank-by-worst-of-10, not landscape smoothness) -- log's pre-fix mean win with a tied tail is the signature of cubed's tail gain being eaten by its convergence cost; Study D's paired cubed_vs_log CVaR95 decides it, with a log-train/tail-select hybrid as the conditional follow-up.

## Live campaign results (post-fix, deployed DV mean / CVaR95 m/s, 100% capture unless noted)

**Classical (01):** FNPAG **124.3 / 144.0** (the surprise — 2026-06 density-fix made it the best classical, near-NN; converged by gen 59 so 371 gens is plenty), pred_guid 167.4 / 227.1, FTC 170.7 / 244.1 (FIXED-ref; joint-ref/07 recovers it to 126.2/142.9 -- see Study E), energy_controller 176.7 / 245.8 (99.6%), eqglide 200.3 / 327.6 (99.5%), piecewise 258.3 / 421.1 (99.8%). Ordering flipped from pre-fix (pred_guid sign-fix + FNPAG density-fix).

**Optimizer × budget (02, dense_p3998):** GA best @150 (118.0) & @300 (120.4) but **GA@60 collapsed (166.3** — n_pop=60 starves the 4000-dim search; "GA dominates at every budget" REFUTED). islands budget-robust (123.7/120.1/122.2). Headline NN = ga_300 (120.4 / 137.6).

**Optimizer × dimensionality (03):** at **26p (FTC) all optimizers tie ~170** (GA worst on tail; CMA-ES-low-dim hypothesis refuted); GA separates only at 515/3998. Optimizer matters for NN, not classical-gain tuning.

**Study B output-param (03, all GA):** atan2 117.4/128.7 > delta 119.9/141.6 > scaledpi 122.2/140.4 — pre-fix 25 m/s gap was a different-optimizer artifact; real edge is ~12 m/s on the TAIL.

**Study C seed-strategy (04) — THE result:** GA fixed→rotating→adaptive 160.3→120.0→118.0 (−42 m/s); CMA-ES flat ~127; islands 145→120; PSO 140→130. GA *needs* non-stationarity (see narrative). Iso-compute clincher above.

**Study C-sub curation (06), far-tail n=10000 — bucket=max VINDICATED:** which seed per cost-CDF bin trains the policy. max (default) dominates the far tail: CVaR99.9 153.0 / max 160.1 vs random 173/190, middle 194/236, min 226/245 (catastrophic — min has best mean 117.8 but blows the extreme: optimize-the-average-blow-the-worst-case). Trim refuted again (trimming extreme deciles hurts the tail). UNIFIED: cubed (transform, Study D) + max (bucket) are the SAME worst-case-leaning mechanism — both compress the design-case extreme tail; quartet legs 3+4 are one idea.

**Study E joint reference (07) -- user hypothesis CONFIRMED:** co-optimizing the constant-bank reference recovers huge DV for table-reading schemes. FTC 170.7->126.2 mean (-44 m/s paired, 100% win, p=3e-165), CVaR95 244->143, CVaR99 310->153; EC 176.7->142.1 (-35); pred_guid 167.3->144.2 (-23). FTC's degradation WAS the reference. The reference-design progression (constant-bank -> PC -> joint) is a clean methodological arc.

**The 3-way RESHAPED -> NN vs joint-FTC vs FNPAG (the new best classical = joint-FTC):** joint-FTC (126.2 mean / CVaR95 142.9) now MATCHES FNPAG (124.3 / 144.0) on accuracy across the whole distribution -- and joint-FTC is ANALYTIC/FAST (~50x faster than FNPAG's predictor). Far-tail n=10000 (sizing depth): NN CVaR99.9 152.9 / max 160.1 < joint-FTC 164.0/170.5 ~= FNPAG 165.0/175.5 (fixed-FTC 353/411, superseded). So **NN beats the best classical (joint-FTC) by ~6 m/s mean (paired -5.8, 76% win, p=1e-69) and ~11 m/s at the design tail, at FTC's compute class.** Headline: NN > best-classical at the fastest compute; joint-FTC = FNPAG accuracy at 50x less compute (the reference methodology). compute-benchmark still PENDING (needs an idle box for clean single-core timing).

## Historical results (PRE-WIPE — directional guidance only, all superseded by the campaign)

All prior `training_output` study dirs were deleted 2026-06-12 (reorg). Old numbers live in spec §5 + git history. Directionally: GA won every budget on the big net (115.4/126.2 @300 pre-fix-log); atan2 ≫ scaled_pi/delta (~20+ m/s); classical FTC ~136 ≫ other classicals; RL ~5× worse; min-bucket worst, middle/max > random; trim refuted; log best mean / cubed best max (tradeoff). Effect sizes of 1-3 m/s sit at ~0.5-1.5 σ_run — hence the seed-repeats study.

**Preserved legacy dirs (PRE-FIX regime, footnote when quoted):** RL (`neural_network_rl` 636/973/1185, `neural_network_gru_ppo` 513/829, `neural_network_atan2_{ppo,rl,best}`, `neural_network_rl_explore`), warm-start/joint (`paper_opt_warmstart` 132.4, `{best_,}neural_network_joint` 125.3/125.7, `neural_gru_joint`, `neural_network_warm`), quantization/pruning (`neural_network_atan2` 119.0/132.0/165.2 base, `_qat8` 125.1, `_qat4` 128.7, all `*pruned*` + bases). Regime-insensitive conclusions; not re-run.

## Defaults in `common.toml` (the ONE campaign regime)

- `cost_transform = "cubed"` · `curation_bucket_selection = "max"` · `algorithm = "ga"` · `seed_strategy = "adaptive"` · `training_n_sims = 10` · `curation_top_k = 1` · `seed_pool_interval = 2` · `validation_n_sims = 1000`.
- Every campaign cell inherits these; per-cell deltas live in the cell config name (`dense_p3998_ga_transform_log.toml`, `dense_p3998_ga_bucket_min.toml`, ...).

## The campaign (`experiments/paper/`, run from repo root, in order)

| Script | Study | Cells / reuse |
|---|---|---|
| `00_prereqs.sh` | corridor + mission ref + PC classical row | canonical `piecewise_constant`, `mars/` |
| `01_classical_baselines.sh` | classical GA @2000×300 | canonical `ftc`, `equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag` |
| `02_optimizer_budget.sh` | **Study A** (6 opt × 3 budgets, dense_p3998) | `paper/optimizer_budget/<opt>_<budget>`; **`ga_300` = headline**, reused by 04/05/06/08/11 |
| `03_optimizer_dimensionality.sh` | **opt × width** (26p FTC / 515p) + **Study B** | `paper/optimizer_dimensionality/*`, `paper/output_param/*`; FTC GA cell = `training_output/ftc` |
| `04_seed_strategy.sh` | **Study C** fixed/rotating | `paper/seed_strategy/*`; adaptive column = 02's @150 row |
| `05_cost_transform.sh` | **Study D** | `paper/cost_transform/{linear,sqrt,log,squared}`; cubed = `ga_300` |
| `06_curation_shaping.sh` | **C-sub** bucket + trim | `paper/curation_shaping/*`; max = `ga_300` |
| `07_joint_reference.sh` | **Study E** | `paper/joint_reference/*`; baselines from 01, same budgets |
| `08_training_n_sims.sh` | **Study F** (rotating noise floor + adaptive allocation) | `paper/training_n_sims/*`; adaptive_10 = `ga_300` |
| `09_capability_floor.sh` | sub-500 collapse | canonical `sweep_dense_p{102,201,298,416}` |
| `10_architecture_sweep.sh` | 6-family Pareto (GA post-fix) | canonical `sweep_<arch>_p<N>` via param_sweep |
| `11_seed_repeats.sh` | σ_run repeats | `paper/seed_repeats/*` (s1 from 01/02/03) |
| `12_collect_results.sh` | committed bundle | `articles/paper/data/runs/<study>/<cell>/` |

All runners: skip-if-done per cell, `--sim-timeout 5`. Never run two cells of the same config TOML concurrently; never regenerate `training_output/mars/` while a ref-tracking scheme trains.

**Reporting rules (locked in by the 2026-06-12 methodology review):** quote p99 + CVaR95 (not sample max); pair all cross-cell tables on the shared 1000-seed final-eval pool (paired bootstrap + Wilcoxon in `aggregate_results.py`); report ACTUAL total sims per run (from JSONL: validations fire on ~58-80% of gens × 1000 sims; curation ≈1000 sims/event every ≤2 gens) next to any "compute-matched" claim; σ_run from 11 calibrates every N=1 comparison; re-quote the final headline model once on a FRESH pool (offset 8M) for the abstract number.

## After all experiments (Phase 2-4 of the plan)

- **Aggregate:** `articles/paper/scripts/aggregate_results.py` (plan v2 Tasks 1-2: `paper_stats.py` helpers + bundle-driven aggregator) reads `articles/paper/data/runs/` → `articles/paper/data/results.json`, with paired stats, actual-sims, σ_run, and the fresh-pool headline re-quote.
- **Figures** (plan v2 Task 3): 10 figures via the shared figlib — Pareto incl. capability floor, optimizer budget+dimensionality, seed-strategy (thesis figure), cost_transform, curation, output-param, training_n_sims, classical-vs-NN CDF, joint-reference, pruning/quant (legacy footnote). Ablation + input report on ga_300 (Task 4); fresh-pool re-quote (Task 5).
- **Typst paper** (plan v2 Tasks 6-14): `articles/paper/` (standalone arXiv-style), GA-quartet narrative, statistical-protocol subsection, ~10 results subsections, voice from `articles/markdown/05`.
- **Finish:** `smart-commit` skill over the branch.

## How to extract numbers (reusable)

Capture = `ifinal==3 & eccentricity<1.0`; DV = `dv_total_m_s` over captured; one `final_eval.parquet` per run dir (under `training_output/paper/<study>/<cell>/`, canonical scheme dirs, or the committed bundle `articles/paper/data/runs/`).

```python
import numpy as np, pyarrow.parquet as pq
def stats(path):
    df = pq.read_table(f"{path}/final_eval.parquet").to_pandas()
    cap = (df["ifinal"]==3) & (df["eccentricity"]<1.0)
    dvc = df.loc[cap,"dv_total_m_s"].to_numpy()
    return dict(n=len(df), cap=round(100*cap.mean(),1), mean=round(dvc.mean(),1),
                p95=round(float(np.percentile(dvc,95)),1), p99=round(float(np.percentile(dvc,99)),1),
                cvar95=round(float(np.sort(dvc)[-max(1,len(dvc)//20):].mean()),1))
```

(`cost_transform` rescales the *training* best-val, so it is NOT comparable across transforms — use deployed DV. Within one transform, training best-val from `run_*.jsonl` `validation.rms_cost` is comparable.)

## Process notes

- Flow used: `brainstorming` → spec → `writing-plans` → `executing-plans` (inline). Currently: **campaign reorganized + outputs wiped; user runs `experiments/paper/00..12` in order; assistant aggregates + drafts.**
- Division of labor: **the user runs the heavy training**; the assistant sets up configs/runners, analyzes results, keeps the spec current. Branch: `feature/parameter_sweep`. Never push.
- New experiment knobs added this project (all in `train.py`/`OptimizerConfig`/`SeedCurator`): `--output-dir`, `--seed-strategy`, `--training-n-sims`, `--seed` (repeats), `[optimizer] curation_trim_fraction`, `[optimizer] curation_bucket_selection`.
