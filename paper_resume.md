# Aerocapture NN Paper — Session Resume

> **Purpose:** let a fresh session pick up the paper work without re-reading the whole history.
> **As of:** 2026-06-12 (post-reorg). **Phase:** clean-slate campaign ready; all training re-runs pending, then figures + Typst drafting.

## TL;DR

Writing a comprehensive **Typst** research paper — the follow-up to **Gelly & Vernis, AIAA GNC 2009** ("Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture"). It benchmarks the repo's NN aerocapture guidance against classical + predictor-corrector schemes, and presents the training methodology. The optimizer narrative **flipped from "islands" to "GA"** mid-project; the spine is now **training a robust policy in a non-stationary, dispersed environment**. On 2026-06-12 the whole campaign was **reorganized for reproducibility and reset to one regime**: study-named runners in `experiments/paper/`, cell-named configs in `configs/training/paper/`, all prior training outputs wiped (except footnoted legacy dirs), everything re-runs under the post-fix defaults.

## Authoritative docs (read first, in order)

1. **Spec (design + all studies + prior results + rerun plan):** `docs/superpowers/specs/2026-06-08-aerocapture-nn-article-design.md` ← study definitions; its §5 numbers are now HISTORICAL (pre-wipe) and will be superseded by the campaign re-runs.
2. **Reorg spec (layout + wipe policy + bundle):** `docs/superpowers/specs/2026-06-12-paper-experiments-reorg-design.md`.
3. **Campaign guide:** `experiments/paper/README.md` ← run order, reuse cells, reporting rules, preserved legacy dirs.
4. **Plan (build order: experiments → aggregate → figures → Typst → smart-commit):** `docs/superpowers/plans/2026-06-08-aerocapture-nn-article.md`.
5. **Source extracts (prose + citations + voice):** `articles/markdown/00..05`.

## The narrative (current)

17-year arc: 2009 feed-forward NN + GA for aerocapture → 2015-17 recurrent NN for speech (CG-LSTM, QPSO, divide-and-conquer, custom losses — the author's own work) → now stateful NN guidance trained for robustness, vs FTC + predictor-correctors on a bit-validated simulator.

**The optimizer/training-for-robustness quartet (the centerpiece):** a robust policy in a moving, dispersed environment needs (1) **GA** — population optimizer; (2) **non-stationary seeds** — adaptive curation; (3) **tail-weighted objective** — `cost_transform = cubed`; (4) **worst-case curation** — `bucket_selection = max`. Plus the new **dimensionality axis** (optimizer ranking vs chromosome width: 26 / 515 / 3998 params) and the **capability floor** (smallest dense net that still guides). Caveats already known from the pre-wipe data: cubed/max trade ~1.5-2 m/s mean for tail gains that only show in the (noisy) sample max — the re-runs must re-justify them on p99/CVaR95 + σ_run error bars, or the paper presents them as worst-case-prior engineering choices.

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

- **Aggregate:** `articles/paper/scripts/aggregate_results.py` (plan Task 5) reads the committed bundle `articles/paper/data/runs/` → `articles/paper/data/results.json`, with paired-difference stats.
- **Figures** (plan Tasks 6-9): param-vs-DV Pareto (incl. capability floor), optimizer scaling + dimensionality, cost_transform/bucket/seed-strategy bars, classical-vs-NN CDF, ablation (**on the new headline model**, not sweep_dense_p515), pruning/quant (legacy, footnoted).
- **Typst paper** (plan Tasks 10-18): `articles/paper/` (standalone arXiv-style), voice from `articles/markdown/05`.
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
