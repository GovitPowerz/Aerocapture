# Paper Experiments Reorganization — Design

**Date:** 2026-06-12
**Status:** Approved (user, 2026-06-12)
**Goal:** anyone with repository access reproduces the paper data — training runs from the
study-named runners, tables/figures from a committed compact results bundle.

## Decisions (user-approved)

1. **Full re-run, one regime.** All paper studies re-run under the post-fix codebase and the
   HEAD defaults (`cost_transform=cubed`, `bucket=max`, `seed_pool_interval=2`,
   `curation_top_k=1`). The DONE-but-old-regime studies D (cost_transform) and C-sub (bucket)
   are re-run too. A step-0 prereq script (piecewise corridor + `make_reference`) makes the
   chain reproducible from a bare clone.
2. **Layout:** runners live in `experiments/paper/NN_<study>.sh`, numbered by dependency
   order, invoked from the repo root. Study identity belongs to the RUNNERS; configs are
   CELLS (net x optimizer / knob value) reused across studies and keep cell names.
3. **Committed data:** per-run `{best_model.json, best_params.json, final_eval.parquet,
   final_selection.json, run.jsonl.gz}` under `articles/paper/data/runs/<study>/<cell>/`
   (~180 MB campaign total). Raw `training_output/` (8.3 GB checkpoints/PDFs) stays out of git.
4. **Wipe casualties:** the 24-config architecture sweep is re-run under GA post-fix
   (upgrades the Pareto figure to the campaign regime). RL / warm-start / QAT / pruning
   legacy dirs are PRESERVED and footnoted as pre-fix regime (their conclusions — RL 5x
   worse, warm-start below plain GA, QAT/pruning deployability — are regime-insensitive).

## Scripts (`experiments/paper/`)

| # | script | study | cells (output under `training_output/paper/<study>/`) |
|---|--------|-------|--------|
| 00 | `00_prereqs.sh` | corridor + mission reference | canonical `training_output/{piecewise_constant,mars}` (skip-if-exists) |
| 01 | `01_classical_baselines.sh` | classical GA @2000x300 | canonical `training_output/{ftc,equilibrium_glide,energy_controller,pred_guid,fnpag}` |
| 02 | `02_optimizer_budget.sh` | Study A | `optimizer_budget/{islands,pso,de,qpso,cmaes,ga}_{60,150,300}` |
| 03 | `03_optimizer_dimensionality.sh` | optimizer x width + Study B | `optimizer_dimensionality/{ftc_<opt>, dense_p515_<opt>}`, `output_param/{scaledpi,delta}` |
| 04 | `04_seed_strategy.sh` | Study C | `seed_strategy/<opt>_{fixed,rotating}` (adaptive col = 02's @150 row) |
| 05 | `05_cost_transform.sh` | Study D | `cost_transform/{linear,sqrt,log,squared}` (cubed = 02 `ga_300`) |
| 06 | `06_curation_shaping.sh` | C-sub bucket + trim | `curation_shaping/{bucket_min,bucket_middle,bucket_random,trim_10,trim_20}` (max = 02 `ga_300`) |
| 07 | `07_joint_reference.sh` | Study E | `joint_reference/{ftc,energy_controller,pred_guid}` (budgets = 01) |
| 08 | `08_training_n_sims.sh` | Study F | `training_n_sims/{rotating_<n>, adaptive_<n>}` (adaptive_10 = 02 `ga_300`) |
| 09 | `09_capability_floor.sh` | sub-500 collapse | canonical `sweep_dense_p{102,201,298,416}` (param_sweep floor manifest) |
| 10 | `10_architecture_sweep.sh` | arch Pareto | canonical `sweep_<arch>_p<N>` via `param_sweep --train --from-scratch` |
| 11 | `11_seed_repeats.sh` | sigma_run | `seed_repeats/{ga_300,islands_300,ftc_{ga,cmaes,islands},small_{ga,cmaes,islands}}_s{2,3}` |
| 12 | `12_collect_results.sh` | bundle | `articles/paper/data/runs/` |

All runners: skip-if-`final_eval.parquet` guards, `--sim-timeout 5`, `set -euo pipefail`.
`README.md` in the same dir maps studies -> scripts -> configs -> output dirs, states the run
order, the regime notes for preserved legacy dirs, and the reuse cells.

## Configs (`configs/training/paper/`, renamed via git mv)

- `opt_<o>.toml` -> `dense_p515_<o>.toml`; `optbig_<o>.toml` -> `dense_p3998_<o>.toml`
  (o in ga/pso/de/cmaes/qpso/islands); `opt_warmstart.toml` -> `dense_p515_warmstart.toml`
  (kept as provenance for the preserved legacy run; not re-run).
- `optbig_ga_<t>.toml` -> `dense_p3998_ga_transform_<t>.toml` (linear/sqrt/log/squared).
- `optbig_ga_bucket_{min,middle}.toml` -> `dense_p3998_ga_bucket_*.toml`; NEW
  `dense_p3998_ga_bucket_random.toml` (random is no longer the default).
- `optbig_ga_trim{10,20}.toml` -> `dense_p3998_ga_trim*.toml`.
- `out_*.toml` -> `outparam_{scaledpi,delta}.toml`.
- DELETED: `opt_ga_nsim.toml` (superseded by `--training-n-sims`), `opt_rl.toml` (RL rows
  come from the canonical RL configs), `optbig_ga_cubed.toml` + `optbig_ga_bucket_max.toml`
  (identical to the base GA config now that cubed+max are defaults).
- Each config's `[data] neural_network` fallback deploy path moves to
  `training_output/paper/cells/<config-stem>/best_model.json` (runners always pass
  `--output-dir`; report.py pins the eval to the run-local model since e0ed7df).
- `configs/training/sweep/manifest_floor.json` anchors re-pointed at
  `paper/optimizer_dimensionality/dense_p515_ga` and `paper/optimizer_budget/ga_300`.

## training_output wipe (one-time, executed at implementation)

DELETE: all `paper_*` study dirs, `sweep_*`, loose `dense_p515`/`dense_p3998`, classical
(`ftc`, `ftc_islands`, `fnpag`, `fnpag_islands`, `equilibrium_glide`, `energy_controller`,
`pred_guid`, `msr_aller_*`, `neural_network_islands`, `ftc_joint_ref_prefix_INVALID`),
superseded pre-sweep arch one-offs (`neural_network_{gru,lstm,mamba,transformer,window}_*`
except RL/pruned), `comparison_results.json`, debug PNGs.

PRESERVE: `mars/` (committed reference + corridor; regenerable via 00), and the legacy
footnoted dirs: `neural_network_rl`, `neural_network_gru_ppo`, `neural_network_atan2_ppo`,
`neural_network_atan2_rl`, `neural_network_rl_explore`, `neural_network_atan2_best`,
`neural_network_atan2` (+ `_qat4`, `_qat8`), all `*pruned*` dirs + their bases
(`neural_network_scaledpi_pso`, `neural_network_delta_pso`), the joint/warm-start dirs
(`best_neural_network_joint`, `neural_network_joint`, `neural_gru_joint`,
`neural_network_warm`, `paper_opt_warmstart`).

The suspended Tuesday `run_paper_experiments2.sh` bash (old islands version) and its fnpag
child are killed before the old scripts are deleted.

## Committed bundle

`.gitignore` gains a negation for `articles/paper/data/**` (the global `*.json` rule would
silently drop the bundle). `12_collect_results.sh` wraps
`articles/paper/scripts/collect_runs.py`: walks `training_output/paper/**` plus the
canonical classical + `sweep_*` dirs, copies the 4 quotable artifacts and gzips the newest
`run_*.jsonl` per run into `articles/paper/data/runs/<study>/<cell>/`.

## Docs

`paper_resume.md` runner table and run order rewritten to the new scripts; CLAUDE.md /
README.md references to `run_paper_experiments{4,9}.sh` updated; all 15 old
`run_paper_experiments*.sh` deleted.

## Out of scope

`aggregate_results.py` / figures (plan Tasks 5-9) — they will read the committed bundle, but
their implementation is the next phase of the paper plan, unchanged here.
