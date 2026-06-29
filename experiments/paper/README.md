# Paper experiment campaign

Reproduces every training run behind the paper (Gelly, aerocapture neural-guidance
follow-up to AIAA GNC 2009). All scripts run **from the repo root**, are
**idempotent** (skip-if-`final_eval.parquet` per cell; delete a cell's dir to force a
rerun), and pass `--sim-timeout 5` (never fires for healthy sims; caps the known
NaN-hang failure mode). Prereqs: `./build.sh` (Rust binary + PyO3), `uv sync`.

## Run order

```
./experiments/paper/00_prereqs.sh                   corridor + mission reference (skip: both are committed)
./experiments/paper/01_classical_baselines.sh       classical GA baselines -> canonical training_output/<scheme>
./experiments/paper/02_optimizer_budget.sh          Study A   (18 cells; ga_300 = Study A baseline cell, reused by 04/05/06/08; NOT the deployed headline -- that is Mamba_962 from 10c)
./experiments/paper/03_optimizer_dimensionality.sh  opt x width + Study B (needs 01 for the FTC GA cell)
./experiments/paper/04_seed_strategy.sh             Study C   (adaptive column = 02's @150 row)
./experiments/paper/05_cost_transform.sh            Study D   (cubed cell = 02's ga_300)
./experiments/paper/06_curation_shaping.sh          Study C-sub bucket + trim (max cell = 02's ga_300)
./experiments/paper/07_joint_reference.sh           Study E   (needs 01; same budgets)
./experiments/paper/08_training_n_sims.sh           Study F   (adaptive n=10 anchor = 02's ga_300)
./experiments/paper/09_capability_floor.sh          sub-500 dense collapse sweep
./experiments/paper/10_architecture_sweep.sh        6-family Pareto re-run (GA, post-fix regime)
./experiments/paper/10b_arch_long_challengers.sh    extend best recurrent cells to headline depth (needs 10)
./experiments/paper/10c_tail_sigma_repeats.sh       sigma_run on the sizing tail: mamba_962 vs dense_515 s2/s3 (needs 10b)
./experiments/paper/11_seed_repeats.sh              sigma_run repeats (needs 01/02/03) -- OPTIONAL: 10c already measured run-variance on the tail; only run for a stated optimizer ranking
./experiments/paper/12_collect_results.sh           -> articles/paper/data/runs/ (committed bundle)
./experiments/paper/13_robustness_retrain.sh        OPTIONAL, off-campaign: retrain FTC-joint + Mamba_962 ON the high regime, eval on the 9M stress pool (tests the paper's "widen the NN training regime" future-work line; directional budget by default, scale NGEN_MAMBA for a conclusive run)
./experiments/paper/14_objective_centering.sh        OPTIONAL, off-campaign: objective-centering lever attribution under the high regime (dense_515; Phase 2 Mamba via RUN_MAMBA=1). Tests that worst-case shaping is regime-matched. Spec 2026-06-29.
```

02 is the long pole (18 x ~1-2 h); 01/03's FTC cells are fast (~ms/sim), fnpag is
~50x slower than FTC. Never run two cells **of the same config TOML** concurrently
in one checkout (`[data] neural_network` deploy clobbering; the report pins the
eval to the run-local model, but interim deploys still race), and never regenerate
`training_output/mars/` while a ref-tracking scheme (ftc / energy_controller /
pred_guid) is training.

## Where results land

- Study cells: `training_output/paper/<study>/<cell>/` (study names match the scripts).
- Classical baselines: canonical `training_output/<scheme>/` (compare_guidance,
  Study E and the FTC GA cell expect those names).
- Sweeps: canonical `training_output/sweep_<arch>_p<N>/` (param_sweep manifests:
  `configs/training/sweep/manifest.json` + `manifest_floor.json`).
- Committed bundle (per run: `best_model.json`, `best_params.json`,
  `final_eval.parquet`, `final_selection.json`, `fresh_pool_requote.json`,
  `run.jsonl.gz`): `articles/paper/data/runs/<study>/<cell>/` via
  `12_collect_results.sh`; the preserved legacy dirs land under
  `runs/legacy/<dir>/`. Tables/figures reproduce from the bundle WITHOUT
  re-training. Discipline: any retro `final_select` re-selection must be
  followed by `report.py` on that dir (regenerates `final_eval.parquet`)
  before re-collecting -- the collector skips and warns on dirs whose
  `best_model.json` is newer than their parquet.

## Configs

`configs/training/paper/` holds CELL configs named by what they are
(`dense_p3998_ga.toml`, `dense_p515_cmaes.toml`,
`dense_p3998_ga_transform_log.toml`, `outparam_scaledpi.toml`, ...); study identity
lives in these runners, because cells are reused across studies (e.g.
`dense_p3998_ga` serves Studies A, C, D, C-sub, F and the repeats). All inherit
`configs/training/common.toml` (post-fix defaults: `cost_transform = cubed`,
`curation_bucket_selection = max`, `seed_pool_interval = 2`, `curation_top_k = 1`,
`training_n_sims = 10`, `validation_n_sims = 1000`).

## Reporting rules (from the 2026-06-12 methodology review)

- Sizing metrics: propellant (ergols) tanks are sized for the FAR-tail design
  case (3σ ≈ p99.87 / CVaR99.9 / worst-case), NOT p95. Quote **p99, CVaR99,
  p99.9, CVaR99.9** with bootstrap CIs, estimated on a LARGE pool — the deployed
  cells get an n=10000 re-eval (`far_tail_eval.py`, full reserved 2M pool,
  training-disjoint) so CVaR99.9 (worst 10) / p99.9 (10 samples) are stable;
  at n=1000 they are ~1-sample and unusable. CVaR99.9 is the headline sizing
  metric; the sample max (≈p99.99 at n=10000) is a descriptive bound. (Rationale:
  the design-case DV sizes the ergols and hence mission cost; the tail IS the
  objective, but it is an ESTIMATE, so it gets CIs and a large pool, not a single
  noisy max at n=1000.) NB the cost_transform that minimizes the tail DEPENDS on
  the sizing percentile (Study D): mild transforms win the shallow tail, cubed
  wins the far tail — match the training tail-weight to the sizing percentile.
- All cross-cell tables are **paired** on the shared 1000-seed final-eval pool
  (offset 2M; capture = `ifinal==3 & ecc<1.0`, DV over both-captured seeds).
- "Compute-matched" claims must report **actual sims** per run (from the JSONL:
  training = n_pop x n_sims x n_gen; validations = records with a `validation`
  key x 1000; curations = distinct `last_curation_gen` x top_k x 1000).
- The final headline model (**Mamba_962**, `training_output/mamba_p962_long/`) gets
  one **fresh-pool** MC re-quote (seed offset 8M) for the abstract number (CVaR95 115.2).
- sigma_run on the SIZING TAIL comes from 10c (dense/mamba/lstm s2/s3 at the headline
  allocation); 11_seed_repeats (optimizer-cell mean sigma_run) is OBSOLETE/skipped.

## Legacy dirs (preserved, PRE-FIX regime -- footnote when quoted)

RL: `neural_network_rl`, `neural_network_gru_ppo`, `neural_network_atan2_{ppo,rl,best}`,
`neural_network_rl_explore`. Warm-start/joint: `paper_opt_warmstart`,
`{best_,}neural_network_joint`, `neural_gru_joint`, `neural_network_warm`.
Quantization/pruning: `neural_network_atan2{,_qat4,_qat8}`, all `*pruned*` dirs +
bases (`neural_network_{scaledpi,delta}_pso`). Their conclusions (RL ~5x worse,
warm-start below plain GA, QAT/pruning deployability) are regime-insensitive;
they were NOT re-run under the post-fix defaults.
