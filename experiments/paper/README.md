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
./experiments/paper/14_objective_centering.sh        OPTIONAL, off-campaign: objective-centering lever attribution under the high regime (dense_515; Phase 2 Mamba via RUN_MAMBA=1). Tests that worst-case shaping is regime-matched. Spec 2026-06-29. Sizing-depth requote of the three centered-Mamba seeds + joint-FTC references, both regimes: `make -C articles/paper mc-centered-depth` (#156).
./experiments/paper/18_rl_baseline.sh               Section 5 RL baseline (issue #101): {dense_p515, gru_p1014} x {PPO scratch, PPO warm-started from the per-scenario champion}, protocol-matched to ou_marginal/ft_*, per_draw regime; then report.py on the two ou_marginal champions and 12 (bundle keys rl/*, ou_marginal/*)
./experiments/paper/18_rl_baseline.sh hl            v4 Section 5 RL baseline (issue #175): the same four cells as hl_<cell>_ppo_*, protocol-matched to the #173 scratch cells ou_marginal/hl_dense_p515 / hl_gru_p1014 (the fine-tunes left the main body); ~4 h, resumable; then confirmatory_marginal.py on confirmatory_cells_v4.txt and 12
./experiments/paper/18_rl_baseline.sh hl_repeats    v4 PPO-scratch seed repeats: hl_{dense_p515,gru_p1014}_ppo_scratch_s2 / _s3 ([rl] torch_seed 2 / 3, otherwise the hl protocol; the s1 cells ran unseeded, so a single scratch run is one draw: 180 vs 444 m/s mean for dense on identical code); ~3 h 15, resumable; then confirmatory_marginal.py on confirmatory_cells_v4.txt and 12
./experiments/ou_marginal/classical_campaign.sh     per_draw regime (#172): the nine classical cells of 01 + 07 retuned at the same GA allocation (FNPAG 300 gens) -> training_output/ou_marginal/classical/<cell>, then report.py per cell; resumable, ~18 h under `caffeinate -i` (FNPAG ~11.5 h); sanity table: `quote_marginal.py --manifest experiments/ou_marginal/classical_cells.txt`
./experiments/ou_marginal/classical_campaign.sh classical_ungated   per_draw regime (#188): piecewise_constant + equilibrium_glide rerun as above with `max_violation_rate = 1.0` (the ungated selection of their shared-path parents) -> training_output/ou_marginal/classical_ungated/<cell>; ~45 min; scored by the same manifest
./experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_headline.txt   per_draw regime (#173): the five NN families trained from scratch at the headline allocation (GA 512 x 2, 20000 gens; mamba_962 and dense_515 at three seeds) + the dense_515 fine-tune seeds 2 and 3 -> training_output/ou_marginal/<job>, then report.py per job; resumable, ~68 h under `caffeinate -i`
./experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_heat_load.txt   per_draw regime (#192): the two #173 seed-1 champions fine-tuned 2000 gens at 60 x 10 under max_heat_load 25000 / 27500 / 30000 -> training_output/ou_marginal/hs_<cell>_q<NN>, then report.py per leg; ~2.5 h under `caffeinate -i`; slope table: `heat_load_slope.py` (every leg at its own ceiling and re-flown under the v4 limit, on quote_marginal.py's paired pool)
./experiments/paper/15_state_controls.sh v4        per_draw regime (#176): mechanism and deployment controls on the v4 champion ou_marginal/hl_mamba_p962, quick steps first: reset-state eval at 10^6 (confirmatory_marginal.py --extra-override, label ou_marginal/v4_reset_state), input ablation, fresh-pool re-quote (fresh_pool_requote.py --noise-seeding per_draw), off-nominal stress at n = 10000 (stress_depth_eval.py -> data/stress_depth.json); then campaign.sh on jobs_controls.txt (ctrl_window_p970 + ctrl_mamba_p962_nodv from scratch at GA 512 x 2, 20000 gens, ~7.5 h each) and both at 10^6; pre-registered single-seed rule in the script header; resumable, ~16 h under `caffeinate -i`. `15_state_controls.sh shared` is the arxiv-v3 shared-path study.
```

02 is the long pole (18 x ~1-2 h); 01/03's FTC cells are fast (~ms/sim), fnpag is
~50x slower than FTC. Never run two cells **of the same config TOML** concurrently
in one checkout without distinct `--output-dir`s (the plain invocation's output dir is
the parent of the TOML's `[data] neural_network`; until #170 every `--output-dir` run
also wrote that path, so siblings overwrote each other's deployed model), and never regenerate
`training_output/mars/` while a ref-tracking scheme (ftc / energy_controller /
pred_guid) is training. Only the canonical piecewise_constant run (00) and a
`reference_only` run write `training_output/mars/`; any other `--output-dir`
piecewise_constant run keeps its corridor and reference in its own dir.

## Where results land

- Study cells: `training_output/paper/<study>/<cell>/` (study names match the scripts).
- Classical baselines: canonical `training_output/<scheme>/` (compare_guidance,
  Study E and the FTC GA cell expect those names). Their per-scenario retunes:
  `training_output/ou_marginal/classical/<cell>/`.
- Sweeps: canonical `training_output/sweep_<arch>_p<N>/` (param_sweep manifests:
  `configs/training/sweep/manifest.json` + `manifest_floor.json`).
- Committed bundle (per run: `best_model.json`, `best_params.json`,
  `final_eval.parquet`, `final_selection.json`, `fresh_pool_requote.json`,
  `run.jsonl.gz`): `articles/paper/data/runs/<study>/<cell>/` via
  `12_collect_results.sh`; the preserved legacy dirs land under
  `runs/legacy/<dir>/`. Tables/figures reproduce from the bundle WITHOUT
  re-training: `make -C articles/paper paper` (fetch-logs -> check-logs -> results.json ->
  confirmatory_marginal.json -> quote_marginal.json -> heat_load_slope.json ->
  figures -> provenance -> pdf; `make -C articles/paper check` verifies the
  checksums, the three per-scenario extracts and the figures; the opt-in `mc-*` targets re-fly
  cells). After a re-collect, `make
  -C articles/paper sums` refreshes `data/SHA256SUMS`.
  Discipline: any retro `final_select` re-selection must be
  followed by `report.py` on that dir (regenerates `final_eval.parquet`)
  before re-collecting -- the collector skips and warns on dirs whose
  `best_model.json` is newer than their parquet.
- Deployed-model audit: `uv run python experiments/paper/audit_deployed_models.py`
  rebuilds every run's winner from its final checkpoint and byte-compares it with the
  deployed `best_model.json`, flags byte-identical models across dirs and compares each
  with its bundle copy; `--repair` rewrites a mismatched model (old file kept as
  `best_model.json.pre-repair`). Run it before collecting or quoting a model.
- Appendix cards: `articles/paper/figures/appendix/<scheme>/` (7 report-style
  SVGs + `stats.json` per scheme, built by `articles/paper/scripts/collect_appendix.py`
  from training_output). Committed -- `appendix.typ` reads each `stats.json` at
  compile time, so the paper must build from a clean checkout (`.gitignore`
  exempts them from the global `*.json` rule).
- Noise regime: every committed cell under `runs/` was trained and evaluated
  under the shared noise path (`noise_seeding = "legacy"`), the simulator
  default until ADR-0006 (2026-09-16). The eval scripts (`articles/paper/scripts/*`,
  `param_sweep --eval`, `quantize`, the probe drivers) pin `legacy` explicitly through
  `deploy_overrides.LEGACY_NOISE_REGIME` so the bundle reproduces; the training configs inherit the new
  `per_draw` default, so re-running a campaign script trains under
  per-scenario noise -- a new experiment, not a reproduction. The
  per-scenario cells live under `experiments/ou_marginal/` and
  `configs/training/ou_marginal/`.

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
  `aggregate_results.py` folds the 10c triplets into `results.json` under
  `sigma_run.tail_groups` (per-seed CVaR99.9/max + mean/range/std) -- the durable
  source of the paper's 3-seed-mean tail numbers (124.5/129.2/139.2).

## Legacy dirs (preserved, PRE-FIX regime -- footnote when quoted)

RL: `neural_network_rl`, `neural_network_gru_ppo`, `neural_network_atan2_{ppo,rl,best}`,
`neural_network_rl_explore`. Warm-start/joint: `paper_opt_warmstart`,
`{best_,}neural_network_joint`, `neural_gru_joint`, `neural_network_warm`.
Quantization/pruning: `neural_network_atan2{,_qat4,_qat8}`, all `*pruned*` dirs +
bases (`neural_network_{scaledpi,delta}_pso`). Their conclusions (RL ~5x worse,
warm-start below plain GA, QAT/pruning deployability) are regime-insensitive;
they were NOT re-run under the post-fix defaults.
