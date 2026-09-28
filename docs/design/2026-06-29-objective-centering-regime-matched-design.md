# Worst-case objective-shaping is regime-matched (centered training under adversarial dispersions)

> **Date:** 2026-06-29. **Status:** design approved, pre-implementation.
> **Topic:** a controlled experiment showing that the GA's worst-case objective-shaping
> (cubed cost transform x max-bucket curation x few sims) is matched to the *medium* training
> regime and *backfires* under a high/adversarial regime, and that "centering" the objective
> recovers a usable selection gradient. Becomes a methodology subsection of the aerocapture-NN paper.

## 1. Motivation

exp-13 (`13_robustness_retrain.sh`) retrained the deployed Mamba_962 headline on the high-dispersion
regime (atmosphere / density_perturbation / navigation / nav_filter = high) to test the paper's
closing "widen the NN training regime is future work" line. It **stalled**: trained to gen 10,739
(`training_output/paper/robustness_retrain/mamba_p962/`), the best validated cost plateaued around
`2.72e12` (cubed-space) with the last ~thousand validations sitting *above* the best (no longer
descending), and validation capture stuck at **~96%** rather than climbing to ~100%. The user stopped
it: stagnant for 6000+ generations, underperforming a high-retrained joint-FTC.

**Diagnosis (grounded in the run data).** Under the high regime ~4% of scenarios are catastrophic
(crash / hyperbolic escape -> virtual DV ~3000+). The `cubed` cost transform makes those few failures
dominate the objective almost entirely; `max`-bucket curation preferentially feeds the *hardest* seeds
into each individual's small sim batch; and with only a few sims the per-individual estimate is a
near-worst-case sample. Two policies that both fail the hard seeds then look equally bad, so selection
has no gradient to climb -- the objective has collapsed into a spiky, near-discrete "how many of the
worst seeds did you survive" signal. Hence the stall.

This refines, rather than contradicts, the paper. The campaign established that in the *medium* regime
the aggressive shaping wins: cubed minimizes the far-tail (Study D), `n_sims=2` over many gens
dominates (Study F), and `max`-bucket beats middle/random (Study C-sub). The optimal tail-weighting is
**matched to the environment's noise and the per-individual sample budget**. When the regime is noisy
enough that the sample budget cannot estimate the tail and failures dominate, the same stack amplifies
noise instead of shaping the tail.

## 2. Hypothesis

Under the high/adversarial regime, the worst-case shaping stack (cubed x max x few-sims) collapses the
GA selection gradient; *centering* the objective -- more sims per individual (a real cost estimate), a
central curation bucket, and a milder cost transform -- recovers the gradient and yields better
off-nominal performance at equal compute. The dominant lever is expected to be the **sample budget**
(`n_sims`): with too few sims you cannot estimate, let alone optimize, the cost distribution.

## 3. What we already have (the medium half of the reversal)

No new medium-regime runs are needed. The reversal's medium half is documented in
`articles/paper/data/results.json`: in the medium regime `cubed` (= `optimizer_budget/ga_300`) wins the
far tail over linear/sqrt/squared/log; `curation_shaping/bucket_*` shows `max` best; and
`training_n_sims/adaptive_2` is the allocation winner. This experiment builds only the **high half +
attribution**, and the paper cross-references the existing medium numbers for the reversal.

## 4. Phase 1 -- attribution on the fast dense_515 vehicle (high regime)

The objective-shaping effect is architecture-independent, so Phase 1 uses the memoryless 515-parameter
dense net (`~2.4` ms/sim; trains far faster than the stateful Mamba). All five cells run under the
high regime (the four domains above set to `level = "high"`; everything else as the controlled
regime). One lever is flipped at a time from the "stacked" control to the fully "centered" cell:

| cell        | `cost_transform` | `curation_bucket_selection` | `training_n_sims` |
|-------------|------------------|-----------------------------|-------------------|
| `stacked`   | cubed            | max                         | 2                 |
| `plus_sims` | cubed            | max                         | 16                |
| `plus_bucket` | cubed          | middle                      | 2                 |
| `plus_transform` | linear      | max                         | 2                 |
| `centered`  | linear           | middle                      | 16                |

**Held fixed across all cells:** dense_515 architecture (17-input atan2, the `sweep/dense_p515.toml`
base), GA optimizer, `seed_strategy = adaptive` (only the curation *bucket* varies, isolating it from
the seed schedule), `n_pop = 256`, and the reserved seed pools (validation 1M, final-eval 2M, stress
9M).

**Iso-compute by construction.** `n_sims` differs 8x (2 vs 16), so cells are matched on the *total
training-sim budget* `B = n_pop * n_sims * n_gen`: the `n_sims=16` cells run ~1/8 the generations of
the `n_sims=2` cells. Starting budget: `stacked` (n=2) runs ~16,000 gens, the `n_sims=16` cells ~2,000
gens (B ~= 256 * 2 * 16000 ~= 8.2M training sims/cell). This is a knob -- enough for `stacked` to
visibly stall and `centered` to plateau. All comparisons are read against **cumulative actual sims**
(training + validation + curation, from the per-gen JSONL), not generations, matching the Study C
iso-compute discipline so no cell is flattered by extra compute.

## 5. Metrics and figure

1. **Convergence (the gradient-recovery evidence):** validation RMS and validation capture-rate vs
   cumulative actual sims, one line per cell. Expectation: `stacked` flat at ~96% capture; `centered`
   descending to ~100%; the single-lever cells rank the levers (hypothesis: `plus_sims` recovers most
   of the gradient).
2. **Deployed off-nominal:** capture %, CVaR95, CVaR99 on the reserved 9M stress pool, per cell, via
   the existing `robustness_retrain_eval.py` machinery (extended to take these cells; same pool and
   high overrides, so numbers are comparable to `robustness_stress.json` / `robustness_retrain.json`).
3. **The reversal:** a compact restatement that in the medium regime the ordering is the opposite
   (stacked >= centered), cited from the existing campaign data -- the "regime-matched" point.

A new figure (paper stage) overlays the convergence curves and the deployed off-nominal bars.

## 6. Phase 2 -- confirm on Mamba_962

Take Phase 1's winning centered recipe, train Mamba_962 under the high regime to comparable depth, and
deploy-eval on the 9M pool. Control = the stopped stacked Mamba run (`mamba_p962/`, checkpoints
retained). Confirms the objective effect transfers to the stateful architecture and shows whether
centering closes the joint-FTC off-nominal gap (a bonus, not the methodology goal).

## 7. Artifacts

- `configs/training/paper/objective_centering/dense_{stacked,plus_sims,plus_bucket,plus_transform,centered}_high.toml`
  -- each base-inherits `sweep/dense_p515.toml`, sets the four high MC levels, and overrides the cell's
  `[optimizer] training_n_sims` / `curation_bucket_selection` and `[cost_function] cost_transform`.
  Isolated `[data] neural_network` deploy paths so training does not clobber other cells.
- `configs/training/paper/objective_centering/mamba_centered_high.toml` (Phase 2, filled with the
  Phase 1 winner's settings).
- `experiments/paper/14_objective_centering.sh` -- idempotent runner (skip-if-`final_eval.parquet`),
  trains the five dense cells with the per-cell `--training-n-sims` and gen counts that hold `B`
  fixed, then runs the eval. Phase 2 Mamba cell gated behind a flag / second invocation.
- Extend `articles/paper/scripts/robustness_retrain_eval.py` (or a sibling
  `objective_centering_eval.py`) to score the cells on the 9M pool and emit
  `articles/paper/data/objective_centering.json` plus the per-cell convergence series for the figure.
- A new `fig_objective_centering` builder (paper stage).

## 8. Scope and non-goals (YAGNI)

- **Reuse** the medium-regime numbers; run no new medium cells.
- **No** full 2^3 factorial (8 cells) -- one-lever-at-a-time (5 cells) is enough to attribute the
  dominant lever given the paper treats the three knobs as one idea.
- **No** seed-strategy lever (adaptive vs rotating) in this experiment -- hold seed strategy fixed at
  adaptive and vary only the bucket, to keep the attribution clean. (Rotating-vs-adaptive under
  adversarial noise is a possible follow-up, noted not built.)
- Closing the joint-FTC off-nominal gap is a **Phase 2 bonus**, not the success criterion -- the goal
  is the methodology finding.

## 9. Risks

- **Low-gen under-exploration.** The iso-compute `n_sims=16` cells run far fewer generations; a GA can
  under-explore in too few gens. Mitigated by choosing `B` large enough that the `n_sims=16` cells get
  >= ~2,000 gens, and by reading convergence on the actual-sims axis (if a centered cell is still
  descending at the budget, extend it -- `train.py` auto-resumes).
- **The effect might be a single lever, not the bundle.** That is a *finding*, not a failure -- the
  attribution design is built to surface exactly which lever dominates.
- **Dense -> Mamba non-transfer.** If the centered recipe helps dense but not Mamba, Phase 2 reports
  that honestly; the methodology finding (objective centering on the fast vehicle) still stands.

## 10. Success criteria

- Phase 1 convergence figure shows the `stacked` cell stalling at ~96% validation capture while the
  `centered` cell (and at least the dominant single-lever cell) descends toward ~100% at equal
  cumulative sims.
- Phase 1 deployed off-nominal: the cells are cleanly ranked on the 9M pool and the per-lever ranking
  identifies the dominant knob. The expected direction is `centered` > `stacked`; if centering recovers
  the training gradient but does *not* improve off-nominal deployment, that dissociation is itself a
  reportable finding (training-signal recovery and off-nominal generalization are distinct).
- The medium-vs-high reversal is stated and cited from the existing campaign data.
- Phase 2 reports whether the direction transfers to Mamba_962 (transfer or non-transfer both stated).

## 11. Final step

After implementation, invoke the `smart-commit` skill, telling it to take the whole git branch into
account.

## 12. Results at sizing depth (2026-09-26, #156)

`articles/paper/scripts/centered_depth_eval.py` (`make -C articles/paper mc-centered-depth`) scores
the three centered-Mamba trainer seeds (`objective_centering/mamba_centered`,
`sigma_extras/mamba_centered_s2`, `_s3`) and the two joint-FTC baselines on the 9M stress pool at
n = 10,000 (the n = 1000 pool is its first 1000 seeds; both baselines reproduce their committed
n = 1000 numbers on it exactly), paired on scenario, bootstrap 95% CIs, under both noise regimes,
into `articles/paper/data/centered_depth.json`. Mean and CVaR95 are over captured scenarios; the
delta columns are the seed minus the joint-FTC retrained on the regime.

Shared noise path (the regime the cells trained under and the figure's regime):

| Cell | Capture % | Mean (m/s) | CVaR95 (m/s) | Delta capture vs retrained (pts) | Delta CVaR95 vs retrained (m/s) |
|---|---|---|---|---|---|
| Mamba s1 | 96.0 [95.7, 96.4] | 135 [133, 136] | 323 [289, 358] | -0.02 [-0.06, +0.02] | -170 [-181, -156] |
| Mamba s2 | 96.0 [95.7, 96.4] | 149 [147, 151] | 345 [313, 379] | -0.01 [-0.05, +0.02] | -147 [-157, -137] |
| Mamba s3 | 95.9 [95.5, 96.3] | 135 [133, 136] | 314 [282, 346] | -0.15 [-0.24, -0.08] | -178 [-186, -169] |
| joint-FTC retrained | 96.1 [95.7, 96.4] | 316 [314, 317] | 492 [466, 520] |  |  |
| joint-FTC medium-deployed | 95.6 [95.2, 96.0] | 170 [169, 172] | 410 [379, 443] |  |  |

Against the medium-deployed joint-FTC: Mamba s1 capture +0.40 [+0.28, +0.54] pts, CVaR95 -88 [-98, -75]; Mamba s2 capture +0.41 [+0.28, +0.55] pts, CVaR95 -65 [-73, -57]; Mamba s3 capture +0.27 [+0.12, +0.43] pts, CVaR95 -97 [-104, -88].

Per-scenario noise (ADR-0006; the cells never trained on it):

| Cell | Capture % | Mean (m/s) | CVaR95 (m/s) | Delta capture vs retrained (pts) | Delta CVaR95 vs retrained (m/s) |
|---|---|---|---|---|---|
| Mamba s1 | 92.2 [91.6, 92.7] | 233 [230, 235] | 604 [580, 629] | -4.15 [-4.58, -3.76] | +48 [+34, +62] |
| Mamba s2 | 94.3 [93.9, 94.8] | 219 [217, 221] | 535 [510, 561] | -1.98 [-2.27, -1.70] | -22 [-31, -13] |
| Mamba s3 | 95.6 [95.2, 96.0] | 206 [204, 209] | 596 [572, 619] | -0.67 [-0.85, -0.49] | +39 [+27, +50] |
| joint-FTC retrained | 96.3 [95.9, 96.7] | 342 [340, 344] | 557 [535, 579] |  |  |
| joint-FTC medium-deployed | 94.6 [94.1, 95.0] | 178 [176, 180] | 457 [429, 485] |  |  |

Against the medium-deployed joint-FTC: Mamba s1 capture -2.42 [-2.88, -1.96] pts, CVaR95 +148 [+131, +165]; Mamba s2 capture -0.25 [-0.64, +0.13] pts, CVaR95 +78 [+67, +89]; Mamba s3 capture +1.06 [+0.73, +1.39] pts, CVaR95 +139 [+124, +155].

Reading. Under the shared path the n = 1000 claim survives the depth: every seed beats both
baselines on the conditional tail with paired CIs excluding zero (147-178 m/s below the retrained
joint-FTC, 65-97 below the medium-deployed one), at capture within half a point of either (the only
capture deficit the depth resolves is seed 3's 0.15 pts to the retrained baseline). The n = 1000
tails (231-273 m/s at 94.8-95.0%) were the low side of wide intervals; at depth they read 314-345 at
95.9-96.0%. Under per-scenario noise the reversal does not survive: the retrained joint-FTC
out-captures every seed (paired deltas -0.67 to -4.15 pts, CIs excluding zero) and out-tails seeds
1 and 3; the medium-deployed joint-FTC holds the best conditional tail of the five (457 m/s, every
seed 78-148 above it) but is out-captured by seed 3, out-captures seed 1 and ties seed 2 on capture.
No seed beats both baselines. Same mechanism as Appendix E: a policy trained on one density history
fits it. The open follow-up is a centered retrain under `per_draw` seeding (TODO.md).

Seed 1's model. The s2 / s3 repeats (`experiments/paper/16_sigma_extras.sh`) train with
`mamba_centered_high.toml`, whose `[data] neural_network` deploy path is seed 1's run directory; the
s3 run left its model there on 2026-07-11, so seed 1's `best_model.json` held seed 3's weights next
to seed 1's own `best_params.json`. It was rebuilt from seed 1's final checkpoint
(`checkpoint_g04000.npz` population row 4, the `final_selection.json` winner `last_gen[4]`) with
`artifacts.write_best_artifacts`: the rebuilt `best_params.json` is byte-identical to seed 1's, and
the model reproduces the committed n = 1000 stress quote exactly (94.9% capture, mean 131.80,
CVaR95 272.80). `centered_depth_eval.py` exits when two seeds share a model file.
