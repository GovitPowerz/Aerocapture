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
