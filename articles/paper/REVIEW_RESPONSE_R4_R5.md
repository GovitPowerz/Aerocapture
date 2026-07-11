# Response to Reviewer Reports 1 and 2 (2026-07-10)

Point-by-point response for the R4/R5 revision on branch `rework_after_review`.
Per-finding execution status: `articles/paper/revision_state.json` (untracked working state);
plan with pre-registered interpretation rules:
`docs/superpowers/plans/2026-07-10-reviewer-4-5-revision.md`.

Status legend: DONE (in the manuscript), PENDING-<what> (awaiting a named compute leg),
REBUTTED (with evidence).

Table numbering note: the revision inserts a scenario-pool table in Section 4, so the
submission's "Table 3" (performance) and "Table 4" (paired) are now Tables 4 and 5.

---

## Reviewer 1 -- major comments

**R1-1 (sizing pool not selection-disjoint).** DONE (data), PENDING-requote (final numbers).
We built the requested pool, stronger than asked: all methodology/architecture/checkpoint
choices frozen at a recorded revision, then a confirmatory pool of 10 replicates x 100,000
scenarios per scheme whose seeds are drawn from [2^31, 2^32) -- structurally disjoint from
every historical draw (all earlier pools, training batches, and curation probes live in
[0, 2^31)), not merely disjoint by birthday bound. Each cell evaluated exactly once. The new
Section 4 pool-roles table states which decisions each pool influenced; the reserved n=1000
in-training pool is renamed a *selection pool* and its adaptive reuse quantified (13,442
queries over the headline run), per the reviewer's observation. Outcome: the development
numbers CONFIRM -- deployed policy CVaR99.9 123.3 +/- 0.11 at n=10^6 vs 122.0 on the
development n=10^4 pool; joint-FTC 165.1 vs 164.0; no detectable selection optimism. The
final requote of every quoted sizing number onto this pool lands with the last compute leg
(FNPAG cell held while training runs occupy the machine).

**R1-2 (CVaR99.9 from ~10 observations).** DONE. On the confirmatory pool CVaR99.9 averages
100 observations per replicate, 1000 pooled, with replicate-level (design-based) standard
errors and paired-difference t-intervals -- no bootstrap assumptions. The empirical estimator
is defined in Section 2.2 (mean of the worst round((1-alpha) n) captured observations, no
interpolation) and every tail statistic is reported with the count it averages. p99.87 is
recorded per replicate alongside CVaR99/CVaR99.9; the sample maximum is labeled descriptive.
Survival curves (new figure, Section 7) show the full tail rather than point statistics.
Replicate pools double as the "repeat the pool generation" request: ten independent designs.

**R1-3 (124.5 is a three-seed mean, not the deployed artifact).** PENDING-requote, resolved
in substance: the abstract will quote the deployed artifact's own confirmatory value
(123.3 +/- 0.1 replicate CI) with the across-retraining range (per-seed 122.2 / 131.0 /
123.3) reported separately in Section 6.2; scenario uncertainty and training-run uncertainty
are never pooled. The confirmatory pool also sharpened the estimand distinction in a way
n=10^4 could not see: one non-deployed retraining seed (s2) physically crashes at a 6.1e-5
rate (61/10^6, individually re-run at 6x timeout, all `Crash` terminations -- committed
classification sidecar), while the deployed artifact captures 10^6/10^6. Figure 1's
"deployed Mamba ensemble" wording is fixed ("200-run Monte Carlo ensemble of the deployed
Mamba policy").

**R1-4 (conditional CVaR over captures).** DONE. All stress-regime tail statistics are
written CVaR95(dv | capture) and read lexicographically (capture probability first,
conditional tail second, no tail win claimed across a capture deficit -- stated in Section
7.2). The Section 7.3 comparison now carries both coordinates for every cell and notes
capture parity within half a point. "100% capture" is defined as "no failures observed" with
the one-sided 95% binomial bound (~3/n) in the performance-table caption.

**R1-5 (state-ablation controls).** DONE -- the reviewer's central experiment, run at
confirmatory grade with rules of interpretation pre-registered before any control trained
(plan Task 19; sequential-seed gate recorded in the runner header). Results (each 10x100k,
100% capture, zero violations):
- *State reset* (deployed weights, state zeroed every guidance tick): CVaR99.9 123.3 -> 414.5.
  The deployed policy computes with its state; it is not a feedforward law in disguise.
- *Matched observation history* (dense + explicit 5-tick window, 970 params, identical budget
  and regime): 142.2 +/- 0.3, worst case 423 m/s -- inside the dense family's seed spread;
  short temporal context does not substitute for learned state.
- *No predicted-dv retrain*: 138.9 +/- 0.2 -- the engineered inputs matter for bulk and tail
  alike; inputs and state are complements.
Pre-registered gate (outside the intact policy's 122.2-131.0 seed range -> single run
suffices) was met with 11-19 m/s margins. Section 6.3 reports all three; Section 9's "control
not run" caveat is deleted -- the mechanism is measured, not hypothesized. The remaining
controls the reviewer listed (state-shuffled, multiple state dimensions) are subsumed: the
zero-state case at matched parameters is the existing dense_972 cell, and the reset control
bounds the state-perturbation family from above.

**R1-6 (classical objective-equivalence).** DONE. Documented in Appendix A ("classical tuning
parity"): every classical scheme is tuned by the same GA on the same cost (identical
penalties, virtual costs, transform), co-optimizing 26 parameters for FTC -- including the
shared navigation and actuator gains -- against the network's weights + 3; budgets stated;
classical searches plateau far inside budget (FNPAG's argmin stopped improving before
generation 60 of 2000). The information asymmetry is tested directly rather than argued: the
no-predicted-dv retrain still beats co-tuned joint-FTC on every reported statistic (CVaR95
125.3 vs 144.3; CVaR99.9 138.9 vs 165.1). The suggested dv-targeting numerical
predictor-corrector variant is noted as future work; the ablation shows the network's edge
does not ride on the privileged observations, which was the objection's substance.

**R1-6b (related work / novelty).** DONE. Both reviewer-supplied references verified real and
added (Zucchelli et al. 2021 two-stage optimization; Matz, Lu, Mendeck & Sostaric 2017 FNPAG
Mars application), plus Rataczak, McMahon & Boyd (JGCD 2025, convex predictor-corrector),
Sonandres, Palazzo & How (2025, ABAMGuid+ and LSTM density estimation inside FNPAG), and
Calkins et al. (2025, risk-aware generative indicator + NPC). A related-work paragraph
situates the paper; the novelty claim is narrowed to "to our knowledge the first for an
MSR-class Mars aerocapture comparing an end-to-end learned policy against co-tuned classical
baselines on paired dispersed scenarios under a far-tail correction-dv risk metric."
REBUTTED in part: the "2026 paper whose title explicitly compares numerical predictive and
machine-learning aerocapture guidance" could not be located under any searchable title after
five targeted searches (the reviewer's two other links carried search-engine attribution
tags and did verify); we cite what exists and welcome a concrete pointer.

**R1-7 (reproducibility).** DONE except release URL. Appendix A now carries: the three-burn
correction equations; the exact per-simulation cost and virtual-cost equations; the L6
aggregate objective; GA operator constants; adaptive-curation pseudocode; the
confirmatory-pool protocol; exact numeric high-regime dispersions; the dispersion rationale
(OU zero-initialization, static-bias-vs-OU frequency separation, inert wind draws). The
simulator validation claim is scoped -- bit-level agreement with the legacy flight code is
regression validation, not physical validation -- and the independent physics checks are
listed, including a new vacuum two-body conservation test (energy and |h| conserved to
~2e-14 / 3e-15 observed, gated at 1e-11). Code, configurations, deployed weights, per-run
evaluation records, and regeneration scripts are released under MIT; the repository URL is
inserted at camera-ready.

**R1-8.1 (monotone-transform claim).** DONE. The incorrect "ranking-neutral" sentence is
replaced by the exact aggregate objective, new equation: J = (mean C_i^6)^(1/2), a monotone
function of the L6 norm of the per-scenario cost vector, with the reason we prefer a smooth
high-moment proxy over direct quantile optimization at two scenarios per individual per
generation. Appendix A cross-references the same equation.

**R1-8.2 (RL discussion).** DONE. The paragraph no longer claims policy-gradient methods
require differentiable rewards; it states the empirical result with numbers from the
committed records: dense PPO 636 m/s mean (1047 CVaR95) and recurrent PPO 513 (893) vs 119
(138) for the population-trained dense network on the same contemporaneous regime, with a
footnote scoping the legacy simulator vintage (the 4-5x gap, not the absolute values, is the
result). Also answers R2-B.

**R1-8.3 (CMA-ES mechanism).** DONE. The flatness (126.9 -> 127.3) is stated empirically;
the mechanism is demoted to a clearly-labeled hypothesis (parameter-space sampling
decorrelates generations; scenario noise perturbs rank-based updates and step-size control,
consistent with the observed self-termination), with "our experiments were not designed to
isolate the mechanism." Also answers R2-C.

**R1-9 (LSTM infeasible seed in the mean; constraint handling).** DONE. A feasibility-first
rule is adopted and stated in Section 6.2 (a run must satisfy every constraint on every pool
it was evaluated on to enter rankings or deployment); the LSTM ranks by its feasible-seeds
mean, ordering unchanged; the performance table carries a daggered LSTM row with both values.
The soft-penalty insufficiency is stated plainly (one of eleven converged runs bought tail
with heat). The 113 m/s floor inconsistency is resolved: the floor at this entry interface is
~105 m/s (undispersed-nominal periapsis raise, committed sidecar); 113 belonged to the 2009
interface and is now scoped to it.

**R1-10 (dispersion model justification).** DONE. Appendix A adds the rationale paragraph
(conservative +/-50% envelope; independence as a stated modeling choice; static bias vs OU as
profile-scale vs along-track frequencies -- no double counting; OU zero-initialized reaching
stationary variance within ~one correlation time; wind draws retained inert for layout
stability) and the exact high-regime numeric presets. The LHS concern is REBUTTED with a
clarification now in Section 2.3: evaluation pools draw one scenario per seed and are
therefore plain independent samples -- LHS stratification only applies within multi-scenario
batches -- so the replicate/bootstrap intervals stand.

**R1-11 (Figure 1 not a reachable set).** DONE. Renamed "empirical trajectory-occupancy
envelope" in figure, caption, legend, and prose; the caption states the quantile edges are
empirical, not a formal reachable set; the undispersed full-lift-up and full-lift-down
constant-bank boundary traces the reviewer requested are overlaid.

**R1-12 (compute claims).** PENDING-idle-box (benchmark re-run with hardware/toolchain
metadata, per-guidance-update cost, and timing spread requires a quiet machine; queued after
the training legs). Already applied: "FNPAG is dominated outright" is scoped to accuracy and
compute with robustness kept a separate axis; a flight-processor scaling paragraph (R2-A)
makes the relative-vs-absolute distinction explicit; Appendix A timing text will absorb the
new metadata when the benchmark reruns.

## Reviewer 1 -- statistical comments

**R1-S1 (optimizer repetitions).** PENDING-16 (the sigma_run extras -- GA/CMA-ES x
fixed/rotating/adaptive seed repeats and the Section 7.3 centered-retrain repeats -- are
training as of this response; results upgrade the Study C and 7.3 statements from single-run
hedges to mean +/- range).

**R1-S2 (tail-delta CIs).** DONE. Paired-resample bootstrap CIs for delta-p95 and
delta-CVaR95 in the paired table (Mamba-vs-classical decisively nonzero; the Mamba-vs-LSTM
CVaR95 interval straddles zero at n=1000 and is reported as such). Far-tail difference
intervals come from confirmatory replicate deltas (e.g. Mamba vs joint-FTC dCVaR99.9 = -41.8
[-42.4, -41.2]).

**R1-S3 (scenario vs training-run uncertainty).** PENDING-requote; the reporting convention
(replicate CI beside per-seed range, never pooled) is fixed and already used in Section 6.3.

**R1-S4 (Wilcoxon pseudo-precision).** DONE -- truncated at 1e-15 with the saturation
footnote.

**R1-S5 (confirmatory vs exploratory).** DONE -- explicit scoping sentence after the
pool-roles table; Sections 4-5 exploratory, frozen-pool quantities confirmatory.

**R1-S6 (Figure 11 connector lines).** DONE -- lines removed; Spearman labeled descriptive
with the non-exchangeability caveat.

## Reviewer 1 -- presentation

Promotional-language pass applied ("dominated outright", "tail one can trust", "canonical
... failure", abstract single-caveat phrasing). "Monte Carlo" attributive throughout.
500 x 11 km defined as apoapsis x periapsis altitude. Heat load in MJ/m2 in text, tables, and
appendix panels. CVaR estimator defined where introduced. Section 8 reframed as closed-loop
input sensitivity, "irreducible scenario noise" withdrawn. Two-column figure sizing and
color-blind styling are deferred to a journal reformat (this revision targets the arXiv
single-column format); title retained for the same reason. Appendix C retained in-paper for
arXiv; will move to supplementary material for a journal submission.

## Reviewer 2

**R2-A (flight-compute realism).** DONE -- Section 7.2 adds the requested scaling discussion:
at a conservative 100x flight-processor factor FNPAG's 86 ms replan approaches its own 2 s
replan period while the network's forward pass stays sub-second; the relative ordering is the
portable result, and worst-case execution time on qualified hardware is explicitly not
established.

**R2-B (PPO numbers).** DONE -- see R1-8.2.

**R2-C (CMA-ES dynamics).** DONE -- see R1-8.3; the revised text follows the
rank-update/step-size framing the reviewer suggested, as hypothesis.

**R2-D (V&V path).** DONE -- Section 9 adds the simplex-monitor discussion (joint-FTC as
onboard analytic monitor with takeover authority; the deployed policy is a fixed
double-precision forward pass with no data-dependent control flow, so WCET/memory bound
trivially; verifying decision behavior across the envelope is the open problem).

**R2 overfitting phrasing (Section 2.B of the report).** DONE -- "does not overfit" softened
to "generalizes across the dispersion distribution rather than memorizing scenarios"
(plateau figure caption and Section 4 text).

**R2 nitpicks.** Three of four REBUTTED as PDF-extraction artifacts, verified against the
rendered PDF: the abstract reads "16.4 m/s in mean and 27.6 m/s at CVaR95" (no "27.6 7.6");
Equation 5 reads wrap_pi(mu_prev + Delta_max tanh(o_1)) with balanced parentheses; Section
2.1 reads 25 MJ/m2. The 27.6-vs-27.5 rounding observation was real and is addressed: the
paired-table caption now states deltas are computed on unrounded per-scenario values and may
differ from differenced rounded marginals by 0.1 m/s. The LSTM feasibility dagger requested
for the performance table is added (see R1-9).

---

## Remaining before resubmission

1. sigma_run extras training (in progress) -> Study C error bars + Section 7.3 upgrade.
2. FNPAG confirmatory cell (held until training completes) -> final requote of every sizing
   number, abstract estimand, survival-figure FNPAG curve, paired far-tail deltas.
3. Idle-box benchmark re-run -> Appendix A timing metadata.
4. Public repository URL in the artifacts paragraph.
