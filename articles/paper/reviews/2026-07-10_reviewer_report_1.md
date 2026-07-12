# Reviewer report

**Recommendation: Major revision — borderline reject-and-resubmit for a selective journal.**

I reviewed the complete 48-page manuscript, including the main text, references, reproduction appendix, architecture probes, and per-scheme mission reports. This review is based on the manuscript and reported results; I did not execute the simulator or inspect the underlying code and data. 

## Overall assessment

This is an ambitious, unusually readable, and potentially important paper. The central idea—evaluating aerocapture guidance by the correction-propellant tail rather than primarily by mean performance—is compelling. The paired Monte Carlo comparisons, attention to constraint violations, co-optimization of classical reference trajectories, and explicit discussion of an off-nominal failure mode are all substantial strengths.

However, the manuscript currently makes stronger claims than the evidence supports. The most serious problems concern:

1. Reuse of evaluation pools during method and architecture selection.
2. Estimation of CVaR(_{99.9}) from only about ten tail observations.
3. Mixing the performance of one deployed policy with averages across independently trained policies.
4. Conditional evaluation of correction (\Delta v) only over successful captures.
5. A causal claim about recurrent state without a state-ablation experiment.
6. Incomplete and potentially objective-mismatched classical baselines.
7. Insufficient simulator, controller, and statistical detail for independent reproduction.
8. Several incorrect or overstated methodological explanations concerning monotone transforms, reinforcement learning, and CMA-ES.

These issues do not make the results uninteresting. They mean the headline numbers and mechanism claims are not yet sufficiently controlled for publication in a rigorous archival journal.

### Indicative scores

| Criterion                        |     Assessment |
| -------------------------------- | -------------: |
| Potential significance           |            4/5 |
| Originality                      |            4/5 |
| Technical soundness as submitted |            2/5 |
| Experimental rigor               |            2/5 |
| Reproducibility                  |            2/5 |
| Clarity and organization         |            4/5 |
| Overall                          | Major revision |

---

# Major comments

## 1. The headline far-tail result does not appear to come from a genuinely untouched test set

The paper repeatedly describes the (n=10{,}000) sizing pool as “training-disjoint.” That is necessary but not sufficient. It must also be **selection-disjoint**.

The same far-tail pool appears to be used to:

* Choose the cubed cost transform in Section 4.2.
* Choose the hardest-seed curation method.
* Compare architecture families.
* Select or justify the Mamba architecture.
* Report the final CVaR(_{99.9}) headline.

Once the pool influences the choice of transform, curation strategy, architecture, or checkpoint, it is no longer an unbiased final test set. Repeated comparison of alternatives against the same 10,000 cases can produce substantial selection optimism, especially when the reported metric depends on only ten extreme observations.

A related issue applies to the reserved (n=1000) “validation” pool. Appendix A says every new argmin is rerun on this pool and promoted on strict improvement. Over a 15,000–20,000-generation search, this is repeated adaptive querying of the validation set. It should therefore be called a **selection set**, not an independent validation set. Figure 6 cannot establish absence of overfitting merely because this repeatedly queried metric continues to improve.

The fresh “re-quote” pool is encouraging, but the manuscript says it confirms the mean and CVaR(*{95}), not the critical CVaR(*{99.9}) result.

### Required revision

Freeze all methodology, architecture, checkpoint-selection rules, and classical tuning before generating a final confirmatory pool. Then evaluate exactly once on a new, untouched pool. Because the simulator is relatively fast, this pool should contain at least (10^5), and preferably (10^6), scenarios for the finalists.

A defensible data split would be:

* Training scenarios.
* Promotion/checkpoint-selection set.
* Hyperparameter and architecture-selection set.
* Completely untouched confirmatory sizing set.

The manuscript should explicitly state which decisions were made using each pool.

---

## 2. CVaR(_{99.9}) is estimated from too few observations for the confidence placed on it

With (n=10{,}000), empirical CVaR(_{99.9}) is essentially the average of the worst ten observations. That is an extremely small effective sample size for the metric carrying the paper’s main conclusion.

This creates several problems:

* The estimate is sensitive to a handful of trajectories.
* Ordinary nonparametric bootstrap confidence intervals can behave poorly for extreme quantiles and expected shortfall with so few exceedances.
* Latin-hypercube samples are not simply interchangeable independent observations, so a naïve row bootstrap may not preserve the sampling design.
* The “sample maximum” is not a bound. It is merely the largest observed value, with very high sampling variability.
* The quoted confidence ranges of approximately (1)–(5) m/s need a complete derivation and validation.

The manuscript also motivates the tail using a conventional (3\sigma) design point, approximately the 99.87th percentile, but then adopts CVaR(_{99.9}). CVaR beyond 99.9% is appreciably more conservative than a 99.87th percentile. This may be a good requirement, but it is not the same requirement and should be justified by mission-level practice rather than presented as approximately equivalent.

### Required revision

For each finalist:

* Use a much larger untouched evaluation pool.
* Repeat the entire Monte Carlo pool generation several times, rather than relying on one LHS design.
* Report the number of observations contributing to every tail statistic.
* Define the finite-sample CVaR estimator, including interpolation and tie conventions.
* Provide confidence intervals for differences in CVaR, not merely intervals for each method separately.
* Consider reporting (p_{99.87}), CVaR(*{99}), CVaR(*{99.9}), and a survival curve together.
* Treat the sample maximum as “maximum observed,” not a design bound.

Extreme-value modeling could supplement the empirical estimate, although it should not replace a larger simulation campaign.

---

## 3. The reported 124.5 m/s value is not clearly the performance of one deployable policy

The abstract says:

> “A 962-parameter recurrent policy … reaches a far-tail CVaR(_{99.9}) of 124.5 m/s.”

However, Table 3 states that the network’s far-tail entries are **three-seed means**. A mean over three independently trained policies is not the performance of one deployed policy unless those policies are actually combined into an ensemble during flight.

The manuscript uses several different estimands:

* A specific selected Mamba policy on the (n=1000) pool.
* The average CVaR(_{99.9}) across three separately trained Mamba policies.
* Paired scenario comparisons involving one selected policy.
* Architecture-level claims intended to generalize across retraining.

These are not interchangeable.

Figure 1 also refers to a “deployed Mamba ensemble,” whereas subsequent text describes one deployed network. The word “ensemble” must be clarified.

### Required revision

Report separately:

1. Performance of the exact retained flight artifact.
2. Performance of every independently trained seed.
3. Mean and dispersion across training runs.
4. Performance of an ensemble only if an ensemble is genuinely deployed, including its command-combination rule and computational cost.

The abstract should quote the performance of the exact deployed artifact. Architecture claims should incorporate both scenario uncertainty and training-run uncertainty, ideally through a hierarchical analysis.

Three training seeds are a useful start, but they are a weak basis for a strong architecture claim. Five to ten repeats for the decisive architectures would be preferable.

---

## 4. Computing correction (\Delta v) only over captured trajectories makes the stress-regime comparisons problematic

Section 2.2 defines correction (\Delta v) only for captured runs. This is unobjectionable when all compared methods capture every test scenario. It becomes problematic whenever capture rates differ.

In the high-dispersion regime, the methods have capture rates around 90–95%, and some retraining configurations perform much worse. CVaR computed only over successful captures is then a **conditional risk measure**. A method can improve its conditional CVaR by failing on the scenarios that would otherwise have had the highest correction costs.

Showing capture rate in a neighboring panel does not create a statistically coherent joint comparison. Statements such as one method “beats” another in the stress regime are therefore not justified by conditional CVaR alone.

The same concern applies, though less severely, to weaker methods in Table 3 with capture rates below 100%.

### Required revision

Use one of the following:

* A lexicographic criterion: maximize capture probability first, then minimize conditional tail cost.
* An unconditional mission-loss variable that assigns failures a physically justified loss.
* A constrained comparison at a fixed required capture probability.
* A Pareto frontier of failure probability versus conditional correction cost.

At minimum, label every such metric explicitly as:

[
\operatorname{CVaR}_{\alpha}(\Delta v\mid\text{capture}).
]

“100% capture” should also be described as an empirical result. With zero failures in 1,000 independent cases, the one-sided 95% upper confidence limit on failure probability is about 0.30%; with zero in 10,000 it is about 0.03%. Thus “no failures observed” is more precise than implying guaranteed capture.

---

## 5. The manuscript does not establish that internal state causes the tail improvement

The paper’s most important architectural claim is that engineered inputs flatten the median while “genuine internal state” compresses the extreme tail. Yet Section 9 acknowledges that the decisive state-ablation control was not performed.

This is not a peripheral omission. Mamba differs from a dense network in more than memory:

* Parameterization.
* Multiplicative input-dependent operations.
* Depth and nonlinear structure.
* Optimization landscape.
* Weight sharing over time.
* Implicit regularization.
* State initialization and actuator interaction.

Matched parameter count does not isolate memory. The result presently supports:

> “The tested Mamba architecture achieved a tighter tail than the tested dense architectures.”

It does not yet support:

> “Internal state is what compresses the tail.”

The one-at-a-time input-zeroing analysis in Section 8 does not solve this problem. Zeroing a normalized feature can create an off-distribution observation and does not measure feature interactions. Because the policy is closed-loop, each ablation also changes all subsequent states and observations. It is useful as a sensitivity experiment, but not as a causal explanation.

### Required revision

At minimum, run:

* The trained Mamba with recurrent state reset at every guidance tick.
* A state-shuffled or state-delayed control.
* A dense policy with a matched observation-history window.
* A dense or feed-forward policy with a parameter and depth budget matched to the Mamba computation graph.
* Several state dimensions, including zero or minimal state.
* Repeated training seeds for these controls.

A stronger input study would retrain policies after removing important feature groups, especially the predicted-(\Delta v) signals. Post-training zeroing and retraining answer different questions and should both be reported.

Until those controls are available, the abstract, title discussion, and conclusion should describe the state explanation as a hypothesis rather than a demonstrated mechanism.

---

## 6. The classical comparison may not be objective-equivalent

The paper deserves credit for co-optimizing the reference trajectory used by FTC and related tracking schemes. The resulting improvement is large and important.

Nevertheless, the neural and classical methods do not appear to receive equivalent objectives or information:

* The neural policy is trained directly on total correction (\Delta v).
* It receives three predicted correction-(\Delta v) components as observations.
* It jointly handles in-plane and out-of-plane motion.
* It co-optimizes three actuator/navigation-side parameters.
* The described FNPAG implementation corrects a constant capture-bank magnitude primarily to match exit apoapsis, with inclination handled by shared roll-reversal logic.

A predictor-corrector aimed at apoapsis error is not necessarily the strongest classical baseline for minimizing the exact combination of periapsis raise, circularization, and plane change used as the neural objective.

The manuscript must also state whether the navigation filter, command shaping, bank-rate parameters, update frequencies, atmospheric adaptation, roll-reversal thresholds, and constraint handling were given equal tuning freedom for each method.

### Required revision

Provide:

* Complete objective functions for every tuned method.
* All tuned parameter values.
* Identical tuning budgets or a justified alternative.
* An ablation in which the neural policy does not receive predicted-(\Delta v) components.
* A predictor-corrector or optimal-control baseline targeting the same terminal correction cost.
* A comparison in which common navigation and actuator parameters are held fixed.
* Another comparison in which each scheme is allowed equal co-optimization of those parameters.

The related-work section also needs substantial updating. Recent work includes two-stage stochastic/robust aerocapture guidance, augmented analytical guidance, convex predictor-corrector aerocapture guidance, and a 2026 paper whose title explicitly compares numerical predictive and machine-learning aerocapture guidance. Therefore, the claim of the “first systematic head-to-head” cannot remain unqualified. A narrower claim—perhaps the first Mars/MSR comparison using paired scenarios and far-tail correction risk—may be defensible after a careful literature review. ([AIAA Journal][1])

The treatment of FNPAG should also engage with its later verification and Mars-application literature, rather than relying almost exclusively on the 2014 and 2015 references. ([AIAA Journal][2])

---

## 7. Reproducibility is insufficient for an archival guidance paper

Appendix A is helpful but far too compact to reproduce the work independently.

The manuscript does not provide enough detail on:

* Equations of motion and coordinate frames.
* Atmospheric density model and nominal profile.
* Aerodynamic coefficient model.
* Gravity-harmonic implementation.
* Heat-flux and integrated-heat models.
* Navigation equations and bias filter.
* Pilot/actuator transfer function.
* Event detection and termination logic.
* Exact correction-(\Delta v) equations and burn sequence.
* Failure virtual-cost formula.
* Soft constraint normalization.
* Classical controllers other than FTC.
* Full Mamba recurrence and discretization.
* Genetic operators, bounds, mutation rates, crossover settings, and initialization.
* Exact adaptive-seed algorithm.
* All selected network weights and normalization constants.

“Every number regenerates from retained records” is not independently verifiable unless those records, code, configurations, and weights are made available.

The simulator-validation statement is also overstated. Agreement with a legacy implementation is valuable regression testing, but it is not physical validation. Bit-identical output establishes that two codes agree; it does not establish that the dynamics, atmosphere, events, or correction-cost equations are correct.

In addition, Section 5 describes an objective produced by a simulator with adaptive integration, whereas the introduction says the campaign uses fixed-step integration. That apparent contradiction needs correction.

### Required revision

Provide a public or review-accessible archive containing:

* Source code or an executable reproducibility package.
* Configuration files.
* Exact network artifacts.
* Monte Carlo seeds or scenario records.
* Scripts regenerating every table and figure.
* A precise software environment and compiler specification.

Add verification tests such as:

* Step-size convergence.
* Vacuum energy and angular-momentum conservation.
* Agreement across several nominal and dispersed trajectories.
* Event-time convergence.
* Independent checks of orbital-element and correction-burn calculations.
* Comparison against published FNPAG or aerocapture benchmark cases.

---

## 8. Several methodological explanations are incorrect or too strong

### 8.1 The monotone-transform statement is mathematically incorrect

Section 4.2 says the cost transform is “ranking-neutral for a deterministic argmin.” Once transformed per-scenario costs are aggregated across scenarios, that is generally false.

For two policies with cost vectors (c_A) and (c_B),

[
\sum_i c_{A,i} < \sum_i c_{B,i}
]

does not imply

[
\sum_i f(c_{A,i}) < \sum_i f(c_{B,i})
]

for nonlinear monotone (f). The transform deliberately changes the risk preference, even on a fixed deterministic scenario set.

Moreover, Appendix A says costs are cubed and then aggregated by root mean square. This produces an objective proportional to

[
\sqrt{\frac{1}{n}\sum_i C_i^6},
]

not merely a cubic risk transform. The paper should describe this as an (L_6)-like high-moment objective and explain why it is preferred to direct CVaR optimization.

### 8.2 The reinforcement-learning discussion is inaccurate

PPO and SAC do not require a differentiable simulator or a differentiable, shaped per-step reward. Policy-gradient methods can optimize nondifferentiable and terminal-only rewards, although sparse terminal rewards may be sample-inefficient.

The manuscript may validly argue that the selected population methods were empirically superior or easier to use. It should not claim that reinforcement learning can only optimize a differentiable shaped surrogate. If RL policies were implemented, their architectures, rewards, budgets, and results should be reported in an appendix; otherwise the discussion should be shortened.

### 8.3 The explanation of CMA-ES is not convincing

CMA-ES samples candidate parameter vectors. It does not automatically resample atmospheric scenarios “through covariance adaptation.” Covariance adaptation occurs in parameter space, not uncertainty space.

The observed insensitivity of CMA-ES to rotating scenarios is an empirical result. The stated mechanism is not established. Training-versus-holdout curves for fixed and rotating scenarios would be much more informative.

Similarly, the claim that the fixed-seed genetic algorithm “overfits” requires showing that it performs well on the fixed training scenarios but poorly on held-out scenarios. Poor held-out performance alone could also result from failed optimization.

---

## 9. Constraint handling is not yet rigorous enough

One of the three LSTM runs violates the integrated heat-load limit in approximately 14–16% of cases. Nevertheless, its tail result enters the three-seed architecture mean.

An infeasible policy should not participate in an unconstrained ranking as though it were comparable to feasible policies. The low correction tail may partly be purchased by violating the thermal constraint, as the manuscript itself recognizes.

This also reveals that the soft-penalty formulation does not reliably enforce feasibility.

### Required revision

Use one of:

* Feasibility-first selection.
* Hard rejection of infeasible policies.
* A constrained evolutionary algorithm.
* An augmented-Lagrangian or adaptive-penalty scheme.
* A chance constraint with a stated allowable violation probability.

Report constraint margins and confidence intervals, not just observed violation percentages. Architecture averages should include only policies satisfying the same deployment criterion, or should report feasibility probability as part of the result.

There is also an internal numerical inconsistency to resolve. Section 2.2 describes an irreducible correction floor of roughly 113 m/s, but the deployed Mamba has a median total correction near 109.6 m/s, and individual periapsis-raise values near 105.9 m/s in Appendix C. Either the “113 m/s floor” is inherited from a different entry interface or definition, or it is not actually a floor. The underlying equations and terminology must be corrected.

---

## 10. The dispersion model needs stronger physical justification

The conclusions are conditional on the assumed uncertainty distribution. The manuscript gives ranges but little basis for them.

Questions include:

* Why is the static atmospheric density factor uniform over ±50%?
* Are aerodynamic coefficient dispersions independent?
* Are navigation errors independent of atmospheric or vehicle errors?
* Does combining a static density bias with an OU perturbation double-count any atmospheric variability?
* How are time-varying perturbations initialized?
* Why are two wind dimensions retained when the wind model is disabled?
* What are the exact numerical “high” settings in the off-nominal regime?
* How sensitive are conclusions to the correlation time and RMS of the density process?
* Does LHS with only two scenarios per generation introduce artificial marginal stratification or antithetic behavior?

The stress regime is dismissed as one a real mission would “design away.” That is too strong. Retargeting can increase nominal margin, but extrapolation and uncertainty beyond a training envelope remain important safety questions.

The conclusion in Section 7.3 that the off-nominal gap “is not intrinsic” is also too definitive. It rests on a single retraining run, (n=1000), differing capture rates, and conditional tail costs. The result is promising preliminary evidence that objective mismatch matters—not proof that the robustness gap has been closed.

---

## 11. Figure 1 is not a demonstrated reachable corridor

Figure 1 is described as a “reachable aerocapture capture corridor,” but its boundaries are empirical quantiles from randomly sampled piecewise-constant bank histories. Such an envelope depends on:

* The distribution of random bank profiles.
* The number and timing of segments.
* Bank-amplitude sampling.
* Roll-reversal sampling.
* The number of Monte Carlo runs.
* The definitions used to classify the lower and upper edges.

A quantile envelope of randomly generated trajectories is not equivalent to a reachable set or a mathematically defined capture corridor.

The upper and lower boundaries also use different conditional populations, making the shaded band particularly difficult to interpret physically.

The figure remains valuable, but it should be renamed something like **empirical trajectory-occupancy envelope** unless a formal reachable-set construction is supplied. The constant-bank overshoot and undershoot boundaries should preferably be shown explicitly.

---

## 12. The compute comparison does not yet establish onboard deployability

The reported timing is wall-clock time for an entire simulation on one unspecified Apple-silicon laptop core. That measurement combines the guidance algorithm with the surrounding simulator and depends on trajectory duration, compiler options, numerical precision, implementation quality, and hardware.

It does not directly provide:

* Inference time per guidance update.
* Worst-case execution time.
* Memory footprint.
* Floating-point operation count.
* Embedded processor timing.
* Determinism and jitter.
* Numerical precision requirements.
* Certification burden.

The 23× ratio is a valid measurement for the stated implementation and host, but it should not be generalized into a flight-computer claim.

Report the exact processor, compiler, optimization flags, thread configuration, numerical precision, number of guidance updates, and timing distribution. A representative embedded target or cycle-level estimate would make the deployability discussion much stronger.

“FNPAG is dominated outright” should also be removed. The paper itself shows that robustness is a separate axis on which the methods differ, so domination is not established across all relevant criteria.

---

# Statistical reporting comments

1. **Optimizer conclusions need repetitions.**
   Many optimizer and training-method comparisons appear to rely on one run per condition. The 40–70 m/s effects are likely real, but the claim that one optimizer is “best” requires replicated runs and uncertainty across runs.

2. **Paired bootstrap intervals should be given for (p_{95}) and CVaR differences.**
   Table 4 supplies an interval only for the mean difference, even though the tail differences are more important.

3. **Scenario uncertainty and training uncertainty should be separated.**
   Bootstrapping scenarios while holding a selected network fixed answers a different question from comparing architecture families across retraining.

4. **The Wilcoxon values are unnecessarily extreme.**
   Reporting (p\approx3\times10^{-165}) conveys pseudo-precision. “(p<10^{-15})” together with the 100% win rate and effect size would be more informative.

5. **Account for multiple comparisons.**
   Numerous transforms, curation methods, architectures, budgets, decoders, and baselines are compared. The manuscript need not apply a simplistic correction everywhere, but confirmatory and exploratory analyses should be distinguished.

6. **Figure 11 requires caution.**
   A Spearman coefficient over eleven runs combines different families and repeated seeds. The points are clustered and not exchangeable. The connecting lines may visually imply trajectories or ordered observations when none exist.

---

# Presentation and organization

## Strong aspects

* The manuscript has a clear narrative.
* Tables 3 and 4 are useful and centralize the key comparisons.
* The paper reports negative and infeasible results rather than hiding them.
* The appendices contain valuable per-scheme diagnostics.
* The figures use consistent visual conventions.
* The distinction between bulk performance and tail performance is communicated effectively.

## Changes recommended

* Reduce promotional language such as:

  * “the tail one can trust,”
  * “FNPAG is dominated outright,”
  * “canonical optimize-the-average, blow-up-the-tail failure,”
  * “the right optimizer,”
  * “one honest caveat remains.”

  These phrases are engaging, but an archival paper should use more neutral language.

* The title is memorable but long. A more conventional alternative would be:

  **Tail-Risk-Aware Stateful Neural Guidance for Mars Aerocapture**

* Move most of Appendix C to supplementary material. The 20-page mission-card appendix is useful for auditability but overwhelms the article.

* Several figure labels and legends will be too small after conversion to a two-column journal format.

* Avoid relying on color alone; use line styles and markers suitable for color-vision deficiencies and grayscale printing.

* Figure 11 should not connect unrelated runs with lines.

* Define whether orbit dimensions such as “500 × 11 km” are altitudes or radii and state apoapsis first consistently.

* Use one unit for heat load throughout, preferably MJ/m² in both text and tables.

* Use “Monte Carlo,” not “Monte-Carlo,” as an attributive noun unless the target journal specifies otherwise.

* If targeting an IEEE journal, convert the author–date bibliography to numbered citations and provide complete publication metadata and DOIs.

---

# Section-specific revision notes

## Abstract

The abstract should:

* Say “no failures observed in (n=\ldots)” rather than unqualified “captures 100%.”
* Quote uncertainty on the headline tail result.
* Identify whether 124.5 m/s belongs to one policy or is a mean over training runs.
* Avoid the unqualified causal statement that state compresses the tail.
* Avoid claiming a single remaining caveat when the paper has several acknowledged limitations.
* Scope the novelty claim carefully.

## Sections 1–3

* Expand the related work substantially.
* Give the full correction-(\Delta v) equations.
* Provide sufficient mathematical descriptions of all classical schemes.
* Explain why the selected FNPAG formulation is the appropriate state-of-the-art comparator.
* Separate empirical occupancy envelopes from formal capture corridors.

## Section 4

* Give pseudocode for adaptive seed curation and promotion.
* State precisely when seeds are replaced or reused.
* Correct the monotonic-transform claim.
* Write the actual aggregate objective as an equation.
* Explain the relationship between the training objective and the reported CVaR metrics.
* Quantify how often the reserved promotion set was queried.

## Section 5

* Correct the reinforcement-learning discussion.
* Correct or remove the explanation that CMA-ES internally resamples scenarios.
* Add repeated runs before ranking optimizers.
* Match total simulator calls and stopping criteria.

## Section 6

* Add the state-ablation experiments.
* Do not average feasible and infeasible policies into one architecture statistic.
* Report every training seed numerically.
* Distinguish performance of the selected artifact from expected performance across retraining.

## Section 7

* Replace conditional-CVaR-only comparisons when capture rates differ.
* Tone down the interpretation of the stress retraining.
* Add current classical and learning-based baselines.
* Report exact high-regime dispersions.
* Clarify whether classical methods were retrained and reselected using separate data.

## Section 8

* Call the analysis closed-loop input sensitivity rather than definitive feature importance.
* Add grouped ablations and retraining without key feature groups.
* Remove the statement that residual cost is “irreducible scenario noise” unless a lower-bound analysis is provided.

## Sections 9–10

The limitations section is candid, but the conclusion then restates several disputed claims too strongly. In particular:

* “Training loss did not pick the tail winner; architecture did” should be qualified.
* “Internal state earns its place only on the hardest scenarios” remains a hypothesis without state ablation.
* “The gap was the objective, not neural guidance” is too categorical for the single-run stress result.

---

# Minimum set of changes needed for acceptance

I would reconsider the paper favorably after the following core revisions:

1. Generate a genuinely untouched, much larger final sizing pool after all choices are frozen.
2. Report the exact deployed policy separately from training-seed averages.
3. Incorporate training-run uncertainty into architecture claims.
4. Use a joint treatment of failures and correction cost.
5. Run the missing state-reset and matched-history controls.
6. Align the neural and classical objectives and tuning freedoms.
7. Add recent aerocapture guidance baselines and revise the novelty claim.
8. Correct the monotone-transform, RL, and CMA-ES explanations.
9. Supply enough equations, parameter values, code, weights, and data for reproduction.
10. Resolve the stated 113 m/s floor versus observed values below 110 m/s.

## Suggested defensible central claim after revision

A more supportable conclusion, assuming the new holdout confirms the result, would be:

> On the specified Mars Sample Return simulation and dispersion model, a compact recurrent neural policy reduced empirical correction-(\Delta v) tail risk relative to the implemented classical guidance baselines. The results suggest that recurrent processing may improve performance in extreme scenarios, although dedicated state-ablation experiments are required to isolate that mechanism.

That would still be a meaningful and publishable result.

[1]: https://arc.aiaa.org/doi/10.2514/6.2021-1569?utm_source=chatgpt.com "Two Stage Optimization for Aerocapture Guidance"
[2]: https://arc.aiaa.org/doi/10.2514/6.2017-1901?utm_source=chatgpt.com "Application of a Fully Numerical Guidance to Mars Aerocapture"
