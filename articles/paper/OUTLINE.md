# Paper Outline — NN Aerocapture Guidance (follow-up to Gelly & Vernis, AIAA GNC 2009)

> Drafted 2026-06-25 from the completed campaign (00-10c + 5b/5c). For the WRITING session:
> this is the skeleton — sections, the one claim each makes, the figure/table it carries, and
> the locked numbers. Prose is the author's (voice: `articles/markdown/05_authorial_voice_and_style.md`).
> All numbers trace to `articles/paper/data/{results.json, far_tail_eval.json, robustness_stress.json,
> compute_benchmark.json}` + the committed bundle `articles/paper/data/runs/`.

## Working title
Neural-network aerocapture guidance, revisited: a recurrent policy that beats classical
predictor-correctors at the tail that sizes the mission — and the moving-environment training
that makes a genetic algorithm the right optimizer.

## The contribution spine (3 threads, in priority order)
1. **Training methodology IS the contribution.** GA is the WORST optimizer under fixed seeds
   (overfits) and the BEST under a *non-stationary* (adaptive-seed) objective: fixed→rotating→
   adaptive = 160.3 → 120.0 → 118.0 m/s (−42 m/s, the campaign's biggest single effect). The
   quartet — GA + adaptive seeds + cubed transform + max-bucket curation — is a matched system.
2. **Architecture: engineered inputs flatten the bulk; internal STATE wins the sizing tail.**
   All architectures tie on the median (~108-112), but on the mission-sizing tail a recurrent
   net (Mamba_962) beats the best dense net beyond σ_run (CVaR99.9 124.5 vs 139.2). The deployed
   headline is **Mamba_962**; a 515-param dense net is the efficiency reference.
3. **NN vs classical: the NN wins the nominal sizing tail at the fast compute class** (beats the
   best classical, joint-FTC, by −16.4 mean / −27.6 CVaR95; 23× faster than FNPAG) — with the
   honest caveat that the analytic joint-FTC is more robust OFF-nominal.

The objective is the **tail of the correction-ΔV distribution** (propellant sizing = mission cost):
lead every comparison with p95 / CVaR95 / CVaR99.9, mean as a footnote.

---

## §1 Abstract
- Claim: a 962-param recurrent NN matches the best classical on the median and BEATS it at the
  sizing tail, at near-FTC compute, trained by GA under a moving MC environment.
- Numbers: Mamba_962 fresh-pool (8M) CVaR95 **115.2**; vs joint-FTC −16.4 mean / −27.6 CVaR95;
  far-tail CVaR99.9 124.5 vs joint-FTC/FNPAG ~164/165; 3.68 ms/sim (23× < FNPAG).

## §2 Introduction
- Claim: aerocapture needs robust guidance; the 2009 paper showed NN feasibility; this work
  benchmarks NN vs modern predictor-correctors AND delivers the training methodology that makes it work.
- Content: the 2009 predecessor (`articles/markdown/01_2009_AIAA_neural_guidance.md`), the gap,
  the three contributions, MSR mission framing.
- No figure. Possibly a teaser of the deployability triangle.

## §3 Problem & objective
- Claim: aerocapture = capture into target orbit via bank-angle modulation; cost = correction ΔV;
  tanks are sized off the TAIL, so the tail is the objective.
- Content: dynamics, the (energy, dynamic-pressure) corridor, MC dispersions (26 dims), the
  sizing-tail rationale (CVaR/3σ). Define capture = `ifinal==3 & ecc<1`.
- **Fig: corridor** (energy vs pdyn corridor with a nominal trajectory) — reuse report.py corridor.
- **Table: MC dispersion domains** (26 dims, levels).

## §4 Guidance schemes
- Claim: 6 classical schemes (FTC analytic apoapsis-enslavement, FNPAG numerical PC, PredGuid,
  EnergyController, EqGlide, PiecewiseConstant) + the NN (35 candidate inputs → bank).
- Content: brief each scheme; the NN input vector (engineered autoregressive inputs:
  predicted_dv1/2/3, bank-history (sin,cos), hdot/pdyn reference, ...), decoders (atan2/acos/
  scaled_pi/delta), v2 architectures (dense/GRU/LSTM/Window/Transformer/Mamba).
- **Table: scheme summary** (signed/unsigned bank, compute class, ref-dependence).
- (Joint-reference improvement to FTC belongs here OR §8 — see fig_joint_reference.)

## §5 Training methodology — THE CONTRIBUTION
- Claim: a moving MC environment converts GA from worst to best optimizer; the quartet is a system.
- §5.1 Seed strategy (Study C): fixed=worst (160.3, overfits), adaptive=best (118.0); CMA-ES flat
  (~127). Iso-compute clincher: GA rotating-vs-fixed +40 m/s at 1.14× compute; CMA-ES +0 at exact
  iso-compute → it's the seed strategy, not compute. **Fig: fig_seed_strategy.**
- §5.2 Cost transform (Study D): cubed compresses the far tail best; vindicated at n=10000.
  **Fig: fig_cost_transform.**
- §5.3 Curation shaping (Study C-sub): max-bucket > middle/random. **Fig: fig_curation.**
- §5.4 Allocation (Study F): many gens × few sims/gen beats balanced (n=2 @ many gens dominates);
  few-sims noise bought out by more non-stationary diversity. **Fig: fig_training_n_sims.**
- Methodology note: training is compute-bound, not overfitting-bound (the non-stationary objective
  never converges) — the val-RMS plateau figure (515 vs 972, headline cells' logs).
  **Fig: fig_plateau** (NEW — val RMS vs gen for dense_515 vs dense_972, from the headline cells' jsonl).

## §6 Optimizer & dimensionality
- Claim: GA + a wide-enough population is the right optimizer; it scales where CMA-ES degrades.
- Study A (02): GA best @150/@300, **GA@60 COLLAPSES at 3998 params** (166.3 — n_pop must scale
  with dim); islands budget-robust. Study 03: CMA-ES degrades at high dim (and self-terminates).
- **Fig: fig_optimizer** (optimizer × budget, dv with the GA@60 collapse visible).
- Stat note: tight ties (GA@150 vs @300) reported "indistinguishable" off the σ_run from 10c
  (exp-11 NOT run — its mean-σ_run is obsolete; see §discussion).

## §7 Architecture — the headline result
- Claim: engineered inputs flatten the bulk; internal STATE wins the sizing tail. Deployed = Mamba_962.
- §7.1 Pareto + capability floor (sweep, 09): all archs 100% capture; dense sweet spot ~515 beats
  3998 (within-family); transformer worst (attention overhead forces tiny d_model); no collapse to
  102 params. **Fig: fig_pareto** (params vs dv_p99 per family, log-x; 2nd panel capture-vs-params floor).
- §7.2 The tail reversal (10b/10c) — THE headline figure. At the headline allocation (n=2/512/
  20000), 3-seed σ_run far-tail: **Mamba_962 CVaR99.9 124.5 < LSTM 129.2 < DENSE 139.2**; max 127.6
  < 132.4 < 159.0. Both recurrent beat dense beyond σ_run (LSTM max [126-138] non-overlapping with
  dense [146-184]). Equal-capacity control: mamba_962 vs dense_972 (~960 params) −8.7 CVaR99.9 →
  architecture, not size. **Fig: fig_arch_tail** (NEW — σ_run box/whisker of CVaR99.9 + max for
  dense/mamba/lstm, the deciding figure).
- §7.3 Why (mechanism): val RMS near-identical (dense 1.326e6 ≈ mamba 1.331e6) but deployed tail
  differs → training loss ≠ sizing tail; state handles the hardest scenarios. Ablation shows both
  lean on the engineered autoregressive inputs (redundant in the bulk). **(ties to §9.)**
- (Optional) §7.x decoder variants (output_param): **fig_output_param** (atan2 vs scaled_pi vs delta).

## §8 Classical vs NN — the deployability triangle
- Claim: NN wins the nominal sizing tail at the fast compute class; joint-FTC is the robust fallback.
- §8.1 Joint-reference recovers FTC (Study E, 07): FTC 170.7 → 126.2 mean / 142.9 CVaR95 (−44 m/s);
  the reference WAS FTC's weakness. Best classical = joint-FTC ≈ FNPAG on accuracy, analytic/fast.
  **Fig: fig_joint_reference.**
- §8.2 The 3-way (NN / joint-FTC / FNPAG): accuracy NN > joint-FTC ≈ FNPAG; far-tail CVaR99.9 NN
  124.5 vs ~164/165 (~40 m/s); paired nn_vs_jointftc −16.4 mean / −27.6 CVaR95 (100.0% win).
  **Fig: fig_classical_vs_nn** (compute-vs-CVaR99.9 scatter: NN-mamba, NN-dense, FTC, joint-FTC, FNPAG).
- §8.3 Compute (5b): NN-mamba 3.68 / NN-dense 2.40 / FTC 1.25 / FNPAG 86.1 ms/sim (NN 23× < FNPAG).
- §8.4 Robustness (5c) — HONEST CAVEAT: off-nominal (9M high-dispersion) the analytic joint-FTC is
  MOST robust (capture drop 5.5% vs NN 9.9%; CVaR95 inflation +197 vs +402). NN wins NOMINAL sizing,
  joint-FTC generalizes better. **Fig: fig_robustness** (NEW — capture drop + CVaR95 inflation bars).

## §9 What the NN uses (ablation / interpretability)
- Claim: the deployed NN leans on the engineered autoregressive inputs — the reason recurrence is
  redundant in the bulk.
- Mamba ablation: eccentricity_excess (+3.81), hdot_nominal (+3.58), pdyn_error (+2.96),
  predicted_dv2/3. Input-report: no failure tail (residual DV is irreducible scenario noise).
- **Fig: fig_ablation** (per-input cost-delta bars for Mamba_962).

## §10 Discussion / limitations / future work
- Off-nominal robustness gap (NN trained on medium regime) → widen training dispersion = future work.
- Stateful runtime (Mamba 1.5× dense) — dense efficiency reference if compute-bound.
- exp-11 (optimizer mean-σ_run) not run; tail-σ_run from 10c is what calibrates the headline.
- Pruning/quantization: NO clean campaign study exists (only `legacy/` pruned cells, pre-fix) →
  scope as FUTURE WORK (deploy-size reduction of the Mamba head), no figure.

## §11 Conclusion
- The methodology (moving environment + GA) and the architecture finding (state wins the tail) +
  the deployable Mamba_962 that beats classical at the metric that sizes the mission.

---

## Figure inventory (→ Task 19)
| fig | section | data source | status |
|---|---|---|---|
| fig_corridor | §3 | report.py corridor / a nominal MC | reuse |
| fig_seed_strategy | §5.1 | results.json (seed_strategy cells) | build |
| fig_cost_transform | §5.2 | results.json (cost_transform/* + ga_300) | build |
| fig_curation | §5.3 | results.json (curation_shaping/*) | build |
| fig_training_n_sims | §5.4 | results.json (training_n_sims/*) | build |
| **fig_plateau** | §5 | headline cells' run.jsonl.gz (val RMS vs gen) | build (NEW) |
| fig_optimizer | §6 | results.json (optimizer_budget/* + dimensionality) | build |
| fig_pareto | §7.1 | bundle architecture_sweep/* parquets + manifest | build |
| **fig_arch_tail** | §7.2 | far_tail_eval.json (dense/mamba/lstm s1-s3) | build (NEW — headline) |
| fig_output_param | §7.x | results.json (output_param/* vs dense_p515_ga) | build (optional) |
| fig_joint_reference | §8.1 | results.json (joint_reference/* vs classical) | build |
| fig_classical_vs_nn | §8.2 | far_tail_eval.json + compute_benchmark.json | build (headline) |
| **fig_robustness** | §8.4 | robustness_stress.json | build (NEW) |
| fig_ablation | §9 | mamba_p962_long/ablation_results.json | build |
| ~~fig_pruning_quant~~ | — | no clean study (legacy only) | DROPPED → future work |

## Tables
- T1 MC dispersion domains (§3); T2 scheme summary (§4); T3 final MC performance (all schemes:
  capture, mean, p95, CVaR95, CVaR99.9, max, heat/g violation %) — from results.json `runs`;
  T4 paired comparisons (nn_vs_*, headline_vs_*) with dMean + dP95 + dCVaR95 + win% + p (results.json `paired`).

## Open decisions for the author
1. ~~fig_pruning_quant~~ RESOLVED: no campaign study (legacy cells only) → dropped, future work.
2. Section order: is §5 (methodology) before or after §7-8 (results)? Spine says methodology is the
   lead contribution → §5 early. Alternative: results-first (§7-8) then "how we trained it" (§5).
3. Headline framing: lead the abstract with the ARCHITECTURE result (Mamba beats classical at the
   tail) or the METHODOLOGY (moving environment)? Both are contributions; pick the spine lead.
4. dense_515 efficiency-reference: a full row everywhere, or a footnote? (It's the parameter-
   efficiency + GA-dimensionality story.)
