// =============================================================================
// PAPER SKELETON -- NN aerocapture guidance (follow-up to Gelly & Vernis 2009).
// Fill in the PROSE (the #todo[...] blocks). Figures + locked numbers are wired.
// Compile:  typst compile articles/paper/paper.typ   (run from repo root)
// Structure + claims + numbers: articles/paper/OUTLINE.md. Voice:
// articles/markdown/05_authorial_voice_and_style.md. State: paper_resume.md.
//
// THREE AUTHOR DECISIONS (see OUTLINE.md "Open decisions"):
//  (1) section order: methodology (§4 here) before results, or results-first?
//  (2) abstract lead: architecture result or methodology?
//  (3) dense_515: full row everywhere or a footnote to the efficiency story?
// The order below is methodology-leaning (the spine). Reorder freely.
// =============================================================================

#set document(title: "Neural-network aerocapture guidance, revisited")
#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 12pt)
#show link: set text(fill: blue.darken(20%))

// Placeholder helper -- visible yellow note for prose to write. Delete each as you fill it.
#let todo(body) = box(fill: yellow.lighten(70%), inset: 4pt, radius: 2pt, width: 100%)[#text(fill: red.darken(20%))[*TODO* ] #body]
// Writing-guide note (claim + numbers for the section). Keep or delete after writing.
#let guide(body) = block(fill: blue.lighten(92%), inset: 5pt, radius: 2pt, width: 100%)[#text(size: 9pt)[#body]]
#let fig(path, cap, lbl) = figure(image("figures/" + path, width: 92%), caption: cap)

#align(center)[
  #text(size: 17pt, weight: "bold")[Neural-network aerocapture guidance, revisited:\
  a recurrent policy that beats classical predictor--correctors at the tail that sizes the mission]
  #v(6pt)
  #text(size: 11pt)[Grégory Gelly #todo[co-authors, affiliations]]
]
#v(4pt)

#align(center, block(width: 90%)[
  #text(weight: "bold")[Abstract] \
  #guide[LEAD with EITHER the architecture result OR the methodology (decision 2). Key numbers,
  all on the SIZING tail (the metric that sizes propellant = mission cost): the deployed
  962-param recurrent (Mamba) policy reaches fresh-pool CVaR#sub[95] *115.2* m/s and far-tail
  CVaR#sub[99.9] *124.5*; it beats the best classical (joint-FTC) by *#sym.minus 16.4* m/s mean /
  *#sym.minus 27.6* CVaR#sub[95] (paired, 99.9% win), at 3.68 ms/sim (23#sym.times faster than
  FNPAG). The training methodology -- a non-stationary (adaptive-seed) MC environment -- converts a
  genetic algorithm from the *worst* optimizer (160.3 m/s under fixed seeds) to the *best* (118.0).
  Honest caveat: off-nominal, the analytic joint-FTC is more robust.]
  #todo[write the abstract, ~200 words]
])
#v(6pt)

= Introduction
#guide[Aerocapture needs robust guidance; Gelly & Vernis (2009) showed NN feasibility; this work
benchmarks NN vs modern predictor--correctors AND delivers the training methodology that makes it
work. THREE contributions: (1) moving-environment training methodology; (2) architecture --
engineered inputs flatten the bulk, internal STATE wins the sizing tail; (3) the NN beats the best
classical at the fast compute class. MSR mission framing.]
#todo[intro prose + the three contributions as a bulleted list]

= Problem and objective
#guide[Aerocapture = capture into target orbit via bank-angle modulation; cost = correction #sym.Delta v;
tanks are sized off the TAIL, so the tail is the objective (CVaR / 3#sym.sigma). Dynamics, the
(energy, dynamic-pressure) corridor, 26-dim MC dispersions. Capture is defined as #raw("ifinal == 3 and ecc < 1").]
#todo[prose: dynamics, corridor, the sizing-tail rationale]
#fig("fig_corridor.svg", [DRAFT: Deployed-policy MC ensemble in the (orbital energy, dynamic
pressure) plane. The vehicle enters hyperbolic ($E>0$) and must bleed energy through the atmosphere
into a bound capture orbit ($E<0$) within the dynamic-pressure / heating corridor.], <fig-corridor>)
#todo[Table 1: the 26 MC dispersion domains + levels (from configs/missions + dispersions.rs)]

= Guidance schemes
#guide[Six classical (FTC analytic apoapsis-enslavement, FNPAG numerical PC, PredGuid,
EnergyController, EqGlide, PiecewiseConstant) + the NN (35 candidate inputs #sym.arrow.r bank).
NN input vector = engineered autoregressive inputs (predicted_dv1/2/3, bank-history (sin,cos),
hdot/pdyn reference, ...), decoders (atan2/acos/scaled_pi/delta), v2 archs
(dense/GRU/LSTM/Window/Transformer/Mamba).]
#todo[prose: brief each scheme; the NN input vector + architectures]
#todo[Table 2: scheme summary (signed/unsigned bank, compute class, reference-dependence)]

= Training methodology
#guide[THE lead contribution. A moving MC environment converts GA from worst to best optimizer;
the quartet (GA + adaptive seeds + cubed transform + max-bucket curation) is a matched system.]

== Seed strategy: the genetic algorithm needs a non-stationary objective
#guide[GA WORST under fixed (160.3 mean / 215.8 CVaR#sub[95], overfits the repeated scenarios),
RESCUED by rotating (120.0 / 144.5); CMA-ES FLAT (126.9 #sym.arrow.r 127.3). Iso-compute clincher:
GA rotating-vs-fixed +40 m/s mean at 1.14#sym.times compute; CMA-ES +0 at exact iso-compute
#sym.arrow.r it is the seed strategy, not compute.]
#fig("fig_seed_strategy.svg", [DRAFT: Fixed vs rotating seeds, per optimizer. GA is worst under
fixed seeds and rescued by rotating (#sym.minus 71 m/s on the tail); CMA-ES is unchanged because it
already resamples internally.], <fig-seed>)
#todo[prose: the seed-strategy result -- the load-bearing methodology contribution]

== Cost transform, curation, and allocation
#guide[Cubed compresses the FAR tail best (Study D, decided at n=10000). Max-bucket curation >
middle/random (Study C-sub). Allocation (Study F): many gens #sym.times few sims/gen beats balanced
(adaptive n=2 dominates, CVaR#sub[95] 117.5). Training is COMPUTE-bound, not overfitting-bound.]
#grid(columns: 2, gutter: 6pt,
  fig("fig_cost_transform.svg", [DRAFT: cost transform vs sizing tail; cubed deployed.], <fig-cost>),
  fig("fig_training_n_sims.svg", [DRAFT: allocation -- adaptive n=2 dominates.], <fig-nsims>))
#fig("fig_curation.svg", [DRAFT: curation bucket/trim; max-bucket deployed.], <fig-curation>)
#fig("fig_plateau.svg", [DRAFT: best validation RMS vs generation. Both nets keep improving for
~15--20k generations then plateau -- the non-stationary objective never overfits; the 515-param net
overtakes the 972 (more dense parameters are harder for the GA).], <fig-plateau>)
#todo[prose: cost transform, curation, allocation, the compute-bound plateau]

= Optimizer and dimensionality
#guide[GA + a wide-enough population is the right optimizer; it scales where CMA-ES degrades.
Study A: GA best at 150/300, GA at 60 COLLAPSES at 3998 params (166.3 mean / 205.6 CVaR#sub[95] --
population must scale with dimension); islands budget-robust. Tight ties (GA 150 vs 300) reported
"indistinguishable" off the #sym.sigma#sub[run] from §7 (exp-11 not run).]
#fig("fig_optimizer.svg", [DRAFT: optimizer #sym.times budget at 3998 params. GA at 60 collapses;
GA at 150 best; islands flattest.], <fig-optimizer>)
#todo[prose: the optimizer + dimensionality story]

= Architecture: the headline result
#guide[Engineered inputs flatten the BULK (all archs median ~108--112); internal STATE wins the
sizing TAIL. Deployed = Mamba_962.]

== Parameter-budget Pareto and capability floor
#guide[Sweep (n=2/5000): all archs 100% capture; dense sweet spot ~515 beats 3998 (within-family);
transformer worst (attention overhead #sym.arrow.r tiny d_model); no collapse to 102 params.]
#fig("fig_pareto.svg", [DRAFT: parameters vs sizing tail per architecture family (left); dense
capability floor, 100% capture throughout (right).], <fig-pareto>)

== The tail reversal (the deciding result)
#guide[At the headline allocation (n=2/512/20000), 3-seed #sym.sigma#sub[run] far-tail (n=10000):
Mamba_962 CVaR#sub[99.9] *124.5* < LSTM_1082 *129.2* < Dense_515 *139.2*; max 127.6 < 132.4 < 159.0.
BOTH recurrent nets beat dense beyond #sym.sigma#sub[run] (LSTM max [126,138] non-overlapping with
dense [146,184]). Equal-capacity control: mamba_962 vs dense_972 (~960 params) #sym.minus 8.7
CVaR#sub[99.9] #sym.arrow.r architecture, not size. Mechanism: val RMS near-identical (dense 1.326e6
#sym.approx mamba 1.331e6) but deployed tail differs #sym.arrow.r training loss #sym.eq.not sizing
tail; state handles the hardest scenarios.]
#fig("fig_arch_tail.svg", [DRAFT: three-seed $sigma_"run"$ on the sizing tail. Mamba is the lowest
and tightest; both recurrent nets beat the dense net beyond run-to-run variance, and all crush the
best classical (band).], <fig-archtail>)
#fig("fig_output_param.svg", [DRAFT: NN bank-decoder variants; atan2 wins.], <fig-outparam>)
#todo[prose: the Pareto, the tail reversal (THE result), the mechanism]

= Classical versus neural network
#guide[NN wins the nominal sizing tail at the fast compute class; joint-FTC is the robust fallback.]

== Joint-reference optimization recovers the predictor--correctors
#guide[Study E: joint-reference recovered FTC from 170.7 to 126.2 mean / 142.9 CVaR#sub[95]
(#sym.minus 44 m/s) -- the reference WAS FTC's weakness. Best classical = joint-FTC #sym.approx FNPAG
on accuracy, analytic/fast.]
#fig("fig_joint_reference.svg", [DRAFT: fixed vs joint reference for the three reference-tracking
schemes; joint-FTC recovers most of the gap.], <fig-joint>)

== The deployability triangle, compute, and robustness
#guide[3-way (NN / joint-FTC / FNPAG): accuracy NN > joint-FTC #sym.approx FNPAG; far-tail
CVaR#sub[99.9] NN 124.5 vs ~164/165 (~40 m/s); paired NN vs joint-FTC #sym.minus 16.4 mean /
#sym.minus 27.6 CVaR#sub[95]. Compute (5b): NN-mamba 3.68 / NN-dense 2.40 / FTC 1.25 / FNPAG 86.1
ms/sim (NN 23#sym.times < FNPAG). Robustness (5c) -- HONEST CAVEAT: off-nominal the analytic
joint-FTC is MOST robust (capture drop 5.5% vs NN 9.9%).]
#grid(columns: 2, gutter: 6pt,
  fig("fig_classical_vs_nn.svg", [DRAFT: deployability -- per-sim compute vs sizing tail. Both NN
  points are lower-left (tighter tail AND cheaper than FNPAG).], <fig-classical>),
  fig("fig_robustness.svg", [DRAFT: off-nominal stress -- joint-FTC is the most robust to
  distribution shift; the medium-trained NN generalizes less well.], <fig-robust>))
#todo[prose: the 3-way comparison, compute, and the honest robustness caveat]
#todo[Table 3: final MC performance, all schemes (capture, mean, p95, CVaR95, CVaR99.9, max,
violation %) -- auto-fill from articles/paper/data/results.json `runs`.]
#todo[Table 4: paired comparisons (#raw("nn_vs_*") , #raw("headline_vs_*")) -- dMean, dP95, dCVaR95,
win-rate, p -- from results.json #raw("paired").]

= What the network uses
#guide[The deployed NN leans on the engineered autoregressive inputs -- the reason recurrence is
redundant in the bulk. Mamba ablation: eccentricity_excess (+3.81), hdot_nominal (+3.58), pdyn_error
(+2.96), predicted_dv2/3. Input-report: no failure tail (residual DV is irreducible scenario noise).]
#fig("fig_ablation.svg", [DRAFT: per-input cost increase when each input is zeroed, for the deployed
Mamba policy. The engineered reference / autoregressive inputs dominate.], <fig-ablation>)
#todo[prose: the ablation / interpretability]

= Discussion and limitations
#guide[Off-nominal robustness gap (NN trained on medium regime) #sym.arrow.r widen training
dispersion = future work. Stateful runtime (Mamba 1.5#sym.times dense) -- dense efficiency reference
if compute-bound. Pruning/quantization of the Mamba head = future work (no clean study). exp-11
(optimizer mean-#sym.sigma#sub[run]) not run; the tail-#sym.sigma#sub[run] from §7 calibrates the headline.]
#todo[prose: limitations + future work]

= Conclusion
#todo[prose: the methodology (moving environment + GA) + the architecture finding (state wins the
tail) + the deployable Mamba_962 that beats classical at the metric that sizes the mission]

// #bibliography("refs.bib")   // uncomment once refs.bib exists (Gelly&Vernis 2009, FNPAG/Lu, etc.)
#todo[bibliography: create articles/paper/refs.bib (Gelly & Vernis 2009; Lu FNPAG; Ng-Harada-Russell
PBRS; Jozefowicz LSTM init; etc.) and uncomment the #bibliography line above]
