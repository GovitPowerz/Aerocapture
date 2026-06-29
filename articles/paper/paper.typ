// =============================================================================
// NN aerocapture guidance, revisited -- follow-up to Gelly & Vernis 2009.
// Compile (from repo ROOT, so figure paths resolve):
//   typst compile articles/paper/paper.typ articles/paper/paper.pdf
// Structure + locked numbers: articles/paper/OUTLINE.md. Data: articles/paper/data/.
// Authorial voice: articles/markdown/05_authorial_voice_and_style.md.
// Section order: methodology-first (the spine). Abstract leads with the architecture
// result. dense_515 carried as a full efficiency-reference row throughout.
// =============================================================================

// NeurIPS-style single-column layout (cf. Vaswani et al. 2017): Times body,
// STIX Two Math for equations, a narrow 5.5in text block, bold numbered headings.
#set document(title: "Neural-network aerocapture guidance, revisited", author: "Grégory Gelly")
#set page(paper: "us-letter", margin: (x: 1.5in, top: 1in, bottom: 1.25in), numbering: "1")
#set text(font: "Times New Roman", size: 10pt)
#show math.equation: set text(font: "STIX Two Math")
#set par(justify: true, leading: 0.58em, spacing: 0.7em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading: set text(weight: "bold")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show link: set text(fill: blue.darken(20%))

// Figure helper: include from figures/, attach the caption and the label.
#let fig(path, cap, lbl) = [#figure(image("figures/" + path, width: 100%), caption: cap)#lbl]

#v(0.15in)
#align(center)[
  #text(size: 16pt, weight: "bold", hyphenate: false)[Neural-network aerocapture guidance, revisited:\
  a recurrent policy that beats classical predictor--correctors at the tail that sizes the mission]
  #v(16pt)
  #text(size: 12pt)[Grégory Gelly]
  #v(1pt)
  #text(size: 10pt)[#link("mailto:gregory.gelly@gmail.com")[gregory.gelly\@gmail.com]]
  #v(3pt)
  #text(size: 9pt, style: "italic")[Preprint, 2026]
]
#v(18pt)

#align(center)[#text(size: 11pt, weight: "bold")[Abstract]]
#v(2pt)
#pad(x: 0.45in)[
  #set par(justify: true, leading: 0.55em)
  #text(size: 9.5pt)[In 2009 we showed that a single-hidden-layer feed-forward network, trained by a
  genetic algorithm, could fly the aerocapture of a Mars Sample Return vehicle more efficiently than
  a Cerimele--Gamble feedback law, and we closed that work by asking for the obvious next step: a
  comparison against predictor--corrector guidance. This paper answers it. We train stateful neural
  guidance policies -- spanning recurrent, gated, attention, and selective state-space cells -- and
  benchmark them, on identical Monte-Carlo scenarios drawn from a bit-validated simulator, against
  six classical schemes including a numerical predictor--corrector (FNPAG) and a reference-tracking
  feedback law (FTC). Because aerocapture propellant is sized off the worst-case correction
  $Delta v$, we lead every comparison with the tail of its distribution, not the mean. A 962-parameter
  recurrent (Mamba) policy reaches a far-tail $"CVaR"_(99.9)$ of $124.5$ m/s and a fresh-pool
  $"CVaR"_95$ of $115.2$ m/s at $100%$ capture; it beats the best classical scheme (FTC with a
  co-optimized reference) by $16.4$ m/s in mean and $27.6$ m/s at $"CVaR"_95$ -- better on every one
  of $1000$ paired scenarios -- while running at $3.68$ ms per simulation, $23 times$ faster than
  FNPAG. The result rests on a training methodology that is itself a contribution: a non-stationary,
  adaptive-seed Monte-Carlo environment turns the genetic algorithm from the *worst* optimizer under
  fixed scenarios ($160.3$ m/s mean) into the *best* ($118.0$). We report one honest caveat -- under a
  deliberately harsher off-nominal regime the analytic reference-tracking law generalizes better than
  the medium-trained network -- and we trace the architecture finding to its mechanism: engineered
  autoregressive inputs flatten the median across all cell types, but genuine internal state is what
  compresses the extreme tail that sizes the tanks.]
]
#v(18pt)

= Introduction

Aerocapture replaces a propulsive orbit-insertion burn with a single pass through a planet's
atmosphere: the vehicle enters on a hyperbolic trajectory and bleeds its excess energy through
aerodynamic drag, modulating only the bank angle to steer the lift vector and arrive at a bound
target orbit. The maneuver is attractive precisely because it is propellant-free, but it is also
unforgiving. The control authority is proportional to dynamic pressure, the entry corridor is
narrow, and the dispersions -- atmospheric density first among them -- are large. There is no
closed-form optimal bank-angle law, so the classical practice is to make strong simplifying
assumptions (a decoupling of in-plane and out-of-plane motion, a reference trajectory built on
apoapsis control alone) in order to obtain a tractable feedback or predictor--corrector law. These
simplifications are exactly what costs correction propellant downstream.

Neural networks are especially valuable for guidance problems that admit no analytic solution, and
aerocapture is a textbook case. In 2009 we showed that a feed-forward network with a single hidden
layer, trained by a genetic algorithm without any reference trajectory or input--output pairs, could
perform the aerocapture of a Mars Sample Return (MSR) vehicle at a mean correction cost of
$116.7$ m/s against $144.8$ m/s for a Cerimele--Gamble-derived feedback law -- a $19%$ reduction,
approaching the $113$ m/s periapsis-raise floor @gelly2009neural, itself building on earlier
aerocapture-corridor and soft-landing guidance studies @vernis2004tradeoff @gelly2007guidance. The
mechanism was that the network
learned the in-plane and out-of-plane logic *jointly*, where the classical law treats them
separately. We closed that paper with an explicit hook: "the next step would be to extend our work
on the aerocapture to skip-entry missions and evaluate the performance of neural guidance compared
to classic algorithms such as the predictor--corrector schemes." This paper is that next step,
seventeen years later.

In the interval we carried the same evolutionary-training philosophy to recurrent networks for
speech, where a coordinated-gate LSTM trained by quantum-behaved particle-swarm optimization
@sun2004qpso, with a divide-and-conquer initialization and task-aligned differentiable losses, became
the working method @gelly2015mwe @gelly2016divide @gelly2017angular. Stateful cells, swarm training, smart
initialization, and custom losses are precisely the machinery this paper brings back to where it
started. With their innate ability to exploit the trajectory history, recurrent and selective
state-space policies are natural candidates for a guidance law that must anticipate the atmospheric
pass rather than merely react to the current state.

We make three contributions:

+ *A training methodology, not just a policy.* We show that a genetic algorithm is the *worst*
  optimizer for this problem under a fixed set of Monte-Carlo scenarios -- it overfits the repeated
  cases -- and the *best* under a non-stationary, adaptive-seed environment that keeps moving the
  scenarios beneath it. Switching the seed schedule from fixed to adaptive moves the mean correction
  cost from $160.3$ to $118.0$ m/s, the single largest effect in the campaign. The moving
  environment, a tail-weighting cost transform, and hardest-case seed curation form one matched
  system.

+ *An architecture finding with a mechanism.* Across dense, gated (GRU), recurrent (LSTM), windowed,
  attention (Transformer), and selective state-space (Mamba) cells, the engineered autoregressive
  inputs flatten the *median* correction cost to a common $108$--$112$ m/s. The separation appears only
  on the *tail*: a $962$-parameter Mamba policy beats the best dense network beyond run-to-run
  variance at the $99.9$th-percentile depth where the propellant tanks are sized. Training loss does
  not predict the sizing tail; internal state does.

+ *The first systematic head-to-head of neural versus predictor--corrector aerocapture guidance.* On
  identical dispersions, the deployed network beats the best classical scheme by $16.4$ m/s in mean
  and $27.6$ m/s at $"CVaR"_95$, at a per-simulation compute cost $23 times$ below the numerical
  predictor--corrector -- with the honest caveat that the analytic law is more robust off-nominal.

All of this rests on a high-fidelity simulator -- EKF navigation, altitude-dependent winds,
Gauss--Markov density perturbations, $J_2$/$J_3$/$J_4$ gravity, thermal limits, and adaptive
integration -- validated bit-for-bit against a legacy reference across $725$ time steps. The next
section formalizes the aerocapture problem and the sizing-tail objective; Section 3 describes the
guidance schemes; Section 4 presents the training methodology; Sections 5--7 give the optimizer,
architecture, and classical-versus-neural results; and Section 8 reports what the deployed network
actually uses.

= Problem and objective

== Dynamics and the aerocapture corridor

We study the same robotic MSR mission as the 2009 work. The vehicle reaches the atmospheric entry
interface at an altitude of $130$ km with a relative velocity of $5687$ m/s, a flight-path angle of
$-10.81 degree$, and an azimuth of $38.04 degree$; it carries a mass of $1089$ kg on a $14.7$ m#super[2]
reference area and is statically trimmed at a fixed angle of attack, so the only control is the bank
angle $mu$, slewed at up to $15 degree$/s. The target is a $500 times 11$ km capture orbit at
$50 degree$ inclination. Energy is dissipated by drag during a single atmospheric pass; the bank
angle rotates the lift vector to manage how much energy is shed and in which plane.

The natural state space for the maneuver is the (orbital energy, dynamic pressure) plane
(@fig-corridor). The vehicle enters with positive orbital energy (hyperbolic, $E approx +4.9$ MJ/kg)
and must exit with negative energy (bound, $E approx -5.9$ MJ/kg) without either skipping back out --
a hyperbolic escape -- or descending so deep that it crashes or violates a thermal limit. The set of
trajectories that capture is bounded by two constant-bank profiles: an overshoot boundary (full
lift-up, the limit against escape) and an undershoot boundary (full lift-down, the limit against
crash). A practical *restricted corridor* tightens these to the bank profiles that land within a
bounded apoapsis error of the target. Control authority is proportional to dynamic pressure, so the
guidance has the most leverage deep in the pass and almost none on the thin entry and exit legs.
The vehicle must respect a peak heat-flux limit of $200$ kW/m#super[2], a $4$ g load limit, a dynamic-
pressure limit of $1.08$ kPa, and an integrated heat-load limit of $25$ MJ/m#super[2].

#fig("fig_corridor.svg", [Deployed-policy Monte-Carlo ensemble in the (orbital energy, dynamic
pressure) plane. The vehicle enters hyperbolic ($E > 0$) and must bleed energy through the atmosphere
into a bound capture orbit ($E < 0$) while staying inside the dynamic-pressure and heating corridor.
The shaded band is the restricted corridor; the heavy line is the undispersed nominal.], <fig-corridor>)

== The objective: the tail that sizes the mission

Performance is measured by the *correction $Delta v$* -- the propellant cost to repair the captured
orbit to the target, summed over the apoapsis-lowering, periapsis-raising, and inclination-change
burns. At atmosphere exit the periapsis always sits inside the atmosphere and must be raised, so the
correction cost has an irreducible floor of roughly $113$ m/s set by the nominal periapsis raise; a
guidance law cannot do better than deliver the vehicle to that floor across all dispersions. We
define a run as a *capture* when it terminates in a bound orbit (#raw("ifinal == 3") and eccentricity
$< 1$) and compute $Delta v$ over captured runs only.

The point we want to make sharply, because it governs every comparison in this paper, is that the
*mean* correction cost is operationally almost irrelevant. Aerocapture propellant tanks are sized
for the worst credible case, conventionally the $3 sigma$ design point ($approx$ the $99.87$th
percentile), not the median. Two guidance laws with the same mean but different tails are not equally
good: the one with the heavier tail forces larger tanks and a heavier, more expensive mission. We
therefore treat the *tail* of the $Delta v$ distribution as the objective, and report the conditional
value-at-risk -- the mean cost in the worst $(1-alpha)$ fraction of cases, $"CVaR"_alpha$
@rockafellar2000cvar -- at $alpha = 95%$ and, for sizing decisions, at the far-tail depth
$"CVaR"_(99.9)$, together with the $95$th and $99$th percentiles and the sample maximum. The far tail
cannot be estimated from a $1000$-case ensemble (one to ten samples beyond $p_(99.9)$), so every
sizing number in this paper is computed on a dedicated $n = 10\,000$ pool, training-disjoint by
construction. We lead with the tail and relegate the mean to a footnote.

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, center, left, left),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header(
      [*Domain*], [*Dims*], [*Distribution*], [*Dispersion (controlled regime)*],
    ),
    [Entry state], [6], [Gaussian], [altitude $plus.minus 0.3$ km, velocity $plus.minus 3$ m/s, flight-path/azimuth $plus.minus 0.15$--$0.3degree$ ($3sigma$; medium)],
    [Atmospheric density], [1], [Uniform], [$plus.minus 50%$ multiplicative bias (medium)],
    [Aerodynamics], [3], [Uniform], [drag $plus.minus 5%$, lift $plus.minus 10%$, angle of attack $plus.minus 1degree$ (medium)],
    [Navigation errors], [7], [Gaussian], [position $approx plus.minus 2$ km, velocity $plus.minus 1.2$ m/s, drag-accel $plus.minus 0.3$ m/s#super[2] ($3sigma$; medium)],
    [Mass], [1], [Uniform], [$plus.minus 1%$ (medium)],
    [Vehicle], [2], [Uniform], [reference area $plus.minus 2%$, max bank rate $plus.minus 10%$ (medium)],
    [Pilot dynamics], [3], [Uniform], [time constant / damping / frequency $plus.minus 10%$ (medium)],
    [Nav-filter gain], [1], [Gaussian], [$plus.minus 0.3$ absolute ($3sigma$; medium)],
    [Winds], [2], [Uniform], [speed $times [0.7, 1.3]$, direction $plus.minus 5degree$ (low)],
    [Density perturbation], [OU], [Gauss--Markov], [correlation time $120$ s, $5%$ RMS, time-varying (low)],
  ),
  caption: [The dispersion model. Twenty-six static draws across ten domains, plus a time-varying
  Gauss--Markov density perturbation. The controlled-study regime uses medium presets except for the
  winds and density perturbation (low). Gaussian dispersions are quoted at $3sigma$; uniform
  dispersions at their full range.],
) <tbl-dispersions>

The mission is flown under $26$ dispersed parameters across ten domains, summarized in
@tbl-dispersions, plus a time-varying Gauss--Markov (Ornstein--Uhlenbeck) density perturbation that
evolves during each run. The controlled-study regime uses medium presets for the initial-state,
atmospheric, aerodynamic, navigation, mass, vehicle, pilot, and navigation-filter domains and low
presets for the winds and the density perturbation. The atmospheric density bias alone spans
$plus.minus 50%$ -- it is the dominant driver of apoapsis error, and a guidance law blind to it
cannot reject it. The winds follow a parametric Mars profile @forget1999mars and the density
perturbation an Ornstein--Uhlenbeck process layered on the static bias. Draws are generated by
Latin-hypercube sampling for space-filling coverage.

= Guidance schemes

We benchmark the neural policies against six classical schemes that span the spectrum of aerocapture
guidance, from analytic feedback through numerical prediction. All schemes share the same simulator,
the same navigation chain, and -- where they need one -- the same kind of reference trajectory, so
the comparison isolates the guidance law itself (@tbl-schemes).

== Classical schemes

*Feedback Trajectory Control (FTC)* is the Cerimele--Gamble-derived scheme from the 2009 study
@cerimele1985simplified @gelly2009neural and our continuity baseline. It builds a virtual reference
trajectory on apoapsis control alone -- a constant-bank profile that reaches the target apoapsis
without inclination control, tabulated as $cos mu_"ref"$, $dot(h)_"ref"$, and $q_"ref"$ versus orbital
energy -- and enslaves the commanded bank to it through a proportional law on the altitude-rate and
dynamic-pressure errors,
$ cos mu_"com" = cos mu_"ref" + G_(dot(h)) (dot(h) - dot(h)_"ref") / q + G_q (q - q_"ref") / q, $ <eq-ftc>
where $q$ is the dynamic pressure and $G_(dot(h))$, $G_q$ are the altitude-rate and dynamic-pressure
feedback gains; the command is clamped to $0degree$ or $180degree$ when $|cos mu_"com"| > 1$. It decouples in-plane (apoapsis) from
out-of-plane (inclination) motion, the latter handled by a roll-reversal logic that flips the bank
sign whenever the projected inclination error leaves a velocity-referenced corridor. FTC is analytic,
fast, and -- as we will show -- only as good as its reference.

*FNPAG* is Lu's fully numerical predictor--corrector @lu2015fnpag @lu2014predictor. Each guidance cycle
it integrates the equations of motion forward to atmosphere exit and bisects the constant capture-bank
magnitude until the predicted osculating exit apoapsis matches the target, scaling the onboard
atmosphere by the navigation-estimated density factor so the predictor tracks the measured atmosphere
rather than a nominal model. It is the most accurate classical scheme and, at roughly eleven forward
integrations per replan, by far the most expensive.

*PredGuid* is the Apollo/Shuttle-heritage drag-tracking law @bairstow2006reentry: it tracks a
drag-versus-energy reference profile with negative feedback. *EnergyController* tracks an energy-
dissipation reference through dynamic-pressure and altitude-rate feedback. *Equilibrium glide* holds
the equilibrium-glide condition with altitude-rate damping and a velocity bias, using the
navigation-filtered density rather than a static table. *Piecewise-constant* flies an $N$-segment
constant-bank profile; it is the simplest scheme, produces the reference trajectory and corridor the
other schemes consume, and -- like the full-neural network -- emits a *signed* bank, so it bypasses
the shared roll-reversal, exit, and thermal-limiter logic.

== Neural guidance

The neural policy maps an observation vector to a bank command. We generalize the 2009 single-hidden-
layer feed-forward network -- five hand-picked inputs (orbital energy, eccentricity, inclination,
velocity, non-gravitational acceleration) and a two-output bank decoder -- in two directions.

First, the *inputs*. The modern policy draws from a $35$-element candidate vector behind a learned
input mask. Sixteen entries carry the instantaneous orbital, aerodynamic, and thermal state; the
remaining nineteen are *engineered, mostly autoregressive* signals that give the policy temporal
context without requiring it to integrate the history itself: reference-trajectory interpolations
($dot(h)$ and $q$ at the current energy), a closed-form exit-bank teacher signal, seam-free
$(sin, cos)$ encodings of the recent bank history, the periapsis altitude, and -- most important, as
the ablation in Section 8 shows -- three *predicted correction-$Delta v$* components evaluated on the
current osculating orbit (the energy-closing, periapsis-correction, and plane-change burns). These
are smooth, causal, and cost-aligned: they tell the network what the maneuver would cost if it stopped
now. The deployed atan2 policies use a $17$-input subset of this vector.

Second, the *bank decoder*. The 2009 paper read the bank from a two-element output through
$ mu = "atan2"(o_1, o_2), $ <eq-atan2>
which we retain as the default (#raw("atan2_signed")). It wastes half its range when the magnitude
alone is needed, so for magnitude-only policies we add $mu = arccos(tanh(o_1))$, a single output
mapping smoothly onto $[0, pi]$. Two further single-output decoders attack the $plus.minus pi$ wrap
seam that a raw angle output suffers near $mu = pi$: $mu = "wrap"_pi(n pi tanh(o_1))$ (#raw("scaled_pi"),
which pushes the seam out of the operating region) and $mu = "wrap"_pi(mu_"prev" + Delta_max tanh(o_1))$
(#raw("delta"), a bounded increment on the previous realized bank).

Third, the *cell type*. Where 2009 had one hidden layer, we span a family of architectures behind a
common runtime: dense feed-forward, gated recurrent (GRU) @cho2014gru, long short-term memory (LSTM)
@hochreiter1997lstm, a fixed windowed buffer, a causal-attention Transformer block @vaswani2017attention,
and a selective state-space (Mamba) core @gu2023mamba. All are trained and deployed through the same
bit-validated Rust runtime, and all are sized so that the comparison across cell types holds the
parameter budget roughly fixed -- the fair-comparison guardrail we have always insisted on.

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, left, center, center, left),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header(
      [*Scheme*], [*Bank command*], [*Reference*], [*Compute*], [*Heritage / note*],
    ),
    [FTC], [magnitude + roll reversal], [yes], [fast], [Cerimele--Gamble apoapsis enslavement],
    [FNPAG], [magnitude], [no], [slow], [onboard forward integration, bisection corrector],
    [PredGuid], [magnitude + roll reversal], [yes], [fast], [Apollo/Shuttle drag tracking],
    [Energy controller], [magnitude + roll reversal], [yes], [fast], [energy-dissipation tracking],
    [Equilibrium glide], [magnitude + roll reversal], [no], [fast], [equilibrium-glide condition, nav density],
    [Piecewise constant], [signed ($N$ segments)], [no], [fast], [produces reference + corridor],
    [Neural network], [signed or magnitude], [no], [fast], [35-input candidate vector, stateful cells],
  ),
  caption: [The benchmarked schemes. "Reference" marks dependence on a tabulated reference trajectory;
  "Compute" classes are quantified in @sec-deployability (fast: $1$--$4$ ms/sim; slow: $86$ ms/sim).
  Signed-bank schemes (full-neural, piecewise-constant) bypass the shared roll-reversal, exit-phase,
  and thermal-limiter logic.],
) <tbl-schemes>

= Training methodology

The training methodology is the load-bearing contribution. The policies are trained without any reference
trajectory or input--output pairs: each candidate network is simulated on a batch of dispersed
Monte-Carlo scenarios, and its fitness is the resulting correction-$Delta v$ cost (with soft
constraint penalties), so the optimizer searches directly on mission performance. The same recipe
trained the 2009 networks. What changed -- and what makes the modern policies work -- is the
realization that *how the scenario batch is chosen, generation to generation, matters more than the
optimizer or the architecture*. The right environment is non-stationary: it keeps moving the
scenarios beneath the population so that no individual can memorize a fixed set. Four design choices
-- a genetic algorithm, an adaptive (moving) seed schedule, a tail-weighting cost transform, and a
hardest-case seed curation -- form one matched system. We take them in turn.

== Seed strategy: the genetic algorithm needs a non-stationary objective

The single largest effect in the entire campaign is the seed strategy. We compared three: *fixed*
(a deterministic scenario batch, identical every generation), *rotating* (a fresh random batch every
generation), and *adaptive* (a curated batch, refreshed on a validated improvement or periodically,
drawn from the cost distribution of the best individuals). Under fixed seeds the genetic algorithm is
the *worst* optimizer we tested -- mean correction cost $160.3$ m/s, $"CVaR"_95$ $215.8$ m/s -- because
it overfits the repeated scenarios, evolving a policy that is excellent on those particular draws and
mediocre elsewhere. Rotating the seeds rescues it outright: $120.0$ m/s mean, $144.5$ m/s $"CVaR"_95$,
a $40$ m/s drop in the mean and a $71$ m/s drop on the tail. Adaptive curation tightens it a little
further, to $118.0$ m/s.

The control that makes this a statement about the *objective* and not about compute is CMA-ES. Run
on the identical fixed-versus-rotating change, CMA-ES does not move -- $126.9 arrow.r 127.3$ m/s --
because it already resamples internally through its covariance adaptation, so a moving scenario batch
tells it nothing new (@fig-seed). The genetic algorithm, by contrast, has no such internal
re-randomization: a fixed objective lets selection converge onto the quirks of the fixed batch.
Rotating costs marginally more compute, but CMA-ES given the same marginal compute gains nothing, so
the lever is the non-stationarity of the objective, not the extra evaluations. The practical
consequence is striking: under a fixed objective one would deploy CMA-ES and discard the genetic
algorithm; under a moving objective the genetic algorithm becomes the best optimizer in the study.
The moving environment does not make the genetic algorithm *robust* to a moving objective -- it makes
the genetic algorithm *need* one.

#fig("fig_seed_strategy.svg", [Fixed versus rotating seeds, per optimizer. The genetic algorithm is
the worst optimizer under fixed seeds and is rescued by rotating them ($-40$ m/s mean, $-71$ m/s at
$"CVaR"_95$); CMA-ES is essentially unchanged because it already resamples internally. The lever is
the non-stationary objective, not the extra compute.], <fig-seed>)

== Cost transform, curation, and allocation

The same worst-case-leaning logic that makes us report the tail also shapes how we train. Because the
mission is sized off the far tail, we apply a monotonic *cost transform* to each per-simulation cost
before aggregating, so that the optimizer feels expensive scenarios more sharply. The transform is
ranking-neutral for a deterministic argmin, but under the noisy, non-stationary objective it changes
which individuals survive selection. Evaluated at the depth that matters -- a far-tail
$n = 10\,000$ pool -- the cubed transform compresses the extreme tail best ($"CVaR"_(99.9)$ $153.0$ m/s,
sample max $160.1$) against linear ($156.7$/$162.2$), square-root ($158.4$/$167.1$), squared
($162.7$/$180.9$), and logarithmic ($162.3$/$180.6$, worst across the shallow and mid tail because it
over-compresses and starves the gradient between captures -- only the squared transform edges it
worse at the very extreme). A shallower metric ($"CVaR"_95$) would have mildly favored
square-root; the deeper we look into the tail, the more the tail-weighting pays, which is exactly why
the sizing depth must decide (@fig-cost).

Seed *curation* is the same mechanism applied to the scenarios rather than the cost. At each
refresh the adaptive strategy bins the cost distribution of the best individuals into quantiles and
picks one representative seed per bin; the choice of representative is the lever. Picking the
*hardest* seed per bin (the "max" bucket) dominates the far tail ($"CVaR"_(99.9)$ $153.0$ m/s) against
the bin median ($193.9$), a random pick ($173.1$), and the *easiest* seed ($225.9$). The easiest-seed
bucket is the cautionary tale: the best mean, $117.8$ m/s, and a catastrophic worst case -- it drops
captures and reaches $245$ m/s, the canonical optimize-the-average, blow-up-the-tail failure. Trimming
the cost distribution helped nothing (@fig-curation). The cost transform
and the curation bucket are therefore one idea -- force the policy onto the hard cases -- expressed
twice.

Finally, *allocation*: given a compute budget, is it better spent on more scenarios per generation or
more generations? Under rotating seeds with a fixed generation count the sweet spot is moderate
($n_"sims" = 10$). But under the adaptive strategy with the budget spent on generations, the answer
flips hard: just $n_"sims" = 2$ per generation, run for many generations, gives the best mean
($109.9$ m/s) and $"CVaR"_95$ ($117.5$), beating $5$, $20$, and $100$ (@fig-nsims). The few-sample
noise per generation is bought back, and then some, by the far greater scenario diversity the moving
environment sees over the run. The deployed headline policy uses this allocation.

#grid(columns: 2, gutter: 6pt,
  fig("fig_cost_transform.svg", [Cost transform versus the sizing tail. The cubed transform minimizes
  the far-tail $"CVaR"_(99.9)$, the metric that sizes the mission, even though a shallow $"CVaR"_95$
  would mildly favor square-root.], <fig-cost>),
  fig("fig_training_n_sims.svg", [Allocation of the compute budget. Under the adaptive schedule, two
  scenarios per generation over many generations dominates larger per-generation batches.], <fig-nsims>))
#fig("fig_curation.svg", [Seed curation. Selecting the hardest seed per cost-CDF bin (the "max"
  bucket) compresses the far tail; the easiest-seed bucket wins the mean but blows up the worst case.], <fig-curation>)

That the optimal allocation pours the budget into generations points to the last methodological fact:
training here is *compute-bound, not overfitting-bound*. A stationary objective eventually overfits,
and one stops early. The moving objective never converges to a fixed landscape, so the validation
error keeps falling for tens of thousands of generations and then plateaus rather than degrading
(@fig-plateau). The plateau also exposes a counter-intuitive dimensionality effect that recurs in the
architecture results: the $972$-parameter dense network learns *faster* early -- more plasticity --
but the $515$-parameter network overtakes it and plateaus *lower* (validation RMS $1.326 times 10^6$
versus $1.433 times 10^6$). For the gradient-free genetic algorithm, the extra dense parameters are
more search burden than added capacity; the "more parameters, learn faster" intuition does not
transfer.

#fig("fig_plateau.svg", [Best validation RMS versus generation for the two dense reference networks.
Both keep improving for roughly ten thousand generations and then plateau -- the non-stationary
objective does not overfit. The $515$-parameter network plateaus below the $972$-parameter one:
beyond a few hundred parameters, extra dense capacity hurts the gradient-free search.], <fig-plateau>)

= Optimizer and dimensionality

Having established that the genetic algorithm is the right optimizer under a moving objective, two
questions remain: does it need a population that scales with the search dimension, and does the
optimizer choice even matter for the low-dimensional classical-gain problems? We compared the genetic
algorithm @goldberg1989genetic against CMA-ES @hansen2001cmaes, particle-swarm optimization
@kennedy1995pso and its quantum-behaved variant @sun2004qpso, differential evolution, and a
three-island heterogeneous model, on an optimizer-by-budget grid at the largest dense network
($3998$ weights) and an optimizer-by-dimension grid spanning the $26$-parameter FTC-gain problem and
the $515$-parameter dense network.

At $3998$ weights the population size is decisive (@fig-optimizer). The genetic algorithm is best at a
population of $150$ ($118.0$ m/s mean) and $300$ ($120.5$ m/s) -- the two are within run-to-run
variance, so we report them as indistinguishable -- but at a population of $60$ it *collapses* to
$166.3$ m/s. Sixty individuals cannot cover a four-thousand-dimensional weight space; selection
drifts. So the often-repeated claim that the genetic algorithm "dominates at every budget" is wrong:
at a starved population it is no better than a single restart. The corrective is simple and worth
stating plainly -- the population must scale with the search dimension. CMA-ES improves smoothly with
budget ($133.3 arrow.r 126.3 arrow.r 121.8$ m/s) but never reaches the genetic algorithm's optimum and
self-terminates on the noisy objective before exhausting a generous generation count, an asymmetry to
keep in mind for compute-matched comparisons. The three-island heterogeneous trainer (particle-swarm
+ genetic + differential evolution with periodic migration) is the most budget-robust of all,
$120$--$124$ m/s across every budget, but the well-populated single genetic algorithm edges it.

The dimensionality grid carries the more useful lesson. On the $26$-parameter FTC-gain problem every
gradient-free optimizer we ran lands in a tight band, $170$--$178$ m/s -- CMA-ES $171.7$, particle
swarm $170.9$, quantum-behaved swarm $172.9$, differential evolution $178.2$, islands $169.8$ --
statistically indistinguishable. Optimizer choice barely matters when there are only twenty-six gains
to tune. It is at neural-network dimensionality that the optimizers separate: at $515$ weights the
genetic algorithm is best ($117.4$ m/s), the islands next ($118.6$), and particle swarm worst
($129.8$), a $12$ m/s spread. The honest confound is that the $26$-parameter cell is a different
guidance scheme (FTC), not a narrowed version of the same network, so the comparison conflates
dimension with law; but the direction is clear -- the optimizer earns its keep on the high-dimensional
weight search, not on low-dimensional gain tuning.

#fig("fig_optimizer.svg", [Optimizer by population budget at $3998$ weights. The genetic algorithm is
best at populations of $150$--$300$ but collapses at $60$ (the population must scale with the search
dimension); the heterogeneous island model is the most budget-robust; CMA-ES improves with budget but
trails the genetic optimum.], <fig-optimizer>)

= Architecture: the headline result

We now sweep the cell type. The finding, which the next two subsections establish, comes in two
halves: the engineered autoregressive inputs flatten the *median* correction cost to a common
$108$--$112$ m/s across every architecture, and the separation appears only on the *tail*, where a
recurrent or selective state-space policy wins.

== Parameter budget and the capability floor

A parameter-budget sweep across six families (dense, GRU, LSTM, windowed, Transformer, Mamba), each
trained identically at two scenarios per generation for five thousand generations, makes the first
half of the thesis concrete (@fig-pareto). Every cell in the sweep captures $100%$ of the
time -- there is no capability collapse, and a dense network with as few as $102$ parameters still
guides the maneuver at $100%$ capture and $120.8$ m/s mean. Within the dense family the cost is flat
from a few hundred to a few thousand parameters ($515 arrow.r 972 arrow.r 1957$ weights give
$117.4 arrow.r 116.9 arrow.r 116.8$ m/s) and a $3998$-weight network gains nothing further -- it is
over-parameterized for the genetic search, consistent with the plateau result of the training
methodology (Section 4).
At a matched budget the recurrent and state-space families edge the best dense cell (GRU $112.8$,
Mamba $114.9$, LSTM $116.0$ versus the best dense $116.8$), while the Transformer pays an
attention-overhead penalty at small budgets -- its worst cell, at $762$ parameters, is the worst in
the whole sweep ($121.9$ m/s), and not for lack of parameters ($762 > 515$): a small budget forces the
attention block to a tiny model width and two dimensions per head, which the genetic algorithm trains
poorly. The Pareto front is therefore shallow on the mean; the interesting structure is in the tail,
which a five-thousand-generation sweep cannot resolve.

#fig("fig_pareto.svg", [Parameter budget versus sizing tail per architecture family (left) and the
dense-family capability floor (right). Every cell captures $100%$; the dense cost is flat above a few
hundred parameters; recurrent and state-space families edge dense at matched budget; the Transformer
is penalized at small budgets. The left panel plots the $99$th-percentile cost; the body quotes
medians, which order the families identically.], <fig-pareto>)

== The tail reversal

To resolve the tail we took the strongest recurrent and state-space candidates plus the dense
reference to convergence -- the headline allocation of two scenarios per generation, a population of
$512$, run for roughly fifteen to twenty thousand generations until each plateaued -- and evaluated
each on the far-tail $n = 10\,000$ pool. Because a single run carries real run-to-run scatter, we
repeated the deciding cells over three independent seeds and report the mean and range (@fig-archtail).

On the far-tail $"CVaR"_(99.9)$, the depth at which the tanks are sized, the ordering is
$ underbrace("Mamba"_962, 124.5) < underbrace("LSTM"_1082, 129.2) < underbrace("Dense"_515, 139.2) $
m/s (three-seed means), and on the sample maximum $127.6 < 132.4 < 159.0$. The recurrent advantage
clears run-to-run variance *on the sample maximum*: the LSTM's worst run maxes at $138$ m/s, the dense
reference's best at $146$, non-overlapping, and the Mamba is tighter still. On $"CVaR"_(99.9)$ the
three-seed mean ordering is just as clean, though there the per-seed ranges overlap. Crucially, the
advantage is invisible at shallow depth -- on the shared $n = 1000$ pool the Mamba and the dense
reference are a statistical tie on the mean ($+0.1$ m/s) and the Mamba leads by only $1.6$ m/s at
$"CVaR"_95$ -- and it grows monotonically with tail depth, to $14.7$ m/s at $"CVaR"_(99.9)$. The dense
network's tight median ($109.2$ m/s) masks a fat, high-variance extreme tail: its three runs span
$"CVaR"_(99.9)$ from $128$ to $150$ m/s, and its worst hit $184$. The Mamba's tail is both lower and
half as variable. Sizing from a single dense run could quote $128$ m/s and still be unlucky in flight;
the recurrent state delivers a tail one can trust.

The control that pins this on architecture rather than parameter count is the equal-capacity pair.
At roughly $960$ parameters, the Mamba ($124.5$ m/s $"CVaR"_(99.9)$) beats the dense network of the
same size ($972$ weights, $130.7$ m/s) -- the state, not the parameters, buys the tighter tail. And
the dense family does not reward extra capacity: the $972$-weight network plateaus to a worse
validation loss than the half-size $515$-weight reference and is beaten by it on the $n = 1000$ pool
($-2.7$ m/s mean, $-4.7$ m/s at $"CVaR"_95$, paired), exactly the dimensionality effect from the
plateau. (On the far-tail metric itself the two dense nets are not cleanly separable: the $972$-weight
net's single far-tail run, $130.7$ m/s, sits inside the $515$-weight net's three-seed spread, and we
did not repeat the $972$ net, so its tail carries no measured scatter -- the dense-versus-dense
comparison is honest only on validation loss and the median.) More dense parameters do not buy a better
policy; internal state does, and the Mamba's $124.5$ undercuts both dense nets regardless.

#fig("fig_arch_tail.svg", [Three-seed run-to-run distribution on the sizing tail ($n = 10\,000$).
Mamba is lowest and tightest; both recurrent policies beat the dense reference beyond run-to-run
variance on the sample maximum (and lead it in three-seed-mean $"CVaR"_(99.9)$), and all of them crush
the best classical band.], <fig-archtail>)

== Why: training loss does not predict the sizing tail

The mechanism is the most surprising part. The three policies reach nearly the same training
objective -- validation RMS $1.331 times 10^6$ (Mamba), $1.326 times 10^6$ (dense), and $1.276 times 10^6$
(LSTM, the *lowest* training loss of the three) -- yet their deployed far tails order Mamba below LSTM
below dense. Training loss, even on a held-out validation pool, does not rank the policies the way the
sizing tail does. The reason is visible in the median: every architecture, dense included, reaches the
same $108$--$112$ m/s typical cost, because the engineered autoregressive inputs (the predicted-$Delta v$
components above all) already encode most of what a memory cell could recover. Recurrence is redundant
in the bulk. It earns its place only on the handful of hardest scenarios -- the deep-tail draws where
the static inputs are not enough and genuine internal state, carried across the pass, lets the policy
anticipate rather than react. That is precisely the part of the distribution that sizes the mission,
and precisely the part a validation-RMS objective under-weights.

The deployed headline policy is therefore the Mamba network: a $962$-parameter
dense-to-selective-state-space-to-dense stack (a $17$-input dense encoder, a Mamba core of inner
width $16$ and state size $12$, a two-output dense decoder with the atan2 bank decoder), trained under
the full methodology of Section 4. It captures $100%$ of the time at $109.9$ m/s mean and $115.2$ m/s
$"CVaR"_95$ on a fresh, never-trained-or-selected-on pool -- within rounding of its $2$M-pool numbers
(the $115.4$ of @tbl-perf), so there is no selection optimism in the headline. The $515$-parameter dense network remains the
*efficiency reference*: half the parameters, no internal state, and a competitive median, at the cost
of the fat tail just described. If compute or simplicity is the binding constraint it is the better
pick; if the mission is sized off the tail, the Mamba wins.

A final architectural detail concerns the bank decoder. Among the single-output decoders that attack
the $plus.minus pi$ wrap seam, the classical two-output atan2 decoder of @eq-atan2 still
wins: it reaches $117.4$ m/s mean and $128.7$ m/s $"CVaR"_95$ against the delta decoder ($119.9$/$141.6$)
and the scaled-$pi$ decoder ($122.2$/$140.4$), with the edge concentrated on the tail (roughly $12$--$13$
m/s at $"CVaR"_95$, paired) (@fig-outparam). The decoder we inherited from 2009 is still the right one.

#fig("fig_output_param.svg", [Bank-decoder variants on the $515$-parameter dense network. The
two-output atan2 decoder inherited from the 2009 work wins, with most of its advantage on the tail.], <fig-outparam>)

= Classical versus neural network

We now place the neural policy against the classical schemes on identical Monte-Carlo scenarios --
the same seed pools, the same dispersions, the fair comparison we have always insisted on. Two things
have to be established: that the classical baselines are tuned to their best, and that the comparison
is read at the depth that sizes the mission.

== A co-optimized reference recovers the predictor--correctors

Before comparing against the classical schemes we must give them their best shot. The
reference-tracking laws (FTC, the energy controller, PredGuid) are only as good as the reference
trajectory they enslave to, and the legacy constant-bank reference is not optimal. We therefore let
the genetic algorithm co-optimize the reference: a single extra gene sets the constant bank angle that
generates the reference table, regenerated per individual, so the law and the trajectory it tracks
adapt together. The effect is large (@fig-joint). FTC falls from $170.7$ to $126.3$ m/s mean and from
$244.1$ to $142.9$ m/s at $"CVaR"_95$ -- a $44$ m/s improvement, and the network beats it on every one
of $1000$ paired scenarios ($p approx 3 times 10^(-165)$). The energy controller recovers by $35$ m/s
and PredGuid by $23$. The reference *was* FTC's weakness: a feedback law tracking a poor target cannot
out-perform the target.

With its reference repaired, FTC ($126.3$ m/s / $142.9$ $"CVaR"_95$) becomes the best classical scheme,
within about $2$ m/s in mean of the far more expensive FNPAG ($124.3$ / $144.0$) -- the paired gap is
statistically resolvable ($p approx 10^(-23)$) but operationally negligible -- and it is analytic and
fast. The aerocapture story we wrote in 2009, in which FTC was the baseline to beat, becomes: a
*well-referenced* FTC is the classical state of the art, and it costs a fraction of a numerical
predictor--corrector to run.

#fig("fig_joint_reference.svg", [Fixed versus co-optimized reference for the three reference-tracking
schemes. Co-optimizing the constant-bank reference recovers most of FTC's deficit and lifts it to the
best classical scheme.], <fig-joint>)

== The deployability triangle <sec-deployability>

Three schemes define the deployability frontier: the neural network, the well-referenced FTC, and
FNPAG. They trade off along three axes -- accuracy, compute, and robustness -- and no single scheme
dominates all three.

*Accuracy.* On the sizing tail the network wins outright. Its far-tail $"CVaR"_(99.9)$ of $124.5$ m/s
sits roughly $40$ m/s below joint-FTC ($164$) and FNPAG ($165$). On the shared paired pool it beats
joint-FTC by $16.4$ m/s in mean, $23.8$ at $p_95$, and $27.6$ at $"CVaR"_95$, winning all $1000$
scenarios; against FNPAG the margins are $14.4$ / $23.4$ / $28.7$ m/s, winning $998$ of $1000$
(@tbl-paired). The tail margin is consistently *larger* than the mean margin -- the network's
advantage is precisely where the mission is sized.

*Compute.* On a single idle core, the dense network runs at $2.40$ ms per simulation and the stateful
Mamba at $3.68$ ms, against $1.25$ ms for FTC and $86.1$ ms for FNPAG (@fig-classical). The network is
roughly three times FTC -- the same fast class -- and $23 times$ faster than the numerical
predictor--corrector. FNPAG is dominated outright: joint-FTC matches its accuracy and the network beats
it, both at a small fraction of its cost. The selective-state-space core costs about $1.5 times$ the
dense network, the price of the tail it buys.

*Robustness -- the honest caveat.* We trained the network on the medium dispersion regime. Under a
deliberately harsher off-nominal regime (atmosphere, density perturbation, navigation, and filter all
set high), the picture inverts on robustness, and we report it plainly because it is the one place the
network loses (@fig-robust). The analytic joint-FTC degrades least -- its capture rate falls by
$5.5$ points and its $"CVaR"_95$ inflates by $197$ m/s -- against the network's $9.9$-point capture
drop and $+402$ m/s inflation; FNPAG ($-7.1$ pts, $+490$) and PredGuid ($-9.3$ pts, $+297$) sit
between, and the *fixed*-reference FTC collapses entirely ($-33$ points), which again ties the
robustness of FTC to its reference. The lesson is not that the network is fragile -- it captures
$90%$ even far outside its training regime -- but that a training-free analytic law extrapolates
better than a policy trained on a narrower distribution. The network wins the nominal sizing tail it
was trained for; widening its training regime to recover off-nominal robustness is future work, not a
property we can claim.

#grid(columns: 2, gutter: 6pt,
  fig("fig_classical_vs_nn.svg", [Deployability: per-simulation compute versus the sizing tail. Both
  network points sit lower-left -- a tighter tail than every classical scheme and far cheaper than
  FNPAG.], <fig-classical>),
  fig("fig_robustness.svg", [Off-nominal stress. The analytic joint-FTC is the most robust to
  distribution shift; the medium-trained network generalizes less well, the paper's honest caveat.], <fig-robust>))

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header(
      [*Scheme*], [*Capture %*], [*Mean*], [$bold(p_95)$], [$bold("CVaR"_95)$], [$bold("CVaR"_(99.9))$†],
    ),
    [NN -- Mamba (deployed)], [100.0], [109.9], [114.0], [*115.4*], [*124.5*],
    [NN -- dense (efficiency ref.)], [100.0], [109.7], [114.9], [117.0], [139.2],
    [FTC + joint reference], [100.0], [126.3], [137.8], [142.9], [164.0],
    [FNPAG], [100.0], [124.3], [137.4], [144.0], [165.0],
    [PredGuid], [100.0], [167.4], [209.8], [227.1], [---],
    [FTC (fixed reference)], [100.0], [170.7], [208.9], [244.1], [353.1],
    [Energy controller], [99.6], [176.7], [226.0], [245.8], [---],
    [Equilibrium glide], [99.5], [200.3], [290.0], [327.6], [---],
    [Piecewise constant], [99.8], [258.3], [374.6], [421.1], [---],
  ),
  caption: [Final Monte-Carlo performance, correction $Delta v$ in m/s, ordered by $"CVaR"_95$.
  Capture / mean / $p_95$ / $"CVaR"_95$ are on the $n = 1000$ final-evaluation pool; †$"CVaR"_(99.9)$
  is the far-tail sizing metric on a dedicated $n = 10\,000$ pool (network values are three-seed
  means). All schemes meet the heat-flux, g-load, and heat-load limits with margin; sub-$100%$
  captures are the off-corridor draws of the weaker schemes. The mean is reported for continuity with
  the 2009 work but is operationally secondary to the tail.],
) <tbl-perf>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 5pt,
    table.header(
      [*Comparison (A vs B)*], [$bold(Delta"mean")$], [$bold(Delta p_95)$], [$bold(Delta"CVaR"_95)$], [*A-win %*], [*p*],
    ),
    [Mamba vs FTC (fixed ref.)], [$-60.8$], [$-95.0$], [$-128.8$], [100.0], [$3 times 10^(-165)$],
    [Mamba vs FTC (joint ref.)], [$-16.4$], [$-23.8$], [$-27.6$], [100.0], [$3 times 10^(-165)$],
    [Mamba vs FNPAG], [$-14.4$], [$-23.4$], [$-28.7$], [99.8], [$3 times 10^(-165)$],
    [Mamba vs dense (eff. ref.)], [$+0.1$], [$-0.9$], [$-1.6$], [44.9#super[‡]], [$0.02$],
    [Mamba vs LSTM], [$+1.4$], [$-0.0$], [$-0.6$], [29.2#super[‡]], [$3 times 10^(-46)$],
    [FTC: joint vs fixed reference], [$-44.4$], [$-71.2$], [$-101.2$], [100.0], [$3 times 10^(-165)$],
    [FTC (joint) vs FNPAG], [$+2.0$], [$+0.4$], [$-1.1$], [33.9], [$1 times 10^(-23)$],
  ),
  caption: [Paired comparisons on the shared $n = 1000$ pool, correction $Delta v$ in m/s; negative
  $Delta$ favors A. Win-rate and $p$ (Wilcoxon signed-rank) are computed on the per-scenario cost.
  #super[‡]For the two intra-network rows the win-rate reflects the *mean*-level comparison, where
  the LSTM and dense networks are competitive; the headline ordering lives in the tail
  ($Delta"CVaR"_95 < 0$ and, at the far-tail sizing depth, $Delta"CVaR"_(99.9) = -14.7$ for Mamba vs
  the dense reference).],
) <tbl-paired>

= What the network uses

To close the architecture argument we ask which inputs the deployed Mamba policy actually relies on,
by zeroing each input in turn and measuring the resulting cost increase (@fig-ablation). The ranking
is informative. The largest degradations come from the orbital tracking signals -- the eccentricity
excess relative to the target, the nominal altitude rate, and the dynamic-pressure error -- the
quantities a reference-tracking law would feed back on. Immediately behind them are two of the
engineered, autoregressive *predicted correction-$Delta v$* components: the periapsis-correction burn
(#raw("predicted_dv2")) and the plane-change burn (#raw("predicted_dv3")), each evaluated on the
current osculating orbit. The energy-closing burn (#raw("predicted_dv1")) and the bank-history
encodings contribute much less.

This is the mechanism behind the bulk-versus-tail split of Section 6 seen from the inside. The
network's two strongest dependencies after the orbital errors are signals that tell it, causally and
smoothly, what the maneuver would cost if it stopped now -- a cost-to-go surrogate handed to the
policy as an input. Because those signals already encode most of the trajectory's relevant history,
a dense network without any memory matches the recurrent ones on the median: the engineered inputs do
the work that recurrence would otherwise do. What they cannot capture is the small set of hardest
scenarios where the future of the pass depends on more than the present osculating orbit, and that is
exactly where the Mamba's internal state pays off and the dense tail frays. A per-input behavior
report over the deployed policy shows no distinct failure mode: the residual correction cost is
irreducible scenario noise, not a pocket of mishandled cases. The network is using the physics we
handed it, and reserving its memory for the tail.

#fig("fig_ablation.svg", [Per-input cost increase when each input is zeroed, for the deployed Mamba
policy. The orbital tracking errors (eccentricity excess, altitude rate, dynamic-pressure error)
dominate, followed by the engineered autoregressive predicted-$Delta v$ components.], <fig-ablation>)

= Discussion and limitations

The clearest limitation is the robustness gap of @sec-deployability. The deployed network wins the nominal
sizing tail it was trained for and loses, off-nominal, to a training-free analytic law. We think this
is a property of the training distribution, not of neural guidance as such: the network was optimized
on the medium dispersion regime and asked to extrapolate to a high one. The methodology already
contains the remedy -- the moving environment that prevents overfitting to a fixed scenario batch can
just as well move across a wider regime -- so widening the training dispersion, and re-measuring the
off-nominal degradation, is the natural next experiment. We did not run it, and we do not claim the
robustness it might recover.

A second tradeoff is the cost of state. The deployed Mamba runs at $3.68$ ms per simulation against
$2.40$ ms for the dense network -- about $1.5 times$ for the selective-state-space core -- which is
the price of the tighter tail. Both remain in the fast compute class, an order of magnitude below the
numerical predictor--corrector, so the choice is between the network and FTC, not between the network
and FNPAG. If on-board compute or implementation simplicity is the binding constraint, the memoryless
$515$-parameter dense network is the efficiency reference: a competitive median at half the
parameters, paying only on the tail.

We deliberately leave two threads as future work. We have no clean campaign study of pruning or
quantizing the deployed head -- the only such cells predate the simulator fixes in this work and are
not comparable -- so deploy-size reduction of the Mamba policy is open. And we calibrated run-to-run
variance only at the tail, through the three-seed architecture repeats, which is what the headline
needs; a dedicated study of the mean-level variance across the optimizer cells was not run, so we
report tight optimizer differences (the genetic algorithm at populations of $150$ versus $300$, for
instance) as indistinguishable rather than ranking them.

Finally, a methodological note for anyone reproducing this. The training is not bit-reproducible from
a seed alone -- it never was -- because the non-stationary objective and the operator randomness make
each run a fresh draw; the deployed policy is reproduced from its saved weights and checkpoints, not
re-derived. This is the same property that forced us to measure run-to-run variance directly rather
than assume it away.

= Conclusion

Seventeen years ago we showed that a feed-forward network trained by a genetic algorithm could fly an
MSR aerocapture more efficiently than a Cerimele--Gamble feedback law, and we asked for a comparison
against predictor--correctors. This paper delivers it, and the answer is favorable to neural guidance
on the metric that matters. A $962$-parameter recurrent (Mamba) policy captures $100%$ of the time and,
on the far tail that sizes the propellant tanks, reaches $"CVaR"_(99.9) = 124.5$ m/s -- some $40$ m/s
below the best classical schemes and beating a well-referenced FTC by $16.4$ m/s in mean and $27.6$ at
$"CVaR"_95$, on every one of a thousand paired scenarios, at $23 times$ less compute than the numerical
predictor--corrector.

Two findings carry beyond the headline number. The first is methodological: a genetic algorithm is the
wrong optimizer for a fixed objective and the right one for a moving one, and the moving
Monte-Carlo environment -- adaptive seeds, a tail-weighting cost transform, and hardest-case curation
-- is a matched system that converts it from the worst optimizer we tested ($160.3$ m/s) to the best
($118.0$). The second is architectural: engineered, cost-aligned, autoregressive inputs flatten the
median across every cell type we tried, so the network's internal state earns its place only on the
hardest scenarios -- the extreme tail -- which is exactly the part of the distribution that sizes the
mission and exactly the part a validation-loss objective under-weights. Training loss did not predict
the sizing tail; architecture did.

The honest drawback that closed the 2009 paper -- that the training is too heavy to run on board --
remains, but its sting is gone: the deployed policy is a fixed forward pass that costs a few
milliseconds, and the heavy optimization happens once, on the ground. The next step would be to widen
the training environment until the network's off-nominal robustness matches its nominal accuracy, and
to carry these stateful policies beyond the single capture maneuver -- to skip-entry and Earth-return
legs, and to on-line adaptation of the deployed policy in flight.

#bibliography("refs.bib")
