// =============================================================================
// NN aerocapture guidance, revisited -- follow-up to Gelly & Vernis 2009.
// Compile (from repo ROOT, so figure paths resolve):
//   typst compile articles/paper/paper.typ articles/paper/paper.pdf
// Structure + locked numbers: articles/paper/OUTLINE.md. Data: articles/paper/data/.
// Authorial voice: articles/markdown/05_authorial_voice_and_style.md.
// Section order: methodology-first (the spine). Abstract leads with the architecture
// result. dense_515 carried as a full efficiency-reference row throughout.
// =============================================================================

// arXiv-preprint layout matched to the Mamba-3 PDF (arXiv:2603.15569v1): Linux
// Libertine body (Typst's bundled Libertinus Serif) over a wide ~6.5in text block,
// Computer Modern math (standing in for newtxmath), booktabs-style tables with the
// caption ABOVE, hyperref-style colors (maroon citations, blue internal refs/links).
#set document(title: "Seventeen years later: stateful neural guidance and the tail that sizes a Mars aerocapture mission", author: "Grégory Gelly")
#set page(paper: "us-letter", margin: (x: 1in, top: 1.1in, bottom: 1.15in), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(font: "DejaVu Sans Mono", size: 0.82em)  // cmtt stand-in for literals/URLs
// Mamba-3 body metrics: 10pt over ~12.4pt leading, flush-left paragraphs separated
// by vertical space (no indent).
#set par(justify: true, leading: 0.56em, spacing: 0.95em)
#set enum(indent: 1.2em, body-indent: 0.5em)
#set list(indent: 1.2em, body-indent: 0.5em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
// Headings: section \Large-bold (~14pt), subsection \large (~12pt),
// subsubsection \normalsize (~10.5pt), with roomier LaTeX-article skips.
#show heading: set text(weight: "bold")
#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 2): set text(size: 12pt)
#show heading.where(level: 3): set text(size: 10.5pt)
#show heading.where(level: 1): set block(above: 1.7em, below: 1.0em)
#show heading.where(level: 2): set block(above: 1.5em, below: 0.9em)
#show heading.where(level: 3): set block(above: 1.3em, below: 0.8em)
// hyperref-style colors from the Mamba-3 PDF: citations in a crimson/maroon,
// internal refs (sections, tables, equations) and URLs in blue.
#let citecolor = rgb("#b83a5c")
#let linkcolor = rgb("#2b5ba8")
#show cite: set text(fill: citecolor)
#show ref: set text(fill: linkcolor)
#show link: set text(fill: linkcolor)
// Booktabs tables: no vertical rules, no cell grid; each table adds its own
// top/mid/bottom hlines. Table captions sit ABOVE the table (LaTeX convention).
#set table(stroke: none, inset: (x: 7pt, y: 4.5pt))
#show figure.where(kind: table): set figure.caption(position: top)
#show figure: set block(above: 1.5em, below: 1.5em)
#set figure(gap: 0.8em)
#show grid: set block(above: 1.5em, below: 1.5em)

// Figure helper: include from figures/, attach the caption and the label.
#let fig(path, cap, lbl) = [#figure(image("figures/" + path, width: 100%), caption: cap)#lbl]

#v(0.15in)
#align(center)[
  #text(size: 16pt, weight: "bold", hyphenate: false)[Seventeen years later: stateful neural guidance\
  and the tail that sizes a Mars aerocapture mission]
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
  #set par(justify: true, leading: 0.44em)
  #text(size: 10pt)[In 2009 we showed that a single-hidden-layer feed-forward network, trained by a
  genetic algorithm, could fly the aerocapture of a Mars Sample Return vehicle more efficiently than
  a Cerimele--Gamble feedback law, and we closed that work by asking for the obvious next step: a
  comparison against predictor--corrector guidance. This paper answers it. We train stateful neural
  guidance policies and benchmark them, on identical Monte Carlo scenarios drawn from a bit-validated
  simulator, against
  six classical schemes including a numerical predictor--corrector (FNPAG) and a reference-tracking
  feedback law (FTC). Because the mission's correction propellant is sized off the worst-case
  $Delta v$, we lead every comparison with the tail of its distribution, not the mean. A 962-parameter
  recurrent (Mamba) policy captures every one of $10^6$ frozen confirmatory scenarios (a $95%$
  upper bound of $3 times 10^(-6)$ on its failure probability) and reaches a far-tail
  $"CVaR"_(99.9)$ of #box[$123.3 plus.minus 0.1$ m/s]; independent retraining seeds span
  $122$--$131$. It beats the best classical scheme (FTC with a
  co-optimized reference) by #box[$16.4$ m/s] in mean and #box[$27.6$ m/s] at $"CVaR"_95$, better on
  every one of $1000$ paired scenarios, at #box[$3.68$ ms] per simulation -- $23 times$ faster than
  FNPAG. The result rests on a training methodology that is itself a contribution: a non-stationary,
  adaptive-seed Monte Carlo environment turns the genetic algorithm from the *worst* optimizer under
  fixed scenarios ($154$ m/s three-seed mean) into the *best* ($120$). Across cell types, engineered,
  cost-aligned inputs flatten the median; ablation controls -- state reset, matched history, input
  removal -- show it is genuine internal state that compresses the extreme tail
  that sizes the tanks. The main deployment caveat: under a deliberately harsher off-nominal regime
  the analytic law generalizes better than the medium-trained network -- a gap we trace to the
  training objective, not to neural guidance itself.]
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
on the aerocapture to skip entry missions and evaluate the performance of neural guidance compared
to classic algorithms such as the predictor-corrector schemes." This paper is that next step,
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
  optimizer for this problem under a fixed set of Monte Carlo scenarios -- it overfits the repeated
  cases -- and the *best* under a non-stationary, adaptive-seed environment that keeps moving the
  scenarios beneath it. Switching the seed schedule from fixed to moving takes the mean correction
  cost from $154$ to $120$ m/s in three-seed means (first seeds $160.3 arrow.r 118.0$), the single
  largest effect in the campaign. The moving
  environment, a tail-weighting cost transform, and hardest-case seed curation form one matched
  system.

+ *An architecture finding with a mechanism.* Across dense, gated-recurrent (GRU, LSTM), windowed,
  attention (Transformer), and selective state-space (Mamba) cells, the engineered, cost-aligned
  inputs flatten the *median* correction cost to a common $108$--$112$ m/s for every architecture we
  trained to convergence. The separation appears only on the *tail*: a #box[$962$-parameter] Mamba policy
  beats the best dense network at the far-tail depth where the propellant tanks are sized -- by
  $15.0$ m/s in three-seed-mean $"CVaR"_(99.9)$ on the frozen confirmatory pool, with the deployed
  artifact below every dense retraining seed. Training loss does not pick the tail winner; internal
  state does -- a mechanism the ablation controls of Section 6.3 measure directly.

+ *A systematic head-to-head of neural versus predictor--corrector aerocapture guidance* -- to our
  knowledge the first for an MSR-class Mars aerocapture to compare an end-to-end learned policy
  against classical baselines co-tuned on the same objective, on paired dispersed scenarios, under
  a far-tail correction-$Delta v$ risk metric. On
  identical dispersions, the deployed network beats the best classical scheme by $16.4$ m/s in mean
  and $27.6$ m/s at $"CVaR"_95$, at a per-simulation compute cost $23 times$ below the numerical
  predictor--corrector -- with the caveat that the analytic law is more robust off-nominal, under a
  deliberately harsh stress regime we return to in Section 7.3.

This answer lands in a field that has moved since 2009. On the classical side, FNPAG @lu2015fnpag
set the numerical predictor--corrector standard and was carried across Mars aerocapture mission and
vehicle design maps @matz2017mars; more recent work replans under uncertainty explicitly --
two-stage stochastic and robust formulations that beat a deterministic predictor--corrector under
density perturbations @zucchelli2021twostage -- solves the full constrained replan by convexification
@rataczak2025cpag, or augments the bank channel with angle-of-attack modulation through optimal
control @sonandres2025abamguid. Machine learning has so far entered the loop mostly as a *component*
inside a classical scheme: LSTM density estimation feeding FNPAG @sonandres2025density, generative
failure-mode indicators steering a predictor--corrector @calkins2025riskaware. End-to-end learned
aerocapture guidance evaluated against classical baselines on common dispersed scenarios -- with the
classical side given the same objective and tuning freedom, and the comparison read at the far-tail
depth that sizes the propellant -- is, to our knowledge, the gap this paper fills.

All of this rests on a high-fidelity simulator -- $J_2$/$J_3$/$J_4$ gravity, Gauss--Markov density
perturbations, thermal limits, pilot dynamics -- regression-validated against a legacy reference
implementation
across all $725$ time steps of a guided trajectory, $22$ of $24$ output channels bit-identical (the
two mismatches trace to uninitialized variables in the reference; Appendix A lists the independent
physics checks). The simulator also carries EKF
navigation, an altitude-dependent wind model, and adaptive integration; this campaign does not
exercise them, so every run here uses the bias-filter navigation and fixed-step integration the
validation covers. The next
section formalizes the aerocapture problem and the sizing-tail objective; Section 3 describes the
guidance schemes; Section 4 presents the training methodology; Sections 5--7 give the optimizer,
architecture, and classical-versus-neural results; and Section 8 reports what the deployed network
actually uses.

= Problem and objective

== Dynamics and the aerocapture corridor

We study the same robotic MSR mission concept as the 2009 work. The vehicle reaches the atmospheric
entry interface at an altitude of $130$ km with a relative velocity of $5687$ m/s, a flight-path
angle of $-10.81 degree$, and an azimuth of $38.04 degree$ (the 2009 study placed the interface at
$120$ km with a $-10.24 degree$ flight-path angle; vehicle and target are unchanged); it carries a mass of $1089$ kg on a $14.7$ m#super[2]
reference area and is statically trimmed at a fixed angle of attack, so the only control is the bank
angle $mu$, slewed at up to $15 degree$/s. The guidance targets a $500 times 11$ km orbit at
atmosphere exit (apoapsis $times$ periapsis altitude), at $50 degree$ inclination, subsequently corrected to a $500$ km circular parking
orbit. Energy is dissipated by drag during a single atmospheric pass; the bank
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
The vehicle must respect a peak heat-flux limit of $200$ kW/m#super[2], a $4$ g load limit, and an
integrated heat-load limit of $25$ MJ/m#super[2].

#fig("fig_corridor.svg", [Empirical trajectory-occupancy envelope of the aerocapture corridor in the
(orbital energy, dynamic pressure) plane, traced by a $1\,000\,000$-run dispersed Monte Carlo of
randomized signed piecewise-constant bank profiles (roll reversals included), with the undispersed
full-lift-up and full-lift-down constant-bank boundary traces overlaid (dashed). The shaded band
spans the occupied corridor: the upper edge is the $p_(99.9)$ dynamic pressure of all capturing
trajectories (the crash-side limit), the lower edge the $p_(0.5)$ of trajectories capturing below a
$5000$ km apoapsis (the escape-side limit); as quantiles of sampled profiles these edges are
empirical, not a formal reachable set. The vehicle enters hyperbolic ($E > 0$, right) and bleeds
energy into a bound orbit ($E < 0$, left); the $200$-run Monte Carlo ensemble of the deployed Mamba
policy and its undispersed nominal (heavy line) fly well inside the envelope.], <fig-corridor>)

== The objective: the tail that sizes the mission

Performance is measured by the *correction $Delta v$* -- the propellant cost to carry the captured
orbit to the $500$ km circular parking orbit, summed over the periapsis-raising, circularization,
and plane-change burns. At atmosphere exit the periapsis always sits inside the atmosphere and must be raised, so the
correction cost has a floor of roughly $105$ m/s at this entry interface, set by the nominal
periapsis raise (dispersed cases spread a few m/s around it; the 2009 study's interface put the
same floor near $113$ m/s); a
guidance law cannot do better than deliver the vehicle to that floor across all dispersions. We
define a run as a *capture* when it terminates in a bound orbit (the simulator's terminal flag
$i_"final" = 3$ with eccentricity $< 1$) and compute $Delta v$ over captured runs only.

The point we want to make sharply, because it governs every comparison in this paper, is that the
*mean* correction cost is operationally almost irrelevant. Aerocapture propellant tanks are sized
for the worst credible case, conventionally the $3 sigma$ design point ($approx$ the $99.87$th
percentile), not the median. Two guidance laws with the same mean but different tails are not equally
good: the one with the heavier tail forces larger tanks and a heavier, more expensive mission. We
therefore treat the *tail* of the $Delta v$ distribution as the objective, and report the conditional
value-at-risk -- the mean cost in the worst $(1-alpha)$ fraction of cases, $"CVaR"_alpha$
@rockafellar2000cvar -- at $alpha = 95%$ and, for sizing decisions, at the far-tail depth
$"CVaR"_(99.9)$, together with the $95$th and $99$th percentiles and the sample maximum (a
descriptive bound, $approx p_(99.99)$ at $n = 10\,000$). Empirically $"CVaR"_alpha$ is the mean of
the worst $max(1, "round"((1-alpha) n))$ captured observations, without interpolation; every tail
statistic is reported with the number of observations it averages. The far tail
cannot be estimated from a #box[$1000$-case] ensemble (a single sample beyond $p_(99.9)$), so every
sizing number in this paper is computed on a dedicated $n = 10\,000$ pool, training-disjoint by
construction. We lead with the tail; the mean is reported for continuity with the 2009 work.

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, center, left, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Domain*], [*Dims*], [*Distribution*], [*Dispersion (controlled regime)*],
    ),
    table.hline(stroke: 0.35pt),
    [Entry state], [6], [Gaussian], [altitude $plus.minus 0.3$ km, velocity $plus.minus 3$ m/s, flight-path/azimuth $plus.minus 0.15$--$0.3degree$ ($3sigma$; medium)],
    [Atmospheric density], [1], [Uniform], [$plus.minus 50%$ multiplicative bias (medium)],
    [Aerodynamics], [3], [Uniform], [drag $plus.minus 5%$, lift $plus.minus 10%$, angle of attack $plus.minus 1degree$ (medium)],
    [Navigation errors], [7], [Gaussian], [altitude $plus.minus 2$ km, horizontal $approx plus.minus 9$ km per axis, velocity $plus.minus 1.2$ m/s, drag-accel $plus.minus 0.3$ m/s#super[2] ($3sigma$; medium)],
    [Mass], [1], [Uniform], [$plus.minus 1%$ (medium)],
    [Vehicle], [2], [Uniform], [reference area $plus.minus 2%$, max bank rate $plus.minus 10%$ (medium)],
    [Pilot dynamics], [3], [Uniform], [time constant / damping / frequency $plus.minus 10%$ (medium)],
    [Nav-filter gain], [1], [Gaussian], [$plus.minus 0.3$ absolute ($3sigma$; medium)],
    [Winds], [2], [Uniform], [speed $times [0.7, 1.3]$, direction $plus.minus 5degree$ (low; model disabled -- draws inert)],
    [Density perturbation], [process], [Gauss--Markov], [correlation time $120$ s, $5%$ RMS, time-varying (low)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The dispersion model. Twenty-six static draws across nine domains, plus a time-varying
  Gauss--Markov density perturbation (the tenth row). The controlled-study regime uses medium presets
  except for the winds and density perturbation (low); the wind model itself is disabled in this
  campaign, so its two draws are inert. Gaussian dispersions are quoted at $3sigma$; uniform
  dispersions at their full range.],
) <tbl-dispersions>

The mission is flown under $26$ dispersed parameters across nine domains, summarized in
@tbl-dispersions, plus a time-varying Gauss--Markov (Ornstein--Uhlenbeck) density perturbation that
evolves during each run. The controlled-study regime uses medium presets for the initial-state,
atmospheric, aerodynamic, navigation, mass, vehicle, pilot, and navigation-filter domains and low
presets for the winds and the density perturbation. The atmospheric density bias alone spans
$plus.minus 50%$ -- it is the dominant driver of apoapsis error, and a guidance law blind to it
cannot reject it. The simulator's wind model follows a parametric Mars profile @forget1999mars but is disabled for
this campaign (its two draw dimensions are carried but inert); the density perturbation is an
Ornstein--Uhlenbeck process layered on the static bias. Within a multi-scenario batch, draws are
generated by Latin-hypercube sampling @mckay1979lhs for space-filling coverage; the evaluation
pools draw one scenario per seed and are therefore plain independent samples -- which is what the
replicate- and bootstrap-based intervals of Sections 6--7 assume.

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
sign whenever the inclination error, projected a few seconds ahead, exceeds a threshold inside an
energy-gated activation window. FTC is analytic,
fast, and -- as we will show -- only as good as its reference.

*FNPAG* is Lu's fully numerical predictor--corrector @lu2015fnpag @lu2014predictor, subsequently
carried across Mars aerocapture mission and vehicle design maps @matz2017mars. Every two-second
replan cycle it integrates the equations of motion forward to atmosphere exit and bisects the
constant capture-bank magnitude until the predicted osculating exit apoapsis matches the target,
scaling the onboard atmosphere by the navigation-estimated density factor so the predictor tracks
the measured atmosphere rather than a nominal model; the command is held between replans. It is the most accurate classical scheme and, at roughly eleven forward
integrations per replan, by far the most expensive.

*PredGuid* is the Apollo/Shuttle-heritage drag-tracking law @bairstow2006reentry @harpold1979shuttle:
it tracks a drag-versus-energy reference profile with negative feedback.

*Energy controller* tracks an energy-dissipation reference through dynamic-pressure and altitude-rate feedback.

*Equilibrium glide* holds the equilibrium-glide condition with altitude-rate damping and a velocity bias, using the
navigation-filtered density rather than a static table.

*Piecewise constant* flies an $N$-segment constant-bank profile; it is the simplest scheme, produces the reference trajectory and corridor the
other schemes consume, and -- like the full-neural network -- emits a *signed* bank, so it bypasses
the shared roll-reversal, exit, and thermal-limiter logic.

== Neural guidance

The neural policy maps an observation vector to a bank command. We generalize the 2009
single-hidden-layer feed-forward network -- five hand-picked inputs (orbital energy, eccentricity,
inclination, velocity, non-gravitational acceleration) feeding a two-output bank decoder -- along
three independent axes: the *inputs* the policy reads, the *bank decoder* it writes through, and the
*cell type* that carries state across the pass. We take them in turn.

=== Inputs

The modern policy draws from a #box[$35$-element] candidate vector; a learned input mask selects
the subset that actually reaches the network (the deployed atan2 policies use $17$). Sixteen entries
are instantaneous orbital, aerodynamic, and thermal state variables. The other nineteen are
*engineered* signals that hand the policy temporal context and a cost-aligned summary of the flown
trajectory, so it need not integrate the history itself. A few are genuinely autoregressive --
seam-free $(sin, cos)$ encodings of the recent bank history and the roll-reversal telemetry -- but
most are instantaneous functions of the current osculating orbit: reference-trajectory
interpolations ($dot(h)$ and $q$ at the current energy), a closed-form exit-bank teacher signal, the
periapsis altitude, and -- most important, as the ablation of Section 8 shows -- the three
*predicted correction-$Delta v$* components (the energy-closing, periapsis-correction, and
plane-change burns). Those last three are smooth, causal, and cost-aligned: they tell the network
what the maneuver would cost if it stopped now.

=== Bank decoder

The 2009 paper read the bank from a two-element output,
$ mu = "atan2"(o_1, o_2), $ <eq-atan2>
which we keep as the default (#raw("atan2_signed")). It wastes half its range when only the magnitude
is needed, so for magnitude-only policies the #raw("acos_tanh") decoder maps a single output smoothly
onto $[0, pi]$,
$ mu = arccos(tanh(o_1)). $
Two further single-output decoders attack the $plus.minus pi$ wrap seam a raw angle output suffers
near $mu = pi$. The #raw("scaled_pi") decoder pushes the seam out of the operating region,
$ mu = "wrap"_pi(n pi tanh(o_1)), $
while the #raw("delta") decoder applies a bounded increment on the previous realized bank,
$ mu = "wrap"_pi(mu_"prev" + Delta_max tanh(o_1)). $

=== Cell type

Where 2009 had a single hidden layer, we span six architecture families behind one
common runtime, each carrying a different kind of internal state across the atmospheric pass:

- *Dense* feed-forward -- memoryless; maps the current observation straight to a command. The
  2009-style baseline.
- *GRU* @cho2014gru -- a gated recurrent cell carrying one hidden-state vector, updated each tick
  through reset and update gates.
- *LSTM* @hochreiter1997lstm -- a recurrent cell that adds a separate long-term cell state alongside
  the hidden state, regulated by input, forget, and output gates.
- *Windowed* -- a zero-parameter FIFO buffer of the last $N$ observations flattened into a dense
  stack: explicit recent history rather than a learned state.
- *Transformer* @vaswani2017attention -- a causal-attention block attending over a fixed-length
  key/value window of recent steps.
- *Mamba* @gu2023mamba -- a selective state-space core: a linear recurrence whose gates depend on the
  input, carrying a compact SSM state.

We use *recurrent* loosely for any cell that carries internal state across the pass. All six train
and deploy through the same bit-validated Rust runtime, and all are sized so the comparison across
cell types holds the parameter budget roughly fixed. Appendix B probes three further recent
recurrent families -- closed-form continuous-time cells @hasani2022cfc, the exponential-gated and
matrix-memory xLSTM variants @beck2024xlstm, and the discretization and complex-state axes of
Mamba-3 @lahoti2026mamba3 -- against these cells at matched budget; none improves on them.

#figure(
  table(
    columns: (auto, auto, auto, auto, 1fr),
    align: (left, left, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Scheme*], [*Bank command*], [*Reference*], [*Compute*], [*Heritage / note*],
    ),
    table.hline(stroke: 0.35pt),
    [FTC], [magnitude + roll reversal], [yes], [fast], [Cerimele--Gamble apoapsis enslavement],
    [FNPAG], [magnitude + roll reversal], [no], [slow], [onboard forward integration, bisection corrector],
    [PredGuid], [magnitude + roll reversal], [yes], [fast], [Apollo/Shuttle drag tracking],
    [Energy controller], [magnitude + roll reversal], [yes], [fast], [energy-dissipation tracking],
    [Equilibrium glide], [magnitude + roll reversal], [no], [fast], [equilibrium-glide condition, nav density],
    [Piecewise constant], [signed ($N$ segments)], [no], [fast], [produces reference + corridor],
    [Neural network], [signed or magnitude], [inputs], [fast], [35-input candidate vector, stateful cells],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The benchmarked schemes. "Reference" marks dependence on a tabulated reference trajectory
  ("inputs": the network reads two reference interpolations as observations but does not enslave to
  them); "Compute" classes are quantified in @sec-deployability (fast: $1$--$4$ ms/sim; slow: $86$ ms/sim).
  Signed-bank schemes (full-neural, piecewise-constant) bypass the shared roll-reversal, exit-phase,
  and thermal-limiter logic.],
) <tbl-schemes>

= Training methodology

The training methodology is the load-bearing contribution. The policies are trained without
input--output pairs or any reference-tracking objective (two observation inputs interpolate the
reference table; nothing enslaves to it): each candidate network is simulated on a batch of dispersed
Monte Carlo scenarios, and its fitness is the resulting correction-$Delta v$ cost (with soft
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
drawn from the cost distribution of the best individuals). The decisive cells carry three
independent training seeds each (this revision's repeats). Under fixed seeds the genetic algorithm
is the *worst* optimizer we tested and an unstable one -- three trainings land at $160.3$, $166.9$,
and $133.5$ m/s mean ($"CVaR"_95$ $215.8$ on the first seed) -- because
it overfits the repeated scenarios, evolving a policy that is excellent on those particular draws and
mediocre elsewhere. Rotating the seeds rescues it outright and *reliably*: $119.4$ m/s three-seed
mean with a $3.5$ m/s seed range (first seed $120.0$ mean, $144.5$ $"CVaR"_95$) -- a $34$ m/s drop
in three-seed means, a $71$ m/s drop on the tail on the first seeds, and the two distributions do
not touch: the fixed schedule's luckiest seed sits $12.5$ m/s above the moving schedule's worst.
Adaptive curation is indistinguishable from rotating on the *mean* at this allocation ($119.6$
versus $119.4$, seed ranges overlapping; the first-seed $118.0$ was a favorable draw); its
contribution is the hardest-seed tail shaping of Section 4.2, which a plain rotating schedule
cannot express.

The control that makes this a statement about the *objective* and not about compute is CMA-ES. Run
on the identical fixed-versus-rotating change, CMA-ES does not move -- $125.7$ versus $126.1$ m/s
in three-seed means, seed ranges within $plus.minus 1$ (first seeds $126.9 arrow.r 127.3$)
(@fig-seed). Under fixed seeds the genetic algorithm's selection converges onto the quirks of those
particular draws; CMA-ES neither suffers under the fixed batch nor benefits from the moving one. We
state that asymmetry empirically rather than mechanistically: a plausible reading is that CMA-ES's
continuous re-estimation of its parameter-space search distribution already decorrelates successive
generations, while scenario noise chiefly perturbs its rank-based updates and step-size control --
consistent with its self-termination on the noisy objective (Section 5) -- but our experiments were
not designed to isolate the mechanism.
Rotating costs marginally more compute, but CMA-ES given the same marginal compute gains nothing, so
the lever is the non-stationarity of the objective, not the extra evaluations. The practical
consequence is striking: under a fixed objective one would deploy CMA-ES and discard the genetic
algorithm; under a moving objective the genetic algorithm becomes the best optimizer in the study.
The moving environment does not make the genetic algorithm *robust* to a moving objective -- it makes
the genetic algorithm *need* one.

#fig("fig_seed_strategy.svg", [Fixed versus rotating seeds, per optimizer. The genetic algorithm is
the worst optimizer under fixed seeds and is rescued by rotating them ($-34$ m/s in three-seed
means, $-71$ m/s at $"CVaR"_95$ on the first seeds); CMA-ES is essentially unchanged (candidate
mechanisms are discussed in the text). Black dots: three independent training seeds for the
repeated GA and CMA-ES cells -- the fixed-seed pathology is itself high-variance. The
lever is the non-stationary objective, not the extra compute.], <fig-seed>)

== Cost transform, curation, and allocation

The same worst-case-leaning logic that makes us report the tail also shapes how we train. Because the
mission is sized off the far tail, we apply a monotonic *cost transform* to each per-simulation cost
before aggregating, so that the optimizer feels expensive scenarios more sharply. Applied per
scenario before the root-mean-square aggregation over the individual's batch (Appendix A), the
cubed transform makes the per-individual objective
$ J = ( 1/n sum_(i=1)^n C_i^6 )^(1\/2) , $ <eq-objective>
a monotone function of the $L_6$ norm of the per-scenario cost vector -- a high-moment objective
that weights an individual's worst scenarios far more heavily than its typical ones. (For a single
scenario a monotone transform is ranking-neutral; across a batch it deliberately is not.) We prefer
this smooth high-moment proxy over optimizing an empirical tail quantile directly because the
deployed allocation evaluates as few as two scenarios per individual per generation -- far too few
to estimate a quantile -- while the moving batch supplies tail coverage across generations.
Evaluated at the depth that matters -- a far-tail
$n = 10\,000$ pool -- the cubed transform compresses the extreme tail best ($"CVaR"_(99.9)$ $153.0$ m/s,
sample max $160.1$) against linear ($156.7$/$162.2$), square-root ($158.4$/$167.1$), squared
($162.7$/$180.9$), and logarithmic ($162.3$/$180.6$). The logarithm is worst across the shallow and
mid tail because it over-compresses and starves the gradient between captures; only at the very
extreme is the squared transform worse still. These are single training runs, and the cubed-versus-linear
margin ($3$--$5$ m/s) is within the run-to-run scatter we measure in Section 6, so we read the
direction -- deeper tail-weighting pays at deeper sizing depth -- rather than the exact ranking as
the finding. A shallower metric ($"CVaR"_95$) would have mildly favored
square-root; the deeper we look into the tail, the more the tail-weighting pays, which is exactly why
the sizing depth must decide (@fig-cost).

Seed *curation* is the same mechanism applied to the scenarios rather than the cost. At each
refresh the adaptive strategy bins the cost distribution of the best individuals into quantiles and
picks one representative seed per bin; the choice of representative is the lever. Picking the
*hardest* seed per bin (the "max" bucket) dominates the far tail ($"CVaR"_(99.9)$ $153.0$ m/s) against
the bin median ($193.9$), a random pick ($173.1$), and the *easiest* seed ($225.9$). The easiest-seed
bucket is the cautionary tale: the best mean, $117.8$ m/s, and a catastrophic worst case -- it drops
captures and reaches $245$ m/s, optimizing the average at the expense of the worst case. Trimming
the cost distribution helped nothing (@fig-curation). These too are single runs, but the min- and
middle-bucket gaps ($40$--$70$ m/s) sit far outside run scatter. The cost transform
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
  fig("fig_cost_transform.svg", [Cost transform versus the sizing tail, at the sizing depth
  ($n = 10\,000$ pool). The cubed transform minimizes both the far-tail $"CVaR"_(99.9)$ and the
  sample maximum, even though a shallow $"CVaR"_95$ read would mildly favor square-root.], <fig-cost>),
  fig("fig_training_n_sims.svg", [Allocation of the compute budget ($"CVaR"_95$; the mean orders both
  series identically). Under the adaptive schedule, two scenarios per generation over many
  generations dominates larger per-generation batches.], <fig-nsims>))
#fig("fig_curation.svg", [Seed curation. Left: far-tail $"CVaR"_(99.9)$ ($n = 10\,000$) per curation
  bucket -- selecting the hardest seed per cost-CDF bin (the "max" bucket) compresses the sizing
  tail, and the easiest-seed bucket blows it up (worst case $245$ m/s). Right: mean ($n = 1000$) --
  the easiest bucket wins the mean, the optimize-the-average trap; the trim variants match the max
  bucket and were not carried to the far-tail pool.], <fig-curation>)

That the optimal allocation pours the budget into generations points to the last methodological fact:
training here is *compute-bound, not overfitting-bound*. A stationary objective eventually overfits,
and one stops early. The moving objective never converges to a fixed landscape, so the validation
error -- the RMS on the reserved selection pool of @tbl-pools; we keep the conventional name for
the loss curve -- keeps falling for nearly the whole twenty-thousand-generation run and then
plateaus rather than degrading, i.e. the policy does not memorize scenarios (@fig-plateau). The plateau also exposes a counter-intuitive dimensionality effect that recurs in the
architecture results: the #box[$972$-parameter] dense network learns *faster* early -- more plasticity --
but the #box[$515$-parameter] network overtakes it and plateaus *lower* (validation RMS $1.326 times 10^6$
versus $1.433 times 10^6$). For the gradient-free genetic algorithm, the extra dense parameters are
more search burden than added capacity; the "more parameters, learn faster" intuition does not
transfer.

#fig("fig_plateau.svg", [Best validation RMS versus generation for the two dense reference networks.
Both keep improving until late in the twenty-thousand-generation run and then plateau -- under the
non-stationary objective the policy generalizes across the dispersion distribution rather than
memorizing scenarios. The #box[$515$-parameter] network plateaus below the
#box[$972$-parameter] one: beyond a few hundred parameters, extra dense capacity hurts the
gradient-free search.], <fig-plateau>)

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header([*Pool*], [*n*], [*Queries*], [*Decisions taken on it*]),
    table.hline(stroke: 0.35pt),
    [Training batches], [$2$/gen], [every generation], [weight updates (moving, curated)],
    [Selection pool (offset 1M)], [$1000$], [$13\,442$ over the run], [in-training argmin promotion],
    [Development far tail (offset 2M)], [$10\,000$], [tens], [cost transform, curation bucket, allocation, cell type, headline choice],
    [Fresh re-quote (offset 8M)], [$1000$], [once], [none (reported only)],
    [Confirmatory sizing (Appendix A)], [$10 times 100\,000$], [once, post-freeze], [none -- every quoted sizing number],
    [Off-nominal stress (offset 9M)], [$1000$], [once per policy], [none (robustness probe)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Scenario-pool roles and the decisions each pool influenced. The pools above the last
  two rows are development quantities: the selection pool is adaptively reused by the promotion
  gate, and the development far-tail pool informed the methodology and architecture choices, so
  neither is an unbiased test set. The confirmatory pool was generated from a seed range disjoint
  from every earlier draw, after all methodology, architecture, and checkpoint choices were frozen,
  and each cell was evaluated on it exactly once.],
) <tbl-pools>

The studies of Sections 4--5 are exploratory -- single training runs unless stated otherwise; the
confirmatory statements of this paper are the frozen-pool quantities of Sections 6--7.

= Optimizer and dimensionality

Every optimizer in this study is a gradient-free population method, a deliberate choice rather than
an omission. The training objective is the correction $Delta v$ of a complete atmospheric pass --
produced by a simulator with discrete capture, crash, and atmosphere-exit termination,
threshold-triggered phase transitions, and hard constraint limits -- so it is a black-box function
of the network weights with no usable gradient. Policy-gradient reinforcement learning
(PPO @schulman2017ppo, SAC @haarnoja2018sac) does not require a differentiable simulator or a
differentiable reward -- it estimates gradients from sampled rollouts and can in principle optimize
a terminal-only objective. We implemented and trained both, with potential-based per-step shaping
aligned to the predicted correction cost plus the true terminal cost (the standard remedy for
sparse terminal rewards), and the best policies still underperformed the population methods by a
wide margin: $636$ m/s mean ($1047$ at $"CVaR"_95$) for the dense PPO policy and $513$ ($893$) for
the recurrent one, against $119$ ($138$) for the population-trained dense network on the same
simulator regime#footnote[The reinforcement-learning cells predate two later simulator fixes and
are quoted on their own contemporaneous evaluation pool; the $4$--$5 times$ gap, not the absolute
values, is the result.] -- consistent with the stochastic shaped return optimizing a different
quantity than the deterministic mission cost. Population search on the mission cost itself was
simply the stronger tool here, so throughout we optimize the mission cost directly.

Having established that the genetic algorithm is the right optimizer under a moving objective, two
questions remain: does it need a population that scales with the search dimension, and does the
optimizer choice even matter for the low-dimensional classical-gain problems? We compared the genetic
algorithm @goldberg1989genetic against CMA-ES @hansen2001cmaes, particle-swarm optimization
@kennedy1995pso and its quantum-behaved variant @sun2004qpso, differential evolution @storn1997de, and a
three-island heterogeneous model, on an optimizer-by-budget grid at the largest dense network
($3998$ weights) and an optimizer-by-dimension grid spanning the #box[$26$-parameter] FTC-gain problem and
the #box[$515$-parameter] dense network.

At $3998$ weights the population size is decisive (@fig-optimizer). The genetic algorithm is best at a
population of $150$ ($118.0$ m/s mean; three-seed $119.6$, range $3.0$) and $300$ ($120.5$ m/s) -- a gap smaller than the run-to-run
scatter we measure on retrained cells in Section 6, so we report them as indistinguishable -- but at
a population of $60$ it *collapses* to
$166.3$ m/s. Sixty individuals cannot cover a four-thousand-dimensional weight space; selection
drifts. So the tempting generalization that the genetic algorithm dominates at any budget is wrong:
at a starved population it is no better than a single restart. The corrective is simple and worth
stating plainly -- the population must scale with the search dimension. CMA-ES improves smoothly with
budget ($133.3 arrow.r 126.3 arrow.r 121.8$ m/s) but never reaches the genetic algorithm's optimum and
self-terminates on the noisy objective before exhausting a generous generation count, an asymmetry to
keep in mind for compute-matched comparisons. The three-island heterogeneous trainer (particle swarm,
genetic, and differential evolution, with periodic migration) is the most budget-robust of all,
$120$--$124$ m/s across every budget, but the well-populated single genetic algorithm edges it.

The dimensionality grid carries the more useful lesson. On the #box[$26$-parameter] FTC-gain problem every
gradient-free optimizer we ran lands in a tight band, roughly $170$--$178$ m/s -- islands $169.8$,
particle swarm $170.9$, CMA-ES $171.7$, quantum-behaved swarm $172.9$, differential evolution
$178.2$ -- an $8$ m/s spread we do not attempt to rank (Section 9). Optimizer choice barely matters when there are only twenty-six gains
to tune. It is at neural-network dimensionality that the optimizers separate: at $515$ weights the
genetic algorithm is best ($117.4$ m/s), the islands next ($118.6$), and particle swarm worst
($129.8$), a $12$ m/s spread. The confound is that the #box[$26$-parameter] cell is a different
guidance scheme (FTC), not a narrowed version of the same network, so the comparison conflates
dimension with law; but the direction is clear -- the optimizer earns its keep on the high-dimensional
weight search, not on low-dimensional gain tuning.

#fig("fig_optimizer.svg", [Optimizer by population budget at $3998$ weights. The genetic algorithm is
best at populations of $150$--$300$ but collapses at $60$ (the population must scale with the search
dimension); the heterogeneous island model is the most budget-robust; CMA-ES improves with budget but
trails the genetic optimum.], <fig-optimizer>)

The island model -- the most budget-robust optimizer above -- was built for exactly that robustness
(@fig-islands). It splits one population budget across three islands running complementary operators
(particle swarm, genetic, and differential evolution) on the same scenarios, and every $k$
generations migrates each island's best $n$ individuals into the others, overwriting their worst.
Two properties follow. The heterogeneous operators explore the weight space differently and the
migration cross-pollinates their discoveries, so a sub-population trapped in a local optimum is
pulled out by a migrant from another island instead of having to escape on its own -- the diversity
a single homogeneous method lacks. And because the three strategies share one run's evaluations
through migration, one need not run each optimizer separately and keep the best. The result never
collapses the way a starved genetic algorithm does ($120$--$124$ m/s across every budget); the price
is that at a well-chosen budget a single large genetic algorithm still edges it, so the islands buy
robustness to the budget choice rather than a lower optimum.

#fig("fig_islands.svg", [The three-island heterogeneous optimizer. Three islands run complementary
gradient-free operators (particle swarm, genetic, differential evolution) on the same scenarios;
periodic migration exchanges each island's best individuals for the others' worst. The heterogeneity
plus migration is what escapes local optima, and one population budget covers all three searches.], <fig-islands>)

= Architecture: the headline result

We now sweep the cell type. The finding, which the next two subsections establish, comes in two
halves: a flat median across cell types, and a separation that appears only on the *tail*, where a
policy with internal state wins.

== Parameter budget and the capability floor

A parameter-budget sweep across six families (dense, GRU, LSTM, windowed, Transformer, Mamba), each
trained identically at two scenarios per generation for five thousand generations, makes the first
half of the thesis concrete (@fig-pareto). Every cell in the sweep captures $100%$ of the
time -- there is no capability collapse, and a dense network with as few as $102$ parameters still
guides the maneuver at $100%$ capture and $120.8$ m/s mean. Within the dense family the cost is flat
from a few hundred to a few thousand parameters ($515 arrow.r 972 arrow.r 1957$ weights give
$117.4 arrow.r 116.9 arrow.r 116.8$ m/s) and a #box[$3998$-weight] network gains nothing further -- it is
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
is penalized at small budgets. The left panel plots the $99$th-percentile cost and the right panel
$"CVaR"_95$; the body quotes means, under which the flat cells are indistinguishable.], <fig-pareto>)

== The tail reversal

To resolve the tail we took the strongest recurrent and state-space candidates plus the dense
reference to convergence -- the headline allocation of two scenarios per generation, a population of
$512$, run for roughly fifteen to twenty thousand generations until each plateaued -- and evaluated
each once on the frozen confirmatory pool ($10 times 100\,000$ scenarios; the development
$n = 10\,000$ pool guided the campaign and agrees within $1$--$2$ m/s throughout). Because a single
run carries real run-to-run scatter, we
repeated the deciding cells over three independent seeds and report the mean and range (@fig-archtail).
(The GRU, which had the best sweep mean, was also taken to convergence; its single confirmatory run
lands between the Mamba and the LSTM three-seed means -- $"CVaR"_(99.9)$ $126.4$ $[126.0, 126.8]$
-- but we did not repeat it over seeds, so it stays out of the three-seed comparison.)

On the far-tail $"CVaR"_(99.9)$, the depth at which the tanks are sized, the confirmatory-pool
ordering is
$ underbrace("Mamba"_962, 125.5) < underbrace("LSTM"_1082, 131.5) < underbrace("Dense"_515, 140.5) $
m/s (three-seed means; $10 times 100\,000$ scenarios per seed, replicate standard errors of
$0.1$--$0.9$ m/s, so the per-seed values are essentially exact and the residual spread is
training-run variance). The deployed artifact separates cleanly: its $123.3$ ($95%$ CI
$[123.0, 123.6]$) sits below every dense retraining seed ($128.7$ / $139.2$ / $153.7$) and every
feasible LSTM seed, with a paired margin of $5.4$ $[4.4, 6.4]$ m/s over even the best dense seed.
Across retrainings the three-seed mean gap to dense is $15.0$ m/s against a combined seed-scatter
standard error near $8$; the per-seed ranges touch once (the Mamba's worst seed, $131.0$, against
the dense reference's lucky first seed, $128.7$), so the architecture-level claim is a strong
ordering of means and of consistency -- the Mamba's seeds span $8.8$ m/s where the dense
reference's span $25.0$ -- rather than disjoint ranges. Crucially, the
advantage is invisible at shallow depth -- on the shared $n = 1000$ pool the Mamba and the dense
reference are a statistical tie on the mean ($+0.1$ m/s; $95%$ CI $[-0.1, +0.3]$) and the Mamba leads by only $1.6$ m/s at
$"CVaR"_95$ -- and it grows monotonically with tail depth. The dense
network's tight median masks a fat, high-variance extreme tail: its three runs span
$"CVaR"_(99.9)$ from $129$ to $154$ m/s. Sizing from a single dense run could quote $129$ m/s and
still be unlucky in flight;
the recurrent policy's tail estimate is consistent across retraining. The sample maximum, by
contrast, retires as a comparison statistic at this depth: at $10^6$ scenarios it is a single draw
from the extreme tail -- the Mamba seed with the *lowest* $"CVaR"_(99.9)$ ($122.2$) also logged the
campaign's deepest single excursion ($412$ m/s) -- which is precisely why the sizing metric is an
expected shortfall and the maximum is reported as descriptive only.

One feasibility asterisk belongs on the LSTM. Its best run -- the seed with the lowest training loss
and the tightest tail of its three -- exceeds the integrated heat-load limit on $13.7%$ of the
confirmatory pool ($15.6%$ of the $n = 1000$ pool: its $p_95$ heat load sits above the $25$ MJ/m#super[2] limit),
so its $Delta v$ tail is bought partly with heat. Its two repeats violate nothing, and neither do the
GRU or the Mamba on any pool (the dense reference grazes the limit on $2$ of $10\,000$ development
draws and $0.01%$ of the confirmatory pool). The
LSTM three-seed mean therefore mixes one infeasible run with two feasible ones. We adopt a
*feasibility-first* rule for every ranking and deployment statement: a run must satisfy all three
constraints on every pool it was evaluated on, or it is excluded from the comparison. Under that
rule the LSTM ranks by its feasible-seeds mean ($135.2$ m/s $"CVaR"_(99.9)$) and the ordering
Mamba $<$ LSTM $<$ dense is unchanged -- the Mamba wins the sizing tail entirely inside the
constraint envelope with or without the rule; @tbl-perf's violation
column makes the same check for the classical schemes. The infeasible seed also shows that the soft
constraint penalty does not by itself enforce feasibility -- one of eleven converged runs bought
tail performance with heat -- which is why the deployment rule is feasibility-first rather than
penalty-trusting.

The control that pins this on architecture rather than parameter count is the equal-capacity pair.
At roughly $960$ parameters, the Mamba (three-seed $125.5$ m/s $"CVaR"_(99.9)$; deployed seed
$123.3$) beats the dense network of the
same size ($972$ weights, $131.1$ $[130.5, 131.6]$, single run) -- the state, not the parameters,
buys the tighter tail. And
the dense family does not reward extra capacity: the #box[$972$-weight] network plateaus to a worse
validation loss than the half-size #box[$515$-weight] reference and is beaten by it on the $n = 1000$ pool
($-2.7$ m/s mean, $-4.7$ m/s at $"CVaR"_95$, paired), exactly the dimensionality effect from the
plateau. On the far-tail metric itself the two dense nets are not cleanly separable: the #box[$972$-weight]
net's single confirmatory run, $131.1$ m/s, sits inside the #box[$515$-weight] net's three-seed spread, and we
did not repeat the $972$ net, so its tail carries no measured scatter; the dense-versus-dense
comparison is conclusive only on validation loss and the median. More dense parameters do not buy a better
policy; in this campaign internal state does, and the deployed Mamba's $123.3$ undercuts both dense nets regardless.

#fig("fig_arch_tail.svg", [Per-seed far-tail $"CVaR"_(99.9)$ on the confirmatory pool
($10 times 100\,000$ scenarios per seed; replicate $95%$ CIs are narrower than the markers). Mamba
is lowest and tightest across retraining, the deployed seed sits below every dense seed, and every
network seed sits well below the best classical scheme (the LSTM's best seed carries the heat-load
caveat of the text).], <fig-archtail>)

== Why: training loss does not pick the tail winner

The mechanism is the most surprising part. The three policies reach nearly the same training
objective -- validation RMS $1.331 times 10^6$ (Mamba), $1.326 times 10^6$ (dense), and $1.276 times 10^6$
(LSTM, the *lowest* training loss of the three) -- yet their deployed far tails order Mamba below LSTM
below dense. Validation loss is not blind: across the eleven converged runs it orders the seeds
*within* each family exactly as the tail does (@fig-losstail). What it cannot see are the offsets
*between* families: the lowest-loss run of the campaign (the LSTM) has neither the lowest nor a
feasible tail, and at indistinguishable loss the dense reference concedes $6$ m/s of far tail per run
($15$ in three-seed mean) to the Mamba. Selecting the deployed model on validation loss would have
picked the wrong cell. Why the selective state space edges the gated cell -- input-conditioned
recurrence better matched to the density process, or simply a friendlier search landscape at this
budget -- our three-seed evidence cannot separate, and we claim no mechanism for the intra-recurrent
ordering. What we can say is that more sophisticated memory does not help: Appendix B probes three
further recent recurrent families (CfC, the xLSTM cells, Mamba-3's axes) at matched budget, and none
beats the plain cells -- two are significantly worse. The reason is visible in the median: every architecture we trained to convergence,
dense included, reaches the
same $108$--$112$ m/s typical cost, because the engineered, cost-aligned inputs (the predicted-$Delta v$
components above all) already encode most of what a memory cell could recover. Recurrence is redundant
in the bulk. It earns its place only on the handful of hardest scenarios -- the deep-tail draws where
the static inputs are not enough and genuine internal state, carried across the pass, lets the policy
anticipate rather than react. That is precisely the part of the distribution that sizes the mission,
and precisely the part a validation-RMS objective under-weights.

#fig("fig_loss_vs_tail.svg", [Best validation RMS versus far-tail $"CVaR"_(99.9)$ ($n = 10\,000$) for
the eleven converged runs. Within every family the runs order identically on both axes (overall
Spearman $rho = 0.91$, pooled over families and read as descriptive -- the runs are not
exchangeable across families); what the loss cannot see are the offsets between families -- the lowest-loss
run (the LSTM, starred: the heat-load-infeasible one of Section 6.2) is not the best tail, and the
dense cells sit well above the Mamba at matched loss.], <fig-losstail>)

Three controls, run for this revision and evaluated once on the frozen confirmatory pool of Section
7, pin the mechanism. *State reset*: the deployed Mamba evaluated with its recurrent state zeroed at
every guidance tick collapses from $"CVaR"_(99.9) = 123.3$ to $414.5$ m/s at unchanged $100%$
capture -- the deployed policy computes with its state; it is not a feedforward law in disguise.
*Matched history*: a dense cell fed an explicit five-tick observation window ($970$ parameters,
identical budget and regime) reaches $142.2 plus.minus 0.3$ -- inside the dense family's
seed-to-seed spread, $19$ m/s above the intact policy, with a $423$ m/s worst case at $10^6$
scenarios; short temporal context does not substitute for learned state. *No
predicted-$Delta v$*: the same Mamba architecture retrained without the three cost-aligned inputs
reaches $138.9 plus.minus 0.2$ (median $113.1$) -- the engineered inputs matter for bulk and tail
alike, so inputs and state are complements rather than substitutes; notably, even this stripped
network beats the best classical scheme on every reported statistic ($"CVaR"_95$ $125.3$ versus
joint-FTC's $144.3$), so the network's edge does not ride on its privileged observations. In sum:
state is necessary for the deployed tail (the first two controls) and not sufficient without the
cost-aligned inputs (the third).

The deployed headline policy is therefore the Mamba network: a #box[$962$-parameter]
dense-to-selective-state-space-to-dense stack (a #box[$17$-input] dense encoder, a Mamba core of inner
width $16$ and state size $12$, a two-output dense decoder with the atan2 bank decoder), trained under
the full methodology of Section 4. It captures $100%$ of the time at $109.9$ m/s mean and $115.2$ m/s
$"CVaR"_95$ on a fresh, never-trained-or-selected-on pool -- within rounding of its $2$M-pool numbers
(the $115.4$ of @tbl-perf) -- and, on the frozen confirmatory pool, captures $10^6$ of $10^6$
scenarios at $"CVaR"_(99.9) = 123.3 plus.minus 0.1$, within $1.3$ m/s of the development-pool
estimate: there is no selection optimism in the headline. The #box[$515$-parameter] dense network remains the
*efficiency reference*: half the parameters, no internal state, and a competitive median, at the cost
of the fat tail just described. If compute or simplicity is the binding constraint it is the better
pick; if the mission is sized off the tail, the Mamba wins.

A final architectural detail concerns the bank decoder. Among the single-output decoders that attack
the $plus.minus pi$ wrap seam, the classical two-output atan2 decoder of @eq-atan2 still
wins: it reaches $117.4$ m/s mean and $128.7$ m/s $"CVaR"_95$ against the delta decoder ($119.9$/$141.6$)
and the scaled-$pi$ decoder ($122.2$/$140.4$), with the edge concentrated on the tail (roughly $12$--$13$
m/s at $"CVaR"_95$, paired) (@fig-outparam). The decoder we inherited from 2009 is still the right one.

#fig("fig_output_param.svg", [Bank-decoder variants on the #box[$515$-parameter] dense network. The
two-output atan2 decoder inherited from the 2009 work wins, with most of its advantage on the tail.], <fig-outparam>)

= Classical versus neural network

We now place the neural policy against the classical schemes on identical Monte Carlo scenarios --
the same seed pools, the same dispersions, the fair comparison we have always insisted on. Two things
have to be established: that the classical baselines are tuned to their best, and that the comparison
is read at sizing depth.

== A co-optimized reference recovers the predictor--correctors

Before comparing against the classical schemes we must give them their best shot. The
reference-tracking laws (FTC, the energy controller, PredGuid) are only as good as the reference
trajectory they enslave to, and the legacy constant-bank reference is not optimal. We therefore let
the genetic algorithm co-optimize the reference: a single extra gene sets the constant bank angle that
generates the reference table, regenerated per individual, so the law and the trajectory it tracks
adapt together. The effect is large (@fig-joint). FTC falls from $170.7$ to $126.3$ m/s mean and from
$244.1$ to $142.9$ m/s at $"CVaR"_95$ -- a $44$ m/s improvement, and the network beats it on every one
of $1000$ paired scenarios ($p < 10^(-15)$, saturated). The energy controller recovers by $35$ m/s
and PredGuid by $23$. The reference *was* FTC's weakness: a feedback law tracking a poor target cannot
out-perform the target.

With its reference repaired, FTC ($126.3$ m/s / $142.9$ $"CVaR"_95$) becomes the best classical scheme,
within about $2$ m/s in mean of the far more expensive FNPAG ($124.3$ / $144.0$) -- the paired gap is
statistically resolvable ($p approx 10^(-23)$) but operationally negligible, and at the far-tail
sizing depth the repaired FTC pulls decisively ahead ($165.1$ versus $198.7$, Section 7.2) -- and
it is analytic and
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

*Accuracy.* On the sizing tail the network wins outright. Its far-tail $"CVaR"_(99.9)$ of
$123.3 plus.minus 0.1$ m/s sits $41.8$ m/s $[41.2, 42.4]$ below joint-FTC ($165.1$) and $75.4$
$[71.4, 79.3]$ below FNPAG ($198.7$) on the frozen confirmatory pool. FNPAG's extreme tail fattens
with depth in a way its shallow statistics hide -- $"CVaR"_95$ matches the development pool at
$143$, but $"CVaR"_(99.9)$ grows from $165$ at $n = 10^4$ to $198.7$ at $10^6$, with $163$ physical
crashes in $10^6$ (all surface impacts on individual re-runs at $12 times$ the evaluation timeout,
not censoring) -- so the repaired FTC is decisively the better classical scheme at sizing depth. On
the shared paired pool it beats
joint-FTC by $16.4$ m/s in mean, $23.8$ at $p_95$, and $27.6$ at $"CVaR"_95$, winning all $1000$
scenarios; against FNPAG the margins are $14.4$ / $23.4$ / $28.7$ m/s, winning $998$ of $1000$
(@tbl-paired). The tail margin is consistently *larger* than the mean margin -- the network's
advantage is precisely where the mission is sized.

*Compute.* On a single idle core, the dense network runs at $2.40$ ms per simulation and the stateful
Mamba at $3.68$ ms, against $1.25$ ms for FTC and $86.1$ ms for FNPAG (@fig-classical). The network is
roughly three times FTC -- the same fast class -- and $23 times$ faster than the numerical
predictor--corrector. On accuracy and compute FNPAG is dominated -- joint-FTC matches its accuracy
and the network beats it, both at a small fraction of its cost -- though the off-nominal stress
below keeps robustness a separate axis. The selective-state-space core costs about $1.5 times$ the
dense network, the price of the tail it buys. Flight processors run one to two orders of magnitude
slower than the laptop core measured here; at a conservative $100 times$ scaling the numerical
predictor--corrector's $86$ ms replan would approach its own $2$ s replan period, while the
network's few-millisecond forward pass stays comfortably sub-second. The portable result is the
relative ordering, not the absolute milliseconds -- none of these measurements establish worst-case
execution time or memory on qualified hardware (Section 9).

*Robustness -- the honest caveat.* We trained the network on the medium dispersion regime. Under a
deliberately harsher off-nominal regime (atmosphere, density perturbation, navigation, and filter all
set high), the picture inverts on robustness, and we report it plainly because it is the one place the
network loses (@fig-robust). All stress-regime tail statistics are conditional on capture --
$"CVaR"_95 (Delta v | "capture")$ -- and a conditional tail can improve by failing the hardest
scenarios, so we read every stress comparison lexicographically: capture probability first,
conditional tail cost second, and no tail win is claimed across a capture-rate deficit. The analytic joint-FTC degrades least -- its capture rate falls by
$5.5$ points and its $"CVaR"_95$ inflates by $197$ m/s -- against the network's #box[$9.9$-point] capture
drop and $+402$ m/s inflation; PredGuid ($-9.3$ pts, $+297$) sits between, FNPAG loses less capture
($-7.1$ pts) but inflates its tail the most ($+490$), and the *fixed*-reference FTC collapses entirely ($-33$ points), which again ties the
robustness of FTC to its reference. The lesson is not that the network is fragile -- it captures
$90%$ even far outside its training regime -- but that a training-free analytic law extrapolates
better than a policy trained on a narrower distribution. The network wins the nominal sizing tail it
was trained for; widening its training regime to recover off-nominal robustness is future work, not a
property we can claim.

#grid(columns: 2, gutter: 6pt,
  fig("fig_classical_vs_nn.svg", [Deployability: per-simulation compute versus far-tail
  $"CVaR"_(99.9)$. Both
  network points sit lower-left -- a tighter tail than every classical scheme and far cheaper than
  FNPAG.], <fig-classical>),
  fig("fig_robustness.svg", [Off-nominal stress. The analytic joint-FTC is the most robust to
  distribution shift; the medium-trained network generalizes less well, the paper's robustness caveat.], <fig-robust>))

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Scheme*], [*Capture %*], [*Viol. %*], [*Mean*], [$bold(p_95)$], [$bold("CVaR"_95)$], [$bold("CVaR"_(99.9))$†],
    ),
    table.hline(stroke: 0.35pt),
    [NN -- Mamba (deployed)], [100.0], [0.0], [109.9], [114.0], [*115.4*], [*123.3*],
    [NN -- LSTM#super[‡]], [100.0], [15.6], [108.4], [114.0], [116.0], [135.2],
    [NN -- dense (efficiency ref.)], [100.0], [0.0], [109.7], [114.9], [117.0], [128.7],
    [FTC (joint reference)], [100.0], [0.0], [126.3], [137.8], [142.9], [165.1],
    [FNPAG], [100.0], [0.0], [124.3], [137.4], [144.0], [198.7],
    [PredGuid (joint reference)], [100.0], [0.0], [144.2], [164.2], [172.8], [225.8],
    [Energy controller (joint reference)], [100.0], [0.0], [142.1], [166.3], [178.3], [304.3],
    [PredGuid (fixed reference)], [100.0], [0.0], [167.4], [209.8], [227.1], [301.6],
    [FTC (fixed reference)], [100.0], [0.2], [170.7], [208.9], [244.1], [341.4],
    [Energy controller (fixed reference)], [99.6], [0.0], [176.7], [226.0], [245.8], [308.1],
    [Equilibrium glide], [99.5], [0.5], [200.3], [290.0], [327.6], [410.1],
    [Piecewise constant], [99.8], [1.1], [258.3], [374.6], [421.1], [598.2],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Final Monte Carlo performance, correction $Delta v$ in m/s, ordered by $"CVaR"_95$.
  Capture / Viol. / mean / $p_95$ / $"CVaR"_95$ are on the $n = 1000$ final-evaluation pool;
  †$"CVaR"_(99.9)$ is the far-tail sizing metric on the frozen confirmatory pool
  ($10 times 100\,000$ scenarios per scheme; replicate standard errors $0.1$--$3.6$ m/s; each
  average pools $1000$ tail observations). Network rows tabulate the value the feasibility-first
  rule of Section 6.2 ranks the cell by -- the deployed Mamba seed, the dense efficiency-reference
  seed, and the LSTM feasible-seeds mean -- with per-seed values in Section 6.2.
  Reference-tracking schemes appear in both their fixed- and co-optimized
  (joint) reference forms (Section 7.1). Viol. is the fraction of draws exceeding any of the
  heat-flux, g-load, or heat-load limits: the deployed network, the tuned predictor--correctors, and
  the joint-reference trackers violate none; the fixed-reference FTC, equilibrium glide, and
  piecewise constant carry heat-flux exceedances of at most $1.1%$. Sub-$100%$ captures are the
  off-corridor draws of the weaker schemes; a $100%$ capture rate means "no failures observed" --
  with zero failures in $n$ independent scenarios the one-sided $95%$ upper bound on the failure
  probability is $approx 3\/n$ ($3 times 10^(-4)$ at $n = 1000$; $3 times 10^(-6)$ at the
  confirmatory $n = 10^6$). The mean
  is reported for continuity with the 2009 work but is operationally secondary to the tail.
  #super[‡]The LSTM's best seed -- its lowest-training-loss one -- exceeds the heat-load limit on
  $15.6%$ of this pool ($13.7%$ of the confirmatory pool); the tabulated $135.2$ is its two
  feasible seeds' mean, and the raw three-seed mean including the infeasible seed is $131.5$.
  Confirmatory capture is $100%$ for the network rows, joint-FTC, and fixed-reference FTC;
  $99.85$--$99.98%$ for the remaining classical schemes (their † values are conditional on
  capture; FNPAG's $163$ failures in $10^6$ were individually re-run at $12 times$ the evaluation
  timeout and are all physical crashes).],
) <tbl-perf>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Comparison (A vs B)*], [$bold(Delta"mean")$], [*95% CI*], [$bold(Delta p_95)$], [$bold(Delta"CVaR"_95)$], [*A-win %*], [*p*],
    ),
    table.hline(stroke: 0.35pt),
    [Mamba vs FTC (fixed ref.)], [$-60.8$], [$[-62.4, -59.2]$], [$-95.0$], [$-128.8$], [100.0], [$< 10^(-15)$],
    [Mamba vs FTC (joint ref.)], [$-16.4$], [$[-16.8, -16.0]$], [$-23.8$], [$-27.6$], [100.0], [$< 10^(-15)$],
    [Mamba vs FNPAG], [$-14.4$], [$[-14.8, -14.0]$], [$-23.4$], [$-28.7$], [99.8], [$< 10^(-15)$],
    [Mamba vs dense (eff. ref.)], [$+0.1$], [$[-0.1, +0.3]$], [$-0.9$], [$-1.6$], [44.9#super[‡]], [$0.02$],
    [Mamba vs LSTM], [$+1.4$], [$[+1.2, +1.6]$], [$-0.0$], [$-0.6$], [29.2#super[‡]], [$3 times 10^(-46)$],
    [FTC: joint vs fixed reference], [$-44.4$], [$[-45.9, -42.9]$], [$-71.2$], [$-101.2$], [100.0], [$< 10^(-15)$],
    [FTC (joint) vs FNPAG], [$+2.0$], [$[+1.5, +2.5]$], [$+0.4$], [$-1.1$], [33.9], [$1 times 10^(-23)$],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Paired comparisons on the shared $n = 1000$ pool, correction $Delta v$ in m/s; negative
  $Delta$ favors A. The CI is a $10\,000$-resample bootstrap on the paired mean difference; the
  tail deltas carry the same paired-resample construction ($Delta p_95$ / $Delta"CVaR"_95$ $95%$
  CIs -- Mamba vs joint-FTC $[-24.9, -22.5]$ / $[-29.7, -25.5]$; vs FNPAG $[-25.1, -21.5]$ /
  $[-31.4, -26.0]$; vs dense $[-1.6, -0.1]$ / $[-2.6, -0.6]$; vs LSTM $[-0.6, +0.5]$ /
  $[-1.3, +0.1]$, the one interval straddling zero -- the intra-network separation lives at the
  far-tail depth, not here).
  Win-rate and $p$ (Wilcoxon signed-rank) are computed on the per-scenario cost; $p$ is truncated
  at $10^(-15)$ -- the normal-approximation statistic saturates (near $10^(-165)$) at or near sign
  unanimity at $n = 1000$, certifying a (near-)unanimous direction, not a resolved tail
  probability. Deltas are computed on unrounded per-scenario values, so a delta may differ from the
  difference of the rounded marginals in @tbl-perf by $0.1$ m/s.
  #super[‡]For the two intra-network rows the win-rate is driven by the bulk of the per-scenario
  differences, where the dense and LSTM networks match or slightly beat the Mamba; the headline
  ordering lives in the tail ($Delta"CVaR"_95 < 0$ and, at the far-tail sizing depth, confirmatory
  paired replicate deltas of $Delta"CVaR"_(99.9) = -41.8$ $[-42.4, -41.2]$ vs joint-FTC, $-5.4$
  $[-6.4, -4.4]$ vs the dense reference seed, and $-15.0$ in three-seed means). The LSTM row
  additionally carries
  the heat-load feasibility caveat of Section 6.2.],
) <tbl-paired>

#fig("fig_survival.svg", [Empirical survival curves of the correction $Delta v$ on the confirmatory
sizing pool ($10 times 100\,000$ scenarios per scheme; log scale, curves subsampled every
$approx 100$th order statistic). The network-to-classical separation grows with tail depth -- the
sizing thesis in one picture.], <fig-survival>)

== Matching the objective to the regime closes the gap <sec-objcenter>

The robustness caveat of @sec-deployability invites a question: is the off-nominal gap intrinsic to
neural guidance, or an artifact of a training objective tuned for the wrong regime? Before answering
it, the regime itself deserves scrutiny. The high-dispersion stress pool is not a realistic operating
point -- no mission would be flown on an entry profile that fails roughly one pass in twenty under
*any* guidance scheme. In practice the entry interface would simply be re-targeted so that the $3sigma$
dispersion envelope still captures, restoring a near-certain margin by construction. We therefore read
this regime not as a deployment scenario but as a deliberate *stress probe of the training method*: it
pushes the optimizer into a regime where roughly $5%$ of scenarios are catastrophic, and asks whether
the genetic algorithm can still make progress.

Under that probe, it cannot -- not with the objective that wins in the medium regime. Retraining the
Mamba head on the high regime with the deployed objective (the cubed transform and hardest-seed
curation of Section 4) stalled: the validation cost plateaued for thousands of generations, and the
deployed policy came out *worse* than the medium-trained one it was meant to replace -- $27%$ capture
on the stress pool with a $"CVaR"_95$ of $1216$ m/s, against the medium-trained network's $90%$ and
$518$. The mechanism is the one the medium regime hides: when failures dominate, cubing the cost makes
the objective a spiky near-discrete failure count, max-bucket curation feeds the optimizer only the
hardest seeds, and a per-individual scenario batch as small as the deployed allocation's two cannot
estimate any of it -- so selection has no gradient to climb. The worst-case shaping that is a virtue
in the medium regime becomes a liability once the noise outruns the sample budget.

*Centering* the objective -- more scenarios per individual (a real cost estimate), a central curation
bucket, and a milder transform -- restores the gradient. On the dense vehicle a one-lever-at-a-time
sweep (@fig-objcenter) attributes the recovery: more sims is the clean lever (it halves the tail,
$"CVaR"_95$ $1031 arrow.r 523$, while *holding* capture at $95%$); a milder transform helps moderately;
and the central bucket *alone* is a trap -- it drops capture to $84.5%$, because one cannot pick a
central representative from a cost distribution two samples cannot resolve. The three knobs are a
coupled system, exactly as in the medium regime, but now the coupling has teeth: only all three
together recover both capture and the tail (dense $"CVaR"_95$ $1031 arrow.r 276$, mean $574 arrow.r 157$).
The effect transfers cleanly to the deployed architecture -- the centered Mamba reaches $"CVaR"_95$
$273$ m/s at $94.9%$ capture against the stalled stack's $1216$ at $27%$ -- so the stall was the
objective, not the cell type.

The payoff, stated with its caveats, is that on this evidence the off-nominal gap is not intrinsic
to neural guidance. With a
regime-matched objective the centered Mamba ($"CVaR"_95$ $273$ m/s at $94.9%$ capture) beats the best
classical scheme on the very regime where the medium-trained network lost to it -- joint-FTC retrained
on the same regime sits at $424$ m/s at $95.0%$ capture, and the medium-deployed joint-FTC at $340$
at $94.5%$: capture parity within half a point, so the lexicographic comparison is decided by the
conditional tail. The analytic law's
edge in @sec-deployability was a property of the *mismatched* training objective, not of neural
guidance. The result now carries measured run-to-run scatter: three independent centered-Mamba
retrainings hold capture at $94.8$--$95.0%$ with $"CVaR"_95 (Delta v | "capture")$ spanning
$231$--$273$ m/s -- every seed beating both the retrained joint-FTC ($424$) and the
medium-deployed one ($340$) -- so the reversal is stable across training runs. The remaining
caveats: these are $n = 1000$ figures (a sizing-grade number wants the $n = 10\,000$ depth used
elsewhere), and the regime that produces the gap
is itself one a real mission would design away. The durable lesson is methodological and reinforces
Section 4: the optimal worst-case weighting is matched to the environment's noise and the per-individual
sample budget, not fixed once.

#fig("fig_objective_centering.svg", [Objective-centering under the high-dispersion stress probe
($n = 1000$ on the 9M pool). Left: deployed capture rate; right: deployed correction-DV $"CVaR"_95$
over captured runs. The medium-regime stack (red) carries an enormous tail; centering recovers it
(green) and transfers to the Mamba, beating the retrained joint-FTC (dashed). The central bucket alone
trades capture for the tail (left, $84.5%$), so the three levers are read together. Capture and tail
are shown separately because the levers trade them off.], <fig-objcenter>)

= What the network uses

To close the architecture argument we ask which inputs the deployed Mamba policy actually relies
on, through a closed-loop input-sensitivity analysis: each input is zeroed in turn and the
resulting cost increase measured (@fig-ablation). Zeroing a normalized input perturbs every
subsequent state of the closed loop, so the deltas measure closed-loop dependence, not isolated
feature importance. The ranking is informative. The largest degradations come from the orbital tracking signals -- the eccentricity
excess relative to the target, the nominal altitude rate, and the dynamic-pressure error -- the
quantities a reference-tracking law would feed back on. Immediately behind them are two of the
engineered, cost-aligned *predicted correction-$Delta v$* components: the periapsis-correction burn
(#raw("predicted_dv2")) and the plane-change burn (#raw("predicted_dv3")), each evaluated on the
current osculating orbit. The energy-closing burn (#raw("predicted_dv1")) and the bank-history
encodings contribute much less.

This is the mechanism behind the bulk-versus-tail split of Section 6 seen from the inside. The
network's two strongest dependencies after the orbital errors are signals that tell it, causally and
smoothly, what the maneuver would cost if it stopped now -- a cost-to-go surrogate handed to the
policy as an input. Because the osculating orbit is itself an integral of the flown trajectory,
those signals already summarize most of the history a memory cell could recover, and a dense network
without any memory matches the recurrent ones on the median: the engineered inputs do
the work that recurrence would otherwise do. What they cannot capture is the small set of hardest
scenarios where the future of the pass depends on more than the present osculating orbit, and that is
exactly where the Mamba's internal state pays off and the dense tail frays. A per-input behavior
report over the deployed policy shows no clustered failure mode; we did not attempt a per-scenario
lower-bound analysis, so we do not claim the residual cost irreducible. The network is using the
physics we handed it, and reserving its memory for the tail.

#fig("fig_ablation.svg", [Closed-loop input sensitivity: per-input cost increase when each input is
zeroed, for the deployed Mamba policy. The orbital tracking errors (eccentricity excess, altitude rate, dynamic-pressure error)
dominate, followed by the engineered, cost-aligned predicted-$Delta v$ components.], <fig-ablation>)

= Discussion and limitations

The clearest limitation is the off-nominal robustness gap of @sec-deployability -- the deployed
network wins the nominal sizing tail it was trained for and loses, off-nominal, to a training-free
analytic law. @sec-objcenter traces that gap to the training objective rather than to neural
guidance, with the markers stated there: the centered-retrain demonstration is now three-seed
(capture steady at $approx 95%$, conditional tail $231$--$273$ m/s, every seed beating both FTC
references) but remains $n = 1000$, and the stress regime is one a real mission would design away.
What remains open is the far-tail depth used for the headline.

A second tradeoff is the cost of state. The deployed Mamba runs at $3.68$ ms per simulation against
$2.40$ ms for the dense network -- about $1.5 times$ for the selective-state-space core -- which is
the price of the tighter tail. Both remain in the fast compute class, an order of magnitude below the
numerical predictor--corrector, so the choice is between the network and FTC, not between the network
and FNPAG. If on-board compute or implementation simplicity is the binding constraint, the memoryless
#box[$515$-parameter] dense network is the efficiency reference: a competitive median at half the
parameters, paying only on the tail.

A flight-qualification path for a stateful network remains open, and the deployability triangle
suggests its shape: a simplex arrangement in which the analytic joint-FTC -- whose off-nominal
robustness Section 7.2 established -- runs in parallel as an onboard monitor with authority to take
over on envelope violation, while the network flies the nominal corridor it wins on. The deployed
policy is a fixed forward pass of $962$ parameters in double precision with no data-dependent
control flow, so worst-case execution time and memory bound trivially; verifying the *decision*
behavior across the dispersion envelope, rather than the code, is the open problem.

We deliberately leave three threads as future work. We have no clean campaign study of pruning or
quantizing the deployed head -- the only such cells predate the simulator fixes in this work and are
not comparable -- so deploy-size reduction of the Mamba policy is open. The state-ablation thread
is now closed by the three controls of Section 6.3 -- state reset, matched history, and no
predicted-$Delta v$ -- so the tail mechanism is measured rather than hypothesized; what remains
open there is only the intra-recurrent ordering (why the selective state space edges the gated
cells), which the three-seed evidence cannot separate. Run-to-run variance is calibrated at the
tail through the three-seed architecture repeats and, for the decisive seed-strategy and CMA-ES
cells, at the mean through dedicated repeats (Section 4.1; the fixed-seed pathology is itself
high-variance, a $33$ m/s seed range); the remaining tight optimizer differences (the genetic
algorithm at populations of $150$ versus $300$, for
instance) stay reported as indistinguishable rather than ranked.

Finally, a methodological note for anyone reproducing this. The training is not bit-reproducible from
a seed alone -- it never was -- because the non-stationary objective and the operator randomness make
each run a fresh draw; the deployed policy is reproduced from its saved weights and checkpoints, not
re-derived. This is the same property that forced us to measure run-to-run variance directly rather
than assume it away.

= Conclusion

Seventeen years ago we showed that a feed-forward network trained by a genetic algorithm could fly an
MSR aerocapture more efficiently than a Cerimele--Gamble feedback law, and we asked for a comparison
against predictor--correctors. This paper delivers it, and the answer is favorable to neural guidance
on the metric that matters. A #box[$962$-parameter] recurrent (Mamba) policy captures every one of
$10^6$ frozen confirmatory scenarios and,
on the far tail that sizes the propellant tanks, reaches $"CVaR"_(99.9) = 123.3 plus.minus 0.1$ m/s
-- $42$ m/s
below the best classical scheme and beating a well-referenced FTC by $16.4$ m/s in mean and $27.6$ at
$"CVaR"_95$, on every one of a thousand paired scenarios, running $23 times$ faster than the numerical
predictor--corrector.

Two findings carry beyond the headline number. The first is methodological: a genetic algorithm is the
wrong optimizer for a fixed objective and the right one for a moving one, and the moving
Monte Carlo environment -- adaptive seeds, a tail-weighting cost transform, and hardest-case curation
-- is a matched system that converts it from the worst optimizer we tested ($154$ m/s three-seed
mean) to the best
($120$). The second is architectural: engineered, cost-aligned inputs flatten the
typical cost across every cell type we tried, so the network's internal state earns its place only on the
hardest scenarios -- the extreme tail -- which is exactly the part of the distribution that sizes the
mission and exactly the part a validation-loss objective under-weights. Training loss did not pick
the tail winner; architecture did.

The honest drawback that closed the 2009 paper -- that the training is too heavy to run on board --
remains, but its sting is gone: the deployed policy is a fixed forward pass that costs a few
milliseconds, and the heavy optimization happens once, on the ground -- ground compute buying
in-flight margin. The next step would be to widen
the training environment -- annealing the tail-weighting and the dispersion envelope over the run, or
stratifying the curated batch across cost quantiles -- until the network's off-nominal robustness
matches its nominal accuracy, and
to carry these stateful policies beyond the single capture maneuver -- to skip-entry and Earth-return
legs, and to on-line adaptation of the deployed policy in flight.

#pagebreak()
#bibliography("refs.bib", title: "References", style: "harvard-cite-them-right")

#pagebreak()
#set heading(numbering: none)
= Appendix A: reproduction details

One compact reference for the settings and systems behind every number.

*Simulator.* All schemes fly through one native (Rust) simulator: fixed-step fourth-order
Runge--Kutta integration (Gill variant), $J_2$--$J_4$ zonal gravity, a tabulated Mars atmosphere
carrying the static Monte Carlo bias and the Ornstein--Uhlenbeck perturbation, pilot dynamics,
thermal tracking, and the navigation--guidance--control chain sequenced on its own cadences; the
bias-filter navigation recovers density through lift-corrected inverse dynamics. The implementation is
regression-validated against the 2009 study's legacy code -- across all $725$ time steps of a guided
trajectory, $22$ of $24$ output channels are bit-identical (the two mismatches trace to
uninitialized variables in the reference) -- which establishes equivalence with the flight-heritage
implementation, not physical validation. Independent physics checks run in the test suite: an
analytic potential-gradient oracle for the $J_2$--$J_4$ gravity expansion, integrator convergence
on analytic systems, vacuum energy and angular-momentum conservation, and cross-language
forward-pass parity for the network runtime at machine epsilon.

*Seed pools.* Every Monte Carlo pool derives from the mission base seed through disjoint reserved
streams: the selection pool (offset $10^6$, $n = 1000$, the in-training promotion gate -- queried
$13\,442$ times over the headline run, so it is adaptively reused and is not an unbiased estimate
of generalization), final evaluation (offset $2 times 10^6$: the $n = 1000$ paired pool of
@tbl-perf and @tbl-paired, extended to $n = 10\,000$ for the development far tail), the fresh
re-quote pool (offset $8 times 10^6$), and the off-nominal stress pool (offset $9 times 10^6$).
Training scenarios are drawn outside every reserved stream; evaluation pools draw one scenario per
seed (independent samples), and multi-scenario batches use Latin-hypercube draws. @tbl-pools states
which decisions each pool influenced.

*Confirmatory sizing pool.* All methodology, architecture, and checkpoint choices were frozen at a
recorded revision before the pool was generated. Seeds are drawn without duplicates from
$[2^31, 2^32)$ -- structurally disjoint from every historical draw, since all earlier pools,
training batches, and curation probes live in $[0, 2^31)$ -- arranged as ten replicate pools of
$100\,000$ scenarios sharing seeds across schemes (so per-replicate differences are paired). Every
cell is evaluated on it exactly once; tail statistics use the estimator of Section 2.2
($"CVaR"_(99.9)$ averages $100$ observations per replicate, $1000$ pooled), and replicate-level
dispersion gives the quoted standard errors and difference intervals with no distributional
assumptions.

*Off-nominal stress regime.* The Section 7 stress pool raises four domains to their high presets:
the atmospheric density bias to $plus.minus 100%$ (from $plus.minus 50%$); the Gauss--Markov
density perturbation to a $30$ s correlation time at $20%$ RMS (from $120$ s at $5%$); navigation
errors to $3sigma$ altitude $plus.minus 3$ km, horizontal $approx plus.minus 18$ km per axis,
velocity $plus.minus 3$ m/s, drag acceleration $plus.minus 0.6$ m/s#super[2]; and the
navigation-filter gain draw to $plus.minus 0.45$ absolute ($3sigma$).

*Dispersion rationale.* The $plus.minus 50%$ nominal density span is a deliberately conservative
envelope for Mars seasonal and dust-loading variability about a tabulated mean profile
@forget1999mars; domains are drawn independently as a modeling choice (no cross-correlations are
claimed). The static bias and the Ornstein--Uhlenbeck process model different frequencies --
profile-scale uncertainty versus along-track variability -- and do not double-count; the process
is zero-initialized at entry and reaches its stationary variance within roughly one correlation
time. The two wind draw dimensions are retained inert to keep the $26$-dimensional draw layout
stable across configurations with and without the wind model.

*Classical tuning parity.* Every classical scheme was tuned by the same genetic algorithm on the
same cost $C$ above -- identical penalties, virtual costs, and transform -- co-optimizing its full
parameter set including the shared navigation-filter and actuator gains the network also tunes
($26$ parameters for FTC against the network's weights plus $3$ actuator-side parameters), at
$2000$ generations $times 300$ individuals $times 10$ scenarios per generation. The classical
searches plateau far inside that budget (FNPAG's best individual stopped improving before
generation $60$), so the network's longer run buys the comparison nothing. The joint-reference
co-optimization of Section 7.1 is the classical counterpart of the network's co-tuned actuator
parameters, and the information asymmetry is tested directly: retrained without the three
predicted-$Delta v$ observations the classical schemes cannot consume, the network still beats the
co-tuned joint-FTC on every reported statistic (Section 6.3, third control).

*Correction burns.* The reported $Delta v$ is the three-burn plan from the captured orbit (apsis
radii $r_a$, $r_p$) to the $500$ km circular parking orbit (radius $r_c$), using the elliptical
apsis speed $v(r_1, r_2) = sqrt(2 mu r_2 \/ (r_1 (r_1 + r_2)))$: a periapsis correction at
apoapsis, $Delta v_1 = v(r_a, r_c) - v(r_a, r_p)$; circularization at the new periapsis,
$Delta v_2 = sqrt(mu \/ r_c) - v(r_c, r_a)$; and a plane change $Delta v_3 = 2 v_n sin(Delta i \/ 2)$
at the cheaper node, $v_n$ the smaller of the target-orbit speeds at the two nodes. The total is
$|Delta v_1| + |Delta v_2| + |Delta v_3|$.

*Cost and objective.* Per simulation the cost is
$ C = Delta v + s(Delta v - T) + s(Delta v - T)^2 / (2 S) + sum_j w_j thin s((x_j - L_j) \/ L_j), $
with $s$ a softplus (a $C^oo$ knee), $T = 1000$ m/s, $S$ the quadratic scale, and the penalty sum
over the peak heat-flux, peak-load, and integrated heat-load exceedances normalized by their limits
$L_j$, weights $w_j = 1$. Non-captures receive a virtual cost above any captured correction:
$10\,000 + v_oo$ m/s for hyperbolic escape ($v_oo$ the excess speed), and
$3000 + 1000 min(|E - E_"target"|, 50) - 500 thin t\/t_max$ for crash and timeout outcomes
(energies in MJ/kg) -- so captures are always preferred while the optimizer keeps a gradient
toward the capture boundary. The per-simulation cost is cubed (the deployed transform) and
aggregated across an individual's scenario batch by root-mean-square -- the $L_6$-norm-equivalent
objective of @eq-objective.

*Optimizer (deployed headline).* pymoo genetic algorithm (SBX crossover $eta = 3$, polynomial
mutation $eta = 5$ at rate $0.15$), population $512$, two scenarios per individual per generation,
run to plateau ($15\,000$--$20\,000$ generations). The adaptive seed curation, precisely: every
second generation or on a promotion, (1) draw $1000$ fresh probe scenarios outside every reserved
stream; (2) score them with the current best individual; (3) sort the per-scenario costs into
$n_"sims"$ equal-count quantile bins; (4) take the hardest scenario of each bin as the next
training batch -- scenarios are replaced as a set, never accumulated. The selection gate re-runs
each new argmin on the reserved $n = 1000$ selection pool and promotes on strict RMS improvement.

*Training harness.* Population evaluation is batched through in-process Python bindings to the
native core: each generation's individuals are simulated scenario-parallel across CPU cores with the
interpreter lock released, network weights pass in memory rather than through files, and the
dispersion-independent tables (atmosphere, winds, reference trajectories) are shared read-only
across the batch. At the measured single-core cost of Section 7.2, the headline run's
$approx 2 times 10^7$ dispersed passes amount to roughly twenty core-hours -- an overnight training
on a laptop.

*Deployed architecture.* Dense($17 -> 16$, swish) $->$ Mamba($d_"inner" = 16$, $d_"state" = 12$)
$->$ Dense($16 -> 2$, asinh), atan2 bank decoder; $962$ trainable parameters. The input mask selects
$17$ of the $35$ candidate observations (indices 0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27--30, 32--34:
instantaneous orbital/aerodynamic state, two reference-trajectory interpolations, the seam-free
bank-history pairs, and the three predicted correction-$Delta v$ components). Each input is
normalized by a calibrated per-input affine or asinh transform mapping its $[p_5, p_95]$ span to
$[-1, 1]$. Three actuator-side parameters -- the navigation density-filter gain and its rate limit,
and the command-shaping acceleration limit -- are co-optimized with the network weights.

*FNPAG.* Two-second replan cycle; bisection corrector on the constant capture-bank magnitude (about
eleven forward integrations per replan); forward-predictor integration step GA-tuned at $3.8$ s; the
onboard atmosphere is scaled by the navigation-estimated density factor so the predictor tracks the
measured atmosphere.

*One runtime, training to flight.* The policy that flies is the artifact that trained: candidates
are evaluated, selected, and deployed through the same native forward pass, so no export or
re-implementation step separates the training loop from the flight code. An independent PyTorch
implementation of every cell type (used by the supervised warm-start path, not by the deployed
runs) is held to numerical agreement with the native runtime by cross-language tests -- maximum
absolute forward-pass difference near machine epsilon ($10^(-16)$--$10^(-14)$) over $100$-step
stateful sequences.

*Timing.* Wall-clock per simulation over $200$ sequential runs of each deployed scheme on one idle
core of an Apple-silicon laptop; no parallelism.

*Artifacts.* The simulator, training harness, analysis code, every configuration, the deployed
network weights, the committed per-run evaluation records behind each table, and the scripts that
regenerate every figure and number are released publicly under an MIT license (repository URL in
the camera-ready). Every number in the tables regenerates, without retraining, from those records.

#pagebreak(weak: true)
= Appendix B: architecture probes -- CfC, xLSTM, and the Mamba-3 axes

A controlled negative result supporting the Section 6 headline: does any recent recurrent family
beat the plain selective SSM on the sizing tail? Three probes, each anchored on a Section 6 cell at
matched parameter budget: a closed-form continuous-time cell (CfC @hasani2022cfc, hypothesis:
input-dependent time constants suit the fast-near-periapsis, static-in-vacuum phase structure)
against the GRU anchor; the exponential-gated sLSTM and matrix-memory mLSTM of xLSTM @beck2024xlstm
(hypothesis: sharp revision of a stored estimate at the bounce or a density shock) against the LSTM
anchor; and a $2 times 2$ over Mamba-3's @lahoti2026mamba3 two axes -- exponential-trapezoidal
discretization and complex (rotational) state -- at the deployed Mamba anchor, whose euler-real arm
is bit-identical to the deployed cell.

Every arm shares one training regime (the genetic algorithm at population $300$ for $5000$
generations, two scenarios per individual, adaptive hardest-seed curation, live actuator
scaffolding) and one reserved evaluation pool (offset $10^7$, $n = 1000$, each arm scored with its
co-trained scaffolding), with three seed-repeats per arm; $sigma_"run"$ is the standard deviation
over repeats. Capture is $99.97$--$100%$ everywhere and every arm passes the feasibility check of
Section 6.2 -- zero heat-flux, g-load, and heat-load violations on every repeat -- so the comparison
is a pure tail-$Delta v$ story. Two scopes apply throughout: these are $p_95$/$"CVaR"_95$ statistics
at $n = 1000$, not the far-tail $"CVaR"_(99.9)$ sizing metric, and the probe budget is deliberately
sub-headline ($1.5$M evaluations versus the deployed $512 times 20\,000$), so the absolute values
sit above the headline numbers and are not mission figures.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Arm*], [*Params*], [*Capture %*], [*Viol. %*], [$bold(p_50)$], [$bold(p_95 plus.minus sigma_"run")$], [$bold("CVaR"_95 plus.minus sigma_"run")$],
    ),
    table.hline(stroke: 0.35pt),
    [Mamba (baseline)], [962], [99.97], [0.0], [114.0], [$121.6 plus.minus 0.5$], [$124.1 plus.minus 0.3$],
    [Mamba-3 trapezoidal], [978], [100.0], [0.0], [115.5], [$124.9 plus.minus 2.1$], [$128.8 plus.minus 2.3$],
    [Mamba-3 complex], [1154], [100.0], [0.0], [113.7], [$121.1 plus.minus 1.8$], [$123.8 plus.minus 2.1$],
    [Mamba-3 both], [1170], [100.0], [0.0], [114.0], [$121.6 plus.minus 1.3$], [$124.2 plus.minus 1.2$],
    [GRU (baseline)], [1014], [100.0], [0.0], [114.7], [$123.7 plus.minus 1.5$], [$126.7 plus.minus 1.3$],
    [CfC], [1003], [100.0], [0.0], [116.5], [$126.5 plus.minus 0.7$], [$130.4 plus.minus 0.1$],
    [LSTM (baseline)], [1082], [100.0], [0.0], [115.4], [$124.3 plus.minus 1.4$], [$127.3 plus.minus 1.4$],
    [sLSTM], [1042], [100.0], [0.0], [115.7], [$124.7 plus.minus 2.6$], [$127.6 plus.minus 3.1$],
    [mLSTM], [1078], [100.0], [0.0], [118.1], [$127.4 plus.minus 3.5$], [$130.8 plus.minus 4.8$],
    table.hline(stroke: 0.7pt),
  ),
  caption: [The nine probe arms: three-seed means, correction $Delta v$ in m/s on the shared
  $n = 1000$ probe pool. Baselines are retrained in-regime (not the deployed champions -- see the
  budget caveat below). Viol. is the any-constraint violation rate; every arm is feasible on every
  repeat.],
) <tbl-probes>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, left),
    table.hline(stroke: 0.7pt),
    table.header(
      [*Treatment vs baseline*], [*Metric*], [*Gap*], [$bold(sigma_"run")$], [*Verdict*],
    ),
    table.hline(stroke: 0.35pt),
    [CfC vs GRU], [$p_95$ / $"CVaR"_95$], [$+2.8$ / $+3.7$], [$1.6$ / $1.3$], [*significantly worse*],
    [Trapezoidal vs Mamba], [$p_95$ / $"CVaR"_95$], [$+3.3$ / $+4.8$], [$2.2$ / $2.3$], [*significantly worse*],
    [Complex vs Mamba], [$p_95$ / $"CVaR"_95$], [$-0.5$ / $-0.3$], [$1.9$ / $2.1$], [within $sigma_"run"$],
    [Both vs Mamba], [$p_95$ / $"CVaR"_95$], [$+0.0$ / $+0.1$], [$1.4$ / $1.2$], [within $sigma_"run"$],
    [sLSTM vs LSTM], [$p_95$ / $"CVaR"_95$], [$+0.4$ / $+0.3$], [$2.9$ / $3.4$], [within $sigma_"run"$],
    [mLSTM vs LSTM], [$p_95$ / $"CVaR"_95$], [$+3.1$ / $+3.5$], [$3.8$ / $5.0$], [within $sigma_"run"$ (high variance)],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Within-family significance, the rigorous claims: gap = treatment minus baseline
  (positive = worse), cleared when $|"gap"| > sqrt(sigma_"base"^2 + sigma_"arm"^2)$ (the tabulated
  $sigma_"run"$).],
) <tbl-probes-sig>

The within-family rows are the rigorous claims. The CfC is significantly worse than the GRU on both
tail statistics, and its tiny $sigma_"run"$ ($0.1$ on $"CVaR"_95$) makes the loss stable and
reproducible, not a bad-luck draw: continuous-time time constants add a harder optimization
landscape for a timescale adaptation the fixed-cadence gates already learn. Mamba-3's trapezoidal
discretization is significantly worse than the plain cell -- the seed-repeats upgrade an earlier
single-run "no benefit" to a measured degradation. Complex state, both axes combined, the sLSTM, and
the mLSTM all sit within run variance of their baselines; the mLSTM is notably high-variance
($sigma_"run"$ up to $4.8$) and trends worse without clearing the bar -- matrix memory buys
instability, not tail robustness. Cross-family, all nine arms share the regime and pool but the
anchors differ slightly ($962$/$1014$/$1082$ parameters), so the ranking is suggestive rather than
matched: the plain Mamba tops the field (tied with its own complex arms), about $2$ m/s ahead of the
GRU and $2.7$ ahead of the LSTM at $p_95$ -- consistent with the deployed headline.

One caveat is load-bearing. Each probe also scored its deployed higher-budget champion as a
reference row, and those sit $4$--$6$ m/s better at $p_95$ than the retrained in-regime baselines
(Mamba $121.6$ versus $116.6$; GRU $123.7$ versus $117.3$; LSTM $124.3$ versus $120.2$) -- a pure
training-budget effect (the champions had roughly $3.4 times$ the evaluations), not architecture.
That gap is exactly why every treatment compares against its retrained in-regime baseline and never
against a champion. Two smaller scopes: the probe cells are trained through the gradient-free path
only (no warm-start), and the sLSTM's $40$-parameter deficit against the LSTM is a cell-definition
cost (single bias), not a budget mismatch -- it does not explain its null.

The consistent null across three independent families points at the task, not the cells. A single
atmospheric pass is a few hundred guidance ticks whose latent state worth remembering is a handful
of slowly-varying dispersion parameters -- density bias, the Ornstein--Uhlenbeck perturbation state,
aerodynamic dispersions -- and the engineered inputs already carry most of the temporal signal
(Section 8). A diagonal selective-SSM state of dimension $12$--$16$ saturates what little internal
memory the problem rewards; the CfC's timescale adaptation, the xLSTM's revision and associative
recall, and Mamba-3's long-context and state-tracking axes all target capacity this smooth,
low-bandwidth control signal never exercises. The deployed cell wins not by more sophisticated
memory but by just enough memory, cheaply. Framed positively, the probes validate the methodology:
the adaptive-seed, tail-led, matched-anchor protocol distinguishes between architectures rather than
rubber-stamping the newest one -- three 2024--2026 recurrent families tried, two rejected as
significantly worse, none better.

#pagebreak(weak: true)
= Appendix C: per-scheme mission reports

Each scheme below gets a two-page mission-performance card on the final-evaluation
Monte Carlo pool ($n = 1000$), pinned to its deployed policy so the statistics
reproduce @tbl-perf; the FTC, PredGuid, and energy-controller cards use their
co-optimized-reference variants (Section 7.1), matching those rows of @tbl-perf. The first page shows the corridor behaviour -- the classified
trajectory ensemble in the (energy, dynamic pressure), (energy, inclination), and
(energy, bank) planes, with the undispersed nominal overlaid; the dynamic-pressure
panel sets the ensemble against the shared occupancy envelope of @fig-corridor,
so one can read at a glance whether each scheme flies inside it. The second page shows
the correction-$Delta v$ distribution (total and its three burns), the thermal and
load-constraint margins, and the full statistics. Panels reuse the training
pipeline's own report charts; captured trajectories are blue, constraint violations
orange, failures red.

#include "appendix.typ"
