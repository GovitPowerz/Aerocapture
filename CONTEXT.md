# Aerocapture

Trajectory simulation and guidance optimization for aerocapture maneuvers: a spacecraft enters a planet's atmosphere at hyperbolic speed and uses aerodynamic lift, modulated by bank angle, to capture into a target orbit.

## Language

### Flight & Trajectory

**Aerocapture**:
A single atmospheric pass that converts a hyperbolic arrival into a bound orbit.

**Entry**:
The atmospheric-interface state (velocity, flight-path angle, position) at which a simulation starts.

**Bounce**:
The trajectory nadir: the minimum-altitude point, after which the vehicle ascends toward atmosphere exit.

**Capture Phase**:
The guidance phase from entry until the post-bounce handoff to exit guidance.

**Exit Phase**:
The guidance phase on the ascending leg, shaping the exit orbit.
_Avoid_: ascent phase

**Captured**:
The terminal outcome where the vehicle exits into a bound orbit. Other terminal outcomes: crash, hyperbolic (escape), timeout, pending crash.
_Avoid_: successful, converged

**Nominal**:
A run with all Monte Carlo dispersions off.
_Avoid_: baseline (reserved for comparison anchors)

**Corridor**:
The flyable envelope between the crash boundary (full lift-down) and the escape boundary (full lift-up), indexed by energy.

**Commanded Bank**:
The bank angle guidance orders after shaping.

**Realized Bank**:
The bank angle the pilot dynamics actually achieve.
_Avoid_: actual bank

### Guidance & Control

**Guidance Scheme**:
One of the selectable bank-angle guidance laws (FTC, NN, EqGlide, Energy Controller, PredGuid, FNPAG, Piecewise Constant, CPAG).
_Avoid_: guidance algorithm ("algorithm" is reserved for the optimizer)

**Optimizer Algorithm**:
The population-based trainer (GA, CMA-ES, DE, PSO, QPSO, islands).
_Avoid_: using bare "algorithm" for a guidance law

**Unsigned-Magnitude Scheme**:
A scheme that emits only |bank|; lateral guidance picks the sign, and exit guidance and the thermal limiter apply.

**Signed-Bank Scheme**:
A scheme that emits a signed bank and bypasses lateral, exit, and thermal-limiter guidance.

**Roll Reversal**:
A commanded bank-sign flip, issued by lateral guidance to bound inclination error.

**Scaffolding**:
The tuned non-NN pipeline (navigation filter, lateral, exit, thermal limiter, command shaping) that an NN scheme reuses frozen or co-optimizes.

**Exit-Bank Teacher**:
The always-live exit-law signal fed to the NN as a candidate input.
_Avoid_: supervisor (a different concept)

**Supervisor**:
A classical scheme whose flown trajectories provide the behavioural-cloning targets during warm-start.
_Avoid_: teacher (reserved for the exit-bank input signal)

**Warm-Start**:
Supervised pretraining of the NN on supervisor trajectories before population optimization begins.

**Onboard Model**:
The atmosphere model available to navigation and guidance, as opposed to the dispersed truth the physics propagates.

### Monte Carlo & Optimization

**Dispersion**:
A per-run random perturbation of one physical or sensor quantity; a draw is the full vector of them.

**Seed Pool**:
A reserved set of Monte Carlo seeds dedicated to one purpose (training, validation, final eval, warm-start, calibration), disjoint from the others.

**Seed Strategy**:
The policy for choosing training seeds each generation: fixed, rotating, or adaptive.

**Curation**:
The adaptive strategy's refresh of the training seed list by quantile-stratified sampling of the probe-cost CDF.

**Cost**:
The scalar per-sim optimizer objective: DV plus constraint penalties, optionally rescaled by a cost transform.
_Avoid_: fitness; loss (reserved for the supervised warm-start objective)

**DV**:
The correction delta-v (m/s) of the post-capture cleanup burns; the physical objective.

**Virtual DV**:
The synthetic DV assigned to non-capture outcomes so every termination stays comparable in cost.

**Sizing Tail**:
The upper tail of the DV distribution (p95, CVaR95 and beyond) that sizes propellant mass; the headline comparison metric.
_Avoid_: leading comparisons with the median

**Validation Gate**:
The in-training re-evaluation of a new candidate on the validation pool; promotion requires strict improvement.

**Champion**:
The best validated individual so far in a training run.
_Avoid_: best, winner (until final selection)

**Final Selection**:
The end-of-training re-ranking of the last generation plus the champion on the validation pool; decides the deployed winner.

**Final Eval**:
Report-only scoring of the deployed winner on the reserved final-eval pool; never chooses anything.
_Avoid_: conflating with final selection

**Reference Trajectory**:
The energy-indexed table (pdyn, hdot, cos-bank) that ref-tracking schemes follow.
_Avoid_: bare "reference"

**Reference Implementation**:
The legacy code the Rust simulator was validated against at bit level.
_Avoid_: bare "reference"

**Cell**:
One deployed training run in a campaign grid (e.g. a sweep cell), pinned by its config and output directory.

**Probe**:
A budget-matched experimental training arm comparing one architecture axis against a sweep-cell anchor.
