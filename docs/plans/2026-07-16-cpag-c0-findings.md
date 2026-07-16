# CPAG Stage C0 findings — formulation + solver spike

Date: 2026-07-16. Deliverables per TODO.md Stage C0: (1) Python prototype of the
paper's convexified formulation on our dynamics with SCP convergence verified
across the corridor; (2) embedded-solver pick with measured wall time and
iteration spread at the real problem size. Both delivered; C1 is unblocked.

Prototype: `src/python/aerocapture/cpag/` (`model.py` dynamics, `scp.py`
formulation + loop, `studies.py` convergence battery, `bench.py` solver bench,
`python -m aerocapture.cpag` driver). Artifacts quoted below live in
`training_output/cpag_c0/` (untracked); regenerate with the driver.

## 1. Source formulation (paywall workaround: the thesis IS the paper)

The JGCD paper (Rataczak, McMahon & I. Boyd — the hypersonics Boyd, not the
convex one — doi:10.2514/1.G008685) is paywalled, but Rataczak's CU Boulder PhD
dissertation Chapter 6 is the same work (matching abstract, section structure,
and results; publication list identifies J2 = the JGCD paper). Free PDF:
https://www.colorado.edu/lab/ngpdl/media/209 — Appendix B carries every
analytic Jacobian, the STM discretization recipe, weights, and scaling. Key
formulation facts adopted:

- **Bank angle is a STATE, bank rate is the control** — one solve returns a
  SIGNED continuous profile; there is no separate lateral/reversal logic at
  all. Reversals emerge from the inclination terms. (The prediction model must
  therefore carry the lateral lift term, unlike FNPAG's zero-lateral
  predictor.)
- **Terminal apoapsis via an energy/angular-momentum surrogate** eps(x_f) =
  E + mu/ra_t − h²/(2 ra_t²): zero exactly when the osculating apoapsis equals
  the target, smooth and finite through escape velocity — kills the
  apoapsis-Jacobian singularity/sentinel-plateau problem that FNPAG dodges with
  bisection. The thesis shows Keplerian exit-DV predictions err up to ~25% at
  Neptune and recommends this form for ANY NPC scheme (cheap FNPAG upgrade,
  noted for C2).
- **Delta formulation about the predictor's own propagated (feasible)
  reference** — zero correction always feasible, no virtual control needed
  (inherited from Tracy's CPEG).
- Path constraints in the paper: load factor (via an a² augmented state) +
  terminal integrated heat load, L1-slacked. Terminal + intermediate
  inclination penalties (cos-space, last 20% of nodes).
- Trust region: penalized per-node SOC (PTR); N = 50 nodes, FOH control, time
  dilation for free final time; Convex.jl + **Clarabel**; ~18.8 ms mean per
  predictor-corrector iteration (Julia, M2 Pro).

## 2. Prototype deviations (deliberate, all documented in code)

On our dynamics: full repo EOM (J2–J4 — richer than the paper's J2-only —
rotation, mission atmosphere table, geodetic-altitude density lookup) with the
onboard density scaled by the nav density factor (the repo's FNPAG lesson,
playing the paper's fading-memory-filter role). Deviations from the paper:

- Free final time by **re-timing the grid through the predictor each
  iteration** (absolute-time ZOH segments, partial last segment at the exit
  event) instead of a time-dilation parameter. Converges fine; simpler.
- ZOH bank-rate instead of FOH; FD-based exact-discretization Jacobians
  (batched RK4) instead of analytic + STM integrals. C1 can go analytic later
  for speed; FD honesty was verified to 1e-4 in scaled units per segment.
- Path constraints: heat flux + g-load rows linearized directly (no a²
  augmentation) + terminal heat load on the Q state. **pdyn deliberately NOT
  enforced**: the mission nominal peaks at 1.63 kPa vs the 1.081 kPa config
  value and the GA cost function never penalizes pdyn — enforcing it saturates
  slacks with an unsatisfiable row and drowns the merit. Config knob exists.
- L1 (not L2-norm) penalty on intermediate inclination slacks, **with a
  deadband at the mission tolerance** (0.5°-equivalent in cos-space). Without
  the deadband the lambda term dominates the merit with physically unavoidable
  mid-arc error re-sampled on a re-timed grid — pure noise that blocked
  convergence.
- **Box trust region + greedy merit accept/reject** ("box" mode, pure QP) as
  the workhorse instead of the paper's PTR. PTR is implemented ("ptr" mode,
  SOC per node, Clarabel-only) but with my weights it either freezes (w_tr >=
  0.1) or diverges through crash references (w_tr = 1e-3): the missing
  time-dilation DOF + weight sensitivity make it strictly worse here. Not
  worth tuning further for C1 given box-mode results.

Three fixes the paper never needed (its references start benign; corridor-edge
replans don't):

- **Bank box |sigma| <= 180 deg per node**: without it the optimizer WINDS the
  bank through full turns (each 360° sweep is merit-free, each linearization
  sees a local gain via the lateral term) — observed sigma_end = 729° before
  stranding.
- **Grid-size-invariant merit** (terminal values + per-node means/peaks, never
  node sums) + a crash-tier offset so any exit outranks any crash: node-sum
  merits penalize surviving LONGER, which blocks recovery from crash-truncated
  references; inclination terms are gated off on crash references (survival
  first).
- **Constant-bank grid seeding** (roll-to-{0,45,75,105,135,180}° shoots,
  ~25 ms each; best merit wins as the initial reference — FNPAG's monotone
  apoapsis-vs-bank bracket recycled as an SCP initializer). Kills the
  stranded-mid-corridor local optima on cold starts. In the C1 guidance loop
  this fires on the first call only; later replans warm-start from the
  previous profile.

## 3. Convergence results (cold replans, hold-current-bank + grid seeds)

States harvested from Rust-simulated trajectories (the truth plant), each
classified against the physically achievable apoapsis bracket (full-lift-up /
full-lift-down max-authority shoots) so "SCP failed" is never conflated with
"target unreachable". Feasibility gate: |apo err| <= 10 km (FNPAG's tolerance;
mission success tolerance is 100 km), inclination err <= 0.5°, enforced path
peaks <= 1.01, exit reached.

Nominal entry replan: converged + feasible in 9 SCP iterations, apoapsis error
**-0.04 km**, inclination error 0.0003°, all constraints inside limits.

Undispersed constant-bank sweep (corridor sentinel range 0-180° x ~60 s
epochs, 55 states):

| class         | n  | settled | feasible | apo err p50/p95 (km) | least-bad gap p50/p95 (km) |
|---------------|----|---------|----------|----------------------|-----------------------------|
| reachable     | 25 | 100%    | 88%      | 0.40 / 1.8           | —                           |
| over_reach    | 14 | 100%    | 0%       | (energy excess)      | 0.000 / 0.005               |
| unrecoverable | 16 | 100%    | 0%       | (crash inevitable)   | —                           |

Dispersed FTC-guided MC batch (40 sims, full medium dispersions, nav density
factor from the trajectory's estimator column, 317 states):

| class       | n   | settled | feasible | apo err p50/p95 (km) | least-bad gap p50/p95 (km) |
|-------------|-----|---------|----------|----------------------|-----------------------------|
| reachable   | 135 | 100%    | 97.8%    | **0.01 / 0.5**       | —                           |
| under_reach | 73  | 100%    | 0%       | 69 / 128             | 0.001 / 1.7                 |
| over_reach  | 109 | 100%    | 0%       | 133 / 251            | 0.002 / 10.4                |

Read: **100% of 372 replans settle** (median 3-9 SCP iterations, p95 <= 14).
When the target is physically reachable the replan hits it to ~10 m median /
500 m p95; when it is not (57% of mid-flight dispersed states at the ±10 km
gate — FTC's own dispersion is ±150 km), the replan saturates within METERS
(p50) of the best physically achievable apoapsis. Residual failures: 6/160
reachable-class states — 3 inclination-only misses (1-3.8°) on states that
flew 150 s of constant bank with no reversals (lateral authority already
spent; a real guidance loop replanning at cadence never reaches such states),
2 marginal ~25 km apo misses on degenerate histories, 1 stranded constant-72°
sweep state (-365 km). None are formulation blockers; all are cold-start
artifacts C1's warm-started loop avoids.

The inclination channel works: entry replan reaches 0.0003° inclination error
with reversals emerging from the solve — no lateral logic anywhere.

## 4. Solver spike (real instances, canonicalized matrices only)

Captured live subproblems (first/mid/last SCP iteration of the entry replan) at
seg_dt = 8 s (N≈50, the paper's size; 480-700 vars) and 4 s (950-1400 vars),
25 repeats each, M-series laptop:

| instance        | vars | Clarabel p50/p95 (ms) | IP iters | OSQP cold (ms) | OSQP warm (ms) |
|-----------------|------|-----------------------|----------|----------------|----------------|
| box dt8 first   | 483  | 3.5 / 3.7             | 15       | 249 @ 20k cap  | 85 @ 6.8k      |
| box dt8 mid     | 696  | 3.9 / 4.1             | 13       | 365 @ 20k cap  | 365 @ 20k cap  |
| box dt8 last    | 696  | 7.0 / 7.4             | 25       | 369 @ 20k cap  | 221 @ 12k      |
| box dt4 mid     | 1379 | 12.0 / 12.2           | 20       | 746 @ 20k cap  | 31 @ 850       |
| ptr dt8 (SOCP)  | ~570 | 4.3-12.9              | 15-39    | n/a            | n/a            |

**Verdict: Clarabel, box-trust QP variant, N≈50.**

- Clarabel solves every instance in 3.5-7 ms p50 with p95 within ~10% of p50
  (tight spread — the WCET story the TODO flags as CPAG's certification risk
  is *measurably* benign at this size: 13-25 IP iterations, never more).
- OSQP (ADMM) hits its iteration cap on most instances and — worse — its
  loose solutions break SCP convergence parity: driving the same loop with
  OSQP at eps 1e-5 strands the entry replan at +11,182 km; tightening to 1e-7
  burns 100k iterations without solving. The long-STM-chain conditioning is
  hostile to first-order methods. Salvageable only with problem-specific
  preconditioning work there is no reason to do:
- a problem-shaped custom QP is **not warranted** — Clarabel is pure Rust
  (crate, no FFI), already meets the 2 s replan cadence with ~2 orders of
  magnitude headroom, and its Python bindings ARE the Rust solver, so these
  timings transfer to C1 directly.

Rust C1 per-replan projection: SCP iterations (median 8-9 cold, 1-3 warm in a
guidance loop) x (QP solve 4-7 ms + shoot/linearize, sub-ms in Rust) ≈ 30-70 ms
cold first call, ~5-20 ms per subsequent replan — consistent with the paper's
18.8 ms/iteration in Julia. FNPAG's ~0.27 ms/replan stays ~30-100x cheaper;
CPAG buys in-loop constraint enforcement + native lateral for that. Training
wall-time (C2) needs a plan: at ~10-30x FNPAG's sim cost, GA budgets must be
sized accordingly (fewer seeds/gen or reduced N during training).

## 5. C1 recommendations

1. Port the box-mode QP formulation exactly as prototyped (variable layout,
   scaling, and canonical (P, q, A, b, cones) construction are in
   `scp.py::build_qp`/`canonicalize`); use the `clarabel` crate directly.
2. Keep: eps terminal surrogate, sigma box, deadbanded inclination corridor,
   grid-invariant hierarchical merit, greedy accept/reject, bracket-grid
   seeding on first call, warm-start from the held profile between replans,
   FNPAG's replan-throttle + nav-density-factor scaling.
3. Analytic Jacobians (thesis Appendix B) are an optimization, not a
   correctness need — FD linearization honesty measured at 1e-4.
4. Skipped-for-C0, revisit in C1/C2: FOH control, time-dilation DOF (would
   likely rehabilitate PTR mode), the paper's first-call ACCD end-bank
   conditioning, eps with J2 correction, pdyn constraint decision, and the
   eps-surrogate upgrade for FNPAG's own corrector.
