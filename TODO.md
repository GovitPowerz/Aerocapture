# TODO

- [x] implement CPAG (Convex Predictor-Corrector Aerocapture Guidance) as the 8th guidance scheme
      (C0-C2 complete 2026-07-19; tail verdict negative -- see the CPAG section below)

---

## Backlog

- [ ] Add neural counterparts for navigation and control
- [ ] Develop ESR (Earth Sample Return) mission profiles
- [ ] Fix pre-existing `cargo clippy --workspace` warnings in `src/rust/aerocapture-py/src/lib.rs`
      (2x `type_complexity`, 1x `needless_range_loop`; `check_all.sh` scopes to `-p aerocapture`)

---

## CPAG -- Convex Predictor-Corrector Aerocapture Guidance (classical track)

**End goal:** implement CPAG (Rataczak, McMahon & Boyd, JGCD 2025, doi:10.2514/1.G008685;
`@rataczak2025cpag` in the paper's related work) as the 8th guidance scheme -- a convexified
constrained replan (bank profile with heat-flux / g-load / heat-load path constraints enforced
in-loop) -- and evaluate it as the classical architecture developed further, replacing FNPAG in
that role. **C2 verdict (2026-07-19): scheme shipped and honest, but the tail evidence does
NOT support the crowning -- see docs/plans/2026-07-19-cpag-c2-results.md.** Rationale: constraint handling is the structural gap in the classical stack (the thermal
limiter is a bolt-on ramp outside guidance; the paper's LSTM feasibility asterisk shows soft
penalties do not enforce feasibility), and the confirmatory campaign demoted FNPAG (deep tail
fattens 165 -> 198.7 at 1e6; bang-bang + bisection has no headroom for in-loop constraints).
Known trade, carried deliberately: less high-fidelity 6-DoF / flight-computer validation in the
public literature than FNPAG -- partly structural (iterative solver, variable iteration count ->
WCET certification risk). This repo's MC harness can generate exactly the missing validation
evidence.

**Paper lessons that transfer on day one:**
- Scale the predictor's atmosphere by the nav-estimated density factor (the fix that took FNPAG
  from losing to FTC to beating it; density is the dominant apoapsis-error driver, corr -0.72).
- GA-co-tune CPAG's knobs (weights, trust region, targets) + nav scaffolding like every other
  scheme -- an untuned CPAG repeats the untuned-baseline fallacy the paper criticizes.
- Benchmark feasibility-first and tail-led: confirmatory-style pools, CVaR99.9, against
  joint-FTC AND the deployed Mamba -- not against FNPAG on means. Expect fatten-with-depth
  until measured otherwise.
- Deployability row from the start: per-replan wall time + iteration-count distribution
  (FNPAG's ~0.27 ms/replan is the bar; the paper's triangle gets a CPAG vertex).

### Stage C0 -- formulation + solver spike (DONE 2026-07-16, findings: docs/plans/2026-07-16-cpag-c0-findings.md)
- [x] Python prototype of the paper's convexified formulation on our dynamics; verify SCP
      convergence across the corridor (undispersed sweep + a dispersed batch)
      -- `src/python/aerocapture/cpag/`; 372 cold replans, 100% settle; reachable-target states
      hit apoapsis to 10 m p50 / 500 m p95 with inclination in-loop (no lateral logic);
      unreachable-target states saturate within meters of the physical optimum
- [ ] Optional dependent follow-up: energy-based eps apoapsis surrogate as a drop-in upgrade for
      FNPAG's own corrector (thesis Sec. 6.5: Keplerian exit-DV predictions err up to ~25%)
- [x] Pick the embedded solver: Clarabel (pure Rust, SOCP) vs OSQP binding vs problem-shaped
      custom QP; measure per-solve wall time and iteration spread at the real problem size
      -- Clarabel, box-trust QP variant, N~50: 3.5-7 ms p50 per solve, 13-25 IP iterations,
      p95 within ~10% of p50; OSQP-ADMM caps out (20k iters) AND breaks SCP parity; custom QP
      not warranted (2 orders of magnitude cadence headroom, pure-Rust crate)

### Stage C1 -- Rust scheme MVP (DONE 2026-07-16)
- [x] `cpag.rs`: SCP replan on a `replan_period` cadence (FNPAG throttle pattern with the PROFILE
      as the held object: plan playback between replans, clamped to the sigma box), onboard
      atmosphere scaled by the nav density factor -- box-trust Clarabel QP per the C0 pick;
      warm replans escalate to the cold budget + constant-bank grid seeding when the held plan
      crashes (without this, capture on the 50-sim medium-dispersion sanity pool was 78%; with
      it 100%, dv p50 150.8 / p95 223.7)
- [x] Path constraints wired to `[flight.constraints]` (heat flux + g-load rows, terminal heat
      load on the Q state; pdyn off by default -- unsatisfiable on this mission); `[guidance.cpag]`
      TOML params + `param_spaces.py` specs (12 genes: 9 CPAG + nav/shaping; cadence knobs deliberately
      not genes) + routing entry + `compare_guidance` + `train_all.sh cpag` registration
- [x] Unit tests (constraint activation with a satisfiable limit, replan throttle, vacuum hold,
      sigma-box playback, crash-tier convergence fallback, eps identities, Clarabel smoke,
      proptest bounds) + golden config `configs/test/test_cpag_golden.toml` (7th golden).
      Verification: check_all green, lint green, 1284 fast Python tests; head-to-head on 100
      identical medium-dispersion scenarios: UNTUNED CPAG dv 157.8 m/s / apo err 21 km / 100%
      capture vs tuned FTC 174.4/108 and tuned FNPAG 124.3/29; nominal apo err 0.22 km, inc err
      0.019 deg, dv 143.3. Wall cost ~3.5 s/sim (~40x FNPAG) -- size C2 budgets accordingly

### Stage C2 -- training + benchmark (DONE 2026-07-19, results: docs/plans/2026-07-19-cpag-c2-results.md)
- [x] Feasibility-aware validation gate (IMPROVEMENTS 9.14) -- `[optimizer] max_violation_rate`
      ceiling enforced at the validation gate, the islands `validate_each`, final selection
      (records-based, `winner_feasible` sidecar flag), AND the pre-loop initial-champion
      validation (the hole was live: the first C2 launch anchored a 5.75%-heat-load-violating
      gen-0 argmin through the ungated path). Campaign tally: 3 feasible promotions vs 21+
      infeasible rejections -- the soft cost penalty does not hold the feasibility line
- [x] GA-tune under the deployed regime -- 400 gens x 48 x 2 sims, adaptive/max, cubed,
      ceiling 1% (seed_pool_interval 2 -> 25 for the wall cost). Final eval (1000 disjoint):
      99.7% capture, dv p50 143.2 / p95 190.3, flux 0% / g 0% / heat load 0.1% violations
- [x] Head-to-head vs joint-FTC / FNPAG / deployed Mamba on shared 10x1000 pools
      (articles/paper/data/cpag_confirmatory.json; tail metric caps at cvar99 at this depth):
      **CPAG LOSES THE TAIL** -- cvar95 228.8 vs jointFTC 144.9 / FNPAG 143.9 / Mamba 115.9
      (paired +83.7/+84.8/-112.7 m/s, SE ~3.9); capture 99.81% (16 physical + 3 timeout of
      10k) vs 99.96-100%; median parity (p50 ~143). The tail is the crash-escalation rescue's
      bill (recovered-but-expensive 300-900 m/s captures). At ~40x FNPAG compute, in-loop
      constraint enforcement did not pay on this mission (constraints not binding: incumbents
      hit 0.00% violations via plain tuning). Verdict + C3 options (matched budget, tail
      mechanism diagnostic, margin-as-gene) in the results doc -- do NOT crown CPAG the
      developed-further classical architecture on this evidence
- [ ] Optional C3 follow-ups (docs/plans/2026-07-19-cpag-c2-results.md section 4): replan
      telemetry on the worst pooled draws; margin-as-gene; matched-budget retrain after a
      faster replan (analytic Jacobians / FOH). Optional: CPAG as constraint-aware warm-start
      supervisor for `magnitude_only` NN training
- [x] Full verification + smart-commit
---

## Stateful NN guidance program -- SHIPPED (2026-04 to 2026-07)

The program this file's phase ledger tracked is complete and published. Shipped: the stateful NN
runtime (JSON v2 tagged-layer format + per-sim `NnState`), five cell types (GRU, LSTM, Window-MLP,
Transformer, Mamba) behind one bit-validated Rust runtime with cross-language equivalence gates at
machine epsilon, PSO training for all five plus PPO-BPTT for GRU/LSTM, the NN-vs-FTC parity bundle
(co-trained scaffolding, `acos_tanh` decoder, multi-supervisor BPTT warm-start), the CfC / xLSTM /
Mamba-3 architecture probes (paper Appendix B), and the quantization campaign (Appendix C).
Outcome: the 962-parameter Mamba is the deployed sizing headline -- CVaR99.9 123.3 +- 0.1 m/s at
100% capture on the 10 x 100,000 confirmatory pool, 41.8 m/s below the best classical scheme.

Paper: `articles/paper/paper.pdf` (tag `arxiv-v2`, GitHub Release). Detailed phase history
(Phases 0 through 4a, task-by-task): this file's git history plus the specs and plans under
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

**Deferred, no current motivation after the paper's results** (the RL track lost decisively to
population search -- paper Section 5 -- and the probes found no tail benefit beyond the plain
cells; revisit only with a new motivation):
- PPO-BPTT for Window / Transformer / Mamba (old Phases 2b.5 / 3b / 4b)
- SAC for stateful cells + recurrent critic (old Phase 1.6 umbrella)
- Full Mamba block: conv1d pre-filter + SiLU gating + expansion linears (`LayerSpec::MambaBlock`,
  old Phase 4c)
- Widen `load_policy_from_json` to accept v1 JSON (only needed if a legacy v1 artifact ever meets
  the torch mirror)
- Multi-layer Transformer stacks at the TOML level
