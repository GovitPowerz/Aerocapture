# TODO

- [ ] implement CPAG (Convex Predictor-Corrector Aerocapture Guidance) as the 8th guidance scheme

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
in-loop) -- and make it the classical architecture developed further, replacing FNPAG in that
role. Rationale: constraint handling is the structural gap in the classical stack (the thermal
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

### Stage C0 -- formulation + solver spike
- [ ] Python prototype of the paper's convexified formulation on our dynamics; verify SCP
      convergence across the corridor (undispersed sweep + a dispersed batch)
- [ ] Pick the embedded solver: Clarabel (pure Rust, SOCP) vs OSQP binding vs problem-shaped
      custom QP; measure per-solve wall time and iteration spread at the real problem size

### Stage C1 -- Rust scheme MVP
- [ ] `cpag.rs`: SCP replan on a `replan_period` cadence (FNPAG throttle pattern: hold between
      replans, re-clamp the held command at current-altitude bank limits), onboard atmosphere
      scaled by the nav density factor
- [ ] Path constraints wired to `[flight.constraints]`; `[guidance.cpag]` TOML params +
      `param_spaces.py` specs + routing-table entry + `compare_guidance` registration
- [ ] Unit tests (constraint activation, replan throttle, convergence fallback) + golden config

### Stage C2 -- training + benchmark
- [ ] GA-tune under the deployed regime (adaptive/max curation, cubed transform); requires the
      feasibility-aware validation gate (IMPROVEMENTS 9.14) for honest promotion
- [ ] Head-to-head vs joint-FTC / FNPAG / deployed Mamba on confirmatory-style pools; update the
      deployability triangle (tail, compute, robustness)
- [ ] Optional follow-up: CPAG as a constraint-aware warm-start supervisor for `magnitude_only`
      NN training
- [ ] Full verification + smart-commit

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
