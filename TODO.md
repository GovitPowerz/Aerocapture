# TODO

## Backlog

- [ ] Add neural counterparts for navigation and control: train neural counterparts for the
      density estimator (replacing the exponential filter) and the pilot model (replacing the
      first/second-order dynamics), compared against the classical algorithms on identical MC
      scenarios.
- [ ] Develop ESR (Earth Sample Return) mission profiles: the simulator has the ESR reference
      (`data/reference_trajectory/esr_aller.dat`) but no validated ESR entry profiles
      (Earth-specific atmosphere dispersions, ~12 km/s entry, thermal constraints); one of the two
      mission extensions named in the paper's conclusion.
- [ ] Skip-entry mission profiles: the 2009 paper's closing hook and the new paper's conclusion
      name skip entry as the next maneuver class for stateful neural guidance (a deliberate
      atmospheric exit and re-entry, where the policy must plan across two passes). Extend the
      capture/exit phase machinery to a skip-exit-reentry sequence, add success criteria and
      reference data, benchmark the stateful policies against a classical skip-entry
      predictor-corrector.
- [ ] Regime-matched objective schedule (off-nominal robustness recovery; paper conclusion +
      section 7.3): the medium-trained network loses off-nominal robustness to the analytic
      joint-FTC, and the objective-centering experiment pinned the gap on the training objective
      (worst-case shaping collapses the GA gradient once failures dominate). Three mechanisms to
      bridge the medium and high regimes without giving up the nominal sizing tail: an annealed
      tail-weighting (schedule the cost-transform exponent on the population's capture rate,
      near-linear while capture is low, cubed once it saturates); a dispersion-envelope curriculum
      (start on medium with the deployed cubed/max-bucket objective and widen the bounds toward the
      high-stress profile every k generations); a stratified curation batch (n=3-4 per individual,
      mixing a `bucket_middle` seed that anchors the gradient with `bucket_max` seeds that pull the
      tail). The centered-retrain result says the ceiling is reachable.
- [ ] On-line adaptation of the deployed policy (the paper's third forward direction): the
      deployed policy is a fixed forward pass and all adaptation happens on the ground. Investigate
      lightweight in-flight adaptation, e.g. adapting only the calibrated input normalization or
      the three co-optimized actuator-side parameters against the navigation-estimated density
      history, weights frozen; complements the training-side regime-matched schedule above.
- [ ] Run-variance calibration beyond the tail: the objective-centering recovery is three-seed
      (capture 94.8-95.0%, conditional tail 231-273 m/s, every seed beating both FTC references)
      and `sigma_extras.json` adds GA/CMA-ES x fixed/rotating seed-strategy repeats, but (a) the
      centered high-regime cells are still n=1000 and need a requote at sizing depth (n=10,000 +
      CIs), and (b) a mean-level sigma_run study across the optimizer-budget cells would let tight
      ties (GA at population 150 vs 300) be ranked or confirmed indistinguishable.
- [ ] Structured pruning of the Mamba head under the post-fix regime: the quantization half
      shipped (paper Appendix C: the 4-bit fine-tuned head is tail-equivalent, 4.9x memory
      reduction, `a_log`/`d_skip` the bottleneck). Re-run structured pruning on the deployed
      Mamba_962 under the post-fix regime, scoring on the far-tail pool with the feasibility gate;
      prune-then-quantize quantifies how small the tail-winning policy can get.
- [ ] Bayesian optimization for the low-dimensional classical schemes (11-26 params): a GP or
      random-forest surrogate for the MC fitness (BoTorch or scikit-optimize as a pymoo-compatible
      backend); the MC fitness is noisy, so a noise-aware acquisition (noisy Expected Improvement)
      is the key challenge. Could cut evaluations on smooth landscapes at the cost of surrogate
      overhead.
- [x] Feasibility-aware validation gate and final selection: shipped as #109 (ADR-0005,
      `[optimizer] max_violation_rate`).

---

## CPAG -- Convex Predictor-Corrector Aerocapture Guidance -- SHELVED (2026-07, negative result)

Built and benchmarked on `feature/cpag-c1-rust-mvp` (C0 prototype, C1 Rust scheme, C2 GA campaign),
then shelved (#135): tuned CPAG matches the classical incumbents at the median but loses the sizing
tail by +84 m/s cvar95 paired, at ~40x FNPAG's compute (legacy noise regime). Verdict, mechanism
and reopen conditions: `docs/design/2026-09-23-cpag-shelved.md`. Its feasibility gate shipped as
#109 / ADR-0005.

---

## Stateful NN guidance program -- SHIPPED (2026-04 to 2026-07)

The program this file's phase ledger tracked is complete and published. Shipped: the stateful NN
runtime (JSON v2 tagged-layer format + per-sim `NnState`), five cell types (GRU, LSTM, Window-MLP,
Transformer, Mamba) behind one bit-validated Rust runtime with cross-language equivalence gates at
machine epsilon, PSO training for all five plus PPO-BPTT for GRU/LSTM, the NN-vs-FTC parity bundle
(co-trained scaffolding, `acos_tanh` decoder, multi-supervisor BPTT warm-start), the CfC / xLSTM /
Mamba-3 architecture probes (paper Appendix B), and the quantization campaign (Appendix C).
Outcome: the 962-parameter Mamba is the deployed sizing headline -- under per-scenario density
noise (the default since ADR-0006) its fine-tune holds CVaR99.9 163.2 +- 1.3 m/s at 99.996% capture
(both three-seed means, the +- one seed standard deviation; the deployed seed captures 99.9995%) on
the 10 x 100,000 confirmatory pool, 73 m/s below the best classical scheme and the best dense
network. The shared-path result it replaces (123.3 +- 0.1 at 100% capture) is the historical
headline of the paper's main body; Appendix E holds the correction.

Paper: `articles/paper/paper.pdf`, recompiled from the Typst source (the arxiv-v3 build is the
`arxiv-v3` tag / GitHub Release; changes since it: `articles/paper/CHANGES.md`). Detailed phase history
(Phases 0 through 4a, task-by-task): this file's git history plus the specs and plans under
`docs/design/` (the implementation plans left the tree in ee1518a: `git show ee1518a^:docs/superpowers/plans/`).

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
