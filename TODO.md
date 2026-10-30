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
- [ ] Two-phase regime curriculum (#229; off-nominal robustness recovery, paper Section 7.3
      follow-up; design `docs/design/2026-10-30-two-phase-regime-curriculum-design.md`, promotion
      rule ADR-0009): fine-tune the three centered high-regime anchors under the medium regime with
      the deployed cost transform and curation bucket behind a regime guard, judged at two-pool
      parity. Step 0 first: score the anchors on the medium 10^6 pool; whether these high-trained
      anchors give up the nominal sizing tail was never measured (`centered_depth_v4.json` scores
      them on the stress pool only). The annealed exponent, the medium-to-high
      widening curriculum and the stratified batch under `cubed` are rejected in the design.
- [ ] On-line adaptation of the deployed policy (the paper's third forward direction): the
      deployed policy is a fixed forward pass and all adaptation happens on the ground. Investigate
      lightweight in-flight adaptation, e.g. adapting only the calibrated input normalization or
      the three co-optimized actuator-side parameters against the navigation-estimated density
      history, weights frozen; complements the training-side two-phase curriculum above.
- [x] Centered retrain under per-scenario seeding: done as #177 (outcome A, paper Section 7.3,
      `centered_depth_v4.json`). Retrained under `per_draw` (GA 256 x 16, 4000 gens, three seeds),
      every centered seed beats both per-scenario-retuned joint-FTC baselines on the 9M stress pool
      at n = 10,000: capture parity against the medium retune (+0.4 pts) with -94 to -135 m/s of
      conditional CVaR95, both statistics against the high retune. The limit: no high-regime cell
      is gate-feasible (heat-flux exceedance 8-18% of draws against 11% for the medium law, heat
      load 1.4-2.1% against 0.5%), so the comparison stands at the exceedance each policy shows.
- [ ] Run-variance calibration at the mean: `sigma_extras.json` adds GA/CMA-ES x fixed/rotating
      seed-strategy repeats, but a mean-level sigma_run study across the optimizer-budget cells
      would let tight ties (GA at population 150 vs 300) be ranked or confirmed indistinguishable.
- [ ] Structured pruning of the Mamba head under per-scenario noise: the quantization half
      shipped (paper Appendix C, #178: the 4-bit QAT fine-tune of the deployed seed holds the tail
      at 10^6, CVaR99.9 153.7 against the 64-bit seed's 173.9 at 100% capture, where
      post-training quantization loses it, 286.5 with 298 non-captures; `a_log`/`d_skip` the
      bottleneck). Re-run structured pruning on the deployed seed under the same regime, scoring on
      the far-tail pool with the feasibility gate; prune-then-quantize quantifies how small the
      tail-winning policy can get.
- [ ] Bayesian optimization for the low-dimensional classical schemes (11-26 params): a GP or
      random-forest surrogate for the MC fitness (BoTorch or scikit-optimize as a pymoo-compatible
      backend); the MC fitness is noisy, so a noise-aware acquisition (noisy Expected Improvement)
      is the key challenge. Could cut evaluations on smooth landscapes at the cost of surrogate
      overhead.
- [x] Feasibility-aware validation gate and final selection: shipped as #109 (ADR-0005,
      `[optimizer] max_violation_rate`).

---

## CPAG guidance -- SHELVED (2026-09-23, negative result)

Shelved (#135); verdict, noise regime and reopen conditions:
`docs/design/2026-09-23-cpag-shelved.md`.

---

## Stateful NN guidance program -- SHIPPED (2026-04 to 2026-07)

The program this file's phase ledger tracked is complete and published. Shipped: the stateful NN
runtime (JSON v2 tagged-layer format + per-sim `NnState`), five cell types (GRU, LSTM, Window-MLP,
Transformer, Mamba) behind one bit-validated Rust runtime with cross-language equivalence gates at
machine epsilon, PSO training for all five plus PPO-BPTT for GRU/LSTM, the NN-vs-FTC parity bundle
(co-trained scaffolding, `acos_tanh` decoder, multi-supervisor BPTT warm-start), the CfC / xLSTM /
Mamba-3 architecture probes (paper Appendix B), and the quantization campaign (Appendix C).
Outcome (paper v4, #169): the 962-parameter Mamba is the deployed sizing headline. Trained from
scratch under per-scenario density noise (the default since ADR-0006) at the headline allocation
(GA 512 x 2, 20000 generations), it holds CVaR99.9 158.5 +- 15.8 m/s over three training seeds
(the +- one seed standard deviation) on the 10 x 100,000 confirmatory pool, 19 m/s below the best
classical law at that statistic (joint-reference FTC, 177.8) and 31 below FNPAG (189.1), at 63x
less compute than FNPAG; the deployed seed (seed 1, fixed by rule before any confirmatory score)
captures every scenario at CVaR99.9 173.9 +- 2.2 and exceeds the heat-load ceiling on 0.11% of
them. The seed spread is the finding's limit: the 515-parameter dense reference reaches
160.5 +- 23.2 on the same pool, so the networks beat the laws by a margin the spread does not erase
and beat one another by a margin it does. The arxiv-v3 results this replaces (shared-path headline
123.3 +- 0.1; its per-scenario fine-tune 163.0 +- 0.3) are the README's historical section.

Paper: `articles/paper/paper.pdf`, v4 (the single-regime rewrite, #181; the `arxiv-v4` tag and
GitHub Release were cut 2026-10-16 on the #216 merge commit, Zenodo DOI 10.5281/zenodo.23422624, #183; the Zenodo
record is the citable publication, an arXiv submission is not planned, #217). The arxiv-v3 build is the
`arxiv-v3` tag / Release; changes since it: `articles/paper/CHANGES.md`. Detailed phase history
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
