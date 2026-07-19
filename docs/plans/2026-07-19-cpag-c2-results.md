# CPAG Stage C2 results — GA campaign + confirmatory-style benchmark

Date: 2026-07-19. Campaign and benchmark per TODO.md Stage C2; setup and the
9.14 feasibility gate in `docs/plans/2026-07-16-cpag-c0-findings.md` + the
`training: feasibility-aware validation gate` commit. Verdict up front: **the
tuned CPAG matches the classical incumbents at the median but loses the
sizing tail decisively (+84 m/s cvar95 vs both, paired), at ~40x their
compute.** On this mission, at this budget, in-loop constraint enforcement
did not pay for its tail. The result is clean, paired, and worth publishing
as a negative — with two specific asterisks that define C3, if there is one.

## 1. Campaign (deployed regime, feasibility-gated)

`configs/training/msr_aller_cpag_train.toml`: GA 48 x 400 gens,
training_n_sims 2, adaptive/max curation (seed_pool_interval overridden 2 ->
25 for CPAG's wall cost), cubed transform, validation 400 sims,
`max_violation_rate = 1%`. Wall: ~2.5 days of elapsed campaign including
pauses; ~40 s per quiet generation, ~10 min per validation fire.

The 9.14 gate dominated the campaign's shape:

- The **pre-loop anchor hole was live**: the first launch validated the gen-0
  argmin at 5.75% heat-load violation and anchored it as champion through the
  ungated initial-validation path. Patched (feasibility check now applies
  there too), campaign restarted.
- **3 feasible promotions vs 21+ infeasible rejections**: the GA repeatedly
  found better-RMS candidates that violate the heat-load limit at 2-6% rates.
  Every one would have deployed silently under the old RMS-only gate. The
  soft cost penalty does not hold the feasibility line; the gate does.
- Final selection (all 48 last-gen candidates, records-based feasibility):
  champion kept — no fresh candidate beat val RMS 1.0853e10 feasibly.

Deployed champion (`training_output/cpag/best_params.json`): the GA
strengthened the apoapsis penalty (alpha3 1000 -> 1678), softened the
intermediate inclination corridor (lambda_di 1000 -> 124) while tightening
its deadband (0.5 -> 0.22 deg), kept trust near default, and pulled
`nav.density_filter_gain` to 0.79. Final eval (1000 disjoint sims): 99.7%
capture, DV p50 143.2 / p95 190.3, apoapsis err p50 -0.6 km, inclination p50
-0.13 deg, heat flux 0%, g-load 0%, heat load 0.1% over.

## 2. Benchmark: shared 10 x 1000 confirmatory-style pools

All four cells evaluated on identical seed pools
(`articles/paper/data/cpag_confirmatory.json`; seeds from [2^31, 2^32),
selection-disjoint; `--sim-timeout 60` for the CPAG cell). CPAG's wall cost
caps pool depth at 10 x 1000 — the tail metric is **cvar99** here; cvar999
per replicate is a single observation and the pooled cvar999/max are quoted
only as raw observations, never as sized claims. The incumbents' real sizing
numbers remain the frozen 10 x 100k file (Mamba cvar999 123.3).

| cell (tuned)        | capture | p95   | cvar95 | p99   | cvar99 | max   | viol% |
|---------------------|---------|-------|--------|-------|--------|-------|-------|
| mamba_p962 (NN)     | 100.00% | 114.3 | 115.9  | 116.8 | 118.5  | 128.2 | 0.00  |
| joint-FTC           | 100.00% | 139.1 | 144.9  | 149.0 | 154.3  | 183.2 | 0.00  |
| FNPAG               |  99.96% | 136.6 | 143.9  | 146.1 | 159.2  | 315.9 | 0.00  |
| **CPAG**            |  99.81% | 196.4 | 228.8  | 242.2 | 294.8  | 889.1 | 0.12  |

Paired deltas (same seeds, t-SE over 10 replicates):

- CPAG − joint-FTC: **+83.7 ± 3.9 m/s cvar95** (max +311.8)
- CPAG − FNPAG: **+84.8 ± 3.9 m/s cvar95** (max +264.1)
- Mamba − CPAG: **−112.7 ± 3.8 m/s cvar95**

Failed-seed classification (19/10,000): re-run at 300 s timeout → **16
physical failures** (crash/pending-crash; the failure states also show wild
apoapsis overshoots) + **3 timeout artifacts** (sims needing >60 s —
replan-storm pathology: crash-escalated replans at the cold budget on nearly
every cycle). Net-of-timeout capture 99.84% — still below the incumbents.

## 3. Deployability triangle (this mission)

- **Tail (cvar99, shared shallow pools)**: Mamba 118.5 < joint-FTC 154.3 ≈
  FNPAG 159.2 « CPAG 294.8. CPAG's tail is recovered-but-expensive: the
  crash-escalation rescue (C1's capture-rate fix) converts would-be crashes
  into 300-900 m/s captures. Capture without it was 78% — the tail IS the
  rescue mechanism's bill.
- **Compute**: CPAG ~3.5 s/sim (~10-15 ms warm replan, Clarabel 4-7 ms/QP,
  tight p95 — the WCET story is genuinely good per-solve) vs FNPAG 87 ms/sim
  (~0.27 ms/replan) vs Mamba's microseconds/tick. ~40x FNPAG end-to-end.
- **Robustness/feasibility**: in-loop enforcement delivered 0.12% heat-load
  violations — but the incumbents sit at 0.00% on the same pools via plain
  GA tuning + (for FTC-family) the bolt-on thermal limiter. On THIS mission
  the constraints are not binding enough for in-loop enforcement to
  differentiate; the paper's Neptune case (Q_max set AT the median) is the
  regime where CPAG's machinery pays.

## 4. Asterisks and what would change the verdict (C3, if wanted)

1. **Budget asymmetry**: CPAG trained on ~38k core sims (400 x 48 x 2); the
   incumbents' campaigns were ~1.2M (2000 x 300 x 2-scale). The tail may be
   trainable — but the claim needs a matched-budget run nobody should expect
   to be cheap (~10 days at current cost) or a faster replan (analytic
   Jacobians, FOH, larger seg_dt) to buy the budget.
2. **Tail mechanism is diagnosable, not diagnosed**: the 300-900 m/s
   captures and the 16 physical failures both live on the crash-escalation
   path. One targeted study — replan telemetry (escalation count, per-replan
   eps residuals) on the worst 50 pooled draws — would show whether the fix
   is late detection (escalate earlier/predictively), authority (sigma_max /
   rate limits), or model mismatch under the density factor's lag.
3. Gate-vs-exploration tension: 21+ rejections suggest the GA wants
   infeasible regions. Margin-as-gene (enforce internal limits BELOW mission
   limits, letting the GA trade margin against DV) is the principled fix.

**Recommendation**: keep CPAG as the 8th scheme (shipped, tested, honest
numbers); do NOT crown it the developed-further classical architecture on
this evidence. The TODO's end-goal framing should be revised to record the
outcome; revisit only via the C3 items above or a mission where the
constraints actually bind.
