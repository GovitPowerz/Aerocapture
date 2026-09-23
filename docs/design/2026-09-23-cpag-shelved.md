# CPAG (Convex Predictor-Corrector Aerocapture Guidance) -- shelved negative result

Date: 2026-09-23 (work done 2026-07-16..19)
Branch: `feature/cpag-c1-rust-mvp`, tip `d19de90`, tagged `cpag-c2-negative` (unmerged, forked from
`main` 2026-07-16)
Status: shelved (#135). The code stays on the branch; this note is the record on `main`.

## What was built

CPAG (Rataczak, McMahon & Boyd, JGCD 2025, doi:10.2514/1.G008685) as an 8th guidance scheme: a
convexified sequential replan of the bank profile with heat-flux / g-load / heat-load path
constraints enforced in-loop, onboard atmosphere scaled by the nav density factor (the FNPAG
lesson), FNPAG-style replan throttle.

- C0 (`docs/plans/2026-07-16-cpag-c0-findings.md` on the branch): Python SCP prototype
  (`src/python/aerocapture/cpag/`), solver pick Clarabel (pure-Rust box-QP, 4-7 ms per solve,
  tight p95).
- C1: `src/rust/src/gnc/guidance/cpag.rs`, `[guidance.cpag]` TOML + param specs, golden config
  `configs/test/test_cpag_golden.toml`. Untuned capture was 78%; crash-escalating warm replans
  took it to 100%.
- C2 (`docs/plans/2026-07-19-cpag-c2-results.md` on the branch): GA 48 x 400 gens under the
  deployed regime (adaptive/max curation, cubed transform), feasibility-gated. The gate blocked
  21+ better-RMS infeasible promotions against 3 feasible ones; that campaign is the evidence
  behind ADR-0005, which shipped separately as #109.

## Verdict

Shared 10 x 1000 confirmatory-style pools (seeds in [2^31, 2^32), selection-disjoint), all four
tuned cells on identical seeds, `--sim-timeout 60` for CPAG:

| cell | capture | p95 | cvar95 | cvar99 | max | viol% |
|---|---|---|---|---|---|---|
| mamba_p962 (NN) | 100.00% | 114.3 | 115.9 | 118.5 | 128.2 | 0.00 |
| joint-FTC | 100.00% | 139.1 | 144.9 | 154.3 | 183.2 | 0.00 |
| FNPAG | 99.96% | 136.6 | 143.9 | 159.2 | 315.9 | 0.00 |
| CPAG | 99.81% | 196.4 | 228.8 | 294.8 | 889.1 | 0.12 |

DV in m/s. Paired over the 10 replicates (t-SE): CPAG - joint-FTC **+83.7 +- 3.9 m/s cvar95**,
CPAG - FNPAG **+84.8 +- 3.9**, Mamba - CPAG **-112.7 +- 3.8**. Median at parity with the
classical incumbents (CPAG final eval p50 143.2). Compute ~3.5 s/sim, about 40x FNPAG's 87 ms.

Mechanism: the fat tail is the bill for the crash-escalation rescue. It converts would-be crashes
into 300-900 m/s captures; 16 of the 19 failed seeds were physical failures on that path, 3 were
replan-storm timeouts. In-loop enforcement bought nothing on this mission: the incumbents sit at
0.00% violation on the same pools through GA tuning plus the thermal limiter.

**Noise regime: legacy.** The branch predates the `[monte_carlo] noise_seeding` knob
(`5eb13de`, 2026-08-27), so every cell flew the single frozen OU density path, which is what
`noise_seeding = "legacy"` reproduces today. The numbers were not requoted under `per_draw`
(ADR-0006) and must not be mixed into a per-draw table. cvar99 is the deepest tail metric the
1000-deep pools support; the incumbents' sizing numbers remain the paper's 10 x 100k file.

## Why shelved rather than landed

- The C2 doc recommended keeping CPAG as the 8th scheme. Landing it now is a port, not a rebase:
  the branch is 139 commits behind `main` as of 2026-09-23 (`deny_unknown_fields` #127, tiered
  PyO3 seam #125, runner split #124, per-draw default #108, dead-gene removal #129), and a
  quotable result would also need a per-draw requote at ~3.5 s/sim.
- The part of the branch with lasting value, the feasibility gate, is already on `main`.
- A scheme that loses the tail at 40x the compute adds maintenance surface (its own golden, the
  config gates, the clarabel dependency) for no deployable benefit.

## What would reopen it

Any of the C2 doc's C3 items: a matched-budget campaign (CPAG trained on ~38k core sims against
~1.2M for the incumbents, about 10 days at current replan cost, or buy it with a faster replan:
analytic Jacobians, FOH, larger segment dt); replan telemetry on the worst 50 draws to tell late
detection from authority limits from density-lag mismatch; margin-as-gene (internal limits below
mission limits). Or a mission where the constraints bind, e.g. the CPAG paper's Neptune case with
Q_max at the median. Start from tag `cpag-c2-negative` and port onto current `main`.
