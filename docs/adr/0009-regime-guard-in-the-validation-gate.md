# ADR-0009: A configured regime guard is a second promotion bar, flown on a second pool

**Status:** accepted · **Date:** 2026-10-30 (design `docs/design/2026-10-30-two-phase-regime-curriculum-design.md`)

## Context

Section 7.3 of the paper pins the off-nominal gap on the training objective: the deployed
shaping (cubed transform, max bucket, two scenarios per individual) stalls on the high regime,
and the centered objective (linear, middle bucket, sixteen scenarios) recovers it, but a policy
trained on either regime is selected on that regime alone. The two-phase curriculum fine-tunes
a centered high-regime anchor under the medium regime with the deployed shaping. Under the
existing rule (ADR-0002, ADR-0005) every promotion in that phase is judged on the medium
validation pool only, so the champion can walk off the high regime one promotion at a time and
nothing in the run records it; final selection would then rank candidates that have already
forgotten what the anchor knew. The alternative weighed was a final-selection-only two-pool
rule: cheaper (no in-training second pool), but it can only choose among a last generation
that the medium-only gate has already steered away from the high regime.

## Decision

When `[optimizer.regime_guard]` is configured, every promotion site (the validation gate, the
pre-loop champion validation, final selection) flies the candidate on a second reserved pool
(`REGIME_GUARD_SEED_OFFSET = 11_000_000`, `n_sims` seeds) under the guard's own Monte Carlo
overrides and cost transform, and requires its RMS to stay at or below a **bar** times
`1 + tolerance`. The bar is the starting champion's RMS on that pool, computed once at prologue
and checkpointed; a resume reads it, never re-flies it. The guard is lexicographically after
feasibility and before the RMS improvement of ADR-0005: a candidate that is feasible and
beats the medium bar but fails the guard is `REFUSED_GUARD`, a gate status distinct from
infeasible; it becomes the last-validated individual (not re-flown), the champion stays, and
every validation record carries the guard RMS, the bar and the verdict. Without the section
nothing changes.

## Consequences

- A guarded run's champion means "best on the training regime among candidates that kept the
  anchor's performance on the guard regime"; quote the guard pool, its regime and the
  tolerance with the run.
- Each validation costs one more MC pass of `n_sims` sims; the guard pool is sized for the
  question it answers (4000 seeds resolve ~0.6 capture points), not for the gate's cadence.
- The guard pool is a reserved stream of its own, disjoint from the 9M stress pool whose first
  10,000 seeds are the final high-regime score, so no selection touches a reported number.
- The tolerance is relative to the anchor, not absolute: a guarded run started from a weak
  anchor is held to a weak bar.
