# ADR-0009: A configured regime guard is a second promotion bar, flown on a second pool

**Status:** accepted, not implemented · **Date:** 2026-10-30 (design `docs/design/2026-10-30-two-phase-regime-curriculum-design.md`; its Step 0 closed the campaign the same day, Section 10 there, so no run configures a guard yet: the rule stands for the next two-regime fine-tune)

## Context

Section 7.3 of the paper pins the off-nominal gap on the training objective: the deployed
shaping (cubed transform, max bucket, two scenarios per individual) stalls on the high regime,
and the centered objective (linear, middle bucket, sixteen scenarios) recovers it, but a policy
trained on either regime is selected on that regime alone. The two-phase curriculum fine-tunes
a centered high-regime anchor under the medium regime with the deployed transform and bucket
(cubed, max) at the fine-tune allocation of ten scenarios per individual. Under the
existing rule (ADR-0002, ADR-0005) every promotion in that phase is judged on the medium
validation pool only, so the champion can walk off the high regime one promotion at a time and
nothing in the run records it; final selection would then rank candidates that have already
forgotten what the anchor knew. The alternative weighed was a final-selection-only two-pool
rule: cheaper (no in-training second pool), but it can only choose among a last generation
that the medium-only gate has already steered away from the high regime.

## Decision

When `[optimizer.regime_guard]` is configured, every promotion site of the single-algorithm
loop (the validation gate, the pre-loop champion validation, final selection) flies a candidate
that would otherwise promote (feasible and strictly below the medium bar) on a second reserved
pool (`REGIME_GUARD_SEED_OFFSET = 12_000_000`, `n_sims` seeds) under the guard's own Monte Carlo
overrides and cost transform, and requires its RMS to stay at or below a **bar** times
`1 + tolerance`. Promotion is the conjunction of feasibility, the guard and the RMS improvement
of ADR-0005, so flying the guard only for would-promote candidates changes no decision; final
selection flies it in medium-RMS order and stops at the first candidate that passes. The guard
needs the single-algorithm loop and a validation pool: with the islands trainer or
`validation_n_sims = 0` the section is a config error. The bar is the starting champion's RMS on
that pool, computed once at prologue and checkpointed; a resume reads it, never re-flies it. A
candidate that is feasible and beats the medium bar but fails the guard is *refused*: the
verdict rides on the `VALIDATED` gate result as fields (`guard_rms`, `guard_bar`, `guard_pass`)
next to `feasible`, not as a new `GateStatus`, because the record (`Champion.consider`) and the
validation log act on `VALIDATED` only. It becomes the last-validated individual (not re-flown),
the champion stays, and every validation record carries the three columns (null where the guard
was not flown). Without the section nothing changes.

## Consequences

- A guarded run's champion means "best on the training regime among candidates that kept the
  anchor's performance on the guard regime"; quote the guard pool, its regime and the
  tolerance with the run.
- Each would-promote validation costs one more MC pass of `n_sims` sims; the gate validates on
  most generations under adaptive seeds (1583 of 2000 in the #192 fine-tune `hs_mamba_p962_q25`)
  but promotes a handful of times per run, so the guard is sized for the question it answers
  (4000 seeds resolve ~0.6 capture points), not for the gate's cadence.
- The guard pool is a reserved stream of its own, drawn with the 9M stress pool's first 10,000
  seeds (the final high-regime score) excluded: independent streams are disjoint only
  probabilistically (~2% odds of a shared scenario at these sizes), so the exclusion is what keeps
  every selection off a reported number.
- The tolerance is relative to the anchor, not absolute: a guarded run started from a weak
  anchor is held to a weak bar.
