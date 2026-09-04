# ADR-0002: The deployed individual is chosen on the validation pool; the final-eval pool only reports

**Status:** accepted · **Date:** 2026-06-10

## Context

The in-training validation gate only ever sees one candidate per generation (the training
argmin). A better-generalizing individual in the last population never got a validation shot, and
the islands path selected its winner among three champions on the final-eval pool, an
`E[min of 3]` selection-on-test bias the single-algorithm numbers did not carry
(`docs/design/2026-06-10-final-selection-design.md`).

## Decision

At the end of training, `final_select.select_final_individual` re-ranks the last generation plus
the running champion on the reserved **validation** pool (candidates deduplicated by exact
identity; the champion's score is reused, never re-simulated) and promotes a fresh candidate only
on **strictly** lower validation RMS. The final-eval pool (offset 2 000 000) evaluates the single
deployed winner and is never used to choose anything. A CLI applies the same rule retroactively
to existing training directories and patches the checkpoint so a resume cannot revert it.

## Consequences

- Quoted final-eval numbers are clean single-candidate test numbers, comparable across the
  single-algorithm and islands paths.
- The deployed individual can never be worse than the champion (ties keep the incumbent).
- `final_selection.json` records the val-RMS distribution of the converged population.
