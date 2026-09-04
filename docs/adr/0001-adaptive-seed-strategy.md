# ADR-0001: Training seeds are non-stationary by default (adaptive curation)

**Status:** accepted · **Date:** 2026-04-14 (design), confirmed by the paper's Study C (2026-07)

## Context

Population optimizers score each individual on a handful of Monte Carlo seeds per generation.
With a fixed seed list the optimizer memorizes those scenarios: training cost keeps improving
while the validation-pool tail does not. The curated-CDF framework and the explicit
`seed_strategy` knob (`docs/design/2026-04-14-curated-cdf-seed-framework-design.md`,
`docs/design/2026-04-14-explicit-seed-strategy-design.md`) made the choice explicit and
mandatory in every training TOML: `fixed`, `rotating`, or `adaptive`.

## Decision

`[optimizer] seed_strategy` is required. The deployed regime is `adaptive`: the training seed
list is refreshed on champion promotion or every `seed_pool_interval` generations by running the
top-K individuals on a probe pool and picking one seed per quantile bin of the cost CDF
(`seed_curator.SeedCurator`, `curation_bucket_selection = "max"` picks the hardest seed per bin).
Training draws always exclude the reserved validation and final-eval pools.

## Consequences

- Cross-generation training costs are not comparable (the landscape moves); the champion is
  decided by the validation gate, never by training cost (see `trainer.observe`).
- Resume restores the checkpointed champion verbatim and never `<`-compares it against a resumed
  population.
- Study C in the paper: GA with fixed seeds is the worst regime; adaptive seeds are the best, by
  tens of m/s on the sizing tail. The adaptive-seed methodology is a load-bearing contribution.
