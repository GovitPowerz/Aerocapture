# ADR-0004: Every training-side evaluation goes through `run_grid`, bit-identical to the per-seed `run_batch` path

**Status:** accepted · **Date:** 2026-06-04 (commit `86a8b93`); reference-override fix 2026-06-12

## Context

The GA hot path used to call `run_batch` once per seed per individual, reloading every data table
from disk and writing a temporary NN JSON per individual. Population evaluation dominated
wall time, and three call sites (population, curation, validation) each built their own
override lists.

## Decision

`aerocapture_rs.run_grid(toml, overrides_list, seeds, weights=...)` evaluates the full
individuals x seeds grid in one GIL-releasing call: each `SimData` is built once with Arc-shared
atmosphere / wind / reference tables and reused across the seed axis; NN weights are passed
in-memory. It is the only `run_grid` call in the package
(`AerocaptureProblem.evaluate_population_per_seed`), and population evaluation, seed curation,
the validation gate, final selection and the islands trainer all route through it. Its output is
gated bit-identical to the per-seed `run_batch` path (`tests/test_run_grid.py`). Overrides that
would break table sharing (`data.atmosphere`, `data.wind_table`, `monte_carlo.seed`,
`simulation.n_sims`) are rejected; a per-individual `data.reference_trajectory` override reloads
that one table (the joint `ref_bank` gene trained dead until this was honored).

## Consequences

- One place to reason about training-side numbers; deploy-side tools use `run_batch` and must
  stay value-identical to it (a documented gate, not an assumption).
- New per-individual data overrides must be added to `SharedTables` explicitly or hard-error.
