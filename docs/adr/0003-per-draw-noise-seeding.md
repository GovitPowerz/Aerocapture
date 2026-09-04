# ADR-0003: Time-varying noise is seeded per dispersion draw (`noise_seeding = "per_draw"`); the legacy regime stays the default for committed numbers

**Status:** accepted · **Date:** 2026-08-27 (commit `5eb13de`)

## Context

The Ornstein-Uhlenbeck density perturbation and the EKF sensor noise were seeded from
`[simulation] random_seed + env_idx * 10_000`, independent of `monte_carlo.seed`. Every
`n_sims = 1` evaluation (training, validation, final eval, requotes, the confirmatory pools) thus
conditioned every scenario on ONE frozen noise path. The networks, trained under that
conditioning, exploited it 2-4x more than the classical laws
(`experiments/ou_marginal/RESULTS.md`, paper Appendix E).

## Decision

`[monte_carlo] noise_seeding` selects the regime. `"legacy"` (default) reproduces the historical
frozen path bit-for-bit, so goldens and every committed number are unchanged. `"per_draw"` seeds
the streams from an FNV-1a hash of the dispersion draw (`RunState::noise_seed`): identical draw
-> identical stream (reproducible), distinct draws -> independent noise (marginalized). Unknown
values hard-error. The headline cells were retrained under `per_draw` and the far-tail
confirmatory re-run; the paper reports both regimes.

## Consequences

- Any new evaluation that claims to marginalize over noise must set `per_draw` explicitly; the
  demo's `--per-draw` flag and the Appendix E scripts do.
- Frozen-regime and per-draw numbers are not comparable; state the regime with every number.
