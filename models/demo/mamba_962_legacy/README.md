# Demo model: Mamba-962, shared-path champion (historical headline, `--legacy`)

Copy of `training_output/mamba_p962_long/` (`best_model.json` + `best_params.json`),
the deployed champion of the paper's architecture study: a 962-parameter
Dense(17->16, swish) -> Mamba(d_inner=16, d_state=12) -> Dense head, trained with GA
(population 512, two scenarios per individual, adaptive seed strategy) via
`configs/training/sweep/mamba_p962.toml` plus a 15000-generation resume
(`experiments/paper/10b_arch_long_challengers.sh`).

Final-eval pool (n = 1000, disjoint from all training/validation seeds):
capture 100%, DV p50 109.6 m/s, p95 114.0, CVaR95 115.4 (`articles/paper/data/results.json`,
key `headline/mamba_p962`).

`best_model.json` is self-describing (architecture, input mask, normalization,
output decoding); `best_params.json` carries the co-trained navigation/shaping
scaffolding, applied as overrides at run time.

This champion was trained and quoted under the historical shared density-noise
path (`noise_seeding = "legacy"`), the conditioning defect the paper's Appendix E
discloses; under per-scenario noise (the default since ADR-0006) it captures 97.9%
with 0.9% heat-load violations. The deployed model is its per-scenario fine-tune,
`models/demo/ft_mamba_962/`.

Consumed by `uv run python -m aerocapture.demo --legacy`, which also pins the
shared-path regime. The demo runs on an arbitrary fixed seed, deliberately outside
every reserved evaluation pool, so its output is illustrative and can never be
confused with the paper's quoted numbers.
