# Demo model: Mamba-962 (paper headline cell)

Copy of `training_output/mamba_p962_long/` (`best_model.json` + `best_params.json`),
the deployed champion of the paper's architecture study: a 962-parameter
Dense(17->16, swish) -> Mamba(d_inner=16, d_state=12) -> Dense head, trained with GA
(n_pop 300, adaptive seed strategy) via `configs/training/sweep/mamba_p962.toml`
plus a 15000-generation resume.

Final-eval pool (n = 1000, disjoint from all training/validation seeds):
capture 100%, DV p50 109.6 m/s, p95 114.0, CVaR95 115.4 (`articles/paper/data/results.json`,
key `headline/mamba_p962`).

`best_model.json` is self-describing (architecture, input mask, normalization,
output decoding); `best_params.json` carries the co-trained navigation/shaping
scaffolding, applied as overrides at run time.

Consumed by `uv run python -m aerocapture.demo`. The demo runs on an arbitrary
fixed seed, deliberately outside every reserved evaluation pool, so its output
is illustrative and can never be confused with the paper's quoted numbers.
