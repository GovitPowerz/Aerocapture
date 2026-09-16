# Demo model: Mamba-962, per-scenario fine-tune (paper Appendix E champion)

Copy of `training_output/ou_marginal/ft_mamba_p962/` (`best_model.json` + `best_params.json`):
the shared-path champion checkpoint (`training_output/mamba_p962_long/`, the historical
headline cell) fine-tuned for 2000 further GA generations under per-scenario density noise
(`noise_seeding = "per_draw"`) via `configs/training/ou_marginal/ft_mamba_p962.toml`
(GA population 60, ten scenarios per individual, adaptive seed strategy;
`experiments/ou_marginal/phase2_campaign.sh`). Architecture unchanged: 962 parameters,
Dense(17->16, swish) -> Mamba(d_inner=16, d_state=12) -> Dense head. `best_model.json`
md5 `1ecc77e492b4f2caa4229b6b69dc2e4b`.

Far-tail confirmatory, 10 x 100,000 pre-registered scenarios under per-scenario noise
(`experiments/ou_marginal/confirmatory_marginal.json`, key `ou_marginal/ft_mamba_p962`):
capture 99.9995% (5 non-captures per 10^6, all genuine crashes), 0% constraint violations,
DV CVaR95 138.7 m/s, CVaR99.9 163.0 +- 0.3 m/s, worst scenario 249 m/s. Two further
fine-tune seeds (`_s2`, `_s3`) give CVaR99.9 164.6 / 161.9: three-seed mean 163.2 +- 1.3 m/s,
73 m/s below both the best classical scheme (FNPAG, 236.7) and the best dense fine-tune
(236.3). This is the deployed seed (s1). Capture is seed-dependent at the 1e-4 level
(5 / 30 / 80 non-captures per 10^6 for s1 / s2 / s3), which is why deployed policies are
confirmatory-screened. Feasible at the strict ceiling of ADR-0005.

`best_model.json` is self-describing (architecture, input mask, normalization, output
decoding); `best_params.json` carries the co-trained navigation/shaping scaffolding, applied
as overrides at run time.

Consumed by `uv run python -m aerocapture.demo` (the default model and regime since
ADR-0006). The historical shared-path champion it was fine-tuned from lives beside it in
`models/demo/mamba_962_legacy/` for `--legacy`. The demo runs on an arbitrary fixed seed,
deliberately outside every reserved evaluation pool, so its output is illustrative and can
never be confused with the paper's quoted numbers.
