# Demo model: Mamba-962, deployed seed (paper v4)

Copy of `training_output/ou_marginal/hl_mamba_p962/` (`best_model.json` + `best_params.json`),
byte-identical to the bundle copy `articles/paper/data/runs/ou_marginal/hl_mamba_p962/`
(gate: `tests/test_demo_models.py`): the paper's deployed policy, trained from scratch under
per-scenario density noise (`noise_seeding = "per_draw"`) at the headline allocation via
`configs/training/ou_marginal/hl_mamba_p962.toml` (GA population 512, two scenarios per
individual, adaptive seed strategy, cubed cost transform, 20000 generations;
`experiments/ou_marginal/campaign.sh experiments/ou_marginal/jobs_headline.txt`, issue #173).
Seed 1 of three is the deployed one by the rule fixed before any confirmatory score existed.
Architecture: 962 parameters, Dense(17->16, swish) -> Mamba(d_inner=16, d_state=12) -> Dense
head. `best_model.json` md5 `fef0f539d1a535319b3a5921963f18d9`.

Far-tail confirmatory, 10 x 100,000 pre-registered scenarios under per-scenario noise
(`articles/paper/data/confirmatory_marginal.json`, key `ou_marginal/hl_mamba_p962`): capture
100% (10^6 of 10^6), heat-load ceiling exceeded on 0.11% of scenarios, DV CVaR95 120.1 m/s (the
lowest of the nine network seeds), CVaR99.9 173.9 +- 2.2 m/s (replicate standard error), worst
scenario 300 m/s. The two other seeds (`_s2`, `_s3`) hold CVaR99.9 164.6 / 142.3: three-seed mean
158.5 +- 15.8 m/s (one seed standard deviation), 19 m/s below the best classical law at this
statistic (joint-reference FTC, 177.8) and 31 below FNPAG (189.1). The seed spread is the
finding's limit: the 515-parameter dense reference reaches 160.5 +- 23.2 on the same pool.

`best_model.json` is self-describing (architecture, input mask, normalization, output
decoding); `best_params.json` carries the co-trained navigation/shaping scaffolding, applied
as overrides at run time.

Consumed by `uv run python -m aerocapture.demo` (the default model). Beside it:
`models/demo/ft_mamba_962/` (`--finetune`), the arxiv-v3 per-scenario fine-tune of the
shared-path champion, and `models/demo/mamba_962_legacy/` (`--legacy`), that champion under the
shared noise path. The demo runs on an arbitrary fixed seed, deliberately outside every reserved
evaluation pool, so its output is illustrative and can never be confused with the paper's
quoted numbers.
