# Committee of the headline seeds (#232): negative

Design and verdict: `docs/design/2026-11-02-nn-committee-design.md`. The three headline Mamba 962
seeds (and, in a second arm, the three headline dense 515 seeds too) flown as one artifact, their
atan2 heads combined before the decode, scored against the deployed seed. Every arm fails the
pre-registered win rule on the 2000-scenario pilot; the seeds fly independent lateral strategies
(bank-sign agreement at chance), so their commands do not average.

## Files

- `build_cells.sh`: assembles the four pre-registered cells and the two post-hoc `sign_leader`
  cells under `training_output/committee/` with `python -m aerocapture.training.committee`
  (members = the deployed `best_model.json` files, sha256s recorded in the model and in the
  `committee.json` sidecar, member 0's `best_params.json` copied so the cell flies its scaffolding).
- `committee_cells.txt`: the `label|toml|model_dir` manifest for
  `experiments/ou_marginal/confirmatory_marginal.py` (10 x 100,000, per_draw) and for
  `paired_vs_champion.py --cell`; its last two rows are the post-hoc `sign_leader` cells, flown on
  the pilot only.
- `paired_vs_champion.py`: paired per-scenario deltas of one committee cell against the deployed
  seed on the confirmatory pools (`sizing_tail.paired_tail` plus the CVaR99.9 delta), written to
  `paired_vs_champion.json` (tracked; holds the primary arm's pilot pairing quoted below).
- Cell configs: `configs/training/committee/*.toml`, base-inheriting the deployed seed's TOML.

## Pilot (2000 confirmatory-range scenarios, per_draw, 5 s wall clock)

`pilot.py`: the pool is `make_confirmatory_pools(42, 1, 2000)` (2000 seeds of the 10^6 confirmatory
set, not a prefix of replicate 0), every cell on its own scaffolding (the committee cells carry
member 0's `best_params.json`; `pilot.json` holds the rows).

| cell | ms/sim (14 threads) | capture | mean | CVaR95 | max | paired mean vs champion |
|---|---|---|---|---|---|---|
| champion `ou_marginal/hl_mamba_p962` | 0.54 | 100.00% | 110.6 | 119.5 | 151 |  |
| `hl_mamba3_circular_mean` (primary) | 0.64 | 72.15% | 144.2 | 208.5 | 314 | +33.5 m/s, better on 0.2% |
| `hl_mamba3_raw_sum` | 0.67 | 79.35% | 149.2 | 188.7 | 249 | +38.6 m/s, better on 0.1% |
| `hl_mamba3_medoid` | 0.67 | 80.30% | 156.6 | 272.1 | 300 | +46.0 m/s, better on 0.4% |
| `hl_mixed6_circular_mean` | 1.04 | 100.00% | 118.0 | 138.3 | 198 | +7.4 m/s, better on 7.1% |
| `hl_mamba3_sign_leader` (post hoc) | 0.84 | 99.45% | 152.9 | 267.5 | 328 | +42.4 m/s, better on 5.5% |
| `hl_mixed6_sign_leader` (post hoc) | 0.93 | 99.40% | 137.5 | 202.0 | 290 | +26.9 m/s, better on 1.2% |

Paired deltas of the primary arm (`paired_vs_champion.py --replicates 1 --n 2000`, 1443 pairs
captured by both, `paired_vs_champion.json`): mean +33.5, p95 and CVaR95 +89.0, capture
-27.9 points, CVaR99.9 +168.4.

Before review the builder copied no scaffolding, so the first pilot flew the base TOML's navigation
filter and command shaping (`density_filter_gain` 0.349 / `density_gain_max_delta` 0.140 /
`max_bank_acceleration` 12.67 against the deployed seed's 0.573 / 0.396 / 14.42): primary arm
72.70% capture, CVaR95 203.8, the six-member arm 99.95% and 137.6. The corrected rows above move
by under 1.5 points and 5 m/s; the verdict did not.

Runtime check: a committee of the deployed seed three times reproduces the champion (`medoid`
bit-identical; `raw_sum` and `circular_mean` within 1.7e-2 m/s of DV over 300 scenarios, the
last-ulp effect of a scaled head through atan2).

## Mechanism (40 shared scenarios, each seed alone)

| pair | bank-sign agreement (both |bank| > 5 deg) |
|---|---|
| mamba s1 vs s2 | 0.489 |
| mamba s1 vs s3 | 0.574 |
| mamba s2 vs s3 | 0.410 |
| mamba s1 vs dense s1 | 0.517 |
| mamba s3 vs dense s1 | 0.803 |

Mean |bank| 133 / 127 / 114 deg (mamba s1 / s2 / s3), 123 (dense s1); sign flips per flight 25 /
145 / 15 / 38. Same magnitude profile, independent lateral plans.

## Compute (per-scheme tick bench, `src/rust/benches/tick.rs`, extra rows via `AEROCAPTURE_TICK_CONFIGS`)

| cell | guidance call (us, M4 Pro) | vs single Mamba |
|---|---|---|
| dense 515 | 1.01 | 0.38x |
| Mamba 962 | 2.68 | 1x |
| committee of 3 Mamba 962 | 6.54 | 2.4x |
| committee of 3 Mamba + 3 dense | 7.52 | 2.8x |

Per guidance call (the per-flight criterion time divided by the flight's call count: the
three-Mamba committee crashes the undispersed nominal after 178 calls, so its per-flight time is
not comparable). Still the fast compute class, an order of magnitude under FNPAG.

## The 10^6 row

The pre-registered flight of the primary arm through the confirmatory driver
(`confirmatory_marginal.json`, label `committee/hl_mamba3_circular_mean`):

| cell | capture | non-captures | heat load > limit (per 10^6) | CVaR95 | CVaR99.9 +- s.e. | max |
|---|---|---|---|---|---|---|
| `committee/hl_mamba3_circular_mean` | 72.43% | 275,749 (500 re-flown without the wall clock: 500 crashes) | 30 | 212.6 | 294.1 +- 0.8 | 373 |
| deployed seed `ou_marginal/hl_mamba_p962` (paper Table 6) | 100.0000% | 0 | 1148 | 120.1 | 173.9 +- 2.2 | 300 |

The averaged command is a lift-down command no member issued: the 27% non-captures are ground
crashes, not skip-outs. Flown on the corrected cell (member 0's scaffolding) at commit 1e8e711f, model sha256 41bf4dfb...; the first flight, without the scaffolding, gave 72.82% / 209.3 / 338.5 +- 3.7. The other three
pre-registered 10^6 flights were not flown: every arm fails the capture clause of the win rule on
the pilot (the champion loses 0 of 10^6), and the tail of a policy that loses one scenario in four
is not a sizing number.
