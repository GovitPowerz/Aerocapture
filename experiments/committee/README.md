# Committee of the headline seeds (#232): negative

Design and verdict: `docs/design/2026-11-02-nn-committee-design.md`. The three headline Mamba 962
seeds (and, in a second arm, the three headline dense 515 seeds too) flown as one artifact, their
atan2 heads combined before the decode, scored against the deployed seed. Every arm fails the
pre-registered win rule on the 2000-scenario pilot; the seeds fly independent lateral strategies
(bank-sign agreement at chance), so their commands do not average.

## Files

- `build_cells.sh`: assembles the four pre-registered cells under `training_output/committee/`
  with `python -m aerocapture.training.committee` (members = the deployed `best_model.json`
  files, sha256s recorded in the model and in the `committee.json` sidecar).
- `committee_cells.txt`: the `label|toml|model_dir` manifest for
  `experiments/ou_marginal/confirmatory_marginal.py` (10 x 100,000, per_draw).
- `paired_vs_champion.py`: paired per-scenario deltas of one committee cell against the deployed
  seed on the confirmatory pools (`sizing_tail.paired_tail` plus the CVaR99.9 delta).
- Cell configs: `configs/training/committee/*.toml`, base-inheriting the deployed seed's TOML.

## Pilot (first 2000 scenarios of confirmatory replicate 0, per_draw, 5 s wall clock)

| cell | ms/sim (14 threads) | capture | mean | CVaR95 | max | paired mean vs champion |
|---|---|---|---|---|---|---|
| champion `ou_marginal/hl_mamba_p962` | 0.47 | 100.00% | 110.6 | 119.5 | 151 | |
| `hl_mamba3_circular_mean` (primary) | 0.67 | 72.70% | 143.0 | 203.8 | 311 | +32.3 m/s, committee better on 0.3% |
| `hl_mamba3_raw_sum` | 0.65 | 79.80% | 150.3 | 190.8 | 250 | |
| `hl_mamba3_medoid` | 0.70 | 81.90% | 155.4 | 271.0 | 296 | |
| `hl_mixed6_circular_mean` | 1.03 | 99.95% | 118.0 | 137.6 | 198 | |
| `hl_mamba3_sign_leader` (post hoc) | 0.84 | 99.40% | 151.7 | 266.5 | 338 | +41.2 m/s, better on 6.1% |
| `hl_mixed6_sign_leader` (post hoc) | 0.91 | 99.45% | 136.6 | 201.0 | 314 | +26.0 m/s, better on 1.0% |

Paired deltas of the primary arm (`paired_vs_champion.py --replicates 1 --n 2000`, 1454 pairs
captured by both): mean +32.3 [31.3, 33.4], p95 +62.4, CVaR95 +84.4 [76.4, 93.0], capture
-27.3 points, CVaR99.9 +165.5.

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
| `committee/hl_mamba3_circular_mean` | 72.82% | 271,801 (500 re-flown without the wall clock: 500 crashes) | 72 | 209.3 | 338.5 +- 3.7 | 1174 |
| deployed seed `ou_marginal/hl_mamba_p962` (paper Table 6) | 100.0000% | 0 | 1148 | 120.1 | 173.9 +- 2.2 | 300 |

The averaged command is a lift-down command no member issued: the 27% non-captures are ground
crashes, not skip-outs. Evaluated at commit 355a9bd9 (model sha256 41bf4dfb...). The other three
pre-registered 10^6 flights were not flown: every arm fails the capture clause of the win rule on
the pilot (the champion loses 0 of 10^6), and the tail of a policy that loses one scenario in four
is not a sizing number.
