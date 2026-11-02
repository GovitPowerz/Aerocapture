# Tail-aware final selection on the headline populations (#234)

Design and pre-registration: `docs/design/2026-11-02-tail-aware-selection-design.md`, issue #234
(filed before any flight). Question: on the six headline runs (three Mamba 962 seeds, three dense
515 seeds, `training_output/ou_marginal/hl_*`), does a final selection that ranks the feasible
candidates of the final checkpoint on CVaR95 of the correction DV pick a better far tail than the
validation-RMS rule did? No training; the RMS picks' confirmatory rows already exist.

## Files

- `run_selection.sh`: the three arms per run through the retro CLI
  (`python -m aerocapture.training.final_select ... --output-dir`): `rms_validation` (the ADR-0002
  rule, must reproduce the deployed model), `cvar95_validation` (the tail criterion on the run's
  1000-seed validation pool), `cvar95_tail_select` (primary: gate on the validation pool, tail on
  the 5000-seed `TAIL_SELECT_SEED_OFFSET` pool). Picks land under `training_output/tail_select/<run>/<arm>/`
  (untracked, like every training output); the deployed artifacts are not written.
- `compare_picks.py --manifest`: byte-compares the picks, writes `tail_select_cells.txt` (the picks
  that differ from the RMS pick) for `experiments/ou_marginal/confirmatory_marginal.py`.
- `compare_picks.py`: the pre-registered read from `confirmatory_marginal.json`, written to
  `tail_select.json` (tracked) and printed as the table below.

## Picks (training_output/tail_select/selection.log, 2026-11-02)

The `rms_validation` arm reproduced the deployed model byte for byte in all six runs. The tail
criterion changed the pick in the two runs that had a choice: `hl_mamba_p962` (5 feasible
last-generation candidates; both pools pick `last_gen[275]`) and `hl_dense_p515_s2` (4 feasible;
the 5000-seed pool picks `last_gen[458]`, the validation pool `last_gen[312]`). In
`hl_mamba_p962_s2` (325 feasible) both pools keep the champion; in the three runs with no feasible
last-generation candidate nothing could change.

Three confirmatory flights (`confirmatory_marginal.json`, labels `tail_select/...`, 10 x 100,000
per_draw, flown at commit 355a9bd9 plus this branch's uncommitted selection code, which the flight
path does not touch).

## Result (`compare_picks.py`, `tail_select.json`)

| run | feasible last-gen | RMS pick | tail pick (5000 pool) | pool CVaR95 RMS / tail | capture RMS / tail | heat load > limit RMS / tail (per 10^6) | CVaR99.9 RMS / tail (+- s.e.) | delta CVaR99.9 |
|---|---|---|---|---|---|---|---|---|
| `hl_mamba_p962` | 5 | champion | last_gen[275] | 119.61 / 118.06 | 100.0000% / 99.9990% | 1148 / 1471 | 173.9 +- 2.2 / 135.2 +- 0.7 | -38.8 |
| `hl_mamba_p962_s2` | 325 | champion | champion (same model) | 131.33 / 131.33 | 99.9996% / 99.9996% | 437 / 437 | 159.3 +- 0.6 / 159.3 +- 0.6 | +0.0 |
| `hl_mamba_p962_s3` | 0 | champion | champion (same model) | 122.89 / 122.89 | 100.0000% / 100.0000% | 567 / 567 | 142.3 +- 0.5 / 142.3 +- 0.5 | +0.0 |
| `hl_dense_p515` | 0 | champion | champion (same model) | 123.31 / 123.31 | 100.0000% / 100.0000% | 1465 / 1465 | 140.1 +- 0.4 / 140.1 +- 0.4 | +0.0 |
| `hl_dense_p515_s2` | 4 | champion | last_gen[458] | 128.81 / 123.7 | 99.9999% / 100.0000% | 2112 / 1525 | 185.8 +- 0.6 / 150.8 +- 0.4 | -35.1 |
| `hl_dense_p515_s3` | 0 | champion | champion (same model) | 128.73 / 128.73 | 99.9929% / 99.9929% | 2572 / 2572 | 155.6 +- 1.2 / 155.6 +- 1.2 | +0.0 |

Six-run paired delta: mean -12.3, sd 19.1, 95% t-interval [-32.4, +7.7], signs - 0 0 0 - 0.
Runs with more than one feasible member (3): mean -24.6, sd 21.4, 95% t-interval [-77.8, +28.5].
Validation-pool arm of `hl_dense_p515_s2` (`tail_select/hl_dense_p515_s2_val`, `last_gen[312]`):
CVaR99.9 142.2 +- 0.5, one non-capture (a crash), 2353 heat-load exceedances.

**Verdict under the pre-registered rule: negative.** The interval covers zero because four of the
six runs enter as structural zeros, and the Mamba pick trips both regression clauses: it loses 10
of 10^6 scenarios (8 crashes, 2 pending crashes on the re-fly without the wall clock, against the
deployed seed's 0) and exceeds the heat-load limit 323 more times per 10^6. The dense s2 pick wins
every clause on its own (-35.1 m/s, capture up, exceedances down).

What the table says beyond the rule:

- Where the criterion could choose, it chose a better far tail both times, by 35 to 39 m/s, each
  more than fifteen replicate standard errors; the 5000-pool CVaR95 ordered the 10^6 CVaR99.9
  correctly in both runs, and the champion is never the pool's best CVaR95 among the feasible.
- The far tail it buys is not free on the Mamba side: the pick that cuts CVaR99.9 by 39 m/s crashes
  one flight in 100,000. A 5000-seed pool cannot see a 1e-5 event; CVaR95 on captured scenarios
  does not price it. Any deployment rule built on this criterion needs the capture clause scored on
  a pool large enough to see it, or a scheme-level guarantee.
- The binding constraint is the choice set, not the statistic. The ADR-0005 gate at ceiling 0 on
  1000 scenarios leaves 0 to 5 feasible candidates in five of the six final populations (and 325
  in the sixth, where the champion still won the tail). A tail-aware selection that matters has to
  keep more feasible candidates in reach: the gate's ceiling and pool size, or a population that
  the search keeps feasible, are the levers; re-ranking a handful of survivors is not.

Nothing under `training_output/ou_marginal/` changed (`experiments/paper/audit_deployed_models.py`:
every headline run `ok`; the picks' duplicates it lists are the untracked retro dirs). The trainer's
default criterion stays RMS.
