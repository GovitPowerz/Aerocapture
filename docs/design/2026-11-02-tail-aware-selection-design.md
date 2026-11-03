# Tail-aware final selection on the headline populations

> **Date:** 2026-11-02. **Status:** closed, negative under the pre-registered rule (2026-11-02, Section 5).
> **Record:** `experiments/tail_select/README.md`. Vocabulary: `CONTEXT.md` (Champion, Final
> Selection, Seed Pool, Sizing Tail).

## 1. Hypothesis

The end-of-training selection (ADR-0002) promotes on validation RMS, and paper Section 6.2 finds
that RMS does not order the far tail: the deployed Mamba 962 seed has the lowest validation RMS
of its three seeds and the worst CVaR99.9 (173.9 against the family's 158.5 +- 15.8 on the
10 x 100,000 per_draw confirmatory pool); the lowest-RMS dense 515 seed has the widest tail of
nine runs. Combining seeds at deploy time does not reduce the spread (#232: independent lateral
plans), which leaves selection. The cheapest test needs no training: re-run the final selection
on the six headline populations with a tail statistic in place of RMS, fly the picks that differ.

## 2. The criterion

`python -m aerocapture.training.final_select --criterion cvar95` (`select_tail_individual`;
default `rms`, every existing path byte-identical, the trainer's own `finalize` unchanged).
Lexicographic:

1. the ADR-0005 feasibility gate on the validation pool at the run's `[optimizer]
   max_violation_rate`, unchanged and first;
2. among the feasible, the most scenarios captured on the scoring pool;
3. the lowest CVaR95 of the correction DV over the captured scenarios (`sizing_tail.cvar`: the
   mean of the worst `round(0.05 n)` captured samples);
4. ties, the lowest validation RMS.

The champion is scored like any candidate (its recorded RMS pre-empts nothing); the last
generation is deduplicated as the RMS rule does it. CVaR95 rather than a deeper quantile because
the pool is small: at n = 1000, CVaR99.9 is the single worst sample. With no feasible candidate at
all, the best-RMS infeasible row deploys with a warning and `winner_feasible = false`, as the RMS
rule does without a champion.

## 3. Pools

Two arms per run. `--pool validation`: the run's reserved validation pool (offset 1,000,000,
n = 1000), the pick the trainer could have made at no extra cost. `--pool tail_select --pool-n
5000` (primary): the reserved `TAIL_SELECT_SEED_OFFSET = 12,000,000` stream (`seeds.py`), drawn by
no training, validation, final-eval or confirmatory flight; the gate still runs on the validation
pool, the feasible survivors are then flown on the 5000-seed pool and ranked on its capture count
and CVaR95 (the worst 250 samples). The sidecar records the pool (`tail_pool`), every candidate's
gate verdict and, for the survivors, their tail block with the pool's violation rates.

**Amendment (2026-11-02, before any confirmatory flight; posted on #234).** The pre-registration
re-checked the gate on the 5000-seed pool at the run's ceiling. At ceiling 0 that is a different
and stricter gate than the one the runs were trained under (ADR-0005: the ceiling is a sample-size
question): on `hl_mamba_p962` every full-capture survivor, the deployed champion included, exceeds
the heat-load limit on about 0.1% of draws (1148 per 10^6 on the confirmatory pool), so 5000 draws
always hold a few, every one was declared infeasible, and the arm promoted the one candidate with
no exceedance, which loses 111 of 5000 scenarios (CVaR95 160 against the champion's 120). The
rule is therefore: the gate is the validation pool's, at its size; the tail pool's violation rates
are recorded on each survivor's record and reported in the table, and do not exclude. The capture
clause (most scenarios captured) is unchanged and is what removes that candidate.

`--output-dir` deploys the pick and its `final_selection.json` into a directory of its own and
never patches the checkpoint: the headline runs' deployed `best_model.json` are not written
(`experiments/paper/audit_deployed_models.py` checks this at the end).

## 4. Choice set, known before any flight

The six runs' `final_selection.json` already record the gate on their final checkpoints at
ceiling 0: feasible last-generation candidates 5 (`hl_mamba_p962`), 325 (`hl_mamba_p962_s2`),
0 (`hl_mamba_p962_s3`), 0 (`hl_dense_p515`), 4 (`hl_dense_p515_s2`), 0 (`hl_dense_p515_s3`). Where
there are none, the champion is the only feasible individual and no gated criterion can change the
pick: those runs enter the paired delta as structural zeros. The gate is kept as pre-registered
rather than relaxed to widen the set.

## 5. Win rule and result

Primary: the six-run paired delta of pooled CVaR99.9 on the confirmatory pool, tail pick (5000-pool
arm) minus RMS pick, with its sign per run, mean and standard deviation; runs whose picks coincide
enter as 0. Tail-aware selection wins if the mean is negative and the 95% t-interval over the six
runs excludes zero, and no tail pick captures fewer of the 10^6 scenarios or exceeds the heat-load
limit more often than its RMS pick. A negative mean whose interval covers zero is suggestive, not
decided; a positive mean is negative. Secondary reads: the delta over the runs with a non-singleton
choice set, the per-row replicate standard error, the validation-pool arm, and the pool CVaR95 of
each pick beside its confirmatory CVaR99.9.

### Result (2026-11-02; record `experiments/tail_select/README.md`, `tail_select.json`)

The RMS arm reproduced every deployed model byte for byte. The criterion changed the pick in the
two runs with a choice: `hl_mamba_p962` -> `last_gen[275]` (both pools), `hl_dense_p515_s2` ->
`last_gen[458]` (5000 pool; the validation pool picks `last_gen[312]`); `hl_mamba_p962_s2` kept its
champion on both pools, the other three had nothing to choose.

| run | tail pick | capture RMS / tail | heat load > limit RMS / tail (per 10^6) | CVaR99.9 RMS / tail (+- s.e.) | delta |
|---|---|---|---|---|---|
| `hl_mamba_p962` | last_gen[275] | 100.0000% / 99.9990% | 1148 / 1471 | 173.9 +- 2.2 / 135.2 +- 0.7 | -38.8 |
| `hl_dense_p515_s2` | last_gen[458] | 99.9999% / 100.0000% | 2112 / 1525 | 185.8 +- 0.6 / 150.8 +- 0.4 | -35.1 |
| four other runs | same model | | | | 0 |

Six-run paired delta: mean -12.3, sd 19.1, 95% t-interval [-32.4, +7.7]; over the three runs with
more than one feasible member: -24.6, [-77.8, +28.5].

**Verdict under the pre-registered rule: negative.** The interval covers zero (four structural
zeros) and the Mamba pick trips both regression clauses: 10 non-captures per 10^6 (8 crashes, 2
pending crashes on the re-fly), 323 more heat-load exceedances. The dense s2 pick wins every clause
alone. Two reads survive the verdict: where the criterion could choose it chose a better far tail
both times, by 35 to 39 m/s (the 5000-pool CVaR95 ordered the 10^6 CVaR99.9 correctly), and the
ADR-0005 gate at ceiling 0 on 1000 scenarios is what leaves 0 to 5 candidates to choose from in
five of six populations. A tail-aware selection worth deploying needs a larger feasible choice
set and a capture clause scored on a pool that can see a 1e-5 crash rate; neither is a re-ranking
change. The trainer's default criterion stays RMS; the CLI keeps `--criterion cvar95` as the study
tool.
