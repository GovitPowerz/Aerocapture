# OU-marginal campaign results (2026-08-28)

Protocol: paired n=1000 pool (rng 987654321), marginal regime = per-scenario
`simulation.random_seed = 1000 + 7*i`; raw data `quote_results.json`
(regenerate with `quote_marginal.py`; the paper reads its quoted cells through
`articles/paper/data/quote_marginal.json`: after a re-run, `make -C articles/paper
quote-marginal sums provenance`, commit, then `make -C articles/paper pdf` from the
clean HEAD, `make check` fails until then). `quote_marginal.py` keeps every
`label/regime` already quoted and rewrites the file after each new one: stop it
with Ctrl-C and rerun the same command (without `--force`) to resume; a cell whose
model changed since its quote needs `--only LABEL`; cells outside the discovered
`training_output/ou_marginal/*/` runs come in by `--manifest` (`label|toml[|model_dir]`).
The file holds both lists, so a protocol re-quote is two runs: `--force`, then
`--manifest experiments/ou_marginal/classical_cells.txt` without `--force` (with
it, the fresh file would drop every default row, the paper's included).
All numbers below are the MARGINAL
regime, DV CVaR95 in m/s; every listed run is 100% capture and heat-load
feasible unless flagged.

## Scratch per_draw retrains (3 seed repeats each: s1/s2/s3)

Allocation: every run here (scratch and fine-tune) used the sweep config's
GA n_pop 60 x training_n_sims 10 for 20000 gens. The headline/sweep cells were
launched with `--n-pop 512 --training-n-sims 2` (10b_arch_long_challengers.sh),
so this is the deployed generation count, NOT the deployed allocation; the
fine-tunes shrink the champion population 512 -> 60 on resume.

| cell       | s1 / s2 / s3          | mean +- std  |
|------------|-----------------------|--------------|
| mamba_962  | 146.9 / 150.0 / 147.6 | 148.1 +- 1.6 |
| gru_1014   | 147.3 / 146.6 / 156.8 | 150.2 +- 5.7 |
| dense_972  | 149.4 / 157.9 / 143.9 | 150.4 +- 7.1 |
| lstm_1082  | 145.9 / 168.4 / 147.0 | 153.8 +- 12.7|
| dense_515  | 155.2 / 158.0 / 162.8 | 158.7 +- 3.8 |

Reference: FNPAG 154.3 (best classical, feasible); frozen-trained originals
score 170-228 marginal (and lstm_p1082_long is 17% heat-load infeasible).

## Fine-tunes (frozen champion checkpoint + 2000 per_draw gens, single runs)

| run           | CVaR95 | note                                        |
|---------------|--------|---------------------------------------------|
| ft_dense_515  | 129.8  | best feasible policy overall                |
| ft_lstm_1082  | 128.3  | INFEASIBLE (10.8% heat-load) - inherits s1  |
| ft_mamba_962  | 138.6  | replicates the pilot (138.3) exactly        |
| ft_dense_972  | 145.2  |                                             |
| ft_gru_1014   | 153.4  | fine-tune WORSE than scratch (150.2)        |

## Conclusions

1. **Marginal-OU training restores feasibility and the tail.** Every scratch
   retrain: 100% capture, 0% violations, CVaR95 144-168 vs the frozen-trained
   originals' 170-228 (and it fixes LSTM's heat-load infeasibility).
2. **The architecture tail ranking compresses to sigma_run.** mamba/gru/
   dense_972 means (148.2/150.2/150.4) are indistinguishable at 3 repeats;
   only dense_515 is significantly worse (158.7, gap > combined std). The
   surviving mamba-specific property is run-to-run CONSISTENCY
   (+-1.6 vs +-5.7..12.7) - suggestive at n=3, not conclusive.
3. **Fine-tune-from-frozen is the winning recipe where it is feasible**:
   ft_dense_515 (129.8) and ft_mamba (138.6) beat every scratch run and FNPAG
   (154.3) decisively; but it is not universal (gru regresses, lstm inherits
   infeasibility). Recommended deployed champions: ft_dense_515 primary,
   ft_mamba_962 as the stateful representative.
4. **NN-vs-classical under the honest regime**: scratch means beat FNPAG by
   only ~4-6 m/s (~1 sigma_run); the decisive margin comes from the fine-tune
   recipe (-16 to -25 m/s). Frame v3 accordingly.

## Far-tail confirmatory under per-scenario noise (2026-08-29)

10 x 100k pre-registered pools (Section 4.3 protocol) with
`monte_carlo.noise_seeding = per_draw`; raw data `confirmatory_marginal.json`.
Cells come from `confirmatory_cells.txt` or `--manifest` / `--cells`. Stop the
scorer with Ctrl-C at any time and rerun the same command to resume: each
finished replicate is kept under `confirmatory_marginal.json.partial/<label>/`
and never flown twice, and a cell already in the results file is skipped.
The paper reads its quoted cells through `articles/paper/data/confirmatory_marginal.json`:
after re-running `confirmatory_marginal.py`, run `make -C articles/paper
confirmatory-marginal sums provenance`, commit, then `make -C articles/paper pdf`
from the clean HEAD (`make check` fails until then).
CVaR999 +- se over replicates, m/s over captured scenarios:

| policy                  | capture  | CVaR95 | CVaR999       | max |
|-------------------------|----------|--------|---------------|-----|
| ft_mamba_962            | 99.9995% | 138.7  | 163.0 +- 0.3  | 249 |
| frozen champion (mamba) |  97.93%  | 188.2  | 221.3 +- 0.5  | 270 |
| ft_dense_515            | 100.00%  | 128.8  | 236.3 +- 2.5  | 405 |
| fnpag                   |  99.37%* | 152.5  | 236.7 +- 2.3  | 579 |

(*) physical crashes: a 500-seed sample re-runs ifinal=1 with no timeout censoring.

The million-scenario depth reverses the CVaR95 verdict: the dense fine-tune's
shallow-tail win hides a fat far tail (236 / max 405), while ft_mamba holds
163.0 losing 5 of 10^6 scenarios, with the smallest max (249) - 73 m/s below both
FNPAG and the dense. The recurrent extreme-tail thesis SURVIVES the honest
regime; conclusion 3 above ("ft_dense_515 primary champion") is superseded:
**ft_mamba_962 is the deployed champion at the sizing metric.**

## ft_mamba seed repeats (2026-08-29)

Two further fine-tunes of the frozen champion checkpoint under trainer seeds
2/3 (rng_state stripped from the checkpoint copy -- resume otherwise restores
the saved RNG and silently overrides --seed; the first repeats were bit-identical
replays, caught by md5 of the deployed weights). Far-tail confirmatory
(10 x 100k per_draw):

| seed | capture   | CVaR95 | CVaR999 | max |
|------|-----------|--------|---------|-----|
| s1   | 99.9995%  | 138.7  | 163.0   | 249 |
| s2   | 99.997%   | 140.0  | 164.6   | 240 |
| s3   |  99.992%* | 138.4  | 161.9   | 211 |

(*) 5 / 30 / 80 non-captures per 10^6 (s1 / s2 / s3), all genuine crashes (ifinal 1 or 4, re-run without sim timeout).

CVaR999 three-seed mean 163.2 +- 1.3 -- the 73 m/s margin over dense/FNPAG is
seed-robust; capture is seed-dependent at the 1e-4 level, so deployed policies
must be confirmatory-screened (as the protocol already requires).

## Classical retunes under per-scenario noise (#172, 2026-09-29)

The nine classical cells of `experiments/paper/01_classical_baselines.sh` and
`07_joint_reference.sh`, re-tuned under per_draw at the same GA allocation
(`classical_campaign.sh`), quoted on this file's paired pool next to their
shared-path-tuned parents (`quote_marginal.py --manifest
experiments/ou_marginal/classical_cells.txt`; the joint parents are
`paper/joint_reference/<scheme>`). Marginal regime, CVaR95 in m/s; flux is the
heat-flux violation rate on the same 1000 scenarios (`heat_flux_viol_pct` in
`quote_results.json`; heat-load violations are 0.0% on every row here).

| cell                    | parent capture / CVaR95 / flux | retuned capture / CVaR95 / flux |
|-------------------------|--------------------------------|---------------------------------|
| ftc                     | 100.0% / 261.3 / 0.3%          | 100.0% / 241.2 / 0.2%           |
| energy_controller       |  99.6% / 265.0 / 0.0%          |  99.8% / 259.7 / 0.0%           |
| pred_guid               |  99.9% / 243.6 / 0.0%          | 100.0% / 242.9 / 0.0%           |
| ftc_joint               | 100.0% / 152.0 / 0.0%          | 100.0% / 149.6 / 0.0%           |
| energy_controller_joint | 100.0% / 279.4 / 0.0%          | 100.0% / 203.2 / 0.0%           |
| pred_guid_joint         | 100.0% / 254.6 / 0.0%          | 100.0% / 195.7 / 0.0%           |
| equilibrium_glide       |  99.6% / 331.4 / 0.8%          |  99.8% / 346.5 / 0.4%           |
| piecewise_constant      | 100.0% / 435.2 / 4.7%          | 100.0% / 1044.6 / 0.0%          |
| fnpag                   |  99.4% / 154.3 / 0.0%          | 100.0% / 143.6 / 0.0%           |

- Conclusion 4 above is superseded against the per-scenario-tuned FNPAG: 143.6
  is below every scratch mean (148.1-158.7) and ft_dense_972 (145.2); only
  ft_dense_515 (129.8), ft_mamba_962 (138.6) and the infeasible ft_lstm_1082
  stay below it. The far-tail table above flies the shared-path-tuned FNPAG.
- The pairs differ in two variables, not one: the retunes trained under
  ADR-0005's strict ceiling (`max_violation_rate = 0.0` from
  `configs/training/common.toml`, 1000-sim validation pool) and the June parents
  were never gated. piecewise_constant and equilibrium_glide score worse than
  their parents (the #172 suspect-run signal), and both parents are heat-flux
  infeasible (piecewise_constant also 0.6% g-load): the gate rejected 476
  better-RMS infeasible candidates in the piecewise_constant retune and 1715 in
  equilibrium_glide. The ceiling-matched reruns are below.
- Gated on the validation pool is not feasible out of it: the ftc retune still
  exceeds the heat-flux limit on 0.2% of these scenarios. The equilibrium_glide
  retune (0.4%) never passed the gate: no candidate was feasible on the
  validation pool and final selection deployed the best-RMS infeasible one.

### Ungated reruns (#188, 2026-09-29)

piecewise_constant and equilibrium_glide rerun with `max_violation_rate = 1.0`
(`classical_campaign.sh classical_ungated`), so each rerun selects by validation
RMS alone, the rule its June parent was selected by. Same pool and columns; g is
the g-load violation rate.

| cell               | parent capture / CVaR95 / flux / g | gated retune                  | ungated rerun                |
|--------------------|------------------------------------|-------------------------------|------------------------------|
| equilibrium_glide  |  99.6% / 331.4 / 0.8% / 0.0%       |  99.8% / 346.5 / 0.4% / 0.0%  |  99.7% / 336.2 / 0.4% / 0.0% |
| piecewise_constant | 100.0% / 435.2 / 4.7% / 0.6%       | 100.0% / 1044.6 / 0.0% / 0.0% | 100.0% / 445.1 / 3.1% / 0.3% |

- The regime alone gains nothing for these two: the ungated reruns sit 4.8
  (equilibrium_glide) and 9.9 (piecewise_constant) m/s above their parents.
  One run each, with no seed repeats to size the spread, so this is no
  measurable gain rather than a loss.
- The ceiling is the whole piecewise_constant regression: opened, the retune
  goes from 1044.6 back to 445.1. Under the ceiling the GA's last feasible
  improvement came at generation 258 of 2000 (validation RMS 4.7e8, against
  3.1e7 for the ungated champion).
- For equilibrium_glide the ceiling bought nothing: the gated run found no
  feasible candidate and deployed an infeasible one at the same 0.4% flux,
  10.3 m/s above the ungated rerun.
- Decision (#188): v4 quotes the ungated reruns for both rows, with their
  violation columns; the piecewise_constant row carries a footnote that the
  best feasible tuning under the strict ceiling scores 1044.6 and stopped
  improving at generation 258 of 2000. The other seven rows quote the gated
  retunes.

## Headline allocation (#173)

Every NN cell above trained at GA 60 x 10, and the per-scenario headline is a
fine-tune of a 512 x 2 shared-path checkpoint, so no cell was trained from
scratch under per-scenario noise at the allocation the paper's methodology
recommends. `campaign.sh experiments/ou_marginal/jobs_headline.txt` trains that
control: `hl_<cell>` is the family config under per_draw at GA n_pop 512,
training_n_sims 2, adaptive curation, cubed transform, 20000 gens from scratch
(three seeds for mamba_962 and dense_515, one for lstm_1082, gru_1014,
dense_972); `ft_dense_p515_s2` / `_s3` add the two missing dense fine-tune seeds
at the fine-tune recipe (60 x 10, 2000 gens), comparable with the three
ft_mamba seeds. About 68 h sequential (7.5 h per `hl_*` job, 25 min per
fine-tune).

Pre-registered decision rule, fixed before any 10^6 score of these runs exists:

- The deployed v4 seed of every cell is seed 1. Seeds 2 and 3 report the mean
  and the standard deviation; nothing is selected on the confirmatory pool.
- Recipe choice, read on the 10^6 per-scenario confirmatory of #174 with the
  feasibility-first rule of Section 6.2: let S be the three-seed mean CVaR99.9
  of `hl_mamba_p962` and F the fine-tune's (163.2 +- 1.3), with sd_S and sd_F
  their standard deviations across the three seeds. If
  S <= F + sqrt((sd_S^2 + sd_F^2) / 3), one standard error of the difference of
  the two means, v4's recipe is single-stage scratch training and the
  fine-tunes leave the main body. Otherwise the two-stage recipe
  (shared-path pre-training, then a per-scenario fine-tune) is a stated finding
  with the `hl_*` cells as its control. Either way Section 6's table quotes one
  allocation for every cell.
- Amendment (2026-10-01, posted on #173 before any 10^6 score of an `hl_*`
  cell existed): every trainer seed counts in S and F. Section 6.2's
  feasibility-first exclusion would already drop `hl_mamba_p962` s1 and s3 on
  their 1-in-1000 heat-load violations at n = 1000 (s2: 0 of 1000), which is
  Poisson noise at that pool size, leaving one seed and no sd_S, so the rule
  could not compare the recipes it was written to compare. Every S and F seed
  passed the same 0% validation gate (ADR-0005); each seed's violation counts
  on the 10^6 pool are quoted next to its CVaR99.9, so a scratch win bought
  with heat shows. The formula is unchanged:
  S <= F + sqrt((sd_S^2 + sd_F^2) / 3) on pooled CVaR99.9
  (`confirmatory_marginal.py --table` evaluates it).
- Outcome (2026-10-02, read on #174's pool): single-stage scratch. The
  per-seed numbers and the caveats stated with it are in the v4 results
  section at the end of this file.

Ran 2026-09-29 20:11 to 2026-10-01 14:35 (about 42 h; the `hl_*` jobs took
0.56 to 0.9 s per generation, dense 3.2 h and mamba 5 to 7.6 h per run). Every
job trained at its TOML's allocation (512 x 2 curator bins for `hl_*`, 60 x 10
for the fine-tunes, read from the final checkpoint), deploys run-local, and the
eleven `best_model.json` md5s are distinct; the checks are posted on #173.

n = 1000 per-scenario eval per job (report.py's pool, seed offset 2M, so these
rows are NOT on this file's paired pool above; `quote_marginal.py` auto-discovers
the run dirs for that). Capture 100% on every row. DV in m/s over captured
scenarios, CVaR95 = mean of the scenarios at or above p95; the last two columns
are violation rates on the same 1000 scenarios (g-load 0.0% everywhere).

| job              | p50   | p95   | CVaR95 | max   | heat load > 25 | flux > 200 | gate rejections |
|------------------|-------|-------|--------|-------|----------------|------------|-----------------|
| hl_mamba_p962    | 110.2 | 116.4 | 119.7  | 148.3 | 0.1%           | 0.0%       |  931            |
| hl_mamba_p962_s2 | 116.4 | 127.0 | 132.0  | 144.8 | 0.0%           | 0.0%       |   95            |
| hl_mamba_p962_s3 | 112.5 | 120.2 | 122.9  | 128.2 | 0.1%           | 0.0%       |  462            |
| hl_dense_p515    | 112.7 | 120.6 | 124.1  | 135.9 | 0.2%           | 0.0%       | 1078            |
| hl_dense_p515_s2 | 112.2 | 121.1 | 129.4  | 171.8 | 0.3%           | 0.0%       | 1261            |
| hl_dense_p515_s3 | 114.3 | 124.7 | 128.0  | 141.2 | 0.3%           | 0.0%       |  758            |
| ft_dense_p515_s2 | 111.6 | 121.7 | 126.1  | 135.9 | 0.0%           | 0.0%       |    9            |
| ft_dense_p515_s3 | 112.7 | 126.1 | 134.8  | 232.8 | 0.0%           | 0.0%       |   68            |
| hl_lstm_p1082    | 114.0 | 123.8 | 130.8  | 182.8 | 0.0%           | 0.0%       | 5089            |
| hl_gru_p1014     | 112.4 | 119.4 | 122.1  | 139.9 | 0.0%           | 0.0%       |   86            |
| hl_dense_p972    | 112.2 | 120.8 | 125.7  | 154.6 | 0.3%           | 0.0%       | 1443            |

Gate rejections count the `REJECTED (infeasible ...)` validations in each
`campaign.log`: better-RMS candidates the ADR-0005 ceiling refused.

Readings at n = 1000 (the 10^6 numbers of #174 decide; nothing here selects):

- Three-seed CVaR95 at 512 x 2: mamba 124.9 +- 6.4 (119.7 / 132.0 / 122.9),
  dense_515 127.2 +- 2.8 (124.1 / 129.4 / 128.0). The mamba seed spread is
  four times the 60 x 10 campaign's (+- 1.6 on the paired pool above).
- The ceiling is the active constraint: heat-load p95 sits at 23.9-24.3 MJ/m2
  and max at 24.6-25.3 on every row, seven of eleven runs rejected hundreds of
  better-RMS candidates, and the three dense_515 scratch seeds and dense_972
  deploy with 0.2-0.3% violations on this pool after clearing the 0% gate on
  the 1000-scenario validation pool (the rule-of-three floor of that gate is
  0.3%). The seed that fought the gate least (mamba s2, 95 rejections) has the
  worst DV and the cleanest heat load. #192 measures the slope.
- The LSTM cell rejected 5089 candidates, 3.5 times any other run, and
  deploys with the cleanest heat-load tail of the campaign (max 24.6) and a
  182.8 m/s DV max; its 60 x 10 s1 was 10.8% heat-load infeasible.
- The dense fine-tune's far tail shows on a second seed: ft_dense_p515_s3
  max 232.8 (3-sigma 167), next to s1's confirmatory max 405.

## v4 confirmatory: every paper row at 10^6 per-scenario (#174)

Same ten pools and estimators as the far-tail section above, every row of the
v4 performance table once, `eval_commit` per row. Rows:
`confirmatory_cells_v4.txt`, 47 cells: the eleven #173 cells (nine `hl_*`, two
dense fine-tune seeds), the seven other fine-tunes, the fifteen 60 x 10 scratch
repeats, `mamba_p962_long`, the four PPO cells, the nine classical rows with
#188's ungated reruns. Run, and rerun the same command to resume:

    caffeinate -i uv run python -u experiments/ou_marginal/confirmatory_marginal.py --manifest experiments/ou_marginal/confirmatory_cells_v4.txt 2>&1 | tee -a experiments/ou_marginal/confirmatory_v4.log

- A run is scored once it has finished: its `final_eval.parquet`, or the
  population trainer's end-only `final_selection.json`, written after its last
  checkpoint (the 60 x 10 campaign trained with `--skip-report`, so 20 of its
  22 dirs have no parquet, 17 of them still to score).
- Violations: rows scored from #174 on carry exact per-replicate counts
  (`viol_n`, `heat_flux_viol_n`, `g_load_viol_n`, `heat_load_viol_n`, summed in
  `pooled`) and the table quotes them at four decimals. Older rows carry only
  means of 2-decimal per-replicate rates, so their pooled 0.00 bounds the rate
  below 0.01% of the 10^6 scenarios, not at zero; the five scored before
  ADR-0005 (the three ft_mamba seeds, ft_dense_p515, mamba_p962_long) have only
  the any-constraint and heat-load rates (`-` for heat flux and g-load).
- Non-captures: each scored cell's recorded `failed_seeds` (the first 50 per
  replicate) are re-flown with no wall-clock limit at the current commit, and
  their terminal outcomes land in the cell's `non_captures` (crash, pending
  crash, hyperbolic, timeout at the simulation's own max_time, or capture,
  meaning the non-capture was the scorer's 5 s timeout). "All genuine crashes,
  none a timeout" is quotable for a cell only when `timeout`, `capture` and
  `hyperbolic` are 0 and every non-capture was re-flown. Rows scored from #174
  on record `model_sha256`, and the re-fly refuses a `best_model.json` whose
  bytes differ; a row scored before it re-flies whatever the dir holds at the
  current commit, so a nonzero `capture` there needs a look before it is read
  as a scorer timeout.
- `--table` (same manifest) prints the table, capture from the pooled
  `n_captured / n` (never the per-replicate 2-decimal `capture_pct`), and the
  #173 recipe rule as amended (headline section).
- `eval_commit`: seven rows of the file predate the per-row field (#171) and
  carry none, six of them v4 rows (`fnpag`, the earlier manifests' untuned
  FNPAG, is not one). They were first committed in 4e231643 (ft_mamba_p962,
  ft_dense_p515, mamba_p962_long, fnpag), fd62090b (ft_mamba_p962_s2 / _s3)
  and 42aa397a (ft_gru_p1014); each was flown at that commit's parent tree or
  earlier.

## v4 confirmatory results: the 10^6 table (#174)

Ran 2026-10-01 21:29 to 2026-10-02 05:54 (8 h 25 min) from main at 76162cd1,
the command above, no interruption. 37 rows scored in this run (`eval_commit`
76162cd1), the 10 already scored kept, and the recorded non-captures of all
32 cells with losses re-flown without the wall clock at 76162cd1. The file
holds 48 cells: these 47 and the earlier manifests' untuned `fnpag`. The
paper's extract (`articles/paper/data/confirmatory_marginal.json`) quotes six
cells by fixed keys and is unchanged. Capture from the pooled counts,
violation rates at four decimals where counts exist (2 decimals, `-` where
not: the ten rows scored before #174), DV in m/s over captured scenarios,
manifest order:

| cell | capture | non-captures | viol % any (flux / g / heat load) | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|---|
| ou_marginal/hl_mamba_p962 | 100.0000% | 0 | 0.1148 (0.0000 / 0.0000 / 0.1148) | 120.1 | 173.9 +- 2.2 | 300 |
| ou_marginal/hl_mamba_p962_s2 | 99.9996% | 4: 4 crash | 0.0437 (0.0000 / 0.0000 / 0.0437) | 131.7 | 159.3 +- 0.6 | 621 |
| ou_marginal/hl_mamba_p962_s3 | 100.0000% | 0 | 0.0567 (0.0000 / 0.0000 / 0.0567) | 122.9 | 142.3 +- 0.5 | 266 |
| ou_marginal/hl_dense_p515 | 100.0000% | 0 | 0.1465 (0.0000 / 0.0000 / 0.1465) | 123.4 | 140.1 +- 0.4 | 243 |
| ou_marginal/hl_dense_p515_s2 | 99.9999% | 1: 1 crash | 0.2112 (0.0000 / 0.0000 / 0.2112) | 127.6 | 185.8 +- 0.6 | 255 |
| ou_marginal/hl_dense_p515_s3 | 99.9929% | 71: 71 crash | 0.2572 (0.0000 / 0.0000 / 0.2572) | 129.0 | 155.6 +- 1.2 | 827 |
| ou_marginal/ft_dense_p515_s2 | 100.0000% | 0 | 0.0110 (0.0000 / 0.0000 / 0.0110) | 127.5 | 172.2 +- 1.4 | 668 |
| ou_marginal/ft_dense_p515_s3 | 100.0000% | 0 | 0.0278 (0.0006 / 0.0000 / 0.0272) | 136.3 | 312.0 +- 6.5 | 1184 |
| ou_marginal/hl_lstm_p1082 | 100.0000% | 0 | 0.0046 (0.0000 / 0.0000 / 0.0046) | 132.7 | 174.6 +- 1.0 | 349 |
| ou_marginal/hl_gru_p1014 | 100.0000% | 0 | 0.0159 (0.0000 / 0.0000 / 0.0159) | 122.8 | 147.8 +- 0.9 | 287 |
| ou_marginal/hl_dense_p972 | 99.9999% | 1: 1 crash | 0.2670 (0.0000 / 0.0000 / 0.2670) | 127.1 | 170.0 +- 0.6 | 330 |
| ou_marginal/ft_mamba_p962 | 99.9995% | 5: 3 crash, 2 pending_crash | 0.00 (- / - / 0.00) | 138.7 | 163.0 +- 0.3 | 249 |
| ou_marginal/ft_mamba_p962_s2 | 99.9970% | 30: 28 crash, 2 pending_crash | 0.00 (- / - / 0.00) | 140.0 | 164.6 +- 0.3 | 240 |
| ou_marginal/ft_mamba_p962_s3 | 99.9920% | 80: 80 crash | 0.00 (- / - / 0.00) | 138.4 | 161.9 +- 0.3 | 211 |
| ou_marginal/ft_dense_p515 | 100.0000% | 0 | 0.03 (- / - / 0.03) | 128.8 | 236.3 +- 2.5 | 405 |
| ou_marginal/ft_gru_p1014 | 99.9288% | 712: 500 crash (500 re-flown) | 0.00 (0.00 / 0.00 / 0.00) | 150.4 | 183.2 +- 0.7 | 315 |
| ou_marginal/ft_dense_p972 | 99.9992% | 8: 6 crash, 2 pending_crash | 0.0149 (0.0000 / 0.0000 / 0.0149) | 146.0 | 258.1 +- 5.6 | 805 |
| ou_marginal/ft_lstm_p1082 | 99.9860% | 140: 139 crash, 1 pending_crash | 10.7435 (0.0000 / 0.0000 / 10.7435) | 129.2 | 225.7 +- 2.6 | 1078 |
| ou_marginal/mamba_p962 | 100.0000% | 0 | 0.0749 (0.0000 / 0.0000 / 0.0749) | 150.5 | 193.5 +- 0.6 | 276 |
| ou_marginal/mamba_p962_s2 | 99.9896% | 104: 104 crash | 0.0029 (0.0025 / 0.0000 / 0.0004) | 151.4 | 198.8 +- 2.0 | 769 |
| ou_marginal/mamba_p962_s3 | 100.0000% | 0 | 0.0252 (0.0252 / 0.0000 / 0.0000) | 146.6 | 172.3 +- 0.4 | 299 |
| ou_marginal/dense_p515 | 99.9992% | 8: 7 crash, 1 pending_crash | 0.0006 (0.0001 / 0.0001 / 0.0005) | 155.7 | 187.8 +- 1.4 | 639 |
| ou_marginal/dense_p515_s2 | 99.9969% | 31: 30 crash, 1 pending_crash | 0.0000 (0.0000 / 0.0000 / 0.0000) | 154.5 | 294.0 +- 4.2 | 761 |
| ou_marginal/dense_p515_s3 | 99.9907% | 93: 55 crash, 38 pending_crash | 0.0338 (0.0338 / 0.0000 / 0.0000) | 161.4 | 199.7 +- 0.5 | 365 |
| ou_marginal/lstm_p1082 | 100.0000% | 0 | 0.0001 (0.0001 / 0.0000 / 0.0000) | 144.1 | 167.9 +- 0.6 | 558 |
| ou_marginal/lstm_p1082_s2 | 100.0000% | 0 | 0.0028 (0.0028 / 0.0000 / 0.0000) | 167.3 | 208.4 +- 0.5 | 333 |
| ou_marginal/lstm_p1082_s3 | 99.9990% | 10: 10 crash | 0.0005 (0.0000 / 0.0000 / 0.0005) | 146.3 | 181.7 +- 2.5 | 890 |
| ou_marginal/gru_p1014 | 99.9978% | 22: 22 crash | 0.0019 (0.0000 / 0.0000 / 0.0019) | 148.1 | 200.2 +- 2.2 | 1079 |
| ou_marginal/gru_p1014_s2 | 99.9989% | 11: 10 crash, 1 pending_crash | 0.0023 (0.0000 / 0.0000 / 0.0023) | 145.1 | 175.3 +- 1.2 | 763 |
| ou_marginal/gru_p1014_s3 | 100.0000% | 0 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 158.7 | 195.3 +- 1.1 | 721 |
| ou_marginal/dense_p972 | 99.9104% | 896: 500 crash (500 re-flown) | 0.0000 (0.0000 / 0.0000 / 0.0000) | 147.1 | 211.9 +- 0.8 | 287 |
| ou_marginal/dense_p972_s2 | 99.9998% | 2: 2 crash | 0.1903 (0.1903 / 0.0000 / 0.0000) | 157.1 | 187.2 +- 0.5 | 478 |
| ou_marginal/dense_p972_s3 | 99.9997% | 3: 2 crash, 1 pending_crash | 0.0000 (0.0000 / 0.0000 / 0.0000) | 145.9 | 179.2 +- 1.3 | 747 |
| mamba_p962_long | 97.9330% | 20670: 438 crash, 61 pending_crash, 1 timeout (500 re-flown) | 0.91 (- / - / 0.91) | 188.2 | 221.3 +- 0.5 | 270 |
| paper/rl/dense_p515_ppo_scratch | 99.9690% | 310: 200 crash, 110 pending_crash | 0.69 (0.69 / 0.00 / 0.00) | 271.7 | 372.0 +- 1.4 | 486 |
| paper/rl/dense_p515_ppo_warm | 100.0000% | 0 | 0.04 (0.00 / 0.00 / 0.04) | 130.6 | 197.6 +- 2.2 | 378 |
| paper/rl/gru_p1014_ppo_scratch | 99.9886% | 114: 28 crash, 86 pending_crash | 0.85 (0.85 / 0.00 / 0.00) | 411.1 | 440.0 +- 0.9 | 620 |
| paper/rl/gru_p1014_ppo_warm | 99.9545% | 455: 453 crash (453 re-flown) | 0.00 (0.00 / 0.00 / 0.00) | 159.6 | 185.2 +- 0.5 | 289 |
| ou_marginal/classical/ftc | 99.9992% | 8: 7 crash, 1 pending_crash | 0.1228 (0.1225 / 0.0027 / 0.0000) | 243.8 | 349.6 +- 0.5 | 536 |
| ou_marginal/classical/energy_controller | 99.7564% | 2436: 500 crash (500 re-flown) | 0.0078 (0.0078 / 0.0000 / 0.0000) | 264.8 | 340.4 +- 0.6 | 429 |
| ou_marginal/classical/pred_guid | 99.8880% | 1120: 500 crash (500 re-flown) | 0.0057 (0.0057 / 0.0000 / 0.0000) | 253.1 | 332.6 +- 0.4 | 425 |
| ou_marginal/classical/ftc_joint | 100.0000% | 0 | 0.0000 (0.0000 / 0.0000 / 0.0000) | 147.6 | 177.8 +- 0.5 | 242 |
| ou_marginal/classical/energy_controller_joint | 99.9820% | 180: 180 crash | 0.0182 (0.0001 / 0.0000 / 0.0181) | 200.2 | 299.1 +- 1.7 | 915 |
| ou_marginal/classical/pred_guid_joint | 99.9955% | 45: 45 crash | 0.0057 (0.0018 / 0.0000 / 0.0039) | 195.5 | 340.4 +- 3.9 | 1255 |
| ou_marginal/classical_ungated/piecewise_constant | 99.9183% | 817: 325 crash, 175 pending_crash (500 re-flown) | 3.1284 (3.0928 / 0.2495 / 0.0000) | 445.7 | 632.2 +- 1.8 | 940 |
| ou_marginal/classical_ungated/equilibrium_glide | 99.7936% | 2064: 402 crash, 98 pending_crash (500 re-flown) | 0.5427 (0.5424 / 0.0009 / 0.0000) | 332.9 | 476.7 +- 1.1 | 614 |
| ou_marginal/classical/fnpag | 99.9762% | 238: 238 crash | 0.0010 (0.0000 / 0.0000 / 0.0010) | 144.9 | 189.1 +- 1.8 | 691 |

Non-captures: no re-fly ended in `capture` or `hyperbolic`, so no non-capture
in the file was the scorer's 5 s timeout, and one ended in `timeout` (1 of the
500 re-flown for `mamba_p962_long`, at the simulation's own `max_time`). "All
genuine crashes" (crash or pending crash) holds for every cell whose
non-captures were all re-flown. Eight cells exceed the 50-per-replicate cap
and had 500 classified (453 for `gru_p1014_ppo_warm`): `ft_gru_p1014` (712),
`dense_p972` (896), `energy_controller` (2436), `pred_guid` (1120), the two
ungated classical reruns (817 and 2064), `gru_p1014_ppo_warm` (455) and
`mamba_p962_long` (20670).

Outcome of the #173 recipe rule, read on this pool as amended (every trainer seed counts):

| | CVaR99.9 per seed (m/s) | mean +- sd | heat load > 25 MJ/m2 per 10^6 | max DV |
|---|---|---|---|---|
| S, `hl_mamba_p962` s1 / s2 / s3 (512 x 2 scratch) | 173.93 / 159.34 / 142.34 | 158.54 +- 15.81 | 1148 / 437 / 567 | 300 / 621 / 266 |
| F, `ft_mamba_p962` s1 / s2 / s3 (fine-tune) | 163.02 / 164.55 / 161.89 | 163.15 +- 1.34 | below 100 each (2-decimal rate 0.00) | 249 / 240 / 211 |

Threshold F + sqrt((sd_S^2 + sd_F^2) / 3) = 172.31; S = 158.54 <= 172.31, so the rule selects single-stage scratch training at 512 x 2: v4's recipe is single-stage and the fine-tunes leave the main body, with the `hl_*` cells as the quoted rows.

Stated next to it, as the amendment requires: the three scratch seeds span 31.6 m/s against 2.7 m/s for the fine-tune seeds, so the 4.6 m/s advantage of the scratch mean is inside its own seed spread, and every scratch seed violates the heat-load ceiling more often than any fine-tune seed (437 to 1148 scenarios per 10^6 against fewer than 100). The rule was fixed before these numbers existed and is applied as written; what the paper says about the margin and the violations is a writing decision, not a selection on this pool.

Readings (nothing here selects; the deployed seed of every cell is seed 1):

- The 60 x 10 scratch repeats, three-seed mean +- sd of CVaR99.9: mamba_p962
  188.2 +- 14.0, dense_p515 227.2 +- 58.2, lstm_p1082 186.0 +- 20.6,
  gru_p1014 190.3 +- 13.2, dense_p972 192.8 +- 17.0. Every 512 x 2 `hl_*` row
  beats its family's 60 x 10 mean; `hl_dense_p515` 160.5 +- 23.2 over its
  three seeds.
- The dense fine-tune's far tail: `ft_dense_p515` 236.3 / 172.2 / 312.0 over
  its three seeds (max 405 / 668 / 1184), against the `hl_dense_p515` seeds
  140.1 / 185.8 / 155.6.
- `ft_lstm_p1082` is heat-load infeasible on this pool (10.74%);
  `dense_p972_s2` (0.19%) and `classical/ftc` (0.12%) violate heat flux, not
  heat load.
- The retuned FNPAG (`classical/fnpag`, #172): CVaR99.9 189.1 +- 1.8, 238
  non-captures (all crashes), 0.001% violations; the untuned `fnpag` row of the
  earlier manifests stood at 236.7 with 6321 non-captures.

## Heat-load ceiling sensitivity: the DV-vs-heat-load slope (#192)

Design in the issue: the #173 seed-1 champions `hl_mamba_p962` and
`hl_dense_p515` continued 2000 gens from their g20000 checkpoint at the
fine-tune recipe (GA 60 x 10, adaptive seeds, per_draw) under
`[flight.constraints] max_heat_load` 25000 (the v4 limit, the control) /
27500 / 30000 kJ/m2, seed 1 on every leg (configs `hs_<cell>_q<NN>.toml`,
jobs `jobs_heat_load.txt`, PR #198). Ran 2026-10-02 10:50 to 12:51 (2 h 1 min,
about 20 min per leg, from main at 7c61ec53), no runner stop: the control
legs deploy the copied champion byte for byte, and no two legs of a cell ended
on the same checkpoint. v4 keeps `max_heat_load = 25000` whatever the slope;
this section changes text, not the limit.

What the trainer did under each ceiling (the ADR-0005 gate is 0% violations on
the 1000-scenario validation pool at the TOML's own ceiling):

| leg | ceiling (kJ/m2) | champion val RMS at the end | deployed model | best last-generation candidate |
|---|---|---|---|---|
| hs_mamba_p962_q25 | 25000 | 1.358e+06 | the copied champion (same bytes) | 1.282e+06, infeasible (36.7% heat load) |
| hs_mamba_p962_q27 | 27500 | 1.268e+06 | promoted during the 2000 gens | 1.291e+06, feasible |
| hs_mamba_p962_q30 | 30000 | 1.291e+06 | promoted during the 2000 gens | 1.295e+06, feasible |
| hs_dense_p515_q25 | 25000 | 1.471e+06 | the copied champion (same bytes) | 1.374e+06, infeasible (13.1% heat load) |
| hs_dense_p515_q27 | 27500 | 1.365e+06 | promoted during the 2000 gens | 1.387e+06, feasible |
| hs_dense_p515_q30 | 30000 | 1.354e+06 | promoted during the 2000 gens | 1.366e+06, feasible |

At the v4 ceiling neither control leg found a feasible improvement in 2000
gens: the best last-generation candidate of each had the lower RMS and failed
the heat-load gate. At 27500 and 30000 a lower-RMS candidate passed the gate
and was promoted, so the relaxed legs deploy a new policy.

Scored by `heat_load_slope.py` on `quote_marginal.py`'s paired n = 1000
marginal pool (per-scenario noise through its per-seed `simulation.random_seed`
override, `heat_load_slope.json`): every leg at its own ceiling (the TOML as
report.py flies it) and re-flown under the v4 limit through an explicit
`flight.constraints.max_heat_load = 25000` override, which also sets the limit
the violation column is scored against; the six `hl_<cell>{,_s2,_s3}`
sources once at 25000 on the same pool. DV in m/s over captured scenarios.

| run | flown at | capture | p50 | CVaR95 | max | heat load p95 / max (MJ/m2) | heat load > ceiling | flux > 200 |
|---|---|---|---|---|---|---|---|---|
| hl_mamba_p962 | 25.0 | 100.0% | 110.2 | 119.6 | 159.3 | 24.11 / 25.15 | 0.1% | 0.0% |
| hl_mamba_p962_s2 | 25.0 | 100.0% | 115.8 | 130.8 | 139.2 | 23.90 / 24.89 | 0.0% | 0.0% |
| hl_mamba_p962_s3 | 25.0 | 100.0% | 112.3 | 123.0 | 133.8 | 24.10 / 24.74 | 0.0% | 0.0% |
| hs_mamba_p962_q25 | 25.0 | 100.0% | 110.2 | 119.6 | 159.3 | 24.11 / 25.15 | 0.1% | 0.0% |
| hs_mamba_p962_q25 | 25.0 (v4) | 100.0% | 110.2 | 119.6 | 159.3 | 24.11 / 25.15 | 0.1% | 0.0% |
| hs_mamba_p962_q27 | 27.5 | 100.0% | 107.7 | 116.5 | 153.4 | 25.93 / 26.67 | 0.0% | 0.0% |
| hs_mamba_p962_q27 | 25.0 (v4) | 99.5% | 123.2 | 132.2 | 136.6 | 25.25 / 25.92 | 11.7% | 0.0% |
| hs_mamba_p962_q30 | 30.0 | 100.0% | 108.3 | 120.3 | 189.3 | 25.72 / 26.72 | 0.0% | 0.0% |
| hs_mamba_p962_q30 | 25.0 (v4) | 98.9% | 154.2 | 187.8 | 213.1 | 24.72 / 25.74 | 1.7% | 0.0% |
| hl_dense_p515 | 25.0 | 100.0% | 112.8 | 122.3 | 140.2 | 24.23 / 24.94 | 0.0% | 0.0% |
| hl_dense_p515_s2 | 25.0 | 100.0% | 112.3 | 127.1 | 188.3 | 24.33 / 25.08 | 0.1% | 0.0% |
| hl_dense_p515_s3 | 25.0 | 100.0% | 113.9 | 130.0 | 155.2 | 24.25 / 25.05 | 0.2% | 0.0% |
| hs_dense_p515_q25 | 25.0 | 100.0% | 112.8 | 122.3 | 140.2 | 24.23 / 24.94 | 0.0% | 0.0% |
| hs_dense_p515_q25 | 25.0 (v4) | 100.0% | 112.8 | 122.3 | 140.2 | 24.23 / 24.94 | 0.0% | 0.0% |
| hs_dense_p515_q27 | 27.5 | 100.0% | 110.1 | 120.2 | 142.7 | 25.73 / 26.73 | 0.0% | 0.0% |
| hs_dense_p515_q27 | 25.0 (v4) | 100.0% | 110.7 | 120.9 | 127.5 | 25.65 / 26.66 | 16.7% | 0.0% |
| hs_dense_p515_q30 | 30.0 | 100.0% | 110.1 | 119.3 | 146.7 | 25.82 / 26.86 | 0.0% | 0.0% |
| hs_dense_p515_q30 | 25.0 (v4) | 100.0% | 111.3 | 120.9 | 127.8 | 25.53 / 26.48 | 13.5% | 0.0% |

mamba_p962: CVaR95 q30 - q25 = 0.67 m/s (0.134 m/s per MJ/m2); source seeds 119.6 / 130.8 / 123.0, sd 5.73: inside the seed spread

dense_p515: CVaR95 q30 - q25 = -2.97 m/s (-0.594 m/s per MJ/m2); source seeds 122.3 / 127.1 / 130.0, sd 3.9: inside the seed spread

Pre-registered reading (issue #192): the slope is CVaR95 at q30 minus q25 per
MJ/m2, read against the source cell's three-seed CVaR95 spread on the same
pool. mamba_p962: +0.67 m/s over 5 MJ/m2 (+0.134 m/s per MJ/m2)
against a seed sd of 5.73; dense_p515: -2.97 m/s (-0.594 per
MJ/m2) against 3.9. Both inside the seed spread, so by the rule Section 6
states that the ceiling costs nothing measurable at the sizing tail and the
question closes; TODO.md does not gain the TPS-mass model. The relaxed legs
did use the headroom (heat-load p95 from 24.1-24.2 to 25.7-25.9 MJ/m2 at
their own ceilings) and bought 2 to 3 m/s of median DV for it, nothing at
CVaR95.

Beyond the rule, the re-fly under the v4 limit: the relaxed policies are not
deployable there. Heat-load violations 11.7% (mamba q27), 16.7% and 13.5%
(dense q27, q30), and mamba q30 loses 1.1% capture with CVaR95 187.8 against
120.3 at its own ceiling. That is the caveat the issue stated before the runs:
NN input 7 is `cumulative_heat_load / max_heat_load` (`tick.rs`) and the
thermal limiter's ramp starts at a fraction of the same limit, so a policy
carries its training ceiling inside it, and the "25.0 (v4)" rows measure that
mismatch, not a pure constraint trade. A single seed per leg; the source
seeds' spread (5.7 and 3.9 m/s CVaR95) bounds what one run can resolve, and
these are n = 1000 numbers, not the 10^6 pool.
