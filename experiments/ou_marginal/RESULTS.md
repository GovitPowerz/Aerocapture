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

## v4 per-scenario quotes: the paired pool and the 2M pool (#175)

The paired tables (Section 7, the Section 6.4 conditioning table, the Section 5
RL sentence) read n = 1000 paired pools. `quote_marginal.py` discovers the
eleven #173 cells (`hl_*`, `ft_dense_p515_s2` / `_s3`; the `hs_*` ceiling legs
of #192 are skipped: `heat_load_slope.py` quotes them) and scored them under
both regimes on the paired pool (2026-10-02, from main at dc040085). The five
shared-path champions and the shared-path-tuned classicals were already in
`quote_results.json` (the Dense 515 row re-flown from the repaired model under
#170, the classical retunes under #172 / #188).

Shared-path (frozen) / per-scenario (marginal) regime, both on the same 1000 seeds:

### Conditioning table data: the five shared-path champions, both regimes

| cell | capture % shared / per-scenario | p50 | CVaR95 | heat-load viol % |
|---|---|---|---|---|
| Mamba 962 (`mamba_p962_long`) | 100.0 / 98.0 | 109.7 / 128.8 | 115.9 / 194.4 | 0.0 / 1.2 |
| LSTM 1082 | 100.0 / 100.0 | 108.1 / 112.4 | 116.3 / 170.4 | 14.6 / 17.0 |
| GRU 1014 | 100.0 / 98.0 | 111.2 / 128.9 | 119.3 / 198.1 | 0.0 / 1.5 |
| dense 972 | 100.0 / 99.3 | 111.9 / 128.2 | 121.3 / 194.9 | 0.0 / 0.5 |
| dense 515 (repaired, #170) | 100.0 / 100.0 | 109.5 / 115.5 | 117.4 / 201.8 | 0.0 / 0.3 |

### Headline-allocation cells (#173) on the paired pool

| cell | capture % shared / per-scenario | p50 | CVaR95 | heat-load viol % |
|---|---|---|---|---|
| `ou_hl_mamba_p962`: Mamba 962 s1 (deployed) | 100.0 / 100.0 | 110.7 / 110.2 | 118.5 / 119.6 | 0.0 / 0.1 |
| `ou_hl_mamba_p962_s2`: Mamba 962 s2 | 100.0 / 100.0 | 115.8 / 115.8 | 127.2 / 130.8 | 0.0 / 0.0 |
| `ou_hl_mamba_p962_s3`: Mamba 962 s3 | 100.0 / 100.0 | 113.3 / 112.3 | 121.9 / 123.0 | 0.0 / 0.0 |
| `ou_hl_dense_p515`: dense 515 s1 (deployed) | 100.0 / 100.0 | 113.8 / 112.8 | 124.0 / 122.3 | 0.0 / 0.0 |
| `ou_hl_dense_p515_s2`: dense 515 s2 | 100.0 / 100.0 | 113.1 / 112.3 | 124.0 / 127.1 | 0.0 / 0.1 |
| `ou_hl_dense_p515_s3`: dense 515 s3 | 100.0 / 100.0 | 114.5 / 113.9 | 128.4 / 130.0 | 0.0 / 0.2 |
| `ou_hl_lstm_p1082`: LSTM 1082 | 100.0 / 100.0 | 114.4 / 113.7 | 129.7 / 130.4 | 0.0 / 0.0 |
| `ou_hl_gru_p1014`: GRU 1014 | 100.0 / 100.0 | 112.6 / 112.3 | 121.6 / 123.0 | 0.0 / 0.0 |
| `ou_hl_dense_p972`: dense 972 | 100.0 / 100.0 | 112.6 / 112.3 | 124.3 / 125.3 | 0.0 / 0.0 |
| `ou_ft_dense_p515_s2`: ft dense 515 s2 | 100.0 / 100.0 | 112.3 / 111.5 | 124.4 / 128.2 | 0.0 / 0.0 |
| `ou_ft_dense_p515_s3`: ft dense 515 s3 | 100.0 / 100.0 | 112.0 / 112.3 | 132.8 / 136.3 | 0.0 / 0.0 |

Per-scenario CVaR95 across the three seeds: Mamba 962 119.6 / 130.8 / 123.0
(124.5 +- 5.7), dense 515 122.3 / 127.1 / 130.0 (126.5 +- 3.9). The three Mamba
numbers are the `heat_load_slope.json` source rows, bit-identical (same pool,
same code path). Every cell trained under per-scenario noise scores the two
regimes within a few m/s of each other; the shared-path champions lose
54-84 m/s of CVaR95 when the noise path varies (LSTM 54, the others 74-84), the conditioning finding of
Section 6.4 restated on the repaired Dense 515.

Bundle: `collect_runs.py` walks every `training_output/ou_marginal/**/final_eval.parquet`
except the `hs_*` legs (24 cells: the eleven #173 cells, the two arxiv-v3
fine-tunes, the nine #172 retunes and the two #188 ungated reruns) into
`articles/paper/data/runs/ou_marginal/<path>/`, run logs included since #179
(stripped with `strip_run_logs.py`; they join the v4 Release asset, #183).
`aggregate_results.PAIRED` gains the `v4_*` rows: Mamba 962 (seed 1)
against joint-FTC (retuned), FNPAG (retuned), dense 515, LSTM 1082, GRU 1014,
dense 972 and fixed-reference FTC (retuned); joint-FTC against fixed-reference FTC
(both retuned) and against FNPAG; the PPO pairs follow the `18_rl_baseline.sh hl`
retrain. All nine v4 pairs share the dispersion fingerprint (the same 1000
scenarios of report.py's 2M pool, NOT the paired pool of the two tables above:
hl_mamba_p962_s2 scores per-scenario CVaR95 130.8 there and 132.0 in its bundle
parquet).

## v4 Section 5 RL baseline: PPO against the #173 cells, with seed repeats (#175)

The four PPO cells retrained against the #173 scratch cells (`18_rl_baseline.sh
hl`, 2026-10-02) and the PPO-scratch seed repeats (`18_rl_baseline.sh
hl_repeats`, `[rl] torch_seed` 2 / 3). The s1 cells ran before torch was seeded:
policy init and exploration noise were unseeded draws, and the first dense
scratch run ended at 444 m/s mean against 180 for the arxiv-v3 run of the same
code. Every policy flies within 1 m/s mean under either scaffolding, so the gap
is run-to-run spread, not the protocol. Population references: `hl_dense_p515` /
`hl_gru_p1014`.

Paired on report.py's 2M pool (n = 1000, `aggregate_results.PAIRED` `v4_ppo_*`),
PPO minus population, m/s:

| cell | Delta mean [95% CI] | Delta p95 | Delta CVaR95 | PPO-win % |
|---|---|---|---|---|
| dense scratch s1 (unseeded) | +330.2 [326.2, 334.1] | +435.0 | +451.5 | 0.0 |
| dense scratch s2 | +73.0 [71.8, 74.2] | +95.0 | +99.7 | 0.0 |
| dense scratch s3 | +82.5 [80.4, 84.6] | +120.0 | +138.1 | 0.3 |
| dense warm | -0.2 [-0.4, +0.0] | -0.9 | -0.5 | 54.3 |
| GRU scratch s1 (unseeded) | +338.2 [333.8, 342.4] | +431.8 | +446.5 | 0.0 |
| GRU scratch s2 | +292.6 [288.7, 296.4] | +361.8 | +371.3 | 0.0 |
| GRU scratch s3 | +107.5 [104.6, 110.5] | +174.3 | +225.8 | 0.0 |
| GRU warm | +0.4 [+0.3, +0.5] | +0.4 | +0.7 | 36.6 |

Confirmatory, 10 x 100k per-scenario (`confirmatory_cells_v4.txt`, 2026-10-05,
eval commit c3433c52), DV in m/s over captured scenarios:

| cell | capture % | p95 | CVaR95 | CVaR99.9 | heat-load viol % |
|---|---|---|---|---|---|
| population dense 515 | 100 | 120.3 | 123.4 | 140.1 | 0.15 |
| dense scratch s1 (unseeded) | 99.9999 | 555.7 | 573.3 | 621.3 | 26.56 |
| dense scratch s2 | 99.93 | 217.1 | 225.2 | 261.8 | 0.12 |
| dense scratch s3 | 99.95 | 239.7 | 266.5 | 446.1 | 0.69 |
| dense warm | 100 | 120.1 | 123.7 | 149.3 | 0.23 |
| population GRU 1014 | 100 | 119.8 | 122.8 | 147.8 | 0.02 |
| GRU scratch s1 (unseeded) | 100 | 549.7 | 566.0 | 610.8 | 0.0 |
| GRU scratch s2 | 99.995 | 477.9 | 488.2 | 517.2 | 0.0 |
| GRU scratch s3 | 99.995 | 294.7 | 350.7 | 506.4 | 0.0 |
| GRU warm | 100 | 120.3 | 123.6 | 155.6 | 0.01 |

The re-flown non-captures (no sim timeout) are all physical, crash or pending
crash, no timeouts; dense s2's 659 are classified on the first 500 re-flown.

Reading: every PPO-scratch run loses to its population cell on essentially every
scenario, by 73 to 338 m/s mean depending on the seed; the seed spread (dense
186-444, GRU 221-451 m/s mean at n = 1000) is wider than any architecture effect,
so the scratch row is quoted as the three-seed spread, not one run. The
warm-started cells deploy a checkpoint tied with the population cell at CVaR95
(dense CI straddles zero, GRU +0.5) but 8-9 m/s worse at CVaR99.9; their policy
gradient then walks off it, as in arxiv-v3.

## v4 champion controls (#176)

`experiments/paper/15_state_controls.sh v4` on the deployed champion
`ou_marginal/hl_mamba_p962` (seed 1 of #173's rule; `best_model.json` sha256
`d92c6bbd...bc517d4d`, the model #174 scored), per-scenario noise throughout.
Started 2026-10-06 from `feature/v4-champion-controls` at f3ee1406 (#202): the
four quick steps below took about 7 minutes, then the two control retrains
began; the controls (record 2) and the rustc check (record 6) close the section.
Posted on #176.

### Reset state at 10^6

`confirmatory_marginal.py --cells ou_marginal/v4_reset_state:<champion toml>:<champion dir>
--extra-override guidance.neural_network.reset_state_every_tick=true` (eval
commit f3ee1406), against the champion's #174 row; the reset state's non-captures
re-flown without the wall clock. DV in m/s over captured scenarios:

| cell | capture | non-captures | viol % any (flux / g / heat load) | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|---|
| ou_marginal/hl_mamba_p962 | 100.0000% | 0 | 0.1148 (0.0000 / 0.0000 / 0.1148) | 120.1 | 173.9 +- 2.2 | 300 |
| ou_marginal/v4_reset_state | 99.9899% | 101: 99 crash, 2 timeout | 0.0264 (0.0000 / 0.0000 / 0.0264) | 807.7 | 860.3 +- 0.4 | 920 |

Median per replicate 110.2 against 694.6-695.4; p95 116.5 against 789.1.

Reading: capture holds to within 0.01 pts, but the whole distribution moves,
not only the tail: the median goes from 110 to 695 m/s. The issue expected
capture parity with a collapsed tail; without its state the policy still
captures, at six times the median DV, so the state carries the bulk as well as
the tail.

### Input sensitivity

`python -m aerocapture.training.ablation <champion dir> --toml <champion toml>
--n-sims 1000 --sim-timeout 5 --cost-transform log`: the config's own Monte
Carlo, per_draw (stated in `ablation_results.json`), costs in the log transform
the paper's ablation figure reads (the config trains on cubed). Baseline cost
4.716; every one of the 17 masked inputs costs something when zeroed:

| rank | input | cost increase | rank | input | cost increase |
|---|---|---|---|---|---|
| 1 | orbital_energy | 4.193 | 10 | drag_accel | 0.616 |
| 2 | predicted_dv1 | 1.799 | 11 | predicted_dv3 | 0.527 |
| 3 | predicted_dv2 | 1.790 | 12 | lift_accel | 0.509 |
| 4 | hdot_nominal | 1.368 | 13 | prev_bank_signed_cos | 0.495 |
| 5 | radial_velocity | 1.289 | 14 | prev_realized_cos | 0.455 |
| 6 | pdyn_error | 1.227 | 15 | eccentricity_excess | 0.259 |
| 7 | accel_magnitude | 1.087 | 16 | prev_bank_signed_sin | 0.239 |
| 8 | heat_flux_fraction | 0.788 | 17 | prev_realized_sin | 0.029 |
| 9 | heat_load_fraction | 0.662 | | | |

The arxiv-v3 headline (`runs/headline/mamba_p962`, n = 500) ranked
eccentricity_excess first; the v4 champion ranks it 15th and leans on
orbital_energy and the first two predicted-DV inputs. Section 8's
input-sensitivity text changes with it.

### Fresh-pool re-quote

`fresh_pool_requote.py <champion dir> --toml <champion toml> --n-sims 1000
--noise-seeding per_draw` (`<champion>/fresh_pool_requote.json`), against the
champion's report.py `final_eval.parquet` on the 2M pool (also per_draw, n = 1000),
same estimators (CVaR95 = mean of the top 50 captured DVs):

| pool | capture | mean | p50 | p95 | p99 | CVaR95 | max |
|---|---|---|---|---|---|---|---|
| 2M (report.py final eval) | 100.0% | 110.65 | 110.19 | 116.36 | 120.75 | 119.65 | 148.3 |
| 8M (fresh) | 100.0% | 111.03 | 110.47 | 116.88 | 120.63 | 122.75 | 185.2 |

The fresh pool sits 0.4 m/s above on the mean and 3.1 m/s above on CVaR95, an
average of 50 values at this n; the 10^6 pool's CVaR95 is 120.1.

### Off-nominal stress at depth

`stress_depth_eval.py --n-sims 10000` (`make -C articles/paper mc-stress-depth`,
`articles/paper/data/stress_depth.json`): the champion and the four #172 retunes
on the reserved 9M stress pool with atmosphere, density perturbation, navigation
and nav filter at `high`, per_draw only, 2000 bootstrap resamples. DV in m/s over
captured scenarios:

| scheme | capture [95% CI] | mean [95% CI] | p95 | CVaR95 [95% CI] |
|---|---|---|---|---|
| NN (champion) | 85.29% [84.63, 85.98] | 221.6 [218.6, 224.7] | 445.0 | 658.4 [625.0, 692.1] |
| joint-FTC | 95.88% [95.50, 96.26] | 183.7 [181.9, 185.7] | 304.8 | 468.4 [442.4, 497.0] |
| FTC-fixed | 92.42% [91.91, 92.93] | 254.1 [252.0, 256.4] | 409.3 | 569.9 [543.1, 597.5] |
| PredGuid | 89.30% [88.69, 89.88] | 273.9 [271.7, 276.1] | 418.5 | 572.4 [546.3, 598.7] |
| FNPAG | 92.85% [92.34, 93.35] | 189.9 [186.7, 193.1] | 317.1 | 716.1 [665.1, 767.7] |

Paired on scenario, NN minus each classical:

| vs | capture pts [CI] | CVaR95 [CI] | both-captured mean [CI] (pairs) | NN win rate |
|---|---|---|---|---|
| joint-FTC | -10.59 [-11.23, -9.97] | +190.1 [+158.1, +222.5] | +43.3 [+40.6, +45.8] (8491) | 0.41 |
| FTC-fixed | -7.13 [-7.83, -6.42] | +88.6 [+62.2, +113.4] | -24.7 [-27.3, -22.0] (8215) | 0.67 |
| PredGuid | -4.01 [-4.75, -3.26] | +86.1 [+54.8, +117.4] | -46.9 [-49.6, -44.2] (7973) | 0.73 |
| FNPAG | -7.56 [-8.26, -6.88] | -57.7 [-108.1, -5.0] | +25.7 [+22.3, +29.3] (8238) | 0.38 |

Reading: under the high regime the champion has the lowest capture of the five,
4 to 11 pts below every classical scheme with every CI clear of zero, and its
CVaR95 is worse than every classical scheme except FNPAG. For reference only
(different cells, legacy noise, n = 1000, unpaired across the two files):
`robustness_stress.json` had the shared-path NN at 90.1% against joint-FTC's
94.5%. Section 7.2's off-nominal probe reports the deployed NN as the least
robust of the five on capture.

### Mechanism controls, seed 1 at 10^6

`campaign.sh experiments/ou_marginal/jobs_controls.txt`, both from scratch at
the champion's allocation (GA 512 x 2, 20000 gens, adaptive seeds, cubed),
trainer seed 1, per_draw: `ctrl_window_p970` ran 11:53 to 16:06 (4 h 13 min);
`ctrl_mamba_p962_nodv` ran 16:07 to 23:07, stopped by Ctrl-C at gen 8996
(18:22) and resumed from that checkpoint at 20:15 (5 h 7 min of training).
Then both on the 10^6 pool by the runner (eval commit 3c6fe51d: the checkout
had moved to #196's branch, whose one commit touches `animate.py`, its test and
the README only). DV in m/s over captured scenarios:

| cell | capture | viol % any (flux / g / heat load) | p95 | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|---|
| ou_marginal/hl_mamba_p962 (champion) | 100.0000% | 0.1148 (0.0000 / 0.0000 / 0.1148) | 116.5 | 120.1 | 173.9 +- 2.2 | 300 |
| ou_marginal/hl_mamba_p962_s2 | 99.9996% | 0.0437 (0.0000 / 0.0000 / 0.0437) | 127.2 | 131.7 | 159.3 +- 0.6 | 621 |
| ou_marginal/hl_mamba_p962_s3 | 100.0000% | 0.0567 (0.0000 / 0.0000 / 0.0567) | 120.1 | 122.9 | 142.3 +- 0.5 | 266 |
| ou_marginal/ctrl_window_p970 | 100.0000% | 0.1521 (0.0000 / 0.0000 / 0.1521) | 122.8 | 127.0 | 158.3 +- 0.6 | 258 |
| ou_marginal/ctrl_mamba_p962_nodv | 100.0000% | 0.2471 (0.0000 / 0.0000 / 0.2471) | 122.7 | 125.8 | 145.6 +- 0.5 | 236 |

Pre-registered rule (`confirmatory_marginal.py --table`, `#176 controls rule`):
the champion's three-seed CVaR99.9 range is [142.3, 173.9]; the window control
(158.3) and the no-predicted-DV control (145.6) both land inside it, so both owe
seeds 2 and 3 before the paper reads them. The four repeats are registered in
`jobs_controls.txt` (`ctrl_*_s2`, `ctrl_*_s3`, trainer seeds 2 and 3).

On one seed each, neither removed ingredient shows a tail cost: the stateless
window policy sits at the champion seeds' mean CVaR99.9 (158.5), the network
without the predicted-DV inputs next to the best seed. Their bulk is a few m/s
behind seed 1 (replicate median 113.8 and 114.3 against 110.2) and inside the
seeds' CVaR95 range (120.1 to 131.7). The reset-state record shows that the
trained Mamba depends on its own state; these two show that a policy trained
without that state, or without those inputs, reaches the same tail. Section
6.3's claim waits on the repeats.

### Mechanism controls, three seeds at 10^6

The rule's repeats, `15_state_controls.sh v4` rerun from `feature/v4-champion-controls`
at ad3998eb (the checkout moved to main at 2479d491, the #202 merge with the same
tree, before the scoring): `ctrl_window_p970_s2` 2026-10-06 23:40 to 10-07 03:31,
`ctrl_mamba_p962_nodv_s2` 03:32 to 08:35, `ctrl_window_p970_s3` 08:36 to 12:36,
`ctrl_mamba_p962_nodv_s3` 12:37 to 18:11, none interrupted; the four scored at
10^6 by 18:29 (eval commit 2479d491), non-captures re-flown without the wall
clock. DV in m/s over captured scenarios:

| cell | capture | non-captures | viol % any (flux / g / heat load) | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|---|
| ou_marginal/ctrl_window_p970_s2 | 99.9999% | 1: 1 crash | 0.3263 (0.0000 / 0.0000 / 0.3263) | 131.2 | 226.7 +- 1.4 | 379 |
| ou_marginal/ctrl_window_p970_s3 | 100.0000% | 0 | 0.1151 (0.0000 / 0.0000 / 0.1151) | 124.8 | 151.8 +- 0.6 | 240 |
| ou_marginal/ctrl_mamba_p962_nodv_s2 | 99.9972% | 28: 28 crash | 0.1211 (0.0000 / 0.0000 / 0.1211) | 129.5 | 181.3 +- 1.2 | 977 |
| ou_marginal/ctrl_mamba_p962_nodv_s3 | 100.0000% | 0 | 0.1448 (0.0000 / 0.0000 / 0.1448) | 130.3 | 156.5 +- 0.5 | 212 |

By family, trainer seeds 1 / 2 / 3 (the seed-1 and champion rows are in the
seed-1 table of the previous subsection):

| family | CVaR99.9 s1 / s2 / s3 | CVaR99.9 mean +- sd | CVaR95 mean +- sd | replicate median s1 / s2 / s3 | non-captures per 10^6 |
|---|---|---|---|---|---|
| champion (`hl_mamba_p962`) | 173.9 / 159.3 / 142.3 | 158.5 +- 15.8 | 124.9 +- 6.0 | 110.2 / 116.1 / 112.6 | 0 / 4 / 0 |
| window, no learned state | 158.3 / 226.7 / 151.8 | 178.9 +- 41.5 | 127.7 +- 3.3 | 113.8 / 113.7 / 112.8 | 0 / 1 / 0 |
| no predicted-DV inputs | 145.6 / 181.3 / 156.5 | 161.1 +- 18.3 | 128.5 +- 2.4 | 114.3 / 114.8 / 115.8 | 0 / 28 / 0 |

Every non-capture re-flown is a crash.

Reading (descriptive: the pre-registered rule says when repeats are owed, not
how three seeds decide): against the champion's three seeds, the window
control's mean CVaR99.9 is 20.4 m/s higher and the no-predicted-DV control's
2.6 m/s higher, Welch t 0.80 and 0.19 on three seeds a side. Neither difference
is resolved. The window's mean rests on one seed (s2, 226.7); its other two sit
inside the champion's range, and its seed spread (sd 41.5) is the widest of the
three families. CVaR95 and the median are also inside the seed spread (CVaR95
+2.8 and +3.7, t 0.71 and 0.98; the champion's s2 median 116.1 is above every
control seed's). So at the champion's allocation and budget, neither learned
state nor the three predicted-DV inputs buys a tail advantage that three seeds
resolve. The reset-state record still holds: the trained Mamba depends on its
own state. What these controls do not support is that a stateful architecture
is needed to reach that tail. Section 6.3's mechanism claim is restated on these numbers
(#181).

### Compute benchmark

The previous `compute_benchmark.json` was measured with rustc 1.98.1
(2026-09-01); the toolchain is now rustc 1.99.0 (2026-09-28), so step 6 re-ran
it: `make -C articles/paper mc-compute-benchmark` on an idle M4 Pro under
`caffeinate -i`, every scheme in one session (200 sims each, n_threads = 1,
median of 5 timed repeats after a warmup). The installed extension carries
rustc 1.99.0's std paths (`/rustc/b940084d7`). ms per simulation:

| scheme | rustc 1.98.1 | rustc 1.99.0 | change | repeat sd / median |
|---|---|---|---|---|
| NN-mamba | 3.07 | 2.84 | -7.4% | 0.5% |
| NN-dense | 1.98 | 1.86 | -6.2% | 0.8% |
| FTC | 0.91 | 0.85 | -6.7% | 0.2% |
| FNPAG | 85.54 | 82.39 | -3.7% | 0.3% |

Ratios: FNPAG / NN-mamba 27.9 -> 29.0, NN-mamba / FTC 3.39 -> 3.36, NN-mamba /
NN-dense 1.55 -> 1.53. The script times the arxiv-v3 cells (`mamba_p962_long`,
`dense_p515_ga_paper_best`, `ftc`, `fnpag`) under the shared-path regime. The
champion `ou_marginal/hl_mamba_p962` flies 771.1 s on average against
`mamba_p962_long`'s 734.1 s (final-eval pools), so its ms per simulation is not
measured here.

## v4 quantization finalists at 10^6 (#178)

`17_quantization.sh v4`, on the champion `ou_marginal/hl_mamba_p962` under per_draw:
`ptq` 2026-10-09 19:07 (verdict 4b per_tensor proj_only, the only 4-bit cell at full
capture on the n = 1000 pool); `qat_finetune` 23:37 to 00:39 (the champion's
g20000 checkpoint continued to g23000 under 4-bit fitness); `qat_scratch`
2026-10-10 00:49 to 06:23 (512 x 2 x 20000, not interrupted); `finalists` scored
the three new rows of `quant_cells_v4.txt` by 10:34 (eval commit 93574445, the
#210 merge), non-captures re-flown without the wall clock. DV in m/s over
captured scenarios:

| cell | capture | non-captures | viol % any (flux / g / heat load) | CVaR95 | CVaR99.9 +- se | max |
|---|---|---|---|---|---|---|
| ou_marginal/hl_mamba_p962 (champion) | 100.0000% | 0 | 0.1148 (0.0000 / 0.0000 / 0.1148) | 120.1 | 173.9 +- 2.2 | 300 |
| quant_v4/ptq4_verdict | 99.9702% | 298: 256 crash, 42 pending_crash | 0.0098 (0.0001 / 0.0000 / 0.0097) | 220.2 | 286.5 +- 0.7 | 371 |
| quant_v4/qat4_finetune | 100.0000% | 0 | 0.1019 (0.0000 / 0.0000 / 0.1019) | 123.8 | 153.7 +- 0.7 | 267 |
| quant_v4/qat4_scratch | 100.0000% | 0 | 0.0505 (0.0000 / 0.0000 / 0.0505) | 122.9 | 175.4 +- 2.6 | 349 |

Paired replicate deltas against the champion (finalist minus champion, mean and
t(9) 95% interval over the 10 pools; `articles/paper/data/quant_v4/confirmatory_marginal.json`):

| finalist | CVaR95 | CVaR99.9 | p99.9 | max |
|---|---|---|---|---|
| ptq4_verdict | +100.1 [99.9, 100.3] | +112.7 [109.2, 116.3] | +127.0 [124.9, 129.0] | +74.4 [56.5, 92.4] |
| qat4_finetune | +3.8 [3.6, 3.9] | -20.0 [-24.3, -15.7] | -1.4 [-4.0, 1.2] | -38.3 [-52.7, -24.0] |
| qat4_scratch | +2.8 [2.7, 3.0] | +1.7 [-5.2, 8.6] | -7.5 [-10.4, -4.6] | +37.5 [18.3, 56.7] |

Reading. Post-training 4-bit rounding is not deployable: +100 m/s CVaR95, +113
CVaR99.9 and 298 crashes per 10^6 at the verdict cell, and that cell's cost is
one rounding draw (`experiments/quant_jitter/quant_jitter.json`: CVaR95 130 to
349 at n = 1000 under a 0 to 10% larger step). Both QAT arms reach the
champion's tail. The fine-tune costs 3.8 m/s at CVaR95 and gains 20.0 at
CVaR99.9; its deployed head is the validation gate's last promotion at g20119,
so 119 generations of 4-bit fitness, and the gate kept it while the final
population went heat-load infeasible (0 of 512 candidates passed). The scratch
arm matches the champion's CVaR99.9 (+1.7, interval across zero) at +2.8 CVaR95;
its head is the g15716 promotion. Heat-load violations per 10^6: champion 1148,
fine-tune 1019, scratch 505. The intervals condition on this champion, the worst
tail of the three fp seeds (CVaR99.9 142.3 / 159.3 / 173.9), and both QAT rows
sit inside that range: a 4-bit proj_only head is tail-equivalent to the fp
family at sizing depth, by fine-tune or from scratch, not better than it.
