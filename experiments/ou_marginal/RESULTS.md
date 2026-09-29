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
heat-flux violation rate on the same 1000 scenarios, from a separate pass
(`quote_results.json` carries heat-load violations only, 0.0% on every row here).

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
  equilibrium_glide. A regime-only delta needs a ceiling-matched rerun.
- Gated on the validation pool is not feasible out of it: the ftc and
  equilibrium_glide retunes still exceed the heat-flux limit on 0.2% / 0.4% of
  these scenarios.
