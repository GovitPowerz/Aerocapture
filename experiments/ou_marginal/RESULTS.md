# OU-marginal campaign results (2026-08-28)

Protocol: paired n=1000 pool (rng 987654321), marginal regime = per-scenario
`simulation.random_seed = 1000 + 7*i`; raw data `quote_results.json`
(regenerate with `quote_marginal.py`). All numbers below are the MARGINAL
regime, DV CVaR95 in m/s; every listed run is 100% capture and heat-load
feasible unless flagged.

## Scratch per_draw retrains (3 seed repeats each: s1/s2/s3)

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
