# Architecture probes: CfC, xLSTM, and Mamba-3 axes vs vanilla Mamba

Status: DRAFT source-of-truth for a writer session. Numbers are final (from the
committed `probe_results.json` runs); prose is notes, not finished paper text.
Intended placement: appendix (negative-result architecture ablation) supporting
the main-text headline that the deployed recurrent cell is the plain selective
SSM (Mamba).

Provenance:
- Branch `feature/cfc-xlstm`. Drivers `aerocapture.training.experiments.{mamba3,cfc,xlstm}_probe`.
- Runs: mamba3 eval 2026-07-09, cfc eval 2026-07-09, xlstm eval 2026-07-10.
- Each arm: 3 seed-repeats (`monte_carlo.seed = 20260707 + r`), GA `n_pop=300`,
  `n_gen=5000`, `training_n_sims=2`, `seed_strategy="adaptive"` +
  `curation_bucket_selection="max"`, `scaffolding="live"`.
- Scored on the shared reserved pool `PROBE_EVAL_SEED_OFFSET = 10_000_000`, 1000
  held-out sims, each model WITH its co-trained nav/shaping scaffolding
  (`best_params.json`).
- All DV figures are m/s of correction delta-v on captured trajectories.

## One-line result

At matched ~1k-param budgets, under the paper's load-bearing adaptive-max GA
regime, no recent recurrent family beats the plain selective SSM on the
propellant-sizing tail. Two treatments are significantly WORSE (CfC vs GRU;
Mamba-3 trapezoidal vs Mamba); the rest are statistically indistinguishable from
their baselines. The bottleneck is not the memory mechanism.

## What was tested and why

Three probes, each a controlled matched-budget comparison anchored on an existing
paper sweep cell, sharing one eval pool and one training regime so all nine arms
are directly comparable:

- **cfc_probe** (hypothesis: input-dependent continuous-time constants help the
  fast-near-periapsis / static-in-vacuum phase structure). Baseline arm is the
  sweep cell `gru_p1014` verbatim (Dense 17->11 -> GRU(11,11) -> Dense 11->2);
  treatment swaps the GRU for a CfC cell (ncps "default" mode) at matched budget.
- **xlstm_probe** (hypothesis: exponential gating lets the cell sharply revise a
  stored estimate at the bounce / a density shock; matrix memory vs Mamba's
  diagonal state). Baseline is `lstm_p1082` verbatim; treatments are sLSTM
  (exponential gating, same H) and mLSTM (matrix memory, H widened to hold budget).
- **mamba3_probe** (hypothesis: Mamba-3's two axes - exponential-trapezoidal
  discretization, complex/rotational state - help a smooth low-bandwidth signal).
  2x2 of `discretization in {euler, trapezoidal}` x `state_mode in {real, complex}`
  at the deployed Mamba_962 anchor; the euler+real arm is bit-identical to the
  deployed Mamba cell.

All probe layers are PSO/GA-trained through the Rust runtime; the PyTorch
warm-start path is intentionally not implemented for them (out of scope).

## Results

Tail metrics (p95, CVaR95) are the sizing statistics and lead the reporting; p50
is shown for context only. `+-` is sigma_run over the 3 seed-repeats. Capture
rate was 99.9-100% for every arm, so this is purely a tail-DV story (no capability
differences to trade off).

### mamba3_probe (2x2 at the 962 anchor)

| arm | disc / state | params | cap% | dvP50 | dvP95 +- sig | CVaR95 +- sig |
|---|---|---|---|---|---|---|
| baseline | euler / real | 962 | 99.97 | 114.0 | **121.6 +- 0.5** | **124.1 +- 0.3** |
| trapz | trapz / real | 978 | 100.00 | 115.5 | 124.9 +- 2.1 | 128.8 +- 2.3 |
| complex | euler / complex | 1154 | 100.00 | 113.7 | 121.1 +- 1.8 | 123.8 +- 2.1 |
| both | trapz / complex | 1170 | 100.00 | 114.0 | 121.6 +- 1.3 | 124.2 +- 1.2 |

### cfc_probe (cfc vs gru, matched budget)

| arm | params | cap% | dvP50 | dvP95 +- sig | CVaR95 +- sig |
|---|---|---|---|---|---|
| gru (baseline) | 1014 | 100.00 | 114.7 | **123.7 +- 1.5** | **126.7 +- 1.3** |
| cfc | 1003 | 100.00 | 116.5 | 126.5 +- 0.7 | 130.4 +- 0.1 |

### xlstm_probe (lstm vs slstm vs mlstm)

| arm | params | cap% | dvP50 | dvP95 +- sig | CVaR95 +- sig |
|---|---|---|---|---|---|
| lstm (baseline) | 1082 | 100.00 | 115.4 | **124.3 +- 1.4** | **127.3 +- 1.4** |
| slstm | 1042 | 100.00 | 115.7 | 124.7 +- 2.6 | 127.6 +- 3.1 |
| mlstm | 1078 | 100.00 | 118.1 | 127.4 +- 3.5 | 130.8 +- 4.8 |

## Significance (the rigorous, within-family claims)

A treatment clears sigma_run only if `|gap| > sqrt(sigma_base^2 + sigma_arm^2)`.
Gaps are treatment minus baseline; positive = worse (more DV).

| comparison | metric | gap | sigma_run | verdict |
|---|---|---|---|---|
| cfc vs gru | dvP95 | +2.8 | 1.66 | **significantly WORSE** |
| cfc vs gru | CVaR95 | +3.7 | 1.30 | **significantly WORSE** |
| trapz vs mamba | dvP95 | +3.3 | 2.16 | **significantly WORSE** |
| trapz vs mamba | CVaR95 | +4.7 | 2.32 | **significantly WORSE** |
| complex vs mamba | dvP95 | -0.5 | 1.87 | within sigma_run |
| complex vs mamba | CVaR95 | -0.3 | 2.12 | within sigma_run |
| both vs mamba | dvP95 | 0.0 | 1.39 | within sigma_run |
| both vs mamba | CVaR95 | +0.1 | 1.24 | within sigma_run |
| slstm vs lstm | dvP95 | +0.4 | 2.95 | within sigma_run |
| slstm vs lstm | CVaR95 | +0.3 | 3.40 | within sigma_run |
| mlstm vs lstm | dvP95 | +3.1 | 3.77 | within sigma_run (high variance) |
| mlstm vs lstm | CVaR95 | +3.5 | 5.00 | within sigma_run (high variance) |

Reading:
- **CfC is significantly worse than GRU** on both tail metrics. Continuous-time
  time-constants do not help; they hurt (and CfC's low sigma_run - 0.1 on CVaR95 -
  says this is a stable, reproducible loss, not a bad-luck draw).
- **Mamba-3 trapezoidal is significantly worse than plain Mamba.** With sigma_run
  this now clears the bar as a real degradation (the earlier single-run 962
  campaign could only say "no benefit"; the probe upgrades that to "worse").
- **Complex, both, sLSTM, mLSTM are all within sigma_run of their baselines**: no
  benefit. mLSTM is notably high-variance (sigma_run up to 4.8) and trends worse
  without clearing the bar - matrix memory adds instability, not tail robustness.

## Cross-family ranking (suggestive, not matched)

All nine arms share the regime and eval pool, so the ranking is fair to first
order, but the anchors differ slightly (962 / 1014 / 1082 params; sandwich
input-widths 16 / 11 / 10), so treat cross-family gaps as suggestive and the
within-family significance table above as the rigorous claim.

By dvP95 (lower = better): complex 121.1, both 121.6, **mamba-baseline 121.6**,
gru 123.7, lstm 124.3, slstm 124.7, trapz 124.9, cfc 126.5, mlstm 127.4.

The plain Mamba cell is at the top of the recurrent field (tied with its own
complex/both arms, which are within sigma_run of it), ~2 m/s ahead of GRU and
~2.7 ahead of LSTM on p95. Consistent with the main-text finding that the
deployed recurrent headline is the selective SSM.

## Caveats a writer must preserve

1. **Reference rows are budget-confounded; do not cross-compare against them.**
   Each probe also scored its deployed sweep cell / 962 arms as reference rows.
   Those are systematically better than the in-regime baselines:

   | in-regime baseline (300 x 5000) | reference (higher budget) | dvP95 gap |
   |---|---|---|
   | mamba baseline 121.6 | 962_baseline 116.6 (512 x 10000) | +5.0 |
   | gru 123.7 | gru_p1014_sweep 117.3 | +6.4 |
   | lstm 124.3 | lstm_p1082_sweep 120.2 | +4.1 |

   This 4-6 m/s gap is a pure training-budget effect (the 962 cells had ~3.4x the
   individual-evaluations), NOT architecture. It is exactly why the probes retrain
   in-regime baselines rather than comparing treatments against the deployed
   champions - and it validates that design choice. Any paper sentence must
   compare treatment-vs-in-regime-baseline, never treatment-vs-champion.

2. **Probe regime is deliberately sub-headline budget.** 300 x 5000 (1.5M evals)
   vs the headline 512 x 20000. The probes answer "does axis X help at a fixed,
   controlled budget," not "what is the best achievable number." The absolute
   DV here (~121-127 p95) is higher than the deployed Mamba headline for that
   reason; do not quote probe absolutes as the mission number.

3. **PSO/GA-only.** The probe cells were not warm-started or PPO-trained. The
   claim is scoped to the deployed optimizer path.

4. **slstm's -40 params vs lstm** (single bias vs LSTM's double bias at matched H)
   is an inherent cell-definition cost, not a budget mismatch - it does not
   explain the null (slstm is within noise, not worse-by-capacity).

## Interpretation (for the discussion section)

The consistent null across three independent families points at the task, not the
cells. Aerocapture guidance is a single atmospheric pass of a few hundred ticks;
the latent state worth remembering is a handful of slowly-varying dispersion
parameters (density bias, Ornstein-Uhlenbeck perturbation state, wind scale, aero
dispersions), and the engineered autoregressive inputs (predicted correction-DV
components, bank-history sin/cos, hdot/pdyn nominals) already carry most of the
temporal signal. A diagonal selective-SSM state of dimension ~12-16 saturates what
little internal memory the problem rewards. Against that backdrop:

- continuous-time time-constants (CfC) add a harder optimization landscape for a
  timescale-adaptation the fixed-cadence gates already learn,
- exponential gating and matrix memory (xLSTM) buy revision/associative-recall
  capacity the smooth signal never exercises,
- trapezoidal accuracy and complex state-tracking (Mamba-3) target long-context
  and state-tracking regimes this control signal does not occupy.

None addresses the actual bottleneck, so at matched budget they are flat-to-worse.
This is the same mechanism that makes the plain selective SSM the headline: the
win is not "more sophisticated memory," it is "just enough memory, cheaply."

Framed positively, the negative is a methodology validation: the adaptive-seed +
tail-metric + matched-anchor protocol distinguishes among architectures rather
than rubber-stamping the newest one - three 2024-2026 recurrent families were
tried and the protocol cleanly rejected them.

## Reproduce

```bash
uv run python -m aerocapture.training.experiments.mamba3_probe --eval --report --repeats 3 --n-sims 1000
uv run python -m aerocapture.training.experiments.cfc_probe   --eval --report --repeats 3 --n-sims 1000
uv run python -m aerocapture.training.experiments.xlstm_probe --eval --report --repeats 3 --n-sims 1000
```

Raw numbers live in `training_output/{mamba3,cfc,xlstm}_probe/probe_results.json`.
Design spec: `docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md`.
Related full-budget Mamba-3 result: `configs/training/mamba3_962/`,
`aerocapture.training.experiments.mamba3_962_compare`.
