# TODO

## Branch `feature/density-estimation-fixes` - Training Regression Analysis

### Problem

All guidance schemes show degraded training performance on this branch.

### Root Causes

#### 1. FTC Analytical Gain Model: Defaults Are ~10,000x Off (FTC-specific, CRITICAL)

The old `pdyn_table` was a 26-entry empirically-tuned lookup table with `coeff_a` values
ranging from -0.197 (at 45 km) to -0.001 (at 81 km). The new analytical model
`pressure_coeff = base * exp(-alt_km / scale_height)` with defaults
`base = -0.001, scale_height = 10.0` produces values of -1e-5 at 45 km and -3e-7 at
81 km -- roughly **10,000x smaller in magnitude** across the FTC operating range.

Since `gain_dynamic_pressure` divides by `pressure_coeff`, the new defaults produce
gains that are 10,000x too large, making FTC guidance wildly unstable out of the box.

The GA parameter bounds (`base` in [-0.01, -0.0001], `scale_height` in [5, 20]) cannot
recover the old table's behavior. Even at the most aggressive bound (`base = -0.01,
scale_height = 20`), the model produces -0.0008 at 50 km vs the old table's -0.08 to
-0.11. That's still **100x too small**. The analytical model with these bounds is
structurally incapable of expressing the gain profile the old table provided.

**FIXED (option c):** Scipy curve_fit on the operating range (45-81 km) gives
`base = -134.4, scale_height = 6.9` with ~12% mean relative error (vs 10,000x before).
GA bounds updated to `base` in [-500, -10], `scale_height` in [4, 15].
Golden files regenerated. All tests pass.

#### 2. Seed Pool Keep-Hardest Eviction: Difficulty Ratchet (All schemes with --adaptive-seeds)

The old gap-closure eviction removed one of two adjacent-difficulty seeds, preserving
coverage across the difficulty spectrum. The new keep-hardest eviction always drops the
easiest seed. Over time this creates a **difficulty ratchet** -- the pool fills with
only the hardest MC scenarios, making fitness increasingly adversarial.

Combined with stress tests (every 5 gens: probe 200 fresh seeds, inject the 20 hardest),
the pool drifts toward worst-case scenarios. The GA then over-optimizes for edge cases
instead of average performance. Training metrics look worse because the evaluation
population is harder, and the resulting solution may actually be worse on typical
scenarios (over-fit to adversarial seeds).

This is a textbook adversarial curriculum collapse: the curriculum keeps getting harder
but the optimizer doesn't get better at the average case -- it sacrifices average
performance to handle the injected outliers.

**Fix options:**
- **(a) Revert to gap-closure eviction**: It was a better strategy for maintaining
  spectrum coverage. Or use a mixed strategy (e.g., evict the easiest seed only if
  there's a nearby seed within some difficulty epsilon).
- **(b) Cap difficulty bias**: Keep a minimum fraction of "easy" seeds (e.g., always
  keep the 20% easiest to prevent ratcheting).
- **(c) Reduce stress test injection**: Instead of injecting 20 hard seeds every 5
  generations, inject fewer (e.g., 2-3) or inject at the pool's median difficulty
  rather than worst-case. Or stop injecting once pool is at capacity.
- **(d) Separate evaluation from training**: Score individuals on the adaptive pool
  but track a fixed hold-out set for the "real" performance metric shown in TUI/logs.

#### 3. Changed Default Seed (42 -> 1) and Population Size (20 -> 50)

`train.py` defaults changed: `--seed 42` -> `--seed 1` and `--n-pop 20` -> `--n-pop 50`.
If training was re-run with these new defaults:
- Different seed = different GA randomization and different MC scenarios. Results are
  expected to differ, but not systematically worse -- this is noise, not regression.
- Larger population (50 vs 20) should generally improve GA convergence, but with the
  same `n_gen` it means 2.5x more evaluations per generation. If `n_gen` was kept the
  same, the GA has seen the same number of generations but with more diversity per
  generation, which could be fine or could need more generations to converge.

These defaults shouldn't degrade training by themselves. They're red herrings unless
the user didn't control for them when comparing.

**Action:** Revert to the old defaults (42, 20) or document as intentional changes.

### Impact Breakdown by Scheme

| Change | FTC | EqGlide | EnergyCtrl | PredGuid | FNPAG | NN | PiecewiseConst |
|--------|-----|---------|------------|----------|-------|----|----------------|
| Analytical gain model | CRITICAL | - | - | - | - | - | - |
| Keep-hardest eviction | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | - |
| Stress test injection | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | if --adaptive-seeds | - |
| Default seed/n-pop | if re-run from scratch | if re-run from scratch | if re-run from scratch | if re-run from scratch | if re-run from scratch | if re-run from scratch | if re-run from scratch |

If "all schemes worse" and `--adaptive-seeds` was used, the seed pool changes (items 2-3)
are the most likely universal cause. The FTC analytical model is a separate, FTC-specific
regression layered on top.

### Non-Regression Changes (Good)

These changes on the branch are improvements and should be kept:
- **compare_guidance.py per-scheme TOML**: Correct fix. Each scheme should use its own
  training TOML for comparison.
- **Prefixed param routing in compare_guidance.py and train.py**: Correct fix. Params
  with `lateral.`, `exit.`, `nav.`, `thermal.` prefixes now route to the right TOML
  sections.
- **`max_reversals` rounding**: Correct fix (integer parameter stored as float in GA).
- **NN always-prefer best_model.json**: Correct fix for comparison.
- **Parameter evolution chart (gen-best + ParamSpec normalization)**: Good improvement
  for training visualization.
- **Per-seed capture rate in TUI**: Good metric, but note it's biased when pool is
  adversarial (see item 2).
- **Hash-based seed generation**: Fine change. SHA-256 spread is better than sequential
  seeds. Not a performance issue.
- **Logger gen_best_params / gen_best_chromosome**: Good for diagnostics.

### Recommended Fix Order

1. Revert seed pool eviction to gap-closure (or implement mixed strategy)
2. Reduce stress test aggressiveness (fewer injections, or disable by default)
3. ~~Fix FTC analytical model bounds~~ -- DONE (fitted to old table: base=-134.4, sh=6.9)
4. Revert default seed/n-pop to 42/20 (or document the change)
5. Re-run training for all schemes and compare against baseline

### Minor Issues

- `param_spaces.py:53-54` comment says `_NAV_PARAMS` routes to `[guidance.ftc]` but
  the actual code in `evaluate.py` correctly routes to `[navigation]`. Comment is stale.

---

## Backlog

- [ ] update comparison script
- [ ] Improve FNPAG predictor fidelity (add J2, actual atmo table)
- [ ] Add bank angle rate/acceleration limits to guidance
- [ ] Improve roll reversal logic (predictive instead of corridor-based)
- [ ] Event detection (root-finding for atmo entry/exit)
- [ ] Improve density estimation filter (gain saturation, outlier rejection)
- [ ] Better drag acceleration extraction (decompose drag/lift)
- [ ] Advanced MC sampling (LHS, Sobol, importance sampling)
- [ ] Sensitivity analysis (Sobol indices, tornado diagrams)
- [ ] Output format improvements (HDF5/Parquet, metadata, dispersions in final CSV)
- [ ] Switch to real-valued GA + alternative optimizers (CMA-ES, PSO, RL)
- [ ] Explore LSTM / Transformer architectures for guidance
- [ ] Add neural counterparts for navigation and control
- [ ] Develop ESR (Earth Sample Return) mission profiles
