# Sizing-tail scorer: one tail record behind `evaluate_cell` (#220)

Status: Approved (user, 2026-10-30: estimator stamped on records, `viol_pct` = union).

## Problem

`cell_eval.evaluate_cell` is the one deploy-side flight, and the scoring stopped at its
return value: nine callers turned a `CellResult` into statistics themselves. CVaR95 had
three estimators (the mean of the worst `round(0.05 n)` samples in `paper_stats.cvar` and
three inline copies; the mean of the samples at or above the 95th percentile in
`probe_common.cvar95`, `quantize`, the two OU-marginal scorers and the demo; the worst
`n // 20` in `fresh_pool_requote`), and no record said which. `viol_pct` was the union of
the three constraints in the probes and `far_tail_eval` and the max over constraints in
`quantize`. Flux and g limits defaulted to None (never violated) in `report._read_constraint_limits`
and to 200 / 15 in `cost.build_cost_kwargs`. The final-record column table lived in `charts`
(matplotlib), so `evaluate.constraint_violation_rates` imported a rendering module.

## The module

`aerocapture/training/sizing_tail.py`: pure numpy + scipy (`wilcoxon`), imports without
`aerocapture_rs` and without matplotlib. The vocabulary is `CONTEXT.md`'s: Sizing Tail, Cell,
Seed Pool, DV.

| function | input | output |
|---|---|---|
| `cvar_tail_size(n, level)` | sample count, level | `max(1, round((1 - level) n))`, the one place the tail width is computed |
| `cvar(x, level)` | sorted or unsorted DV | mean of the worst `cvar_tail_size` samples (the former `paper_stats.cvar`, arithmetic unchanged) |
| `bootstrap_ci(x, stat, n_boot, seed, alpha)` | | percentile CI over resamples (moved from `paper_stats`) |
| `capture_mask(ifinal, ecc)` | parquet columns | the ADR capture definition (moved from `paper_stats`) |
| `dv_tail(dvc, *, n_boot=None, seed=0)` | the captured DV | `dv_mean dv_p50 dv_p95 dv_p99 dv_p999 dv_p9987 dv_cvar95 dv_cvar99 dv_cvar999 dv_max_descriptive`, plus `dv_mean_ci dv_p95_ci dv_cvar95_ci dv_p99_ci dv_cvar99_ci` when `n_boot` is given |
| `run_stats(ifinal, ecc, dv, *, n_boot=None, seed=0)` | parquet columns | `n n_captured capture_pct` (+ `capture_pct_ci`) and `dv_tail` of the captured rows |
| `constraint_exceedances(final_records, cost_kwargs)` | one flight's records | per-constraint boolean masks, limits from `cost_kwargs` (the one violation rule) |
| `tail_record(final_records, cost_kwargs, *, n_boot=None, seed=0)` | one flight's records | `n n_captured capture_pct` (+ `capture_pct_ci`), `dv_tail`, `viol_pct viol_n`, `<c>_viol_pct <c>_viol_n` per constraint, `rms_cost` (the `[cost_function]` objective), `estimator` |
| `score_cell(result, cost_kwargs, *, pool, n_boot=None, seed=0)` | a `CellResult` | `tail_record` + `noise_seeding` (the regime actually flown: the override, else the TOML's `[monte_carlo]`) + `pool` (the caller's label) |
| `paired_comparison(dv_a, cap_a, dv_b, cap_b, n_boot, seed)` | two flights on one pool | `n_pairs delta_mean delta_mean_ci win_rate_a wilcoxon_p` (moved from `paper_stats`) |
| `paired_tail(dv_a, cap_a, dv_b, cap_b, *, n_boot, seed)` | two flights on one pool | `paired_comparison` + `delta_p95 delta_p95_ci delta_cvar95 delta_cvar95_ci` (the scenario-resampled bootstrap of `aggregate_results`) + `delta_capture_pts delta_capture_pts_ci` |

Every DV statistic is rounded to two decimals (the rule every writer already applied) and is
computed on the unclipped correction DV of the captured runs (`CellResult.dv`), as
`results.json` always was; the report's `[1, 5000]` display clip stays in `report` / `charts`.

The violation rule is `sizing_tail.constraint_exceedances` (per-constraint boolean masks over the
final records, limits from `cost_kwargs`, columns `cell_eval.FR_*`), and
`evaluate.constraint_violation_rates` (the training gate's) is its means. It lives in the scorer,
not in `evaluate`, whose import pulls the optimizer stack (pymoo, matplotlib). `viol_pct` is the union: the share of draws that
exceed at least one configured limit. The limits come from `cost.build_cost_kwargs` everywhere
(`report._read_constraint_limits` delegates to it): a config that omits a limit is scored at the
cost function's default, as the gate always scored it.

## Decisions

1. **One estimator, A.** `cvar` keeps `paper_stats.cvar`'s arithmetic. Checked before the change:
   on every committed `final_eval.parquet` and every paired entry of `results.json` the inline
   `round(0.05 n)` copies agree with it to the cent, so `results.json` and the 19 figures do not
   move. The two differ at half-integer `0.05 n` (n = 970: 49 against 48), which the estimator
   test pins.
2. **Stored records produced by the other estimators are not rewritten.** `quote_marginal.json`,
   `heat_load_slope.json`, `quant/`, `quant_v4/quantization_results.json`,
   `quant_v4/finalists_results.json`, `probes/` (percentile-threshold mean) and
   `fresh_pool_requote.json` (`n // 20`) stay as the paper quoted them; `CHANGES.md` names them.
   Every record the scorers write from now on carries `estimator`, `pool` and `noise_seeding`;
   the OU-marginal scorers put the estimator in their protocol block, so a `--only` merge into a
   file quoted under another estimator is refused like any other protocol change, and the
   extracts copy it through when present.
3. **`viol_pct` is the union.** `quantize._max_viol_pct` is removed. The stored quant_v4
   confirmatory rows are heat-load-only (flux 1 draw in 10^6 on the PTQ cell), so union and max
   read the same at two decimals there.
4. **File dialects stay where a committed reader depends on them.** `results.json` keeps its
   frozen run and paired shapes (`aggregate_results` filters the record through the key tuples
   `check_results_schema` pins). `confirmatory_eval` / `confirmatory_marginal` keep the
   unprefixed `cvar95 / cvar999 / p999 / max` keys that `results.typ`, `figlib` and the two
   extracts read; `probe_common` keeps `METRIC_KEYS` and `quantize` keeps `capture_rate`; each
   maps the record once at its write site. The computation is one; the spelling of four files
   is theirs.

## What moves

- `paper_stats.py` keeps `actual_sims` (run-log accounting); `cvar`, `bootstrap_ci`,
  `capture_mask`, `paired_comparison`, `run_stats` move to `sizing_tail` (`run_stats` gains
  `n_captured` and its CIs become opt-in through `n_boot`); `aggregate_results` filters `dv_tail`
  through its frozen key tuples.
- The final-record column table moves from `charts._FR_*` to `cell_eval.FR_*` (already the home
  of `FR_ECC / FR_IFINAL / FR_DV_TOTAL`, pinned to the Rust `final_record_indices()`); `charts`
  aliases them. No non-rendering module imports `charts` for a column.
- Consumers: `probe_common.score_model`, `quantize._score_variant`, `mamba3_962_compare`,
  `demo`, `collect_appendix`, `aggregate_results`, `far_tail_eval`, `confirmatory_eval`,
  `centered_depth_eval`, `stress_depth_eval`, `robustness_stress`, `objective_centering_eval`,
  `fresh_pool_requote`, `fig_pareto`, `quote_marginal`, `heat_load_slope`, `measure_obs_lag`,
  `world_model`.

## Out of scope

`run_grid`, the training-side cost, the Rust simulator, the goldens; re-flying any cell; the
checkpoint reader and the run-log loader (separate review candidates).
