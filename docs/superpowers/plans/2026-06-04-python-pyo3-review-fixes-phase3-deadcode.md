# Phase 3 — Dead-Code Removal (R1-R5) Implementation Plan

> Execute via subagent-driven; each item is verify-then-delete, gated by the full non-slow suite staying green (no FAILURES; the count drops only by the dead tests intentionally removed). Stage only each item's files; commit per item with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer; never `git add -A`.

**Goal:** Remove confirmed-dead code surfaced by the review. All targets had their references verified (zero importers/callers, or tests-only) before this plan was written.

**Baseline:** suite at 959 passed (HEAD `cba60de`). After Phase 3 the count drops by the 2 dead tests removed (R3, R4) — that is expected, not a regression.

## Decisions (two deviate from the original roadmap — evidence-driven)
- **R3 is a DELETE, not a wire-in.** The sensitivity pipeline (`sensitivity.py::run_full_analysis`) produces only `{morris, sobol}` — there is NO convergence data (`sample_sizes`/`S1_series`/`ST_series`) for `chart_sobol_convergence` to consume. Wiring it in would require adding Sobol-convergence sampling to the pipeline, which is a FEATURE (violates the remediation's "no new features" non-goal). So delete the orphaned chart + its test. (A future, separately-scoped feature could add convergence sampling and resurrect it.)
- **animate `costs_{k}`/`n_subpops` reader: KEEP (do NOT delete).** It is backward-compat for legacy-format on-disk checkpoints, paired with `_discover_checkpoints`'s legacy `checkpoint_r*_g*.json` glob; the test fixtures (`test_animate.py:31,201`) still exercise it. Not provably dead — removing it risks breaking animation of pre-existing training runs. Left intact by design; this plan does not touch animate.py.

## Tasks

### R1: delete the `plotting/` package
- **Verified:** zero importers of `aerocapture.plotting` anywhere in `src/python` or `tests`; no `test_plotting*` file. 8 files (`__init__.py` + 7 modules, ~674 lines), built on the legacy photo-file schema, superseded by `charts.py` + `corridor.py`.
- **Do:** `git rm -r src/python/aerocapture/plotting/`. Confirm nothing imports it: `rg -rn "aerocapture\.plotting|import plotting" src/python tests` → empty.
- **Gate:** `uv run pytest -q -m "not slow"` (no new failures; count unchanged — no tests covered it).
- **Commit:** `chore: delete dead plotting/ package (legacy photo-file plots, zero importers) (R1)`

### R2: delete `problem.evaluate_population_per_seed`
- **Verified:** zero callers (`rg evaluate_population_per_seed` → only the def). The 4th copy of the temp-NN-JSON pattern.
- **Do:** remove the method from `src/python/aerocapture/training/problem.py`. Confirm no test references it (`rg -n evaluate_population_per_seed tests` → empty).
- **Gate:** `uv run pytest tests/test_problem.py -q` + full non-slow suite.
- **Commit:** `chore: remove unused AerocaptureProblem.evaluate_population_per_seed (R2)`

### R3: delete the orphaned `chart_sobol_convergence`
- **Verified:** referenced ONLY by `tests/test_charts.py` (an import + one test); never wired into `report._generate_sensitivity_charts`; no data source exists (see Decisions above).
- **Do:** remove `chart_sobol_convergence` from `src/python/aerocapture/training/charts.py` AND its import + test from `tests/test_charts.py`. Confirm: `rg -rn chart_sobol_convergence src/python tests` → empty.
- **Gate:** `uv run pytest tests/test_charts.py -q` + full suite (count drops by 1 — the removed test).
- **Commit:** `chore: delete orphaned chart_sobol_convergence (no data source; wiring-in would be a feature) (R3)`

### R4: delete `calibrate_inputs.derive_asinh_scale`
- **Verified:** referenced ONLY by `tests/test_calibrate_inputs.py` (import + `test_..._puts_p99_at_one`); superseded by `derive_asinh_endpoints` (the live path via `choose_transform`).
- **Do:** remove `derive_asinh_scale` from `src/python/aerocapture/training/calibrate_inputs.py` AND its import + test from `tests/test_calibrate_inputs.py`. Confirm `rg -rn derive_asinh_scale src/python tests` → empty.
- **Gate:** `uv run pytest tests/test_calibrate_inputs.py -q` + full suite (count drops by 1).
- **Commit:** `chore: delete superseded calibrate_inputs.derive_asinh_scale (R4)`

### R5: delete `metrics.CAPTURE_COST_THRESHOLD` + fix the stale `capture_rate` docstring
- **Verified:** `CAPTURE_COST_THRESHOLD` has zero references outside its definition. Separately, `capture_rate`'s docstring (metrics.py:64) references the deprecated `log_cap`.
- **Do:** remove the `CAPTURE_COST_THRESHOLD` constant from `src/python/aerocapture/training/metrics.py`; rewrite the `capture_rate` docstring line so it no longer cites `log_cap` (cite the live `dv_cost` / the `CRASH_FLOOR=3000` semantics instead, matching how the default `capture_threshold=3000.0` is actually used). Confirm `rg -rn "CAPTURE_COST_THRESHOLD|log_cap" src/python/aerocapture/training/metrics.py` → empty.
- **Gate:** `uv run pytest tests/ -q -k metrics -m "not slow"` + full suite.
- **Commit:** `chore: remove unused CAPTURE_COST_THRESHOLD + fix stale log_cap docstring (R5)`

## After R1-R5
Full non-slow suite green (expected ~957: 959 minus the 2 removed dead tests, no failures); `ruff`/`mypy` clean on every touched file; zero Rust touched (goldens trivially intact). Update the resume note: Phase 3 complete, next is Phase 4 (PyO3 run_grid + core Arc).
