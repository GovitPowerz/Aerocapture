# Python + PyO3 Review Remediation — Design

- **Date:** 2026-06-04
- **Branch:** `feature/python-pyo3-review-fixes`
- **Status:** Approved (design); pending implementation plan
- **Source:** Critical review of the Python codebase (`src/python/aerocapture/**`) and the PyO3 binding crate (`src/rust/aerocapture-py/**`), conducted via 7 parallel review agents + author verification of every Critical/High finding against source.

## 1. Context & Scope

The Aerocapture project is a bit-validated Rust trajectory simulator with a Python analysis/training stack and a PyO3 binding. A thorough review surfaced: 6 confirmed defects (2 silent-wrong), a pervasive duplication family (same contracts re-encoded 3–16x), ~1,400 lines of dead/broken code, five god-modules, and a PyO3 interface that is correct but leaves the largest throughput lever unused (GIL held across compute + full `SimData` rebuild per individual x seed).

In scope: all findings, Critical through Low. The PyO3 rework is the **full** variant (includes a core-crate change to share atmosphere/wind/reference-trajectory tables). God-module decomposition is **full**, behavior-preserving and regression-gated. The dead `plotting/` package is **deleted**.

The core Rust simulator is validated to bit-level against the legacy reference (22/24 photo columns across 725 timesteps; 6 guidance golden files). **No remediation may change simulator numerics.** Bit-identity is the governing contract for every refactor phase; the single edit that touches the core crate (Phase 4a) is gated by the full golden + `cargo test` suite plus a characterization run.

## 2. Goals / Non-Goals

**Goals**
- Fix all 6 confirmed defects with test-first regression coverage.
- Establish single sources of truth for the duplicated contracts.
- Remove confirmed dead code.
- Rework the PyO3 interface for throughput (eliminate per-seed config re-read + SimData rebuild, release the GIL, pass NN weights in-memory) without changing numerics.
- Decompose the five god-modules into cohesive, independently-testable units.
- Land the Low-severity nits.

**Non-Goals**
- Any change to simulator physics, GNC algorithms, or output schema.
- Migrating SAC to `V2Policy` (a feature, not a cleanup) — so `GaussianPolicy` stays.
- New analysis features, new guidance schemes, or new layer types.
- Performance work beyond the identified PyO3 levers.

## 3. Finding Inventory (authoritative; IDs used by the plan)

### Defects (Phase 1) — test-first
- **D1 [Critical]** `train.py:961, 1267-1270, 1382-1394` — single-algo training with `validation_n_sims == 0` freezes `best_overall_individual` at the gen-0 argmin (the only in-loop update at 1394 is gated on `val_seeds is not None`); deploys an untrained model. Islands path already has the fallback (1763). Fix: promote the finite training argmin when `val_seeds is None`, mirroring islands.
- **D2 [Critical]** `animate.py:38-40` — checkpoint discovery globs the obsolete `checkpoint_r*_g*.json`; trainer writes `checkpoint_g{:05d}.json`. CLI raises `FileNotFoundError` on every real run. Fix: glob `checkpoint_g*.json` with the `checkpoint_r*` fallback; re-verify the npz subpop reader (`animate.py:62-72`) against the current npz schema.
- **D3 [High]** `report.py:326, 694, 829, 891` — `_read_constraint_limits` returns only `(heat_flux, g_load)`; trajectory classification omits `heat_load_limit`, so the PDF colors heat-load-only violators as OK while the stats block (261) counts them as violations. Fix: thread the heat-load limit (already read at 348) into all `classify_trajectories` calls.
- **D4 [High]** `rl/train.py:241`, `rl/rewards.py:96` — RL validation gate + terminal cost call `compute_cost` with defaults, while `_run_final_eval` honors TOML `cost_kwargs`; `best_model.json` is promoted under a different metric than the report measures. Fix: thread `read_cost_kwargs(toml_path)` into both.
- **D5 [Medium]** `rl/train.py:179` — `_describe_rl_architecture` hardcodes the GRU 3-gate param formula and applies it to LSTM/Transformer/Mamba (wrong startup banner for the committed LSTM config). Fix: delete the private size/param helpers; call `config.describe_architecture`.
- **D6 [Medium]** `problem.py:65-71` — `_evaluate` swallows all exceptions into a flat `1e9`, masking systemic failures as a fake optimum. Fix: log the exception type once per occurrence and abort after N consecutive all-failure batches.

### Shared single-sources-of-truth (Phase 2)
- **H1** `is_captured(final_records)` — replace the `(ifinal==3)&(ecc<1.0)` open-coding (~16 sites across report.py, charts.py, corridor.py, compare_guidance.py); reconcile with corridor.py's separate string-label classifier and its divergent column constants.
- **H2** `route_param(key, value) -> (dotpath, value)` — one home (param_spaces.py) for the `lateral./exit./nav./thermal./shaping.` routing + integer coercion; replace the 3 divergent copies (problem.py, compare_guidance.py x2, evaluate.py). Fixes the int-rounding skew.
- **H3** per-layer `from_flat()` on all six layer modules (mirroring Rust `LayerWeights::from_flat`); collapse the hand-rolled inverse in `warm_start._seed_policy_init` (95 lines) and `model_io.load_policy_from_json`.
- **H4** chart panel-builders: `_time_series_panel(...)` + `_corridor_panel(...)` + named `(N,17)` trajectory column constants (`_TC_*`); collapse ~250 lines of clones in charts.py and the re-implementations in animate.py.
- **H5** `typst_utils`: shared `compile_typst(...)`, `_read_constraint_limits`, `_check_typst` (3 copies across report.py / warm_start_report.py / warm_start_compare.py).
- **H6** seed-offset registry: one module owning all six offsets (1M–6M); fix `nn_input_report` to use `make_reserved_seeds` instead of raw `offset+i`; add a disjointness test.
- **H7** record-index map: centralize the 52-element final-record indices in the core crate (next to `finalize.rs`), reference from `results.rs` + `env.rs`, and expose `NN_FULL_INPUT_SIZE` + `DISPERSION_DRAW_LEN` from the PyO3 module; add Python drift tests (`len(NN_INPUT_NAMES) == aerocapture_rs.NN_FULL_INPUT_SIZE`, etc.). Replace inline `41`/`9`/`30`... magic in sensitivity.py, ablation.py, parquet_output.py.
- **H8** `run_validation_gate(...)` — shared by single-algo (train.py) and islands (island_model.py); folds in the D-class all-inf argmin guard (train.py:1360 currently bare `np.argmin`, island_model.py:378 has the guard). One implementation.
- **H9** bin-and-reduce primitive `binned_reduce(x, y, n_bins, reducer, min_count)` — collapse `_compute_envelope` / `binned_band` / `CorridorAccumulator._update_envelope`.
- **H10** NN-JSON emission: route `train.py:467` through `evaluate.write_nn_json` so all warm-start JSON shares one wrapper (and the normalization-embed); reconcile with `rl/export.export_v2_policy_to_json`.
- **H11** `sns.set_theme` — one `apply_theme()` (avoid the charts.py import-time global side effect); 4 verbatim copies.

### Dead code (Phase 3)
- **R1** `src/python/aerocapture/plotting/` — entire package (7 modules, ~674 lines), zero in-repo importers, legacy photo-file schema. Delete.
- **R2** `problem.evaluate_population_per_seed` — zero callers. Delete.
- **R3** `charts.chart_sobol_convergence` — referenced only by tests; **wire into** `_generate_sensitivity_charts` + report template (the sensitivity section is real).
- **R4** `calibrate_inputs.derive_asinh_scale` — superseded by `derive_asinh_endpoints`. Delete (+ its test) if only tests reference it.
- **R5** `metrics.CAPTURE_COST_THRESHOLD` — unused. Delete; fix the `capture_rate` docstring's stale `log_cap` reference.

### PyO3 rework (Phase 4) — full variant
- **P1 [Critical perf]** GIL held across `run`/`run_mc`/`run_batch`/`run_with_draws` compute (`lib.rs:68,113,159,213`); `detach` only in collect_supervised/collect_nn_inputs.
- **P2 [Critical perf]** per-seed `run_batch` loop (`problem.py:121`) → base TOML re-read + `SimData` rebuild (incl. atmosphere/wind/ref `.dat` reloads at `data/mod.rs:459,536,617,650`) per individual x seed x generation.
- **P3** numpy getters clone twice (`results.rs:123,146,166`) per access.
- **P4** `run_with_draws` `.get([i,j]).unwrap()` (`lib.rs:233`) panics on strided/transposed arrays.
- **P5** fresh Rayon pool per `run_batch` call (`batch.rs:64`).
- **P6** in-memory NN weights: avoid temp-JSON-per-individual disk round-trip (`problem.py:114-118` + Rust reload).

### God-module decomposition (Phase 5)
- **G1** `train.py::train()` (~850 lines) → `_setup_param_specs`, `_build_initial_population`, `_resume_state`, `_emit_warm_start_artifacts`.
- **G2** `rl/train.py::_run_ppo` (~300 lines) + near-duplicate `_run_sac` → `collect_rollout`, `build_critic_from_architecture`, shared update-record/validation/checkpoint tail.
- **G3** `report.py` → compute-payload / render / compile separation; dedupe eval-TOML + scaffolding-override resolution.
- **G4** `warm_start.py::build_warm_start_chromosome` (~240 lines) → `_collect_supervisor_corpus`, `_write_selection_sidecar`, `_encode_and_persist`.
- (charts.py is largely decomposed by H4.)

### Low nits (Phase 6)
- **N1** `warm_start_report.py:240` `except TypeError, ValueError:` — verified to parse on CPython 3.14.5 as a tuple (catches both, no shadowing), so functionally correct; but it reads as the Py2 name-binding trap and SyntaxErrors on <3.14. Add parens.
- **N2** `policy._ACT` missing `asinh` (drift vs Rust + DenseLayer → KeyError on an `asinh` activation).
- **N3** RL has no RNG seeding anywhere; `ppo_update_bptt` shuffles via global `np.random` (`ppo.py:166`). Add a `--seed`/`Generator`, persist in checkpoint. Real reproducibility win for paper-track A/B.
- **N4** narrow the silent excepts (`train.py:866` rng-state restore, `_read_constraint_limits`, `evaluate.py:159,198`, `nn_input_report` mask) — log at minimum.
- **N5** io/ parsers `.stat()` on a possibly-missing file → guard with `.exists()`.
- **N6** `metrics.population_diversity` O(n^2) Python loop every gen (scales with n_params even under `--no-tui`) → vectorize (pdist) or gate on TUI-live.
- **N7** promote `charts._save_line_chart` to public; drop the 6 `# type: ignore` reach-ins in `report_rl.py`.
- **N8** `parquet_output.py:17` stale `runner.rs:833` line citation → cite the function name.
- **N9** `problem.py` 4x temp-NN-JSON write + inline affine decode bypassing `decode_normalized` log-scale handling → fold into one helper reusing `decode_normalized` (largely subsumed by P6/H2).

## 4. Phasing & Sequencing

Ordered so cheap/safe high-value work lands first; riskiest (4a) and highest-churn (5) come after a green baseline.

0. **Safety net** — fresh PyO3 build, full suite green, bit-identity baseline snapshot (6 goldens + a representative training-eval cost vector).
1. **Defects** (D1–D6 + animate D2), test-first.
2. **Shared sources of truth** (H1–H11).
3. **Dead-code removal** (R1–R5).
4. **PyO3 rework** (P1–P6): 4a core Arc-table change → 4b `run_grid` binding + in-memory weights + GIL release + getter/draws/pool fixes → 4c rewire `problem.py`.
5. **God-module decomposition** (G1–G4).
6. **Low nits** (N1–N9).
7. **Docs + smart-commit** over the whole branch.

Phases 2 and 3 are independent of 4; 5 depends on 2 (consumes the new helpers). Within a phase, file-touching work is serialized to avoid worktree races.

## 5. Key Technical Designs

### 5.1 PyO3 `run_grid` + core Arc-table sharing + in-memory weights

**Verified premise:** `run_for_api_with_draws` (runner.rs:319) already runs one `&SimData` across N external draws via `par_iter`. The waste is that `problem._run_batch_pyo3` (problem.py:121) loops seeds in Python and calls `run_batch` (1 sim/call, rebuilds SimData per individual) K times — rebuilding the whole population K times per generation.

**4a — core crate.** `SimData` (mod.rs:155) holds `atmosphere: AtmosphereModel`, `wind_table: Option<WindTable>`, and the reference trajectory by value. Convert these three (the disk-loaded tables) to `Arc<...>`. Read sites are mostly transparent (`Arc<T>: Deref<T>`). Add `SimData::from_toml_with_tables(config, input, shared: SharedTables)` that injects pre-built `Arc`s and skips the loads; keep `from_toml` as the load path (so CLI + every existing caller is unchanged). `neural_net` stays per-individual (not shared). `atmosphere_onboard` is derived from `atmosphere` (a fit, not a disk load) — keep its current derivation. Gate: full golden + `cargo test` + characterization run prove bit-identity.

**4b — binding.** New `run_grid(py, toml_path, overrides_list, seeds, n_threads=None, include_trajectories=False, sim_timeout_secs=None)`:
1. Read + parse + resolve base TOML once.
2. Build `SharedTables` (atmosphere/wind/ref) once into `Arc`s.
3. For each individual: apply overrides, build `SimData` once via `from_toml_with_tables` (cheap; no disk). If the override carries in-memory NN flat weights + architecture, build the `NeuralNetModel` via `from_flat_weights_v2` and inject (no temp JSON).
4. `py.detach(|| ...)`: run the `len(overrides) x len(seeds)` grid in parallel (build the N SimData in a parallel phase, then par_iter the grid reusing `simdatas[i]` by `Arc`/ref + per-seed draw). Each cell reproduces the exact `make_reserved_seeds`-derived draw the current per-seed path would have used.
5. Build numpy results after `detach` returns; shape costs `(n_pop, n_seeds)`.
Also wrap the existing `run*` bodies' compute in `py.detach()`. Fix P3 (allocate `PyArray2` at shape, fill via `as_array_mut()`), P4 (`as_array().rows()`, drop `.unwrap()`), P5 (global Rayon pool when `n_threads is None`).

**4c — Python.** `problem._run_batch_pyo3` drops the `for seed` loop and calls `run_grid` once, aggregating costs by RMS across the seed axis exactly as today. `evaluate.write_nn_json` stays for the *deployed* model only.

**Seed→draw equivalence** is the correctness gate for 4b: a cost-vector diff against the Phase-0 baseline must be bit-identical.

### 5.2 Shared single-sources-of-truth
Each helper (H1–H11) replaces N copies and is covered by the existing suites; numeric helpers (H3 `from_flat`, H7 indices) additionally get the cross-language equivalence tests as their gate. H3 mirrors the canonical Rust per-layer flat ordering already contract-tested by `test_v2_rust_python_equivalence.py` and siblings.

### 5.3 Validation-gate extraction (H8)
One `run_validation_gate(individual, last_validated, best_val_cost, problem, seeds, cost_kwargs) -> (promoted, val_rms, summary, new_last_validated)` with the all-inf `np.any(np.isfinite(F))` guard built in. Single-algo and islands both call it. This simultaneously fixes the single-algo guard divergence and removes the duplicated promotion logic.

### 5.4 Decomposition (G1–G4)
Behavior-preserving extraction of cohesive units with clear inputs/outputs. The resume/validation interplay in `train()` is the most delicate; extracted units are pure where possible and gated by the existing resume regression tests (`test_train_interrupt.py`, etc.).

## 6. Verification & Regression Strategy
- **Bit-identity** governs Phases 2–5: the 6 guidance golden files, the cross-language NN equivalence suite, and the Phase-0 cost-vector snapshot must be unchanged.
- **Phase 4a** (core Arc change) additionally runs the full `cargo test` + `fmt --check` + `clippy` + a golden characterization run before it is trusted.
- **Phase 4b** (`run_grid`) is gated by a bit-identical cost-vector diff + a measured speedup number recorded in the final summary.
- **TDD** for Phase 1 (failing test first) and the new numeric helpers.
- **Per-commit gates:** `ruff` (lint+format) + `mypy` + the relevant `pytest` subset + `cargo test` for Rust-touching changes. `./check_all.sh` and `./lint_code.sh` before the final commit.
- Stage only the files for the unit being committed (never `git add -A`).

## 7. Risk Register
- **Highest — core `SimData` Arc ripple (4a):** mitigated by Deref-transparency, the alternate constructor keeping `from_toml` intact, and the full golden + cargo gate.
- **Medium — decomposition altering control flow (5), esp. train.py resume/validation:** mitigated by pure-unit extraction + existing resume regression tests.
- **Medium — `run_grid` seed→draw mapping drift (4b):** mitigated by reproducing the exact per-cell draw and a bit-identical cost diff.
- **Low — parallel-agent worktree races:** serialize file-touching work per phase; verify every subagent claim against source before commit.

## 8. Out of Scope / Deferred
- SAC → `V2Policy` migration (keeps `GaussianPolicy`; only delete its provably-unreferenced satellites).
- Any simulator numeric/schema change.
- Cross-language `from_flat` for layer types beyond the six already present.

## 9. Branch & Commit Strategy
- Single feature branch `feature/python-pyo3-review-fixes` in an isolated worktree.
- One commit per coherent unit (reviewable history), conventional-commit messages.
- Never commit to `main`; never push.
- Final step: `smart-commit` skill, taking the whole branch into account, to sync CLAUDE.md/README.
