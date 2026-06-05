# Phase 2 — Shared Sources of Truth (H1–H11) Implementation Plan

> **For agentic workers:** Execute task-by-task via superpowers:subagent-driven-development. Each task: TDD where it adds behavior; for pure dedup, the contract is **bit-identical behavior** gated by the existing suites (golden + cross-language + the 914-test non-slow suite). Stage only the task's files; commit with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer; never `git add -A`.

**Goal:** Replace the duplicated contracts (H1–H11 from the spec) with single sources of truth, changing no observable behavior.

**Governing gate:** every task must keep `uv run pytest -q -m "not slow"` green (currently 914) and must not alter the 6 Rust golden files. Per-task gate: `ruff check` + `ruff format` + `mypy` on touched files + the relevant test subset.

**Order = ascending risk** (lower-risk dedups first; the train.py/islands and Rust-touching items last).

---

### Task H6: seed-offset registry
- **Files:** `evaluate.py` (already owns VALIDATION/FINAL_EVAL/RL_TRAINING/WARM_START offsets) gains `CALIBRATION_SEED_OFFSET = 6_000_000` and `NN_INPUT_REPORT_SEED_OFFSET = 5_000_000`; `calibrate_inputs.py` + `nn_input_report.py` import from `evaluate` instead of defining their own. `nn_input_report.py` switches its raw `[OFFSET + i for i in range(n)]` to `make_reserved_seeds(0, NN_INPUT_REPORT_SEED_OFFSET, n)` for parity with `calibrate_inputs`.
- **Contract/acceptance:** new test `tests/test_seed_offsets.py` asserts the six offsets are distinct and equal their documented values, and that `calibrate_inputs`/`nn_input_report` reference the `evaluate` symbols (identity). `nn_input_report` behavior changes (seed stream), so update/relax any test asserting its exact seeds; the report's purpose (diagnose deployed NN) is unaffected. Run `tests/test_calibrate_inputs.py tests/test_nn_input_report.py tests/test_seed_offsets.py`.
- **Commit:** `refactor(seeds): single registry for reserved-pool offsets; nn_input_report uses make_reserved_seeds (H6)`

### Task H11: single `apply_theme`
- **Files:** `charts.py` gains `apply_theme()` wrapping the `sns.set_theme(...)` call; remove the import-time global side effect (call it explicitly from chart entry points / report). `animate.py`, `charts_ablation.py`, `warm_start_report.py` call `charts.apply_theme()` instead of their own verbatim `sns.set_theme(...)`.
- **Contract/acceptance:** SVG output unchanged. Run `tests/test_charts.py` + a smoke of `report`/`animate` chart generation. Confirm no module-import side effect remains (`rg -n "set_theme" src/python` → only inside `apply_theme`).
- **Commit:** `refactor(charts): single apply_theme(), drop import-time global + 4 copies (H11)`

### Task H1: `is_captured`
- **Files:** `charts.py` gains `is_captured(final_records) -> NDArray[bool]` = `(final_records[:, _FR_IFINAL] == 3) & (final_records[:, _FR_ECC] < 1.0)`; `classify_trajectories` and the ~6 open-coded sites in `charts.py`/`report.py` call it. Reconcile `compare_guidance.py`'s 2 sites. Leave `corridor.py`'s separate string-label classifier (different column constants) but add a comment noting the shared definition (or migrate it if its columns match — verify first).
- **Contract/acceptance:** bit-identical classification. Run `tests/test_charts.py tests/test_report.py tests/test_corridor.py` + the full non-slow suite. The captured count in any test must be unchanged.
- **Commit:** `refactor(charts): single is_captured() for the canonical captured definition (H1)`

### Task H9: `binned_reduce`
- **Files:** `charts.py` gains `binned_reduce(x, y, n_bins, reducer, min_count)`; `_compute_envelope` (charts), `binned_band` (charts_nn_inputs), and `CorridorAccumulator._update_envelope` (corridor) delegate to it (matching each one's empty-bin/NaN semantics — verify each before/after).
- **Contract/acceptance:** envelopes byte-identical. Run `tests/test_charts.py tests/test_corridor.py tests/test_nn_input_report.py`. If empty-bin semantics differ subtly across the three, keep per-caller wrappers passing the right `min_count`/reducer rather than forcing a single behavior.
- **Commit:** `refactor(charts): single binned_reduce primitive for envelope/band helpers (H9)`

### Task H5: `typst_utils`
- **Files:** new `src/python/aerocapture/training/typst_utils.py` with `check_typst()` and `compile_typst(template_str, out_pdf, asset_dir, timeout=...)`; `report.py`, `warm_start_report.py`, `warm_start_compare.py` use it. Move the shared `_read_constraint_limits` to one home (keep `report.read_cost_kwargs` as is) — or leave `_read_constraint_limits` in report and have warm_start_compare import it (resolve the "avoid runtime import dep" comment by just importing the function).
- **Contract/acceptance:** add `timeout=` to the typst subprocess (closes a Phase-6 N-item early). Run `tests/test_warm_start_report.py` + report PDF smoke. Typst-absent path still degrades gracefully.
- **Commit:** `refactor(report): shared typst_utils (compile + check + constraint read) (H5)`

### Task H10: unify NN-JSON emission
- **Files:** `train.py:467` (the in-training eval temp-JSON write that calls `flat_weights_to_json` positionally and OMITS the normalization-embed) routes through `evaluate.write_nn_json` instead, so all warm-start/eval JSON shares one wrapper + the normalization policy. Verify `write_nn_json`'s signature covers the args train.py:467 passes (input_mask, output_param, scaled_pi_n, delta_max).
- **Contract/acceptance:** warm-start eval still runs; the emitted JSON now carries `[network.normalization]` consistently (no train/eval scale mismatch). Run `tests/test_warm_start*.py` + a warm-start smoke (`-m "not slow"`).
- **Commit:** `refactor(train): route in-training NN-JSON emission through write_nn_json (H10)`

### Task H2: `route_param`
- **Files:** `param_spaces.py` gains `route_param(key, value) -> (toml_dotpath, coerced_value)` encoding the `lateral./exit./nav./thermal./shaping.` → section map + integer coercion. `problem._build_overrides`, `compare_guidance.run_scheme` (both inline blocks), and `evaluate.py`'s routing call it. Fold the divergent int-rounding (`int(round())` vs `_integer_params`) into the single helper.
- **Contract/acceptance:** **bit-identical overrides.** `tests/test_problem.py` already covers all 6 prefix routes — extend it to assert `route_param` produces the exact dot-paths + integer coercion previously produced at each site. Run `tests/test_problem.py` + a `compare_guidance` smoke. This is the one most prone to silent drift — diff a sample override dict before/after.
- **Commit:** `refactor(params): single route_param() for guidance prefix routing + int coercion (H2)`

### Task H4: chart panel-builders + `(N,17)` column constants
- **Files:** `charts.py` gains named trajectory-column constants `_TC_TIME=7, _TC_ALT=0, _TC_HEAT_FLUX=6, _TC_PDYN=9, _TC_BANK=10, _TC_INCL=11, _TC_GLOAD=12, _TC_NAV_DENS=13, _TC_HEAT_LOAD=15` (verify each against the documented 17-col layout) and two builders: `_time_series_panel(...)` + `_corridor_panel(...)`. The 8 time-domain + 3 corridor public functions become thin wrappers (keep names — the Typst template references them). `animate.py` reuses `_corridor_panel` (envelope-less variant) + the column constants.
- **Contract/acceptance:** **SVG output unchanged.** Run `tests/test_charts.py tests/test_animate.py` + report chart smoke. This is decomposition-via-dedup; verify a rendered panel diff is visually identical (same axes/labels/limits).
- **Commit:** `refactor(charts): parameterized time-series + corridor panel builders, named (N,17) columns (H4)`

### Task H3: per-layer `from_flat`
- **Files:** add `from_flat(slab)` to each of the 6 layer modules (`rl/layers/{dense,gru,lstm,window,transformer,mamba}.py`) mirroring the Rust `LayerWeights::from_flat` ordering (the inverse of each module's existing `to_flat`). `warm_start._seed_policy_init` (95-line hand-rolled inverse) and `model_io.load_policy_from_json` reduce to a cursor walk calling `module.from_flat`.
- **Contract/acceptance:** **gated by the cross-language equivalence suite** (`tests/test_v2_rust_python_equivalence.py` + `test_*_equivalence.py` + `test_warm_start_v2_to_flat_roundtrip.py`). Add a per-module `to_flat∘from_flat == identity` test. Bit-identical to the prior hand-rolled inverse. Run the full equivalence + warm_start suites.
- **Commit:** `refactor(rl/layers): add from_flat() per layer; collapse hand-rolled inverses (H3)`

### Task H7: record-index map + PyO3 const exposure
- **Files:** define the 52-element final-record index map as `const`s in the Rust core crate (next to `finalize.rs`), reference from `aerocapture-py/src/results.rs` + `env.rs` (replace bare literals); export `NN_FULL_INPUT_SIZE` + `DISPERSION_DRAW_LEN` from the PyO3 module. Python: replace inline `41`/etc. in `sensitivity.py`/`ablation.py`/`parquet_output.py` with a shared map (reuse `parse_final.CSV_TO_LEGACY_INDEX` where possible). Add drift tests: `len(NN_INPUT_NAMES) == aerocapture_rs.NN_FULL_INPUT_SIZE`, `len(DISPERSION_COLUMNS) == aerocapture_rs.DISPERSION_DRAW_LEN`.
- **Contract/acceptance:** **touches Rust → rebuild `maturin develop --release` + full `cargo test` + golden bit-identity.** Run the drift tests + `tests/test_parquet_output.py tests/test_sensitivity.py tests/test_ablation.py`.
- **Commit:** `refactor(records): centralize final-record index map; expose NN/dispersion widths to Python (H7)`

### Task H8: `run_validation_gate` (riskiest — train.py + island_model.py)
- **Files:** extract `run_validation_gate(individual, last_validated, best_val_cost, problem, seeds, cost_kwargs) -> (promoted, val_rms, summary, new_last_validated)` with the all-finite `np.any(np.isfinite(F))` guard built in. `train.py`'s single-algo gate AND `island_model.validate_each` call it. This simultaneously lands the single-algo all-inf argmin guard (train.py:1360 currently bare `np.argmin`) that islands already has.
- **Contract/acceptance:** behavior preserved for the normal case; the only behavior CHANGE is the single-algo all-inf path now guarded (add a test: single-algo gen with an all-inf population must not promote `pop[0]`). Run `tests/test_train_interrupt.py tests/test_train_no_validation_promotion.py tests/test_island_model.py` + the full non-slow suite. Diff-review the train.py/islands changes by hand (delicate resume/validation interplay).
- **Commit:** `refactor(train): shared run_validation_gate + single-algo all-inf guard (H8)`

---

## Self-Review
- **Spec coverage:** H1–H11 each have a task. H8 also lands the documented single-algo all-inf guard divergence.
- **Risk ordering:** pure-Python low-risk dedups (H6/H11/H1/H9/H5/H10) first; behavior-sensitive routing (H2), decomposition-via-dedup (H4), cross-language (H3), Rust-touching (H7), and the delicate train/islands gate (H8) last.
- **Gates:** every task names its regression suite; bit-identity is the contract; H7 adds the Rust rebuild + golden gate; H3 leans on the existing cross-language equivalence suite.
