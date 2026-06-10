# End-of-Training Final Selection — Design

**Date**: 2026-06-10
**Status**: Approved (brainstorming complete, awaiting implementation plan)
**Scope**: Generalize the deployed-individual selection at end of training: re-rank the last generation's population (plus the running champion) on the reserved **validation** pool, for both the single-algorithm and islands paths, with a standalone CLI to apply the same rule retroactively to existing training outputs.

## 1. Motivation

Today's deployed `best_model.json` comes from a candidate stream of **one individual per generation** (the training argmin), validation-gated into a running champion. Two gaps:

1. A better-generalizing individual sitting in the final population never gets a validation shot unless it happened to be a gen argmin. For PSO/QPSO the final `pop` is the pbest set — the per-slot best-ever positions — which is a strong candidate pool that we currently ignore at the end.
2. The islands path *selects* its winner on the final-eval pool (`final_eval()` re-ranks 3 champions on the 10k-seed pool whose RMS the paper quotes), a small but systematic `E[min of 3]` selection-on-test bias that single-algorithm numbers don't carry.

The fix for both: a shared end-of-training selection step that re-ranks all final candidates on the **validation pool** (selection pressure is that pool's job), leaving the final-eval pool as a clean test set that only ever evaluates the single deployed winner. Quoted numbers become unbiased and cross-path comparable.

Changing the selection rule mid-study would make the QPSO batch incomparable to batches 2–3, so the rule must be retroactively applicable: checkpoints already store the last-generation population (`population` in single-algo npz; per-island `pop_X` in islands v2 npz), so a standalone tool can re-select existing runs without retraining.

## 2. Goals and Non-Goals

**Goals:**

1. One selection rule, one implementation, three call sites: single-algo end-of-training, islands end-of-training, standalone CLI.
2. Deployed individual can never get worse than today: the last-gen winner displaces the champion only when its validation RMS is strictly lower.
3. Final-eval pool becomes report-only in both paths (fixes the islands selection-on-test wart).
4. Retroactive application to batch-2/3 outputs (including the warm-start column, which requires adaptive-bounds spec reconstruction).
5. Resume-safe: a re-selected run never silently reverts to the pre-selection champion.

**Non-Goals:**

1. Changing the in-training validation gate (per-gen candidate stream unchanged).
2. Changing `report.py` (it keeps evaluating the deployed artifacts).
3. Two-stage screening / candidate subsampling (cost analysis below says brute force is fine).
4. RL trainers, Rust, `compare_guidance.py`.

## 3. Selection Rule

Inputs: candidate matrix `C` (normalized chromosomes), champion `(x*, best_val_cost)` (champion may be absent), validation seed list.

1. **Candidates** = rows of the last-generation population ∪ champion row. Deduplicate by exact row identity (`np.unique(axis=0)` semantics) — pbest pops carry duplicates of gbest; GA elites repeat. The champion's val RMS is `best_val_cost` (already computed on this pool); it is **not** re-simulated.
2. For each non-champion candidate: `costs = problem.evaluate_individual_per_seed(x, val_seeds)`, `val_rms = sqrt(mean(costs^2))` — the same metric `run_validation_gate` promotes on, under the current `cost_kwargs` (including `cost_transform`).
3. Candidates with non-finite val RMS are discarded.
4. **Winner** = argmin val RMS over {finite candidates} ∪ {champion}. A non-champion wins only with `val_rms < best_val_cost` (strict, consistent with the gate). Ties or no finite candidates → champion stands.
5. Result records winner provenance: `"champion"` | `"last_gen[i]"` | islands `"<island>:last_gen[i]"` / `"<island>:champion"`.

**Cost**: ≤ `(n_dedup_candidates) × validation_n_sims` sims, one batch at end of training. Paper scale: @300 single ≈ 300 × 1000 = 300k sims; islands @100/island ≈ 303k. Minutes under Rayon. No screening stage (YAGNI).

**Edge cases:**

| Case | Behavior |
|---|---|
| `validation_n_sims = 0` | Inline: selection skipped entirely (current last-gen-argmin fallback stands). CLI: hard error with a message naming the knob. |
| Champion `None` (possible only with `validation_n_sims = 0`) | Covered by the row above — selection never runs without a pool. |
| All candidates non-finite | Champion stands; outcome recorded as `"champion (all candidates non-finite)"`. |
| KeyboardInterrupt | Inline selection skipped (fast exit); the CLI covers interrupted runs from their checkpoint. |
| Islands `base_mc_seed` mismatch (npz vs TOML) | CLI hard-errors (same guard as `from_checkpoint`). |

## 4. Architecture

### 4.1 New module: `src/python/aerocapture/training/final_select.py`

- `SelectionResult` dataclass: `individual`, `val_rms`, `provenance`, `promoted: bool`, `candidate_rms: list[dict]` (per-candidate `{provenance, val_rms}`), `n_candidates`, `n_deduped`.
- `select_final_individual(problem, candidates, provenances, champion, best_val_cost, val_seeds) -> SelectionResult` — the pure rule from section 3. `problem` only needs `evaluate_individual_per_seed` (the `SeedCurator` / islands `final_eval` duck-type), so unit tests use a mock.
- `write_final_selection_json(save_dir, result, n_val_seeds)` — sidecar writer.
- CLI `main()` (section 4.5).

### 4.2 Shared-helper refactor (targeted, in-scope)

Two pieces of `train.py` logic get factored so the CLI can reuse them instead of duplicating:

1. **Param-spec reconstruction**: `_setup_param_specs(config, toml, verbose)` already exists; add a small `restore_warm_start_bounds(save_dir, param_specs) -> list[ParamSpec]` helper that overlays the weight-slab specs from `<save_dir>/warm_start_bounds.json` when present (the persistence format already exists in `warm_start.py:151-167`; the CLI reads the sidecar **unconditionally** when the file exists — no cache-key gating, because the checkpointed population was encoded under exactly those bounds). Both `train()`'s warm-start path and the CLI use it. Decoding a checkpoint population under any other specs silently corrupts the weights — this is the trap that makes the helper mandatory, and the batch-2/3 `optbig_warmstart` column exercises it.
2. **Artifact writing**: the decode-and-write block inside `save_checkpoint` (NN: `write_nn_json` + scaffolding `best_params.json`; non-NN: `best_params.json`) is extracted to `write_best_artifacts(best_individual, config, param_specs, save_dir, cwd)`, called by `save_checkpoint`, the islands winner write, and the CLI.

### 4.3 Inline integration — single-algorithm (`train.py`)

After the training loop ends normally (not on Ctrl+C), when `val_seeds is not None`:

1. Build candidates from the loop's final `X` (the last `pop.get("X")`), provenances `last_gen[i]`.
2. Run `select_final_individual` with the current `best_overall_individual` / `best_val_cost`.
3. If `promoted`: update `best_overall_individual` and `best_val_cost = winner val_rms`; set `best_overall_cost` to the winner's training cost from the final generation's `costs` row (preserving its existing "training cost at promotion" semantics — see the resume-incomparability rule).
4. **Always save a final checkpoint when `promoted`**, regardless of `last_gen % checkpoint_interval` — the current conditional save would otherwise skip persisting the new champion exactly when the last gen landed on a checkpoint multiple.
5. Print the "Final selection" block; write `final_selection.json`.

### 4.4 Inline integration — islands (`_train_islands` / `island_model.py`)

1. Candidates = union over islands of last-gen `pop_X` rows (provenance `"<island>:last_gen[i]"`) plus each island's `best_overall_individual` (`"<island>:champion"`). The champion baseline for promotion is the lowest `best_val_cost` across islands; other islands' champions enter as ordinary candidates with their stored `best_val_cost` (no re-simulation).
2. Winner from `select_final_individual` decides the artifact write (replacing `final_eval()`'s selection role).
3. `final_eval()` keeps running **for reporting**: the existing 3-champion final-eval records stay (gap diagnostics), plus the winner — if it isn't one of the 3 champions — gets a single final-eval evaluation so the paper has its unbiased test-set number. The printed table notes that artifact selection came from the validation pool.
4. `final_selection.json` written next to the winner's artifacts.

### 4.5 Standalone CLI

`python -m aerocapture.training.final_select <training_dir> --toml <config.toml> [--no-checkpoint-patch]`

1. Resolve TOML (base inheritance), build `TrainingConfig`, `AerocaptureProblem`, param specs via 4.2 (including `warm_start_bounds.json` overlay when present in `<training_dir>`).
2. Detect checkpoint format: latest single-algo `checkpoint_g*.json` + `.npz` (`population`, `costs`, `best_individual`, meta `best_val_cost`) or latest islands v2 `.npz` (pickled `island_states` with `pop_X` / `best_overall_individual` / `best_val_cost`, top-level `base_mc_seed` — verified against the TOML).
3. `val_seeds = make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, validation_n_sims)` from the resolved config.
4. Run selection; rewrite `best_model.json` / `best_params.json` via `write_best_artifacts`; write `final_selection.json`.
5. **Patch the latest checkpoint** (default on): single-algo — rewrite npz `best_individual` and JSON `best_val_cost` atomically (tempfile + rename, both files); islands — rewrite the winning island's `best_overall_individual` / `best_val_cost` inside `island_states` and re-savez atomically. Without the patch, a later resume restores the old champion and the next checkpoint save overwrites the re-selected artifacts — the patch closes that hole. `--no-checkpoint-patch` opts out for read-only experimentation (artifacts still rewritten; warning printed about the resume caveat).
6. The checkpoint `population` / `costs` and all other fields are never modified.

## 5. Outputs

`<save_dir>/final_selection.json`:

```json
{
  "winner": {"provenance": "last_gen[17]", "val_rms": 112.4, "promoted": true},
  "champion_val_rms": 118.9,
  "n_candidates": 301, "n_deduped": 244,
  "validation_n_sims": 1000,
  "candidate_rms": [{"provenance": "champion", "val_rms": 118.9}, ...]
}
```

`candidate_rms` is the full per-candidate list — the val-RMS distribution of the converged population is a free paper figure. Stdout prints a compact block (winner, provenance, delta vs champion, n candidates, sim count). Logger/TUI/`report.py` untouched.

## 6. Testing

- **Pure rule** (`tests/test_final_select.py`, mock problem with deterministic per-seed costs, no Rust): promotion on strict improvement; champion kept on tie/worse; all-inf guard; dedup counts; provenance strings; champion never re-simulated (assert mock call count).
- **CLI round-trip**: synthetic save_dir with a real single-algo checkpoint (written by `save_checkpoint` on a tiny config) → run CLI with a mock-friendly config → artifacts rewritten, checkpoint patched (npz `best_individual` changed, `population` byte-identical), `--no-checkpoint-patch` leaves checkpoint untouched.
- **Islands variant**: union candidates from a 2-island fake `island_states` npz; winning island's state patched.
- **Inline**: single-algo forced-final-save on promotion (checkpoint-interval-multiple last gen); `validation_n_sims = 0` skip.
- **E2E smoke** (verification task): 3-gen eqglide run → "Final selection" block printed, `final_selection.json` present; then the CLI re-run on the same dir is a no-op promotion (winner already deployed).

## 7. Follow-up (ops, not implementation)

After landing: run the CLI over `training_output/paper_opt*` (batches 2–3), `training_output/sweep_dense_p3998` (islands@300), and the batch-2 small-net dirs, so every paper column shares the rule before batch 4 (QPSO) executes.

## 8. Out of Scope

- The in-training per-gen validation gate (unchanged).
- `report.py` internals (it evaluates whatever artifacts are deployed).
- RL trainers, Rust simulator, `compare_guidance.py`.
