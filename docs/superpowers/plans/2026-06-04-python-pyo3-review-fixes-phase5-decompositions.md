# Phase 5 — warm_start bug fix + god-module decompositions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the `warm_start.py` `mse_loss` broadcasting bug (test-first), then decompose four god-modules (`train.py::train`, `rl/train.py::_run_ppo/_run_sac`, `report.py::generate_report`, `warm_start.py::build_warm_start_chromosome`) into cohesive units — **behavior-preserving and regression-gated**. No simulator-numerics change; the 6 guidance goldens + existing test suites are the contract.

**Architecture:** Each decomposition extracts cohesive contiguous blocks into named pure-where-possible helpers, verified by the existing (often `@slow`) regression suites. The riskiest extraction (`train.py`'s resume/validation invariant) is constrained by a hard "do-NOT-extract" boundary. Tasks are ordered by ascending risk so the safe high-value work lands first; if context fills, checkpoint at a clean commit boundary and hand off the rest via RESUME.

**Tech Stack:** Python 3.14 (torch, pymoo, numpy), pytest, ruff, mypy. Pure-Python phase (no Rust rebuild) except where noted.

**Spec:** `docs/superpowers/specs/2026-06-04-python-pyo3-review-fixes-design.md` §3 (G1–G4) + §5.4. Maps in this plan come from a per-module Explore pass (2026-06-04).

**Execution note:** Per-task gate = implement → spec-review → quality-review → fix, staging ONLY the task's files (never `git add -A`), trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`, NEVER push. Run `uv run pytest -q -m "not slow"` (must stay ≥961) + the task's slow subset + `ruff`/`mypy` per task. Worktree: `/Users/govit/Git/Govit/Aerocapture/.claude/worktrees/wizardly-kalam-7cdd6c` (anchor every command with `cd <worktree> && ...`).

---

## Task 1: Fix the `atan2_signed` mse_loss broadcasting bug (TDD) — LOWEST risk, real bug

**Root cause (verified):** `_chunked_bptt_train` (`warm_start.py:412-415`) computes `loss = mse_loss(means, target)` for `output_param == "atan2_signed"` where `target = stack([sin(y), cos(y)])` is `(T,B,2)` but `means` is `(T,B,out_dim)`. `atan2_signed` REQUIRES `out_dim == 2` (sin, cos), but unlike the single-output decoders (acos_tanh/scaled_pi/delta, which validate output_size==1 + tanh and take `means[...,0]`), `atan2_signed` has NO output-size guard. `tests/test_warm_start_failures.py::_basic_cfg` (lines 24-27) builds an `output_size=1` last layer with `output_parameterization="atan2_signed"` → `means=(T,B,1)`, `target=(T,B,2)` → torch broadcasting warning (`[8,48,2]` vs `[8,48,1]`) + a silently-WRONG loss. 100 such warnings at suite runtime.

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py:412-415`
- Modify: `tests/test_warm_start_failures.py` (`_basic_cfg` arch + `test_bptt_length_greater_than_n_seq_raises` arch)
- Test: `tests/test_warm_start_chunked_bptt.py` (add the guard regression test)

- [ ] **Step 1: Write the failing test** (in `tests/test_warm_start_chunked_bptt.py`; mirror the existing call convention there for `_chunked_bptt_train` — it takes `trajectories: list[dict]`, `network: NetworkConfig`, `bptt_length: int`, `n_epochs: int`). Build a trajectory corpus + a `NetworkConfig` with a 1-output last layer and `output_parameterization="atan2_signed"`, and assert it raises:

```python
def test_atan2_signed_requires_two_outputs() -> None:
    """atan2_signed needs a 2-output (sin,cos) head. A 1-output last layer must
    raise a clear error, not silently broadcast (warm_start.py:415 bug)."""
    import numpy as np
    from aerocapture.training.config import NetworkConfig
    from aerocapture.training.warm_start import _chunked_bptt_train

    trajs = [
        {
            "seed": 0,
            "X": np.zeros((16, 4), dtype=np.float64),
            "y_signed": np.linspace(-1.0, 1.0, 16),
            "prev_realized": np.zeros(16),
            "dv": 100.0,
            "captured": True,
            "scheme": "ftc",
        }
    ]
    net = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
            {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},  # WRONG for atan2
        ],
        input_mask=[0, 1, 2, 3],
        output_parameterization="atan2_signed",
    )
    with pytest.raises(ValueError, match="atan2_signed.*output_size"):
        _chunked_bptt_train(trajs, net, bptt_length=8, n_epochs=1)
```
(Confirm the exact `_chunked_bptt_train` keyword/positional convention + the trajectory dict keys it consumes — `X`, `y_signed`, `prev_realized` — against the other tests in that file; adapt the dict if needed. If `_chunked_bptt_train` requires a `"scheme"` or other field, add it.)

- [ ] **Step 2: Run it to verify it FAILS** (currently warns + completes, does NOT raise):

Run: `cd <worktree> && uv run pytest -q "tests/test_warm_start_chunked_bptt.py::test_atan2_signed_requires_two_outputs" 2>&1 | tail -8`
Expected: FAIL (no ValueError raised; a UserWarning about broadcasting is emitted instead).

- [ ] **Step 3: Add the guard** in `warm_start.py`. Replace the `atan2_signed` branch (lines 412-415):

```python
            elif output_param == "atan2_signed":
                # atan2_signed needs a 2-output (sin, cos) head; means must be
                # (T, B, 2). A 1-output last layer would silently broadcast
                # against the (T, B, 2) target and corrupt the loss.
                if means.shape[-1] != 2:
                    raise ValueError(
                        f"atan2_signed requires the network's last layer to emit "
                        f"output_size=2 (sin, cos); got output_size={means.shape[-1]}. "
                        f"Use output_size=2 for atan2_signed, or a single-output decoder "
                        f"(acos_tanh / scaled_pi / delta) for output_size=1."
                    )
                target = torch.stack([torch.sin(y_t), torch.cos(y_t)], dim=-1)
                loss = nn.functional.mse_loss(means, target)
```

- [ ] **Step 4: Fix the misconfigured test fixtures** in `tests/test_warm_start_failures.py` so the failure-path tests use a VALID atan2 config (2-output head) and reach their intended assertions without the broadcasting warning. In `_basic_cfg` (lines 24-27), change the last layer `output_size` from 1 to 2:

```python
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"},
    ]
```
And in `test_bptt_length_greater_than_n_seq_raises` (the inline arch ~lines 162-166), change its last Dense `output_size` from 1 to 2:
```python
    cfg.network.architecture = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "tanh"},
    ]
```
(Rationale: these tests assert FileNotFoundError / "too small" / "clip rate" / "bptt_length.*n_seq" — the head shape is incidental to those failures, so a valid 2-output atan2 head preserves every assertion while eliminating the warning. `test_clip_rate` / `test_adaptive_bounds` reach BPTT, so without this they'd hit the new guard before their intended assertion.)

- [ ] **Step 5: Verify the new test PASSES + the failure tests still pass + the warning is gone:**

```bash
cd /Users/govit/Git/Govit/Aerocapture/.claude/worktrees/wizardly-kalam-7cdd6c
uv run pytest -q "tests/test_warm_start_chunked_bptt.py::test_atan2_signed_requires_two_outputs" 2>&1 | tail -5   # PASS
uv run pytest -q tests/test_warm_start_failures.py -W "error::UserWarning" 2>&1 | tail -8   # PASS, 0 warnings (the -W error turns any stray broadcast warning into a failure)
uv run pytest -q -k warm_start 2>&1 | tail -5   # no broadcasting warnings in the summary
uv run pytest -q -m "not slow" 2>&1 | tail -3   # >= 962 (961 + new test), 0 failed, warning count drops (was 16)
uv run ruff format --check src/python/aerocapture/training/warm_start.py tests/test_warm_start_failures.py tests/test_warm_start_chunked_bptt.py && uv run ruff check <same 3> && uv run mypy --config-file pyproject.toml <same 3 paths>
```

- [ ] **Step 6: Commit** (stage only the 3 files):
```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_failures.py tests/test_warm_start_chunked_bptt.py
git commit -m "$(cat <<'EOF'
fix(warm_start): guard atan2_signed BPTT loss against a non-2-output head

_chunked_bptt_train's atan2_signed branch built a (T,B,2) (sin,cos) target but
fed it to mse_loss against means=(T,B,out_dim); a 1-output head silently
broadcast (the [8,48,2] vs [8,48,1] warning, 100x at suite runtime) and computed
a wrong loss. atan2_signed now requires output_size=2 with a clear error. The
test_warm_start_failures fixtures (which used output_size=1 + atan2_signed to
exercise unrelated failure paths) are corrected to a valid 2-output head.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Follow-up (Phase 6 candidate, NOT this task):** the same misconfig (atan2_signed + output_size≠2) would also `panic`/index-out-of-bounds in the Rust runtime decoder (`atan2(out[0], out[1])`). Consider adding `output_size==2` validation for `atan2_signed` to `NetworkConfig.__post_init__` (Python) and `validate_output_parameterization` (Rust, golden-gated) as defense-in-depth. Out of scope here.

---

## Task 2: G4 — decompose `warm_start.build_warm_start_chromosome` — LOW-MEDIUM risk, well-tested

**Map:** `build_warm_start_chromosome` is `warm_start.py:567-807` (~241 lines), returns `(chromo, weight_specs)`. Extract three cohesive blocks (the cache-check, the `_chunked_bptt_train` call + loss sidecar, and the orchestration stay in the parent):

**Files:** Modify `src/python/aerocapture/training/warm_start.py`. Tests (gates): `tests/test_warm_start_pipeline.py` (@slow), `tests/test_warm_start_end_to_end.py` (fast, mocked `collect_supervised`), `tests/test_warm_start_cache.py` (@slow), `tests/test_warm_start_selection.py`, `tests/test_warm_start_failures.py`.

- [ ] **Extraction contracts** (behavior-preserving; the parent calls them in the same order):
  - **`_collect_supervisor_corpus(cfg, base_mc_seed, resolved_paths) -> tuple[dict[str, list[dict]], list[dict], int]`** — extract lines 603-660 (resolve supervisor paths + per-scheme `_aero_rs.collect_supervised` loop + `_select_best_teacher_per_seed` + the min-corpus threshold). PURE (no I/O, no stdout). Returns `(results_by_scheme, selected, min_corpus_required)`. Does NOT write the selection JSON and does NOT do magnitude_only sign collapse (those stay in the parent / move to the next helper).
  - **`_write_selection_sidecar(save_dir, results_by_scheme, selected, n_warm_seeds) -> None`** — extract lines 662-692 (compute per-scheme selection counts + capture stats, write `warm_start_selection.json`). Side effect: one JSON write. No stdout, no Rust.
  - **`_encode_and_persist(policy, network, weight_specs, cfg, save_dir, cache_key) -> tuple[npt.NDArray[np.float64], list]`** — extract lines 738-807 (flat-weight extraction via `_policy_to_flat_weights_v2` + architecture-shape validation + optional adaptive-bounds derivation + normalize to [0,1] + optional scaffolding-tail concat + write `warm_start_chromosome.npy` / `warm_start_cache_key.json` / `warm_start_bounds.json` + the >5% clip-rate hard-error). Returns the final `(chromo, weight_specs)`.
  - **STAYS in `build_warm_start_chromosome`:** the cache-key compute + cache-hit early return (619-634), the magnitude_only sign collapse (694-703), the `_chunked_bptt_train` call + `warm_start_loss.json` write + MSE-summary print (705-736). The parent threads `selected` → BPTT → `policy` → `_encode_and_persist`.

- [ ] **Steps:** (a) Extract `_write_selection_sidecar` first (most isolated) → run `tests/test_warm_start_end_to_end.py` + `tests/test_warm_start_selection.py`. (b) Extract `_collect_supervisor_corpus` → same tests. (c) Extract `_encode_and_persist` → run `tests/test_warm_start_failures.py` (clip-rate + adaptive-bounds paths live here) + `tests/test_warm_start_end_to_end.py`. (d) Full gate: `uv run pytest -q tests/test_warm_start_end_to_end.py tests/test_warm_start_selection.py tests/test_warm_start_failures.py` + `uv run pytest -q -m slow -k warm_start` (the @slow pipeline/cache/per-arch tests — these are the real bit-identity gate for the chromosome shape + cache key) + `ruff`/`mypy` + `uv run pytest -q -m "not slow"` (≥962). (e) Commit (stage only `warm_start.py`).

- [ ] **Guardrails:** the return contract `(chromo, weight_specs)` must be byte-identical (the @slow pipeline test asserts the normalized vector shape + values; the cache test asserts the cache key + roundtrip). The file-write set + order must be unchanged (selection.json before BPTT, chromosome/cache-key/bounds at encode). Do NOT move the `_chunked_bptt_train` call or its loss-sidecar write into a helper (keeps the BPTT path — incl. the Task-1 guard — in the parent where the @slow per-arch tests exercise it).

---

## Task 3: G3 — `report.py` COMPUTE/RENDER/COMPILE seam + dedupe scaffolding-override — LOW-MEDIUM risk

**Map:** god-function `generate_report()` is `report.py:738-871`. The pure-duplication win: `_load_nn_scaffolding_overrides()` (`report.py:44-75`) is duplicated verbatim in `compare_guidance.py::run_scheme` (lines ~150-169) — same `lateral./exit./nav./thermal./shaping.` routing + the `lateral.max_reversals -> int(round())` special case. Note: H2/H2b (Phase 2) already centralized `route_param_path` in `param_spaces.py`; check whether `_load_nn_scaffolding_overrides` can be expressed in terms of it before adding a new helper.

**Files:** Modify `src/python/aerocapture/training/report.py`; possibly add a shared helper (prefer extending `param_spaces.py`/an existing module over a new `shared_config.py`); modify `compare_guidance.py` to consume it. Tests (gates): `tests/test_report_pdf.py`, `tests/test_training_report.py`, `tests/rl/test_report_rl.py`, `tests/test_route_param_deploy.py` (the H2b deploy-routing lock).

- [ ] **3a (the clear win — do this first, possibly as its own commit):** extract the NN scaffolding-override builder into ONE shared helper and route both `report.py:44-75` and `compare_guidance.py:150-169` through it. Verify against `tests/test_route_param_deploy.py` (which already locks the deploy routing) + `tests/test_report_pdf.py` + `compare_guidance` tests. Reconcile with `route_param_path` (H2) rather than re-encoding the prefix map.
- [ ] **3b (the seam — separate commit):** split `generate_report` into COMPUTE (`load_run_data` + `run_final_evaluation` + `_run_undispersed_nominal` + `_load_corridor_data` + `read_cost_kwargs` → a payload), RENDER (`_generate_{training,trajectory,sensitivity}_charts` + `_build_metadata` + `_build_summary_table` from the payload), COMPILE (`_check_typst` + the `typst compile` subprocess). Extract a `_compute_report_payload(...) -> ReportPayload` and a `_render_report_assets(payload, tmp_dir) -> None`, leaving `generate_report` as the thin COMPUTE→RENDER→COMPILE orchestrator. Gate: `tests/test_report_pdf.py` (chart-gen-to-tmp without Typst, mocked) + `tests/test_training_report.py` (load_run_data dedup/resume, summary/metadata builders). Side effects (parquet write, SVGs, metadata JSON, typst subprocess) must be unchanged.
- [ ] Per-sub-task: `ruff`/`mypy` + `uv run pytest -q -m "not slow"` (≥962). Commit each sub-task separately (3a then 3b), staging only its files.

---

## Task 4: G2 — `rl/train.py` `_run_ppo`/`_run_sac` dedup — MEDIUM-HIGH risk (WEAK gates: SAC untested, PPO @slow)

**Map:** `_run_ppo` (`rl/train.py:452-758`), `_run_sac` (794-950). **WARNING (from the map):** SAC has NO integration test (untested in git, Phase 1.6 defers SAC-GRU); all `_run_ppo` gates are `@slow` (test_gru_ppo_smoke, test_lstm_ppo_smoke, test_ppo_feedforward_regression). Treat SAC changes as unguarded — keep them minimal and mechanical.

**Files:** Modify `src/python/aerocapture/training/rl/train.py` (+ possibly `rl/policy.py`/`rl/schemas.py` for the critic-builder home). Tests (gates): `tests/test_gru_ppo_smoke.py` (@slow), `tests/test_lstm_ppo_smoke.py` (@slow), `tests/test_ppo_feedforward_regression.py` (@slow), `tests/test_ppo_bptt_chunk_invariant.py` (fast).

- [ ] **Extraction contracts:**
  - **`build_critic_from_architecture(architecture, input_dim) -> ValueNetwork`** — extract `rl/train.py:514-531` (mirror trunk widths: DenseSpec→output_size, GruSpec→hidden_size as tanh; append final activation; build ValueNetwork). PURE, zero closure, PPO-only consumer. Lowest-risk of the three — do first.
  - **`collect_rollout(...)`** — extract the PPO rollout loop (`598-675`); thread hidden state (V2Policy) + done-masking + buf snapshotting. The SAC step loop (`858-902`) is structurally similar but pushes to a replay buffer and is stateless — **do NOT force a unified rollout** unless it's clean; the map flags this as the main technical lift. If unification is not clean, extract PPO's `collect_rollout` only and leave SAC's loop (document why).
  - **shared validation/checkpoint/record tail** — the validation gate (PPO 717-726 / SAC 911-920 are identical logic), checkpoint save, record+log+display, end-of-training export differ only by callbacks (validate_fn, checkpoint_fn, export_fn, record_builder_fn). Extract a `_run_training_tail(...)` parameterized on those callbacks ONLY if the PPO @slow gates fully cover it; the SAC record-dict key differences (epochs_run/value_loss vs alpha/q_loss) must be preserved.
- [ ] **Steps:** (a) `build_critic_from_architecture` → `uv run pytest -q -m slow tests/test_ppo_feedforward_regression.py tests/test_gru_ppo_smoke.py`. (b) `collect_rollout` (PPO) → same @slow gates + `tests/test_ppo_bptt_chunk_invariant.py`. (c) the shared tail (only if confidently gated) → all @slow PPO tests. (d) `ruff`/`mypy` + `uv run pytest -q -m "not slow"`. Commit each extraction separately, staging only `rl/train.py` (+ helper module).
- [ ] **Guardrails:** the @slow PPO smoke/regression tests are the ONLY behavioral gate — run them after EVERY extraction (do not batch). If SAC can't be safely deduped (no test), leave `_run_sac` mostly intact and only share the trivially-safe pieces (critic builder is PPO-only anyway). Bit-identity here = the exported `best_model.json` loads + `nn_forward` returns a finite 2-tuple (what the smoke tests assert).

---

## Task 5: G1 — `train.py::train()` decomposition — HIGHEST risk (resume/validation invariant)

**Map:** `train()` is `train.py:692-1538` (~850 lines), LINEAR control flow, 19 phases. `_train_islands` (1541-1934) is fully isolated (does NOT share these blocks). Extract the SAFE blocks; KEEP the delicate resume/validation invariant in `train()`.

**Files:** Modify `src/python/aerocapture/training/train.py`. Tests (gates): `tests/test_train_interrupt.py::TestResumePreservesCheckpointedBest` (THE invariant gate), `tests/test_train_interrupt.py::TestKeyboardInterrupt`, `tests/test_warm_start_optimizer_seeding.py` (param_specs in-place mutation), `tests/test_resume_enhancements.py` (population resize), `tests/test_nn_param_specs_v2.py`, `tests/test_nn_scaffolding_params.py`, `tests/test_warm_start_baseline_writer.py`, `tests/test_warm_start_end_to_end.py`.

- [ ] **Extraction contracts (ascending risk):**
  - **`_setup_param_specs(config, _toml, verbose) -> tuple[list[ParamSpec], int]`** — extract lines 784-846 (NN v2 `nn_param_specs_from_v2` / v1 / piecewise / `PARAM_SPACES` + `active_scaffolding_specs` tail). Side effect: guidance-type prints. LOW risk, do FIRST.
  - **`_emit_warm_start_artifacts(config, base_mc_seed, problem, val_seeds, save_dir, verbose) -> tuple[np.ndarray, list[ParamSpec], np.ndarray | None]`** — sub-extract lines 1041-1169 (build_warm_start_chromosome orchestration + gen-0 baseline + compare/report rendering, with the THREE best-effort try/except blocks preserved). LOW risk (best-effort, file I/O + prints).
  - **`_build_initial_population(resumed, config, param_specs, seed_weights, problem, val_seeds, base_mc_seed, rng, save_dir, verbose, from_scratch) -> tuple[np.ndarray, np.ndarray | None]`** — extract lines 983-1186. **CRITICAL:** it MUTATES `param_specs` IN-PLACE (lines ~1076-1078, `param_specs[j] = warm_weight_specs[j]`) and the caller relies on that visibility (read by `warm_start_algorithm` + `AerocaptureProblem`); Python list-reference semantics preserve it — gated by `tests/test_warm_start_optimizer_seeding.py`. Contains the `_emit_warm_start_artifacts` call. MEDIUM risk.
  - **OPTIONAL pure helpers:** `_load_checkpoint_state()` (851-879) + `_resize_population_on_resume()` (984-1002).
- [ ] **🚫 DO NOT EXTRACT (hard boundary):** the `best_overall_individual` unpacking (lines ~901-912) and the `if best_overall_individual is None:` guard (lines ~1267-1270) MUST stay in `train()`. Moving them breaks the cross-gen training-cost incomparability invariant (a resumed best validated under seed-list A is not comparable to the resumed population under seed-list B). `tests/test_train_interrupt.py::TestResumePreservesCheckpointedBest` will FAIL if violated — run it after EVERY extraction.
- [ ] **Steps:** (a) `_setup_param_specs` → `tests/test_nn_param_specs_v2.py tests/test_nn_scaffolding_params.py`. (b) `_emit_warm_start_artifacts` → `tests/test_warm_start_baseline_writer.py tests/test_warm_start_end_to_end.py`. (c) `_build_initial_population` → `tests/test_warm_start_optimizer_seeding.py tests/test_resume_enhancements.py`. (d) after EACH: `tests/test_train_interrupt.py` (the invariant + interrupt). (e) `ruff`/`mypy` + `uv run pytest -q -m "not slow"` (≥962) + a `@slow` train smoke if one exists. Commit each extraction separately, staging only `train.py`.

---

## Self-review checklist
- [ ] **Spec coverage:** mse_loss bug (Task 1), G4 (Task 2), G3 (Task 3), G2 (Task 4), G1 (Task 5).
- [ ] **Bit-identity / behavior preservation:** 6 guidance goldens unchanged (no Rust touched in Phase 5); the @slow warm_start/rl/train regression suites green; `test_train_interrupt::TestResumePreservesCheckpointedBest` green after every train.py extraction.
- [ ] **Suite:** `uv run pytest -q -m "not slow"` ≥ 962 (961 + the Task-1 guard test); `ruff`+`mypy` clean on every touched file.
- [ ] **Discipline:** one commit per coherent extraction; stage only the task's files; verify every subagent diff against source; NEVER push.

## Sequencing / checkpoint guidance
Tasks are ordered by ascending risk: **Task 1 (bug) is the must-do.** Tasks 2-3 (G4, G3) are clean, well-tested wins. Tasks 4-5 (G2, G1) are the riskiest — G2 has weak SAC coverage, G1 has the resume invariant — and benefit most from fresh context. If context fills, checkpoint at a clean commit boundary, update RESUME, and resume the remaining decompositions in a new session (the established per-phase workflow). Phase 6 (nits N1-N9, incl. the pre-existing lint debt + the atan2 runtime validation noted in Task 1) and Phase 7 (`smart-commit` over the whole branch) follow.
