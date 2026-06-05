# Python + PyO3 Review Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding from the 2026-06-04 review (6 defects, 11 dedup consolidations, 5 dead-code removals, the full PyO3 `run_grid` + Arc-table rework, 4 god-module decompositions, 9 low nits) without changing simulator numerics.

**Architecture:** One feature branch `feature/python-pyo3-review-fixes`, dependency-ordered phases, bit-identity as the governing contract for every refactor. Spec: `docs/superpowers/specs/2026-06-04-python-pyo3-review-fixes-design.md`.

**Tech Stack:** Python 3.14 (numpy, pymoo, torch, pydantic, pytest), Rust 2024 (PyO3, nalgebra, Rayon), `uv` for envs, `maturin` for the binding.

---

## Conventions for every task
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (repo convention).
- **Staging:** stage only the files named in the task (`git add <paths>`). Never `git add -A`.
- **Python tests:** `uv run pytest <path>::<test> -v`. Many tests `importorskip("aerocapture_rs")`, so Task 0 builds it first.
- **Per-task gate:** before each commit run `uv run ruff check <files> && uv run ruff format <files> && uv run mypy <files>` and the task's tests. For Rust-touching tasks add `cargo test` / `cargo clippy` from `src/rust`.
- **Bit-identity:** Phases 2–6 must not change the 6 golden files or the Phase-0 cost snapshot.

---

## Phase Roadmap (this doc details Phase 0 + Phase 1; later phases get their own dated plan, authored just-in-time against the then-current tree)

| Phase | Content | Plan |
|------|---------|------|
| 0 | Safety net: build, green suite, bit-identity baseline | this doc |
| 1 | 6 defects (D1–D6), test-first | this doc |
| 2 | Shared single-sources-of-truth (H1–H11) | `...-phase2-dedup.md` |
| 3 | Dead-code removal (R1–R5) | `...-phase3-deadcode.md` |
| 4 | PyO3 `run_grid` + core Arc tables (P1–P6) | `...-phase4-pyo3.md` |
| 5 | God-module decomposition (G1–G4) | `...-phase5-decompose.md` |
| 6 | Low nits (N1–N9) | `...-phase6-nits.md` |
| 7 | Docs + smart-commit | (smart-commit skill) |

Phases 2 and 3 are independent of 4; 5 depends on 2. Within a phase, file-touching work is serialized.

---

## Phase 0 — Safety Net

### Task 0.1: Build the PyO3 binding and confirm a green baseline

**Files:** none modified.

- [ ] **Step 1: Build the binding (from repo root, with `--manifest-path` — subcrate builds go stale).**

Run:
```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```
Expected: `🛠 Installed aerocapture_rs-0.1.0` (or similar success line).

- [ ] **Step 2: Confirm the Python suite is green (excluding slow).**

Run:
```bash
uv run pytest -q -m "not slow"
```
Expected: all pass (note the count, e.g. `787 passed`).

- [ ] **Step 3: Confirm the Rust suite is green.**

Run:
```bash
cd src/rust && cargo test --release 2>&1 | tail -20
```
Expected: `test result: ok` for each crate.

- [ ] **Step 4: No commit (read-only baseline).**

### Task 0.2: Capture the bit-identity baseline snapshot

**Files:**
- Create: `tests/reference_data/phase0_baseline/` (gitignored scratch — see Step 3)

- [ ] **Step 1: Snapshot the 6 guidance golden configs' output.**

Run (the 6 golden configs already live under `configs/test/`; regenerate into a scratch dir and diff against `tests/reference_data/rust_golden/`):
```bash
mkdir -p /tmp/phase0_baseline
for cfg in $(rg -l "" tests/reference_data/rust_golden --glob '*.csv' | xargs -n1 basename | sed 's/\.csv$//'); do echo "$cfg"; done
```
Expected: lists the 6 golden scheme names (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural).

- [ ] **Step 2: Snapshot a representative training-eval cost vector (the run_grid bit-identity gate for Phase 4).**

Run:
```bash
uv run python - <<'PY'
import numpy as np, aerocapture_rs
from aerocapture.training.evaluate import make_reserved_seeds, compute_cost
seeds = make_reserved_seeds(42, 0, 8)
overrides = [{"monte_carlo.seed": int(s), "simulation.n_sims": 1} for s in seeds]
res = aerocapture_rs.run_batch("configs/test/test_ref_orig.toml", overrides)
costs = np.array([compute_cost(fr.reshape(1, 52)) for fr in res.final_records])
np.save("/tmp/phase0_baseline/ftc_cost_vector.npy", costs)
print("baseline costs:", costs)
PY
```
Expected: prints an 8-element finite cost vector; saves it. (Phase 4 diffs against this exact vector.)

- [ ] **Step 3: Record the baseline in the plan (no source commit).**

Note the pytest pass-count and the cost vector in the execution log. These are the regression anchors. Do not commit `/tmp/phase0_baseline`.

---

## Phase 1 — Confirmed Defects (test-first)

### Task 1.1 (D1): Single-algo must not freeze the gen-0 best when `validation_n_sims == 0`

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (the gen-loop validation block, around lines 1382–1396)
- Test: `tests/test_train_no_validation_promotion.py` (create)

- [ ] **Step 1: Write the failing test** (mirrors the `train()`-in-process pattern in `tests/test_train_interrupt.py`).

Create `tests/test_train_no_validation_promotion.py`:
```python
"""Regression: single-algo training with validation_n_sims=0 must promote a
later generation's best, not freeze the gen-0 argmin (defect D1)."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import TrainingConfig  # noqa: E402
from aerocapture.training.optimizer import OptimizerConfig  # noqa: E402
from aerocapture.training.problem import AerocaptureProblem  # noqa: E402
from aerocapture.training.train import train  # noqa: E402


def test_no_validation_promotes_later_generation(tmp_path: Path) -> None:
    exe_path = tmp_path / "src" / "rust" / "target" / "release"
    exe_path.mkdir(parents=True)
    (tmp_path / "data" / "neural_network").mkdir(parents=True)
    dummy_exe = exe_path / "aerocapture"
    dummy_exe.write_text("#!/bin/sh\nexit 0\n")
    dummy_exe.chmod(dummy_exe.stat().st_mode | stat.S_IEXEC)

    cfg = TrainingConfig(optimizer=OptimizerConfig(seed_strategy="fixed"))
    cfg.optimizer.n_gen = 4
    cfg.optimizer.n_pop = 4
    cfg.optimizer.validation_n_sims = 0  # validation gate OFF -> exercises the D1 path
    cfg.save_dir = str(tmp_path / "training_output")

    # Strictly-decreasing costs across calls: later generations are better, so a
    # correct implementation must end with best_cost well below generation 0's.
    gen0_min = 1000.0
    call_count = 0

    def mock_run_batch(self_prob, X):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        base = max(gen0_min - 100.0 * (call_count - 1), 10.0)
        return base + np.arange(X.shape[0], dtype=np.float64)

    with patch.object(AerocaptureProblem, "_run_batch", mock_run_batch):
        result = train(cfg, seed=42, cwd=str(tmp_path), verbose=False, no_tui=True)

    assert result["interrupted"] is False
    # Buggy behavior freezes at ~gen0_min; fixed behavior promotes a later, lower argmin.
    assert result["best_cost"] < gen0_min - 100.0
```

- [ ] **Step 2: Run it to confirm it fails.**

Run: `uv run pytest tests/test_train_no_validation_promotion.py -v`
Expected: FAIL — `best_cost` stays ~`gen0_min` (frozen gen-0 argmin).

- [ ] **Step 3: Implement the fix in `train.py`.**

Extend the validation block. The current code is:
```python
                if val_seeds is not None and new_gen_best:
                    val_costs, val_records = problem.evaluate_individual_records_per_seed(gen_best_individual, val_seeds)
                    validation_metrics, validation_summary = _build_validation_payload(
                        val_costs, val_records, len(val_seeds), problem.cost_kwargs,
                    )
                    val_rms = validation_metrics["rms_cost"]
                    last_validated_individual = gen_best_individual
                    if val_rms < best_val_cost:
                        best_val_cost = val_rms
                        best_overall_individual = gen_best_individual
                        best_overall_cost = gen_best_cost
                        validated_improvement = True
```
Add an `elif` branch immediately after it:
```python
                elif val_seeds is None and np.isfinite(gen_best_cost):
                    # No validation gate: promote each generation's finite training
                    # argmin directly (mirrors the islands no-validation fallback in
                    # _train_islands). The final MC eval re-ranks on a disjoint pool,
                    # so cross-gen seed incomparability is bounded. Without this the
                    # deployed best_model.json freezes at the gen-0 argmin (defect D1).
                    best_overall_individual = gen_best_individual
                    best_overall_cost = gen_best_cost
```

- [ ] **Step 4: Run the test to confirm it passes.**

Run: `uv run pytest tests/test_train_no_validation_promotion.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the resume-incomparability regression still passes (must not have regressed the gated init).**

Run: `uv run pytest tests/test_train_interrupt.py -v`
Expected: PASS.

- [ ] **Step 6: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/train.py tests/test_train_no_validation_promotion.py
uv run ruff format src/python/aerocapture/training/train.py tests/test_train_no_validation_promotion.py
uv run mypy src/python/aerocapture/training/train.py tests/test_train_no_validation_promotion.py
git add src/python/aerocapture/training/train.py tests/test_train_no_validation_promotion.py
git commit -m "fix(train): promote later-gen best when validation_n_sims=0 (D1)"
```

### Task 1.2 (D2): Fix the broken `animate` checkpoint discovery

**Files:**
- Modify: `src/python/aerocapture/training/animate.py:38-44`
- Modify: `tests/test_animate.py` (update fixture naming + add current-naming regression)

- [ ] **Step 1: Inspect the existing test's checkpoint naming.**

Run: `rg -n "checkpoint_r|checkpoint_g|_discover_checkpoints" tests/test_animate.py`
Expected: shows the fixtures create `checkpoint_r*_g*.json` (the obsolete naming — that's why the broken glob still passed CI).

- [ ] **Step 2: Write the failing regression test** (current `checkpoint_g*` naming). Append to `tests/test_animate.py`:
```python
def test_discover_finds_current_checkpoint_naming(tmp_path: Path) -> None:
    """animate must discover the trainer's current checkpoint_g{NNNNN}.json naming (D2)."""
    import numpy as np

    from aerocapture.training.animate import _discover_checkpoints

    for gen in (0, 1, 2):
        (tmp_path / f"checkpoint_g{gen:05d}.json").write_text(
            '{"best_cost": 1.0, "cost_history": [1.0]}'
        )
        np.savez(tmp_path / f"checkpoint_g{gen:05d}.npz", costs=np.array([1.0, 2.0]))

    found = _discover_checkpoints(tmp_path, every=1)
    assert [c["generation"] for c in found] == [0, 1, 2]
```
(If `tests/test_animate.py` lacks `from pathlib import Path`, add it.)

- [ ] **Step 3: Run it to confirm it fails.**

Run: `uv run pytest tests/test_animate.py::test_discover_finds_current_checkpoint_naming -v`
Expected: FAIL — returns `[]` (glob misses `checkpoint_g*`).

- [ ] **Step 4: Fix `_discover_checkpoints`** (`animate.py:38-44`). Replace:
```python
    pattern = re.compile(r"checkpoint_r\d+_g(\d+)\.json$")
    found: list[tuple[int, Path]] = []
    for p in training_dir.glob("checkpoint_r*_g*.json"):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
```
with:
```python
    # Current trainer naming: checkpoint_g{NNNNN}.json (train.py save_checkpoint).
    pattern = re.compile(r"checkpoint_g(\d+)\.json$")
    found: list[tuple[int, Path]] = []
    for p in training_dir.glob("checkpoint_g*.json"):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    # Legacy fallback: checkpoint_r{R}_g{NNNNN}.json (matches train.py load_checkpoint).
    if not found:
        legacy = re.compile(r"checkpoint_r\d+_g(\d+)\.json$")
        for p in training_dir.glob("checkpoint_r*_g*.json"):
            m = legacy.search(p.name)
            if m:
                found.append((int(m.group(1)), p))
```

- [ ] **Step 5: Run the full animate suite (the new test + the legacy-named existing tests must both pass).**

Run: `uv run pytest tests/test_animate.py -v`
Expected: PASS (new current-naming test + existing legacy-naming tests via the fallback).

- [ ] **Step 6: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/animate.py tests/test_animate.py
uv run ruff format src/python/aerocapture/training/animate.py tests/test_animate.py
uv run mypy src/python/aerocapture/training/animate.py
git add src/python/aerocapture/training/animate.py tests/test_animate.py
git commit -m "fix(animate): discover current checkpoint_g* naming, legacy fallback (D2)"
```

### Task 1.3 (D3): Thread `heat_load_limit` into report.py trajectory classification

**Files:**
- Modify: `src/python/aerocapture/training/report.py` (`_read_constraint_limits:326`, call sites 694/829/891)
- Test: `tests/test_report.py` (add a unit test; create the file if absent — check first with `ls tests/test_report.py`)

- [ ] **Step 1: Write the failing test** for the 3-value return:
```python
def test_read_constraint_limits_includes_heat_load(tmp_path: Path) -> None:
    """_read_constraint_limits must return the heat-load limit so the PDF colors
    heat-load-only violators as constrained, matching the stats block (D3)."""
    from aerocapture.training.report import _read_constraint_limits

    toml = tmp_path / "m.toml"
    toml.write_text(
        "[flight.constraints]\n"
        "max_heat_flux = 200.0\n"
        "max_load_factor = 15.0\n"
        "max_heat_load = 25000.0\n"
    )
    heat_flux, g_load, heat_load = _read_constraint_limits(toml)
    assert heat_flux == 200.0
    assert g_load == 15.0
    assert heat_load == 25000.0
```

- [ ] **Step 2: Run it to confirm it fails.**

Run: `uv run pytest tests/test_report.py::test_read_constraint_limits_includes_heat_load -v`
Expected: FAIL — `ValueError: not enough values to unpack (expected 3, got 2)`.

- [ ] **Step 3: Change `_read_constraint_limits` to a 3-tuple** (`report.py:326-334`):
```python
def _read_constraint_limits(toml_path: Path) -> tuple[float | None, float | None, float | None]:
    """Read heat flux, g-load, and heat-load limits from TOML [flight.constraints]."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    constraints = data.get("flight", {}).get("constraints", {})
    heat_flux: float | None = constraints.get("max_heat_flux")
    g_load: float | None = constraints.get("max_load_factor")
    heat_load: float | None = constraints.get("max_heat_load")
    return heat_flux, g_load, heat_load
```

- [ ] **Step 4: Update the three call sites** to unpack 3 and pass `heat_load_limit`.

Line ~693–694 (`_generate_trajectory_charts`):
```python
    heat_flux_limit, g_load_limit, heat_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None, None)
    traj_class = charts.classify_trajectories(
        final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit, heat_load_limit=heat_load_limit,
    )
```
Line ~829 (summary-table block) — unpack 3 and pass `heat_load_limit=heat_load_limit` into `_build_summary_table` (add the param + a heat-load violation row; see Step 5):
```python
        heat_flux_limit, g_load_limit, heat_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None, None)
        summary = (
            _build_summary_table(final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit, heat_load_limit=heat_load_limit, cost_kwargs=cost_kwargs)
            if final_records is not None
            else None
        )
```
Line ~891 (the RL wrapper):
```python
    heat_flux_limit, g_load_limit, heat_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None, None)
```
Then thread `heat_load_limit` into that block's `_generate_trajectory_charts`/classification call the same way as line 694.

- [ ] **Step 5: Add the heat-load violation row to `_build_summary_table`.**

Read `_build_summary_table` (grep its def: `rg -n "def _build_summary_table" src/python/aerocapture/training/report.py`). Add a `heat_load_limit: float | None = None` parameter and, mirroring the existing g-load/heat-flux violation rows, append a heat-load row gated on `heat_load_limit is not None`, computing the violation fraction as `(final_records[:, _FR_INTEGRATED_FLUX] * 1e3 > heat_load_limit)` over captured trajectories (reuse the same captured mask the function already builds; if it open-codes `(ifinal==3)&(ecc<1)`, that's superseded by H1 in Phase 2 — for now match the existing local idiom).

- [ ] **Step 6: Run the test + the existing report tests.**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS.

- [ ] **Step 7: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/report.py tests/test_report.py
uv run ruff format src/python/aerocapture/training/report.py tests/test_report.py
uv run mypy src/python/aerocapture/training/report.py
git add src/python/aerocapture/training/report.py tests/test_report.py
git commit -m "fix(report): color heat-load-only violators as constrained (D3)"
```

### Task 1.4 (D5): `_describe_rl_architecture` must use the canonical per-layer param counts

**Files:**
- Modify: `src/python/aerocapture/training/rl/train.py:146-190`
- Test: `tests/rl/test_describe_architecture.py` (create; confirm `tests/rl/` exists with `ls tests/rl`)

- [ ] **Step 1: Confirm `config.describe_architecture` / `_layer_n_params` signatures.**

Run: `rg -n "def describe_architecture|def _layer_n_params|def _layer_output_size" src/python/aerocapture/training/config.py`
Expected: shows the canonical helpers covering all six layer types. Note their exact signatures for Step 3.

- [ ] **Step 2: Write the failing test** (LSTM param count is 4-gate, not 3):
```python
"""_describe_rl_architecture must report correct params for non-GRU layers (D5)."""

from __future__ import annotations

import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import _layer_n_params  # noqa: E402
from aerocapture.training.rl.schemas import DenseSpec, LstmSpec  # noqa: E402


def test_lstm_param_count_is_four_gate() -> None:
    # An LSTM(input=16, hidden=32) has 4*h*I + 4*h*h + 8*h params, NOT the
    # 3-gate GRU formula the old _describe_rl_architecture hardcoded.
    spec = LstmSpec(type="lstm", input_size=16, hidden_size=32)
    expected = 4 * 32 * 16 + 4 * 32 * 32 + 8 * 32
    assert _layer_n_params(spec) == expected
    assert _layer_n_params(spec) != 3 * 32 * 16 + 3 * 32 * 32 + 6 * 32
```
(Adjust the `LstmSpec(...)` kwargs to match the schema confirmed in Step 1; if `_layer_n_params` lives under a different name, use the name from Step 1.)

- [ ] **Step 3: Run it to confirm it passes for `_layer_n_params`** (this proves the canonical source is correct), then refactor `_describe_rl_architecture` to use it.

Run: `uv run pytest tests/rl/test_describe_architecture.py -v`
Expected: PASS (canonical helper is already correct). The defect is that `_describe_rl_architecture` does NOT use it.

- [ ] **Step 4: Replace the private helpers in `_describe_rl_architecture`** (`rl/train.py:153-190`). Delete the local `_in_size`/`_out_size`/`_n_params` and route through config:
```python
def _describe_rl_architecture(cfg: RLConfig) -> None:
    """Fail-fast chain check + stdout description of the RL architecture."""
    from aerocapture.training.config import _layer_n_params, _layer_output_size, describe_architecture

    input_mask, architecture, _input_dim = _parse_network_config(cfg)

    # Chain consistency: prev.output == next.input (uses the canonical output-size fn).
    for i in range(len(architecture) - 1):
        prev_out = _layer_output_size(architecture[i])
        next_in = architecture[i + 1].input_size
        if prev_out != next_in:
            raise ValueError(
                f"[network.architecture] chain mismatch at layer {i}->{i + 1}: "
                f"layer {i} ({architecture[i].type}) produces output={prev_out}, "
                f"but layer {i + 1} ({architecture[i + 1].type}) expects input={next_in}"
            )

    print(describe_architecture(architecture), file=sys.stderr)
    print(f"  input_mask: {len(input_mask)} indices", file=sys.stderr)
```
(If `describe_architecture` returns a different shape than a printable string, adapt: build the total via `sum(_layer_n_params(s) for s in architecture)` and the per-layer lines via `_layer_output_size`. Match the exact API from Step 1. The invariant: no hardcoded gate formula remains in `rl/train.py`.)

- [ ] **Step 5: Confirm an LSTM RL config no longer miscounts** by running the existing RL parse/smoke tests.

Run: `uv run pytest tests/rl/ -v -k "describe or parse or architecture"`
Expected: PASS. Also `uv run mypy src/python/aerocapture/training/rl/train.py`.

- [ ] **Step 6: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/rl/train.py tests/rl/test_describe_architecture.py
uv run ruff format src/python/aerocapture/training/rl/train.py tests/rl/test_describe_architecture.py
uv run mypy src/python/aerocapture/training/rl/train.py
git add src/python/aerocapture/training/rl/train.py tests/rl/test_describe_architecture.py
git commit -m "fix(rl): route _describe_rl_architecture through canonical param counts (D5)"
```

### Task 1.5 (D4): RL validation gate + terminal cost must honor TOML `cost_kwargs`

**Files:**
- Modify: `src/python/aerocapture/training/rl/rewards.py` (`compute_terminal_cost`, ~line 90-98)
- Modify: `src/python/aerocapture/training/rl/train.py` (`_validate_deterministic` ~227, `_validate_deterministic_v1` ~246, and their call sites that must pass `cost_kwargs`)
- Test: `tests/rl/test_terminal_cost_kwargs.py` (create)

- [ ] **Step 1: Read the current `compute_terminal_cost` and its caller.**

Run:
```bash
sed -n '1,40p;80,98p' src/python/aerocapture/training/rl/rewards.py
rg -n "compute_terminal_cost|read_cost_kwargs|cost_kwargs" src/python/aerocapture/training/rl/*.py
```
Expected: confirms `compute_terminal_cost(final_record)` calls `compute_cost(...)` with defaults; note its exact signature + the call site in the reward path.

- [ ] **Step 2: Write the failing test** (terminal cost responds to `cost_transform`):
```python
"""RL terminal cost must honor TOML cost_kwargs (D4)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.rewards import compute_terminal_cost  # noqa: E402


def test_terminal_cost_respects_cost_transform() -> None:
    # A captured final-record row (ifinal=3, ecc<1) with some DV.
    fr = np.zeros(52, dtype=np.float64)
    fr[31] = 3.0   # ifinal (captured)
    fr[9] = 0.5    # ecc (bound)
    fr[41] = 1500.0  # dv_total_m_s
    linear = compute_terminal_cost(fr, cost_kwargs={"cost_transform": "linear", "dv_threshold": 1000.0})
    log = compute_terminal_cost(fr, cost_kwargs={"cost_transform": "log", "dv_threshold": 1000.0})
    assert linear != log  # the transform must actually be applied
```
(Adjust the final-record column indices using the names confirmed in Step 1 / `parse_final.py`; `41`=dv_total, `31`=ifinal, `9`=ecc per the review.)

- [ ] **Step 3: Run it to confirm it fails.**

Run: `uv run pytest tests/rl/test_terminal_cost_kwargs.py -v`
Expected: FAIL — `compute_terminal_cost` takes no `cost_kwargs` (TypeError) or ignores it (values equal).

- [ ] **Step 4: Add a `cost_kwargs` parameter to `compute_terminal_cost`** and forward it to `compute_cost`:
```python
def compute_terminal_cost(final_record, cost_kwargs: dict | None = None) -> float:
    kw = cost_kwargs or {}
    return float(compute_cost(final_record.reshape(1, -1), **kw))
```
(Match the existing return/shape conventions; keep the default-None so existing callers don't break, then update them in Step 5.)

- [ ] **Step 5: Thread `read_cost_kwargs(toml_path)` into the validation helpers and the reward caller.**

In `rl/train.py`, `_validate_deterministic` (~241) currently does `rms_cost = float(compute_cost(fr))`. Compute `cost_kwargs = read_cost_kwargs(toml_path)` (import from `aerocapture.training.report`) once near the top of the function and pass `**cost_kwargs`:
```python
    from aerocapture.training.report import read_cost_kwargs
    cost_kwargs = read_cost_kwargs(toml_path)
    ...
    rms_cost = float(compute_cost(fr, **cost_kwargs))
```
Apply the same to `_validate_deterministic_v1`. In the reward path (the `compute_terminal_cost(...)` call found in Step 1), pass the `cost_kwargs` the env/trainer already resolved from `cfg.raw_toml` (thread it from the trainer; if the reward calculator has no toml handle, plumb `cost_kwargs` through its constructor).

- [ ] **Step 6: Run the test + RL smoke.**

Run:
```bash
uv run pytest tests/rl/test_terminal_cost_kwargs.py -v
uv run pytest tests/ -v -k "rl and (smoke or reward)" -m "not slow"
```
Expected: PASS. `uv run mypy src/python/aerocapture/training/rl/rewards.py src/python/aerocapture/training/rl/train.py`.

- [ ] **Step 7: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/rl/rewards.py src/python/aerocapture/training/rl/train.py tests/rl/test_terminal_cost_kwargs.py
uv run ruff format src/python/aerocapture/training/rl/rewards.py src/python/aerocapture/training/rl/train.py tests/rl/test_terminal_cost_kwargs.py
uv run mypy src/python/aerocapture/training/rl/rewards.py src/python/aerocapture/training/rl/train.py
git add src/python/aerocapture/training/rl/rewards.py src/python/aerocapture/training/rl/train.py tests/rl/test_terminal_cost_kwargs.py
git commit -m "fix(rl): honor TOML cost_kwargs in validation gate + terminal cost (D4)"
```

### Task 1.6 (D6): `problem._evaluate` must surface systemic failures, not mask them as 1e9

**Files:**
- Modify: `src/python/aerocapture/training/problem.py` (module const + `_evaluate:65-71`)
- Test: `tests/test_problem.py` (add; confirm with `ls tests/test_problem.py`)

- [ ] **Step 1: Write the failing test:**
```python
def test_evaluate_aborts_after_consecutive_failures(monkeypatch) -> None:
    """A persistent batch-eval failure must raise, not silently return 1e9 forever (D6)."""
    import numpy as np
    import pytest

    from aerocapture.training import problem as problem_mod

    prob = _make_minimal_problem()  # see helper note below

    def always_fail(self, X):
        raise RuntimeError("simulated systemic break")

    monkeypatch.setattr(problem_mod.AerocaptureProblem, "_run_batch", always_fail)

    X = np.zeros((4, prob.n_var), dtype=np.float64)
    out: dict = {}
    # First failures are tolerated (return 1e9); after the threshold it raises.
    for _ in range(problem_mod._MAX_CONSECUTIVE_EVAL_FAILURES - 1):
        prob._evaluate(X, out)
        assert np.all(out["F"] == 1e9)
    with pytest.raises(RuntimeError, match="consecutive"):
        prob._evaluate(X, out)


def test_evaluate_resets_failure_counter_on_success(monkeypatch) -> None:
    import numpy as np

    from aerocapture.training import problem as problem_mod

    prob = _make_minimal_problem()
    calls = {"n": 0}

    def flaky(self, X):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return np.full(X.shape[0], 5.0)

    monkeypatch.setattr(problem_mod.AerocaptureProblem, "_run_batch", flaky)
    X = np.zeros((4, prob.n_var), dtype=np.float64)
    out: dict = {}
    prob._evaluate(X, out)          # failure -> 1e9, counter=1
    prob._evaluate(X, out)          # success -> counter reset
    assert np.all(out["F"] == 5.0)
    assert prob._consecutive_eval_failures == 0
```
Add a `_make_minimal_problem()` helper at the top of the test module that constructs an `AerocaptureProblem` for a non-NN scheme with a tiny `param_specs` (model it on the construction in `tests/test_problem.py` if it already builds one, else use `AerocaptureProblem(scheme="ftc", param_specs=make_ftc_specs(), toml_path=..., seeds=[42], cost_kwargs={})` matching the actual constructor — confirm the signature with `rg -n "def __init__" src/python/aerocapture/training/problem.py`).

- [ ] **Step 2: Run it to confirm it fails.**

Run: `uv run pytest tests/test_problem.py -v -k "consecutive or failure_counter"`
Expected: FAIL — `_MAX_CONSECUTIVE_EVAL_FAILURES` undefined / never raises.

- [ ] **Step 3: Implement the fix in `problem.py`.**

Add a module constant near the top:
```python
_MAX_CONSECUTIVE_EVAL_FAILURES = 5
```
Replace `_evaluate` (65-71):
```python
    def _evaluate(self, X: npt.NDArray[np.float64], out: dict, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        try:
            costs = self._run_batch(X)
            self._consecutive_eval_failures = 0
        except Exception as e:
            self._consecutive_eval_failures = getattr(self, "_consecutive_eval_failures", 0) + 1
            print(
                f"  [problem] batch eval failed ({type(e).__name__}: {e}); penalizing 1e9 "
                f"(consecutive failures: {self._consecutive_eval_failures})",
                file=sys.stderr,
            )
            if self._consecutive_eval_failures >= _MAX_CONSECUTIVE_EVAL_FAILURES:
                raise RuntimeError(
                    f"{self._consecutive_eval_failures} consecutive batch-eval failures; aborting "
                    f"(last: {type(e).__name__}: {e})"
                ) from e
            costs = np.full(X.shape[0], 1e9)
        out["F"] = costs.reshape(-1, 1)
```
Initialize `self._consecutive_eval_failures = 0` in `__init__`, and ensure `import sys` is present at the top of `problem.py`.

- [ ] **Step 4: Run the test to confirm it passes.**

Run: `uv run pytest tests/test_problem.py -v -k "consecutive or failure_counter"`
Expected: PASS.

- [ ] **Step 5: Confirm the broader problem/training suite still passes.**

Run: `uv run pytest tests/test_problem.py tests/test_optimizer.py -v -m "not slow"`
Expected: PASS.

- [ ] **Step 6: Gate + commit.**

```bash
uv run ruff check src/python/aerocapture/training/problem.py tests/test_problem.py
uv run ruff format src/python/aerocapture/training/problem.py tests/test_problem.py
uv run mypy src/python/aerocapture/training/problem.py
git add src/python/aerocapture/training/problem.py tests/test_problem.py
git commit -m "fix(problem): abort on persistent batch-eval failure instead of fake 1e9 optimum (D6)"
```

### Task 1.7: Phase 1 regression sweep

- [ ] **Step 1: Run the full non-slow suite.**

Run: `uv run pytest -q -m "not slow"`
Expected: pass-count >= the Phase-0 baseline (new tests added, none regressed).

- [ ] **Step 2: Confirm goldens untouched** (Phase 1 changed no simulator numerics).

Run: `cd src/rust && cargo test --release guidance 2>&1 | tail -10` (or the golden regression test target).
Expected: golden regression tests PASS.

- [ ] **Step 3: No commit (verification only).** Record the pass-count in the execution log.

---

## Phases 2–6 (roadmap; each authored as its own dated plan before execution)

**Phase 2 — Shared single-sources-of-truth (H1–H11).** For each helper: create it, write its contract test (rendered in full in the Phase-2 plan), migrate the N call sites, run the existing suites as the no-behavior-change gate, commit per helper. Order: H8 `run_validation_gate` (also lands the all-inf argmin guard) → H1 `is_captured` → H2 `route_param` → H3 per-layer `from_flat` (gated by the cross-language equivalence suite) → H7 record-index map + PyO3-exposed `NN_FULL_INPUT_SIZE`/`DISPERSION_DRAW_LEN` + drift tests → H4 chart panel-builders + `(N,17)` column constants → H9 `binned_reduce` → H5 `typst_utils` → H10 NN-JSON emission unification → H6 seed-offset registry + `nn_input_report` `make_reserved_seeds` fix → H11 `apply_theme`.

**Phase 3 — Dead code (R1–R5).** R1 delete `plotting/` (after a final `rg` importer sweep). R2 delete `evaluate_population_per_seed`. R3 wire `chart_sobol_convergence` into `_generate_sensitivity_charts` + `report.typ`. R4 delete `derive_asinh_scale` (+ test) after confirming only-tests reference it. R5 delete `CAPTURE_COST_THRESHOLD` + fix the `capture_rate` docstring. Gate: full suite green.

**Phase 4 — PyO3 rework (P1–P6).** 4a (Rust core): convert `SimData.atmosphere`/`wind_table`/reference-trajectory to `Arc`, add `from_toml_with_tables`; gate on full `cargo test` + golden bit-identity. 4b (binding): `run_grid` (config resolved once, SimData per-individual once, internal seed loop, `py.detach`, in-memory NN weights), GIL release in existing `run*`, P3 zero-extra-copy getters, P4 `as_array().rows()`, P5 global Rayon pool. 4c (Python): rewire `problem._run_batch_pyo3` to one `run_grid` call. **Gate: cost vector bit-identical to `/tmp/phase0_baseline/ftc_cost_vector.npy`; record the measured speedup.**

**Phase 5 — Decomposition (G1–G4).** Behavior-preserving extraction, each gated by the existing resume/regression suites: G1 `train()` → `_setup_param_specs`/`_build_initial_population`/`_resume_state`/`_emit_warm_start_artifacts`; G2 `_run_ppo` → `collect_rollout`/`build_critic_from_architecture`/shared update-record; G3 `report.py` compute/render/compile split; G4 `build_warm_start_chromosome` → `_collect_supervisor_corpus`/`_write_selection_sidecar`/`_encode_and_persist`.

**Phase 6 — Low nits (N1–N9).** N1 except-parens; N2 `_ACT` asinh; N3 RL RNG seeding (+ checkpoint persist); N4 narrow silent excepts; N5 io/ missing-file guard; N6 vectorize `population_diversity`; N7 promote `_save_line_chart`; N8 stale citation; N9 (subsumed by P6/H2 — verify then close).

**Phase 7 — Docs + smart-commit** over the whole branch.

---

## Self-Review

- **Spec coverage:** D1–D6 each have a task; H/R/P/G/N items are scheduled in the roadmap with their gating strategy. No spec item is unmapped.
- **Placeholder scan:** Phase-1 tasks render real test + fix code. Three spots say "confirm signature / adapt to schema in Step 1" (D4 rewards, D5 config API, D6 constructor) — these are deliberate read-then-render points where rendering against an unread signature would be guessing; each names the exact `rg` to run and the invariant to satisfy.
- **Type consistency:** `_read_constraint_limits` returns a 3-tuple consistently across all call sites; `classify_trajectories(heat_load_limit=...)` matches the verified signature in `charts.py:503`; `compute_terminal_cost(fr, cost_kwargs=...)` is used identically in test and fix; `_MAX_CONSECUTIVE_EVAL_FAILURES` / `_consecutive_eval_failures` names match between fix and test.
