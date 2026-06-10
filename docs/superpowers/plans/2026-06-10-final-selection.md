# End-of-Training Final Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-rank the last generation's population (plus running champions) on the reserved validation pool at end of training — single-algorithm and islands — with a standalone CLI to apply the same rule retroactively to existing training outputs.

**Architecture:** One pure selection function in a new `final_select.py` module, three call sites (single-algo end-of-loop, `_train_islands` pre-`final_eval`, CLI). Two small extractions make the CLI possible without duplicating `train.py` logic: the TOML→`TrainingConfig` bootstrap and the artifact-write block. The CLI reads both checkpoint formats and patches the latest checkpoint so resume can't revert re-selected artifacts.

**Tech Stack:** Python 3.14, numpy, pymoo-free core (the selection function only duck-types `evaluate_individual_per_seed`), pytest with mock problems (no Rust in unit tests).

**Spec:** `docs/superpowers/specs/2026-06-10-final-selection-design.md`

**Branch:** `feature/qpso-optimizer` (continues the QPSO branch — coupled paper workflow).

**API refinement vs spec 4.1** (intent-preserving): instead of a single `champion` + `best_val_cost` parameter pair, `select_final_individual` takes `known: list[KnownCandidate]` — pre-scored candidates that are never re-simulated. Single-algo passes `[champion]`; islands passes all 3 champions. The incumbent is the lowest-val-RMS known candidate; `promoted` means a fresh candidate strictly beat the incumbent. This removes the islands special-casing the spec described in prose (4.4.1).

**Conventions for every task:**
- Run all commands from the repo root: `/Users/govit/Git/Govit/Aerocapture/.claude/worktrees/strange-perlman-c3973f`. Use `uv run ...`.
- mypy strict (`disallow_untyped_defs`, tests included); ruff E,F,I,W,UP,B,SIM line-length 160.
- Stage only the task's files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Per-task lint gate: `uv run ruff check <files> && uv run ruff format --check <files> && uv run mypy <files>`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/python/aerocapture/training/final_select.py` | Create | `KnownCandidate`/`SelectionResult`, `select_final_individual`, sidecar writer, summary formatter, checkpoint load/patch (both formats), CLI |
| `tests/test_final_select.py` | Create | All unit tests for the above (mock problem, no Rust) |
| `src/python/aerocapture/training/warm_start.py` | Modify | Extract `load_warm_start_bounds(save_dir)` from `_cache_hit` |
| `src/python/aerocapture/training/train.py` | Modify | Extract `build_cost_kwargs` + `build_training_config_from_toml` + `write_best_artifacts`; inline selection hooks (single-algo + islands) |
| `tests/test_train_config_builder.py` | Create | Tests for the TOML→TrainingConfig builder |

Existing test files (`test_warm_start_*`, `test_train_interrupt`, `test_island_model`, …) guard the refactors.

---

### Task 1: Pure selection core

**Files:**
- Create: `tests/test_final_select.py`
- Create: `src/python/aerocapture/training/final_select.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_final_select.py` with exactly:

```python
"""Unit tests for end-of-training final selection (pure rule, no Rust)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aerocapture.training.final_select import (
    KnownCandidate,
    SelectionResult,
    format_selection_summary,
    select_final_individual,
    write_final_selection_json,
)


class _MockProblem:
    """Per-seed cost = sum(x) + 0.001 * seed. Records every evaluated row."""

    def __init__(self) -> None:
        self.evaluated: list[np.ndarray] = []

    def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
        self.evaluated.append(x.copy())
        return np.array([float(np.sum(x)) + 0.001 * s for s in seeds], dtype=np.float64)


def _rms(problem_free_x: np.ndarray, seeds: list[int]) -> float:
    costs = np.array([float(np.sum(problem_free_x)) + 0.001 * s for s in seeds])
    return float(np.sqrt(np.mean(costs**2)))


SEEDS = [1, 2, 3, 4]


class TestSelectionRule:
    def test_fresh_candidate_promotes_on_strict_improvement(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], known, SEEDS)
        assert sel.promoted
        assert sel.provenance == "last_gen[0]"
        assert sel.winner_index == 0
        assert np.array_equal(sel.individual, cands[0])
        assert sel.val_rms == _rms(cands[0], SEEDS)

    def test_champion_kept_when_no_candidate_beats_it(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.1)
        cands = np.vstack([np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"
        assert sel.winner_index is None
        assert np.array_equal(sel.individual, champ)

    def test_tie_keeps_champion(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.4)
        cands = np.vstack([np.full(4, 0.4) + np.array([0.1, -0.1, 0.0, 0.0])])  # same sum -> same rms
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"

    def test_champion_never_resimulated_and_duplicates_deduped(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([champ, np.full(4, 0.4), np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(
            problem, cands, ["last_gen[0]", "last_gen[1]", "last_gen[2]", "last_gen[3]"], known, SEEDS
        )
        # champ row + one duplicate skipped: only 2 unique fresh rows simulated
        assert len(problem.evaluated) == 2
        assert sel.n_candidates == 4
        assert sel.n_deduped == 2
        assert sel.promoted and sel.provenance == "last_gen[1]"

    def test_all_nonfinite_candidates_keep_champion(self) -> None:
        class _NaNProblem(_MockProblem):
            def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
                self.evaluated.append(x.copy())
                return np.full(len(seeds), np.nan)

        problem = _NaNProblem()
        champ = np.full(4, 0.5)
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, np.vstack([np.full(4, 0.4)]), ["last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"
        # non-finite candidates are visible in the sidecar records as null val_rms
        assert any(e["val_rms"] is None for e in sel.candidate_rms)
        assert np.array_equal(sel.individual, champ)

    def test_no_known_candidates_promotes_best_fresh(self) -> None:
        problem = _MockProblem()
        cands = np.vstack([np.full(4, 0.6), np.full(4, 0.4)])
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], [], SEEDS)
        assert sel.promoted
        assert sel.provenance == "last_gen[1]"

    def test_no_known_and_no_finite_raises(self) -> None:
        class _NaNProblem(_MockProblem):
            def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
                return np.full(len(seeds), np.nan)

        import pytest

        with pytest.raises(ValueError, match="no finite candidate"):
            select_final_individual(_NaNProblem(), np.vstack([np.full(4, 0.4)]), ["last_gen[0]"], [], SEEDS)

    def test_islands_known_multiple_champions_incumbent_is_lowest(self) -> None:
        problem = _MockProblem()
        c_pso = np.full(4, 0.2)
        c_ga = np.full(4, 0.3)
        known = [
            KnownCandidate(x=c_ga, provenance="ga:champion", val_rms=_rms(c_ga, SEEDS)),
            KnownCandidate(x=c_pso, provenance="pso:champion", val_rms=_rms(c_pso, SEEDS)),
        ]
        cands = np.vstack([np.full(4, 0.25)])  # beats ga champion, not pso champion
        sel = select_final_individual(problem, cands, ["de:last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "pso:champion"

    def test_candidate_rms_records_everyone(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)
        provs = {e["provenance"] for e in sel.candidate_rms}
        assert provs == {"champion", "last_gen[0]"}


class TestSidecarAndSummary:
    def _result(self) -> SelectionResult:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        return select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)

    def test_sidecar_schema(self, tmp_path: Path) -> None:
        sel = self._result()
        write_final_selection_json(tmp_path, sel, n_val_seeds=len(SEEDS))
        data = json.loads((tmp_path / "final_selection.json").read_text())
        assert data["winner"]["provenance"] == "last_gen[0]"
        assert data["winner"]["promoted"] is True
        assert data["validation_n_sims"] == 4
        assert data["n_candidates"] == 1
        assert isinstance(data["candidate_rms"], list) and len(data["candidate_rms"]) == 2

    def test_summary_mentions_winner_and_delta(self) -> None:
        sel = self._result()
        text = format_selection_summary(sel)
        assert "last_gen[0]" in text
        assert "Final selection" in text
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_final_select.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'aerocapture.training.final_select'`

- [ ] **Step 1.3: Implement the core**

Create `src/python/aerocapture/training/final_select.py` with exactly:

```python
"""End-of-training final selection (spec: docs/superpowers/specs/2026-06-10-final-selection-design.md).

Re-ranks the last generation's population plus the running champion(s) on the
reserved VALIDATION pool and deploys the winner only on strict val-RMS
improvement. Selection happens on the validation pool by design: the
final-eval pool stays a clean test set that only ever evaluates the single
deployed winner (no min-of-N selection bias on reported numbers).

Three call sites share `select_final_individual`:
- single-algorithm end-of-training hook in train.py (known = [champion]),
- the islands trainer (known = all island champions),
- the standalone CLI (`python -m aerocapture.training.final_select`), which
  re-applies the rule to an existing training directory from its latest
  checkpoint and patches the checkpoint so resume cannot revert the artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class _PerSeedEvaluator(Protocol):
    def evaluate_individual_per_seed(self, x: npt.NDArray[np.float64], seeds: list[int]) -> npt.NDArray[np.float64]: ...


@dataclass
class KnownCandidate:
    """A pre-scored candidate (champion); never re-simulated."""

    x: npt.NDArray[np.float64]
    provenance: str
    val_rms: float


@dataclass
class SelectionResult:
    individual: npt.NDArray[np.float64]
    val_rms: float
    provenance: str
    promoted: bool  # a fresh candidate strictly beat the best known
    winner_index: int | None  # index into the candidates matrix; None when a known candidate won
    n_candidates: int  # fresh rows offered (pre-dedup)
    n_deduped: int  # fresh rows actually simulated
    candidate_rms: list[dict[str, Any]] = field(default_factory=list)  # [{"provenance", "val_rms"}]


def select_final_individual(
    problem: _PerSeedEvaluator,
    candidates: npt.NDArray[np.float64],
    provenances: list[str],
    known: list[KnownCandidate],
    val_seeds: list[int],
) -> SelectionResult:
    """The selection rule (spec section 3) over fresh candidates + known champions.

    Winner = lowest val RMS over {finite fresh candidates} U {known}. A fresh
    candidate displaces the incumbent (lowest-val-RMS known) only with a
    STRICTLY lower val RMS -- ties keep the incumbent, matching the in-training
    validation gate. Fresh rows identical to a known row or to an earlier fresh
    row are deduplicated (never re-simulated).
    """
    records: list[dict[str, Any]] = [{"provenance": k.provenance, "val_rms": k.val_rms} for k in known]

    incumbent: KnownCandidate | None = None
    for k in known:
        if incumbent is None or k.val_rms < incumbent.val_rms:
            incumbent = k

    seen: set[bytes] = {np.ascontiguousarray(k.x).tobytes() for k in known}
    best_fresh_rms = float("inf")
    best_fresh_idx: int | None = None
    n_deduped = 0
    for i in range(candidates.shape[0]):
        key = np.ascontiguousarray(candidates[i]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        n_deduped += 1
        costs = problem.evaluate_individual_per_seed(candidates[i], val_seeds)
        rms = float(np.sqrt(np.mean(np.asarray(costs, dtype=np.float64) ** 2)))
        if not np.isfinite(rms):
            records.append({"provenance": provenances[i], "val_rms": None})
            continue
        records.append({"provenance": provenances[i], "val_rms": rms})
        if rms < best_fresh_rms:
            best_fresh_rms = rms
            best_fresh_idx = i

    incumbent_rms = incumbent.val_rms if incumbent is not None else float("inf")
    if best_fresh_idx is not None and best_fresh_rms < incumbent_rms:
        return SelectionResult(
            individual=candidates[best_fresh_idx].copy(),
            val_rms=best_fresh_rms,
            provenance=provenances[best_fresh_idx],
            promoted=True,
            winner_index=best_fresh_idx,
            n_candidates=int(candidates.shape[0]),
            n_deduped=n_deduped,
            candidate_rms=records,
        )
    if incumbent is None:
        raise ValueError("final selection: no finite candidate and no known champion")
    return SelectionResult(
        individual=incumbent.x.copy(),
        val_rms=incumbent.val_rms,
        provenance=incumbent.provenance,
        promoted=False,
        winner_index=None,
        n_candidates=int(candidates.shape[0]),
        n_deduped=n_deduped,
        candidate_rms=records,
    )


def write_final_selection_json(save_dir: Path, result: SelectionResult, n_val_seeds: int) -> None:
    payload = {
        "winner": {
            "provenance": result.provenance,
            "val_rms": result.val_rms,
            "promoted": result.promoted,
        },
        "n_candidates": result.n_candidates,
        "n_deduped": result.n_deduped,
        "validation_n_sims": n_val_seeds,
        "candidate_rms": result.candidate_rms,
    }
    with open(save_dir / "final_selection.json", "w") as fp:
        json.dump(payload, fp, indent=2)


def format_selection_summary(result: SelectionResult) -> str:
    finite = [e["val_rms"] for e in result.candidate_rms if e["val_rms"] is not None]
    spread = f", candidate val-rms [{min(finite):.4e}, {max(finite):.4e}]" if finite else ""
    verdict = "PROMOTED" if result.promoted else "champion kept"
    return (
        f"  Final selection: {verdict} -> {result.provenance} "
        f"(val_rms={result.val_rms:.4e}; {result.n_deduped}/{result.n_candidates} fresh candidates simulated{spread})"
    )
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_final_select.py -q`
Expected: 11 passed

- [ ] **Step 1.5: Lint gate + commit**

Run the lint gate on both files; fix what it flags. Then:

```bash
git add src/python/aerocapture/training/final_select.py tests/test_final_select.py
git commit -m "feat(train): final-selection core (validation-pool re-rank rule + sidecar)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `load_warm_start_bounds` extraction

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py` (function `_cache_hit`, lines ~136-168)
- Test: `tests/test_final_select.py` (append)

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_final_select.py`:

```python
class TestWarmStartBoundsLoader:
    def test_loads_specs_from_sidecar(self, tmp_path: Path) -> None:
        from aerocapture.training.warm_start import load_warm_start_bounds

        (tmp_path / "warm_start_bounds.json").write_text(
            json.dumps(
                [
                    {"name": "w_0", "p_min": -2.0, "p_max": 2.0, "default": 0.0},
                    {"name": "w_1", "p_min": -0.5, "p_max": 0.5, "default": 0.1, "log_scale": False, "is_integer": False},
                ]
            )
        )
        specs = load_warm_start_bounds(tmp_path)
        assert specs is not None
        assert [s.name for s in specs] == ["w_0", "w_1"]
        assert specs[0].p_min == -2.0 and specs[1].p_max == 0.5

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        from aerocapture.training.warm_start import load_warm_start_bounds

        assert load_warm_start_bounds(tmp_path) is None
```

- [ ] **Step 2.2: Run to verify it fails**

Run: `uv run pytest tests/test_final_select.py::TestWarmStartBoundsLoader -q`
Expected: `ImportError: cannot import name 'load_warm_start_bounds'`

- [ ] **Step 2.3: Extract the loader**

In `src/python/aerocapture/training/warm_start.py`, add a public function ABOVE `_cache_hit` containing exactly the bounds-parsing block currently inside `_cache_hit` (the `bounds_path = save_dir / "warm_start_bounds.json"` block):

```python
def load_warm_start_bounds(save_dir: Path) -> list[ParamSpec] | None:
    """Parse `<save_dir>/warm_start_bounds.json` into ParamSpecs, or None if absent.

    The sidecar records the EXACT weight-slab bounds the chromosome/population
    was encoded under (adaptive bounds). Any consumer decoding a checkpointed
    population (resume, final_select CLI) must overlay these specs -- decoding
    under rebuilt Xavier bounds silently corrupts the weights.
    """
    bounds_path = save_dir / "warm_start_bounds.json"
    if not bounds_path.exists():
        return None
    from aerocapture.training.param_spaces import ParamSpec  # noqa: PLC0415

    raw = json.loads(bounds_path.read_text())
    return [
        ParamSpec(
            name=str(e["name"]),
            p_min=float(e["p_min"]),
            p_max=float(e["p_max"]),
            default=float(e.get("default", 0.0)),
            log_scale=bool(e.get("log_scale", False)),
            is_integer=bool(e.get("is_integer", False)),
        )
        for e in raw
    ]
```

Then replace the corresponding block inside `_cache_hit` with:

```python
    weight_specs = load_warm_start_bounds(save_dir)
```

(keeping `_cache_hit`'s return contract `(chromo, weight_specs)` identical — `weight_specs` is `None` when the sidecar is absent, exactly as before). Match the existing import style in the file (if `ParamSpec` is already imported at module top, drop the local import).

- [ ] **Step 2.4: Run tests to verify they pass (incl. warm-start regression)**

Run: `uv run pytest tests/test_final_select.py -q && uv run pytest tests/ -q -k "warm_start"`
Expected: all pass — the existing warm-start suites prove `_cache_hit` behavior is unchanged.

- [ ] **Step 2.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_final_select.py
git commit -m "refactor(train): extract load_warm_start_bounds from _cache_hit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `write_best_artifacts` + `build_cost_kwargs` extraction

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (`save_checkpoint` lines ~546-585, `_write_winner_artifacts` lines ~2000-2046, cost-kwargs block lines ~1070-1082)
- Test: `tests/test_final_select.py` (append)

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_final_select.py`:

```python
class TestSharedTrainHelpers:
    def test_build_cost_kwargs_defaults_and_overrides(self) -> None:
        from aerocapture.training.train import build_cost_kwargs

        kw = build_cost_kwargs({})
        assert kw["dv_threshold"] == 1000.0
        assert kw["cost_transform"] == "linear"
        kw2 = build_cost_kwargs(
            {
                "cost_function": {"dv_threshold": 500.0, "cost_transform": "log"},
                "flight": {"constraints": {"max_load_factor": 4.0}},
            }
        )
        assert kw2["dv_threshold"] == 500.0
        assert kw2["g_load_limit"] == 4.0
        assert kw2["cost_transform"] == "log"

    def test_write_best_artifacts_non_nn(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.param_spaces import ParamSpec
        from aerocapture.training.train import write_best_artifacts

        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [
            ParamSpec(name="gain", p_min=0.0, p_max=2.0, default=1.0),
            ParamSpec(name="bias", p_min=-1.0, p_max=1.0, default=0.0),
        ]
        write_best_artifacts(np.array([0.5, 0.75]), cfg, specs, tmp_path, cwd=None)
        params = json.loads((tmp_path / "best_params.json").read_text())
        assert params["gain"] == 1.0  # 0.5 of [0, 2]
        assert params["bias"] == 0.5  # 0.75 of [-1, 1]
```

- [ ] **Step 3.2: Run to verify they fail**

Run: `uv run pytest tests/test_final_select.py::TestSharedTrainHelpers -q`
Expected: `ImportError: cannot import name 'build_cost_kwargs'`

- [ ] **Step 3.3: Implement the extractions**

In `src/python/aerocapture/training/train.py`:

(a) **`build_cost_kwargs`** — new module-level function (place near `_decode_nn_weights`), body moved verbatim from the inline block at ~1071-1082:

```python
def build_cost_kwargs(toml_data: dict) -> dict[str, Any]:
    """Cost-function kwargs from a resolved TOML dict ([cost_function] + [flight.constraints])."""
    cost_cfg = toml_data.get("cost_function", {})
    constraints = toml_data.get("flight", {}).get("constraints", {})
    return {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(constraints.get("max_load_factor", 15.0)),
        "heat_flux_limit": float(constraints.get("max_heat_flux", 200.0)),
        "heat_load_limit": float(constraints.get("max_heat_load", 25000.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
        "heat_load_weight": float(cost_cfg.get("heat_load_weight", 1000.0)),
        "cost_transform": str(cost_cfg.get("cost_transform", "linear")),
    }
```

Replace the inline block in `train()` (~1070-1082) with `cost_kwargs = build_cost_kwargs(_toml)` (keep the surrounding `if config.sim.toml_config:` structure: `_toml` load stays, only the dict literal is replaced).

(b) **`write_best_artifacts`** — rename/extend `_write_winner_artifacts` (lines ~2000-2046). New signature and behavior:

```python
def write_best_artifacts(
    best_individual: npt.NDArray[np.float64],
    config: TrainingConfig,
    param_specs: list[ParamSpec],
    save_dir: Path,
    cwd: str | Path | None = None,
    deploy_to_cwd: bool = False,
) -> None:
    """Write best_model.json (NN) / best_params.json from a normalized chromosome.

    Always writes into save_dir. When `deploy_to_cwd` and `cwd` is not None,
    additionally writes the NN model to `cwd / config.sim.nn_param_file`
    (the deploy-path copy save_checkpoint historically maintained).
    """
```

Body = the existing `_write_winner_artifacts` body operating on `best_individual` directly, plus — inside the NN branch, after the `save_dir / "best_model.json"` write — the deploy-path copy from `save_checkpoint`'s block:

```python
        if deploy_to_cwd and cwd is not None:
            nn_path = Path(cwd) / config.sim.nn_param_file
            write_nn_json(
                weights,
                config.network,
                nn_path,
                input_mask=config.network.input_mask,
                output_param=config.network.output_parameterization,
                normalization=cfg_norm,
            )
```

(use one `cfg_norm = _resolve_config_normalization(config, cwd)` local for both writes, as `save_checkpoint` does today).

(c) **`save_checkpoint`** — replace its inline artifact block (`if best_individual is not None:` … through the non-NN `best_params.json` write, lines ~546-585) with:

```python
    if best_individual is not None:
        write_best_artifacts(best_individual, config, param_specs, save_dir, cwd=cwd, deploy_to_cwd=True)
```

(d) **Islands call site** (~1978) — replace `_write_winner_artifacts(winner=winner, config=config, save_dir=save_dir, param_specs=param_specs, cwd=cwd)` with `write_best_artifacts(winner["X"], config, param_specs, save_dir, cwd=cwd)` and delete the old `_write_winner_artifacts` function.

- [ ] **Step 3.4: Run the regression net**

Run: `uv run pytest tests/test_final_select.py tests/test_train_interrupt.py tests/test_island_model.py tests/test_train_no_validation_promotion.py tests/test_resume_enhancements.py -q`
Expected: all pass (these exercise `save_checkpoint` + the islands winner write).

- [ ] **Step 3.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_final_select.py
git commit -m "refactor(train): shared write_best_artifacts + build_cost_kwargs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `build_training_config_from_toml` extraction

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (`main()` TOML block, lines ~2124-2245)
- Create: `tests/test_train_config_builder.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_train_config_builder.py`:

```python
"""TOML -> TrainingConfig builder (extracted from train.main for CLI reuse)."""

from __future__ import annotations

import pytest
from aerocapture.training.train import build_training_config_from_toml


def test_builds_eqglide_config() -> None:
    cfg, toml_data = build_training_config_from_toml("configs/training/msr_aller_eqglide_train.toml")
    assert cfg.guidance_type == "equilibrium_glide"
    assert cfg.sim.toml_config == "configs/training/msr_aller_eqglide_train.toml"
    assert cfg.optimizer.validation_n_sims > 0
    assert "monte_carlo" in toml_data


def test_builds_nn_config_with_network_fields() -> None:
    cfg, _ = build_training_config_from_toml("configs/training/msr_aller_gru_pso_train.toml")
    assert cfg.guidance_type == "neural_network"
    assert cfg.network.architecture is not None


def test_missing_guidance_type_raises_system_exit(tmp_path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[simulation]\nn_sims = 1\n')
    with pytest.raises(SystemExit):
        build_training_config_from_toml(str(bad))
```

- [ ] **Step 4.2: Run to verify they fail**

Run: `uv run pytest tests/test_train_config_builder.py -q`
Expected: `ImportError: cannot import name 'build_training_config_from_toml'`

- [ ] **Step 4.3: Extract the builder**

In `train.py`, create a module-level function:

```python
def build_training_config_from_toml(toml_path: str) -> tuple[TrainingConfig, dict]:
    """TOML -> TrainingConfig (the TOML-derived part of main()'s bootstrap).

    Applies NO CLI overrides: callers overlay n_gen/n_pop/algorithm/sim_timeout
    on the returned config themselves. Raises SystemExit on invalid configs
    (missing/unknown guidance type, bad [checkpoints], warm-start contract
    violations) -- identical messages to the historical main() behavior.
    """
```

Move into it, **verbatim and in order**, the block currently at `main()` lines ~2124-2245, with exactly these deltas:

1. `_toml_data = load_toml_with_bases(Path(args.toml))` → `_toml_data = load_toml_with_bases(Path(toml_path))`
2. DELETE the three CLI-override lines (`if args.n_gen is not None: …`, `if args.n_pop is not None: …`, `if args.algorithm is not None: …`) — they stay in `main()`.
3. `cfg.sim.toml_config = args.toml` → `cfg.sim.toml_config = toml_path`
4. DELETE `cfg.sim.sim_timeout_secs = args.sim_timeout` — stays in `main()`.
5. The function ends after the warm-start contract validation block (the `if warm_start_active:` block ending ~2245) with `return cfg, _toml_data`.

Then replace the moved region in `main()` with:

```python
    cfg, _toml_data = build_training_config_from_toml(args.toml)

    # CLI overrides -- only when explicitly provided (not None / default False)
    if args.n_gen is not None:
        cfg.optimizer.n_gen = args.n_gen
    if args.n_pop is not None:
        cfg.optimizer.n_pop = args.n_pop
    if args.algorithm is not None:
        cfg.optimizer.algorithm = args.algorithm
    cfg.sim.sim_timeout_secs = args.sim_timeout
```

Everything `main()` does AFTER line ~2245 is untouched and continues to read `cfg` / `_toml_data`.

- [ ] **Step 4.4: Run the tests + behavior spot-check**

Run: `uv run pytest tests/test_train_config_builder.py -q`
Expected: 4 passed (3 + the post-review warm-start wiring test).
Then: `uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml --help > /dev/null && echo OK`
Expected: `OK` (main still imports/parses).

- [ ] **Step 4.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_train_config_builder.py
git commit -m "refactor(train): extract build_training_config_from_toml from main()

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Checkpoint load + patch (both formats)

**Files:**
- Modify: `src/python/aerocapture/training/final_select.py`
- Test: `tests/test_final_select.py` (append)

- [ ] **Step 5.1: Write the failing tests**

Append to `tests/test_final_select.py`:

```python
class TestCheckpointIO:
    def _make_single_algo_ckpt(self, d: Path) -> None:
        meta = {"generation": 7, "best_cost": 1.5, "best_val_cost": 2.5, "cost_history": [], "rng_state": None}
        (d / "checkpoint_g00007.json").write_text(json.dumps(meta))
        np.savez(
            d / "checkpoint_g00007.npz",
            population=np.full((4, 3), 0.5),
            costs=np.array([1.0, 2.0, 3.0, 4.0]),
            best_individual=np.full(3, 0.9),
        )

    def test_load_single_algo(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state

        self._make_single_algo_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        assert state.kind == "single"
        assert state.population.shape == (4, 3)
        assert len(state.known) == 1
        assert state.known[0].provenance == "champion"
        assert state.known[0].val_rms == 2.5

    def test_patch_single_algo(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state, patch_checkpoint

        self._make_single_algo_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        new_best = np.full(3, 0.1)
        patch_checkpoint(state, new_best, new_val_rms=1.25)
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], new_best)
        assert np.array_equal(data["population"], np.full((4, 3), 0.5))  # untouched
        meta = json.loads((tmp_path / "checkpoint_g00007.json").read_text())
        assert meta["best_val_cost"] == 1.25
        assert meta["generation"] == 7  # untouched

    def _make_islands_ckpt(self, d: Path) -> None:
        import pickle

        states = [
            {
                "name": "pso",
                "pop_X": np.full((3, 3), 0.4),
                "pop_F": np.array([[1.0], [2.0], [3.0]]),
                "best_overall_individual": np.full(3, 0.45),
                "best_val_cost": 3.0,
            },
            {
                "name": "ga",
                "pop_X": np.full((3, 3), 0.6),
                "pop_F": np.array([[1.0], [2.0], [3.0]]),
                "best_overall_individual": np.full(3, 0.65),
                "best_val_cost": 2.0,
            },
        ]
        np.savez_compressed(
            d / "checkpoint_g00005.npz",
            version=2,
            generation=5,
            base_mc_seed=42,
            cost_transform="linear",
            island_states=np.array(pickle.dumps(states), dtype=object),
        )

    def test_load_islands(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state

        self._make_islands_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        assert state.kind == "islands"
        assert state.population.shape == (6, 3)  # union of both pops
        assert {k.provenance for k in state.known} == {"pso:champion", "ga:champion"}
        assert state.base_mc_seed == 42

    def test_patch_islands_winning_island(self, tmp_path: Path) -> None:
        import pickle

        from aerocapture.training.final_select import load_selection_state, patch_checkpoint

        self._make_islands_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        new_best = np.full(3, 0.2)
        patch_checkpoint(state, new_best, new_val_rms=0.5, island_name="ga")
        data = np.load(tmp_path / "checkpoint_g00005.npz", allow_pickle=True)
        states = pickle.loads(data["island_states"].item())
        ga = next(s for s in states if s["name"] == "ga")
        pso = next(s for s in states if s["name"] == "pso")
        assert np.array_equal(ga["best_overall_individual"], new_best)
        assert ga["best_val_cost"] == 0.5
        assert pso["best_val_cost"] == 3.0  # untouched
        assert np.array_equal(pso["pop_X"], np.full((3, 3), 0.4))  # untouched

    def test_no_checkpoint_raises(self, tmp_path: Path) -> None:
        import pytest

        from aerocapture.training.final_select import load_selection_state

        with pytest.raises(FileNotFoundError):
            load_selection_state(tmp_path)
```

- [ ] **Step 5.2: Run to verify they fail**

Run: `uv run pytest tests/test_final_select.py::TestCheckpointIO -q`
Expected: `ImportError: cannot import name 'load_selection_state'`

- [ ] **Step 5.3: Implement checkpoint IO**

Append to `src/python/aerocapture/training/final_select.py`:

```python
@dataclass
class SelectionState:
    """Everything the CLI needs from a training dir's latest checkpoint."""

    kind: str  # "single" | "islands"
    save_dir: Path
    population: npt.NDArray[np.float64]  # candidate rows (union across islands for "islands")
    provenances: list[str]
    known: list[KnownCandidate]
    base_mc_seed: int | None  # islands npz records it; single-algo derives from TOML
    json_path: Path | None  # single-algo meta path
    npz_path: Path
    island_of_row: list[str] | None  # islands: island name per candidate row


def _latest_islands_npz(save_dir: Path) -> Path | None:
    """Latest checkpoint_g*.npz that carries the islands v2 marker."""
    for p in sorted(save_dir.glob("checkpoint_g*.npz"), reverse=True):
        try:
            with np.load(p, allow_pickle=True) as data:
                if "island_states" in data and int(data["version"]) == 2:
                    return p
        except (OSError, ValueError, KeyError):
            continue
    return None


def load_selection_state(save_dir: Path) -> SelectionState:
    """Load the latest checkpoint (islands v2 preferred when both formats coexist
    and it is the newest; otherwise the single-algo pair)."""
    import pickle  # noqa: PLC0415

    islands_npz = _latest_islands_npz(save_dir)
    json_files = sorted(save_dir.glob("checkpoint_g*.json"))
    single_json = json_files[-1] if json_files else None

    use_islands = islands_npz is not None and (single_json is None or islands_npz.name >= single_json.with_suffix(".npz").name)
    if use_islands:
        assert islands_npz is not None
        with np.load(islands_npz, allow_pickle=True) as data:
            states = pickle.loads(data["island_states"].item())
            base_mc_seed = int(data["base_mc_seed"])
        rows: list[npt.NDArray[np.float64]] = []
        provs: list[str] = []
        row_islands: list[str] = []
        known: list[KnownCandidate] = []
        for s in states:
            name = str(s["name"])
            if s.get("pop_X") is not None:
                pop_x = np.asarray(s["pop_X"], dtype=np.float64)
                for j in range(pop_x.shape[0]):
                    rows.append(pop_x[j])
                    provs.append(f"{name}:last_gen[{j}]")
                    row_islands.append(name)
            best = s.get("best_overall_individual")
            bvc = float(s.get("best_val_cost", float("inf")))
            if best is not None and np.isfinite(bvc):
                known.append(KnownCandidate(x=np.asarray(best, dtype=np.float64), provenance=f"{name}:champion", val_rms=bvc))
        if not rows:
            raise FileNotFoundError(f"islands checkpoint {islands_npz} has no populations")
        return SelectionState(
            kind="islands",
            save_dir=save_dir,
            population=np.vstack(rows),
            provenances=provs,
            known=known,
            base_mc_seed=base_mc_seed,
            json_path=None,
            npz_path=islands_npz,
            island_of_row=row_islands,
        )

    if single_json is None:
        raise FileNotFoundError(f"no checkpoint_g*.json / islands checkpoint_g*.npz found in {save_dir}")
    npz_path = single_json.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"checkpoint npz missing: {npz_path}")
    meta = json.loads(single_json.read_text())
    with np.load(npz_path) as data:
        population = np.asarray(data["population"], dtype=np.float64)
        best = np.asarray(data["best_individual"], dtype=np.float64) if "best_individual" in data else None
    known = []
    bvc = float(meta.get("best_val_cost", float("inf")))
    if best is not None and np.isfinite(bvc):
        known.append(KnownCandidate(x=best, provenance="champion", val_rms=bvc))
    return SelectionState(
        kind="single",
        save_dir=save_dir,
        population=population,
        provenances=[f"last_gen[{i}]" for i in range(population.shape[0])],
        known=known,
        base_mc_seed=None,
        json_path=single_json,
        npz_path=npz_path,
        island_of_row=None,
    )


def patch_checkpoint(
    state: SelectionState,
    new_best: npt.NDArray[np.float64],
    new_val_rms: float,
    island_name: str | None = None,
) -> None:
    """Persist the re-selected best into the latest checkpoint (atomic rewrite).

    Without this, a later resume restores the old champion and the next
    checkpoint save silently overwrites the re-selected artifacts. Only the
    best fields are touched; populations/costs/RNG state are byte-preserved.
    """
    import pickle  # noqa: PLC0415

    if state.kind == "single":
        assert state.json_path is not None
        with np.load(state.npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["best_individual"] = np.asarray(new_best, dtype=np.float64)
        tmp = state.npz_path.with_name(state.npz_path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(state.npz_path)
        meta = json.loads(state.json_path.read_text())
        meta["best_val_cost"] = float(new_val_rms)
        tmp_json = state.json_path.with_suffix(".tmp.json")
        tmp_json.write_text(json.dumps(meta, indent=2))
        tmp_json.rename(state.json_path)
        return

    assert island_name is not None, "islands patch requires the winning island name"
    with np.load(state.npz_path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
        states = pickle.loads(arrays["island_states"].item())
    for s in states:
        if str(s["name"]) == island_name:
            s["best_overall_individual"] = np.asarray(new_best, dtype=np.float64)
            s["best_val_cost"] = float(new_val_rms)
            break
    else:
        raise ValueError(f"island {island_name!r} not found in checkpoint")
    arrays["island_states"] = np.array(pickle.dumps(states), dtype=object)
    tmp = state.npz_path.with_name(state.npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(state.npz_path)
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_final_select.py -q`
Expected: all pass (Task 1's 11 + Task 2's 2 + Task 3's 2 + 5 new = 20).

- [ ] **Step 5.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/final_select.py tests/test_final_select.py
git commit -m "feat(train): final-select checkpoint load/patch for both formats

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Inline hook — single-algorithm

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (end of training loop, ~1546-1569)

- [ ] **Step 6.1: Insert the selection block**

In `train()`, directly after the line `cost_history.extend(gen_best_costs)` (~1546) and BEFORE the `# Always save a final checkpoint` block, insert:

```python
            # End-of-training final selection (spec 2026-06-10-final-selection):
            # re-rank the last generation + champion on the validation pool;
            # deploy the winner only on strict val-RMS improvement. The final-
            # eval pool stays report-only.
            selection_promoted = False
            if val_seeds is not None:
                from aerocapture.training.final_select import (  # noqa: PLC0415
                    KnownCandidate,
                    format_selection_summary,
                    select_final_individual,
                    write_final_selection_json,
                )

                known = []
                if best_overall_individual is not None and np.isfinite(best_val_cost):
                    known.append(KnownCandidate(x=best_overall_individual, provenance="champion", val_rms=float(best_val_cost)))
                try:
                    sel = select_final_individual(
                        problem,
                        X,
                        [f"last_gen[{i}]" for i in range(X.shape[0])],
                        known,
                        val_seeds,
                    )
                except ValueError:
                    # Pathological all-inf run with no champion: nothing to select.
                    sel = None
                if sel is not None:
                    if sel.promoted:
                        best_overall_individual = sel.individual.copy()
                        best_val_cost = sel.val_rms
                        assert sel.winner_index is not None
                        # Training-cost-at-promotion semantics (resume-incomparability rule):
                        # the winner's training cost under the final seed list.
                        best_overall_cost = float(costs[sel.winner_index])
                        selection_promoted = True
                    write_final_selection_json(save_dir, sel, len(val_seeds))
                    if verbose:
                        print(format_selection_summary(sel))
```

Then change the final-checkpoint condition two lines below from:

```python
            if last_gen % checkpoint_interval != 0:
```

to:

```python
            if last_gen % checkpoint_interval != 0 or selection_promoted:
```

(Notes for the implementer: `X` and `costs` are guaranteed defined here — the existing final-checkpoint block already uses them unconditionally. The block sits inside the `try:`, so a Ctrl+C during selection falls through to the existing interrupt handler with the pre-selection champion — by design; the CLI covers interrupted runs.)

- [ ] **Step 6.2: Regression net**

Run: `uv run pytest tests/test_train_interrupt.py tests/test_train_no_validation_promotion.py tests/test_resume_enhancements.py tests/test_final_select.py -q`
Expected: all pass. (`validation_n_sims = 0` paths have `val_seeds = None`, so the new block is skipped and the no-validation fallback behavior is bit-identical — the no-validation tests prove it.)

- [ ] **Step 6.3: Lint gate + commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(train): single-algo end-of-training final selection on the validation pool

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Inline hook — islands

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (`_train_islands` final-eval region, ~1936-1997)

- [ ] **Step 7.1: Insert the selection block and rewire the winner**

Replace the region from the comment `# Final eval + winner selection.` (~1936) down to the `_write_winner_artifacts(`…`)` call (now `write_best_artifacts(...)` after Task 3) with:

```python
    # Validation-pool final selection across islands (spec 2026-06-10-final-selection):
    # union of last-gen pops + champions decides the ARTIFACTS; final_eval below
    # is report-only (winner's fresh final-eval rms is the quoted number).
    selection = None
    if island_model.validation_seeds:
        from aerocapture.training.final_select import (  # noqa: PLC0415
            KnownCandidate,
            format_selection_summary,
            select_final_individual,
            write_final_selection_json,
        )

        known = [
            KnownCandidate(
                x=np.asarray(isl.best_overall_individual, dtype=np.float64),
                provenance=f"{isl.name}:champion",
                val_rms=float(isl.best_val_cost),
            )
            for isl in island_model.islands
            if isl.best_overall_individual is not None and np.isfinite(isl.best_val_cost)
        ]
        cand_rows: list[npt.NDArray[np.float64]] = []
        cand_prov: list[str] = []
        cand_island: list[str] = []
        for isl in island_model.islands:
            pop = isl.algorithm.pop
            if pop is None:
                continue
            pop_x = pop.get("X")
            for j in range(pop_x.shape[0]):
                cand_rows.append(np.asarray(pop_x[j], dtype=np.float64))
                cand_prov.append(f"{isl.name}:last_gen[{j}]")
                cand_island.append(isl.name)
        if known or cand_rows:
            try:
                selection = select_final_individual(
                    problem,
                    np.vstack(cand_rows) if cand_rows else np.empty((0, len(param_specs))),
                    cand_prov,
                    known,
                    island_model.validation_seeds,
                )
            except ValueError:
                # Pathological all-inf run with no champions: fall through to the
                # legacy final_eval / stale-removal path below.
                selection = None
            if selection is not None:
                write_final_selection_json(save_dir, selection, len(island_model.validation_seeds))
                if verbose:
                    print(format_selection_summary(selection))

    # Final eval (report-only when selection ran).
    results = island_model.final_eval()
    if selection is None and not results:
        # validation off AND no island promoted -- legacy stale-artifact removal path.
        if verbose:
            print("  No island had a validated best — skipping final-eval / artifact write.")
        for stale in (
            save_dir / "best_model.json",
            save_dir / "best_params.json",
            Path(cwd or ".") / config.sim.nn_param_file if config.guidance_type == "neural_network" else None,
        ):
            if stale is not None and stale.exists():
                stale.unlink()
                if verbose:
                    print(f"  Removed stale {stale}")
        logger.close()
        return {
            "best_cost": float("inf"),
            "best_individual": None,
            "cost_history": [],
            "interrupted": interrupted,
            "corridor_acc": None,
            "param_specs": param_specs,
            "winner": None,
            "results": [],
            "migration_log": island_model.migration_log,
        }

    if selection is not None:
        # Winner = validation-pool selection. Quote its UNBIASED final-eval rms:
        # reuse the matching champion record when the incumbent won, else run
        # one fresh single-candidate final-eval for a promoted individual.
        match = next((r for r in results if r["island"] + ":champion" == selection.provenance), None)
        if match is not None:
            final_rms = float(match["rms"])
            win_island = str(match["island"])
            capture = float(match["capture_rate"])
        else:
            fe_costs = problem.evaluate_individual_per_seed(selection.individual, island_model.final_eval_seeds)
            final_rms = float(np.sqrt(np.mean(fe_costs**2)))
            win_island = selection.provenance.split(":", 1)[0]
            capture = _capture_rate_from_costs(fe_costs)
        winner = {
            "island": win_island,
            "X": selection.individual.copy(),
            "rms": final_rms,
            "val_rms": float(selection.val_rms),
            "capture_rate": capture,
            "n_sims": len(island_model.final_eval_seeds),
            "selection_provenance": selection.provenance,
        }
    else:
        winner = results[0]

    if verbose:
        gap, overfit = val_generalization_gap(winner["val_rms"], winner["rms"])
        gap_detail = ""
        if winner["val_rms"] < float("inf"):
            gap_detail = f" (val_rms={winner['val_rms']:.4e}, gap={gap:+.1%}{'  [WARN: overfit to validation?]' if overfit else ''})"
        print(
            f"  Winner: {winner['island']} rms={winner['rms']:.4e} cap={winner['capture_rate']:.0%}{gap_detail}",
        )

    write_best_artifacts(winner["X"], config, param_specs, save_dir, cwd=cwd)
```

The tail of `_train_islands` (the `logger.close()` + success return dict, ~1986-1997) is unchanged and now consumes the new `winner` dict (same keys plus `selection_provenance`).

Add the small helper near `_write_winner_artifacts`'s old location (module level):

```python
def _capture_rate_from_costs(costs: npt.NDArray[np.float64]) -> float:
    """Fraction of sims with cost below the hyperbolic/crash floor (mirrors island_model._capture_rate)."""
    from aerocapture.training.island_model import _capture_rate  # noqa: PLC0415

    return _capture_rate(np.asarray(costs))
```

(If `_capture_rate` in `island_model.py` is private but importable, use it directly at the call site instead of this wrapper — implementer's choice, but do not duplicate its threshold logic.)

- [ ] **Step 7.2: Islands regression net**

Run: `uv run pytest tests/test_island_model.py tests/test_train_no_validation_promotion.py tests/test_final_select.py -q`
Expected: all pass. The no-validation islands path (validation_seeds empty) takes `selection is None` → legacy behavior, which those tests pin.

- [ ] **Step 7.3: Lint gate + commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "feat(train): islands artifact winner from validation-pool selection; final_eval report-only

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Standalone CLI

**Files:**
- Modify: `src/python/aerocapture/training/final_select.py` (CLI + orchestrator)
- Test: `tests/test_final_select.py` (append)

- [ ] **Step 8.1: Write the failing tests**

Append to `tests/test_final_select.py`:

```python
class TestRunFinalSelect:
    def _setup_dir(self, d: Path) -> None:
        meta = {"generation": 7, "best_cost": 1.5, "best_val_cost": _rms(np.full(2, 0.9), [1000001, 1000002]), "cost_history": [], "rng_state": None}
        (d / "checkpoint_g00007.json").write_text(json.dumps(meta))
        np.savez(
            d / "checkpoint_g00007.npz",
            population=np.vstack([np.full(2, 0.2), np.full(2, 0.8)]),
            costs=np.array([1.0, 2.0]),
            best_individual=np.full(2, 0.9),
        )

    def test_round_trip_promotes_and_patches(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.final_select import run_final_select
        from aerocapture.training.param_spaces import ParamSpec

        self._setup_dir(tmp_path)
        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5), ParamSpec(name="b", p_min=0.0, p_max=1.0, default=0.5)]

        sel = run_final_select(
            training_dir=tmp_path,
            config=cfg,
            param_specs=specs,
            problem=_MockProblem(),
            val_seeds=[1000001, 1000002],
            patch=True,
        )
        assert sel.promoted and sel.provenance == "last_gen[0]"
        # artifacts rewritten
        params = json.loads((tmp_path / "best_params.json").read_text())
        assert params["a"] == 0.2
        # checkpoint patched
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], np.full(2, 0.2))
        # sidecar present
        assert (tmp_path / "final_selection.json").exists()

    def test_no_patch_leaves_checkpoint(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.final_select import run_final_select
        from aerocapture.training.param_spaces import ParamSpec

        self._setup_dir(tmp_path)
        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5), ParamSpec(name="b", p_min=0.0, p_max=1.0, default=0.5)]
        run_final_select(training_dir=tmp_path, config=cfg, param_specs=specs, problem=_MockProblem(), val_seeds=[1000001, 1000002], patch=False)
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], np.full(2, 0.9))  # untouched
```

- [ ] **Step 8.2: Run to verify they fail**

Run: `uv run pytest tests/test_final_select.py::TestRunFinalSelect -q`
Expected: `ImportError: cannot import name 'run_final_select'`

- [ ] **Step 8.3: Implement orchestrator + CLI**

Append to `final_select.py`:

```python
def run_final_select(
    training_dir: Path,
    config: Any,  # TrainingConfig (typed Any to avoid heavy import at module load)
    param_specs: list[Any],
    problem: _PerSeedEvaluator,
    val_seeds: list[int],
    patch: bool = True,
) -> SelectionResult:
    """Load the latest checkpoint in training_dir, run the selection rule,
    rewrite best artifacts (+ sidecar), and (optionally) patch the checkpoint."""
    from aerocapture.training.train import write_best_artifacts  # noqa: PLC0415

    state = load_selection_state(training_dir)
    sel = select_final_individual(problem, state.population, state.provenances, state.known, val_seeds)
    write_best_artifacts(sel.individual, config, param_specs, training_dir, cwd=None)
    write_final_selection_json(training_dir, sel, len(val_seeds))
    if patch:
        island_name: str | None = None
        if state.kind == "islands":
            if sel.winner_index is not None:
                assert state.island_of_row is not None
                island_name = state.island_of_row[sel.winner_index]
            else:
                island_name = sel.provenance.split(":", 1)[0]
        patch_checkpoint(state, sel.individual, sel.val_rms, island_name=island_name)
    else:
        print("  --no-checkpoint-patch: checkpoint untouched; a later resume will revert these artifacts at its next checkpoint save.")
    print(format_selection_summary(sel))
    return sel


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Re-run end-of-training final selection on an existing training directory.")
    parser.add_argument("training_dir", type=str, help="Directory containing checkpoint_g*.{json,npz} and best artifacts")
    parser.add_argument("--toml", type=str, required=True, help="Training TOML the run used (base inheritance resolved)")
    parser.add_argument("--no-checkpoint-patch", action="store_true", help="Do not write the re-selected best back into the checkpoint")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Per-sim wall-clock timeout (seconds)")
    args = parser.parse_args()

    from aerocapture.training.problem import AerocaptureProblem  # noqa: PLC0415
    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, make_reserved_seeds  # noqa: PLC0415
    from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: PLC0415
    from aerocapture.training.train import _setup_param_specs, build_cost_kwargs, build_training_config_from_toml  # noqa: PLC0415
    from aerocapture.training.warm_start import load_warm_start_bounds  # noqa: PLC0415

    training_dir = Path(args.training_dir)
    config, toml_data = build_training_config_from_toml(args.toml)
    config.sim.sim_timeout_secs = args.sim_timeout

    if config.optimizer.validation_n_sims <= 0:
        raise SystemExit("ERROR: [optimizer] validation_n_sims is 0 -- no validation pool exists to select on. Set validation_n_sims > 0 in the TOML.")

    param_specs, _ = _setup_param_specs(config, toml_data, verbose=False)
    bounds = load_warm_start_bounds(training_dir)
    if bounds is not None:
        # Overlay the EXACT weight-slab bounds the checkpoint population was
        # encoded under (adaptive warm-start bounds). Decoding under rebuilt
        # Xavier bounds would silently corrupt the weights.
        n_weights = len(bounds)
        if n_weights > len(param_specs):
            raise SystemExit(f"ERROR: warm_start_bounds.json has {n_weights} specs but config yields {len(param_specs)} params")
        param_specs = list(bounds) + param_specs[n_weights:]
        print(f"  Overlaid {n_weights} weight-spec bounds from warm_start_bounds.json")

    state = load_selection_state(training_dir)
    base_mc_seed = state.base_mc_seed
    if base_mc_seed is None:
        mc_seed_val = load_toml_with_bases(Path(args.toml)).get("monte_carlo", {}).get("seed")
        base_mc_seed = int(mc_seed_val) if mc_seed_val is not None else 42
    elif toml_data.get("monte_carlo", {}).get("seed") is not None and int(toml_data["monte_carlo"]["seed"]) != base_mc_seed:
        raise SystemExit(
            f"ERROR: checkpoint base_mc_seed={base_mc_seed} != TOML monte_carlo.seed={toml_data['monte_carlo']['seed']} -- wrong TOML for this training dir?"
        )
    val_seeds = make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, config.optimizer.validation_n_sims)

    if state.population.shape[1] != len(param_specs):
        raise SystemExit(
            f"ERROR: checkpoint chromosome width {state.population.shape[1]} != config param count {len(param_specs)} -- wrong TOML for this training dir?"
        )

    problem = AerocaptureProblem(
        param_specs=param_specs,
        toml_path=str(Path(args.toml).resolve()),
        seeds=[base_mc_seed],
        cost_kwargs=build_cost_kwargs(toml_data),
        scheme=config.guidance_type,
        sim_timeout=config.sim.sim_timeout_secs,
        nn_config=config.network if config.guidance_type == "neural_network" else None,
    )

    n_sims_estimate = state.population.shape[0] * len(val_seeds)
    print(f"  Final selection over {state.population.shape[0]} candidates x {len(val_seeds)} validation seeds (<= {n_sims_estimate} sims)...")
    run_final_select(
        training_dir=training_dir,
        config=config,
        param_specs=param_specs,
        problem=problem,
        val_seeds=val_seeds,
        patch=not args.no_checkpoint_patch,
    )


if __name__ == "__main__":
    main()
```

(Note: `run_final_select` calls `load_selection_state` again internally — keep it that way for the simple injectable-test seam; the duplicate load is two small file reads.)

- [ ] **Step 8.4: Run tests**

Run: `uv run pytest tests/test_final_select.py -q`
Expected: all pass (24 total) (post-review fixes).

- [ ] **Step 8.5: Lint gate + commit**

```bash
git add src/python/aerocapture/training/final_select.py tests/test_final_select.py
git commit -m "feat(train): final_select standalone CLI (retro re-selection + checkpoint patch)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Verification

**Files:** none (verification only)

- [ ] **Step 9.1: Lint + full pure-python net**

Run: `./lint_code.sh`
Then: `uv run pytest tests/test_final_select.py tests/test_train_config_builder.py tests/test_train_interrupt.py tests/test_island_model.py tests/test_train_no_validation_promotion.py tests/test_resume_enhancements.py tests/test_training_config.py tests/test_qpso.py tests/test_optimizer.py tests/test_warm_start_optimizer_seeding.py -q`
Expected: clean lint; all tests pass.

- [ ] **Step 9.2: Single-algo E2E smoke (Rust)**

```bash
./build.sh
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml \
    --algorithm qpso --n-gen 3 --n-pop 6 --no-tui --skip-report --final-n-sims 50 \
    --output-dir /tmp/fsel_smoke --from-scratch
```

Expected: training completes; stdout contains a `Final selection:` line; `/tmp/fsel_smoke/final_selection.json` exists with `n_candidates == 6`.

- [ ] **Step 9.3: CLI re-run smoke (idempotence)**

```bash
uv run python -m aerocapture.training.final_select /tmp/fsel_smoke --toml configs/training/msr_aller_eqglide_train.toml
```

Expected: runs the selection again; winner is `champion` with `promoted` false OR re-finds the same individual (the deployed best already won — `best_params.json` content unchanged). Paste the summary line.

- [ ] **Step 9.4: Islands E2E smoke**

```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_eqglide_train.toml \
    --algorithm islands --n-gen 2 --n-pop 6 --no-tui --skip-report --final-n-sims 50 \
    --output-dir /tmp/fsel_islands_smoke --from-scratch
```

Expected: completes; stdout contains `Final selection:` and the `Winner:` line; `/tmp/fsel_islands_smoke/final_selection.json` has provenances prefixed `pso:`/`ga:`/`de:`. Then CLI re-run:

```bash
uv run python -m aerocapture.training.final_select /tmp/fsel_islands_smoke --toml configs/training/msr_aller_eqglide_train.toml
rm -rf /tmp/fsel_smoke /tmp/fsel_islands_smoke
```

Expected: islands checkpoint loaded, selection runs, exits cleanly.

- [ ] **Step 9.5: Commit any fix-ups**

Only if 9.1-9.4 required changes: stage exactly the touched files, commit `fix(train): final-selection verification fix-ups` + the Co-Authored-By trailer.

---

### Task 10: Documentation sync + final commit (smart-commit)

- [ ] **Step 10.1: Invoke the `smart-commit` skill**

Invoke the `smart-commit` skill, telling it to take the **whole git branch** (`feature/qpso-optimizer`, all commits since `main`) into account — both the QPSO work and this final-selection feature. CLAUDE.md needs: the final-selection behavior in the `train.py` bullet (validation-pool re-rank, final-eval now report-only for islands), the new `final_select.py` module bullet, the `final_selection.json` artifact, the CLI invocation, and test-coverage notes. README: a line under Training features + the CLI in the commands section.
