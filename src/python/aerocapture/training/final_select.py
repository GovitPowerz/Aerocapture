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
    """A pre-scored candidate (champion); never re-simulated.

    `provenance` is structured, not just display text: the islands call sites
    use the grammar "<island>:champion" / "<island>:last_gen[i]" and parse the
    island name back out via split(":", 1). Island names must not contain ":".
    """

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
    if candidates.shape[0] != len(provenances):
        raise ValueError(f"candidates/provenances length mismatch: {candidates.shape[0]} != {len(provenances)}")
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


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


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
        except (OSError, ValueError, KeyError):  # fmt: skip
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
        tmp = state.npz_path.with_name(".tmp_" + state.npz_path.name)
        np.savez(tmp, **arrays)
        tmp.rename(state.npz_path)
        meta = json.loads(state.json_path.read_text())
        meta["best_val_cost"] = float(new_val_rms)
        tmp_json = state.json_path.with_name(".tmp_" + state.json_path.name)
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
    tmp = state.npz_path.with_name(".tmp_" + state.npz_path.name)
    np.savez_compressed(tmp, **arrays)
    tmp.rename(state.npz_path)
