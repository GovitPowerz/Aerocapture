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
