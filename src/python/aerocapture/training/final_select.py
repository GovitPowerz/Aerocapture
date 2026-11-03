"""End-of-training final selection (spec: docs/design/2026-06-10-final-selection-design.md).

Re-ranks the last generation's population plus the running champion(s) on the
reserved VALIDATION pool and deploys the winner only on strict val-RMS
improvement. Selection happens on the validation pool by design: the
final-eval pool stays a clean test set that only ever evaluates the single
deployed winner (no min-of-N selection bias on reported numbers).

Three call sites share `select_final_individual`:
- `SingleAlgoTrainer.finalize` (known = [champion]),
- the islands trainer (known = all island champions),
- the standalone CLI (`python -m aerocapture.training.final_select`), which
  re-applies the rule to an existing training directory from its latest
  checkpoint and patches the checkpoint so resume cannot revert the artifacts.

The CLI also offers the tail-aware criterion of #234 (`--criterion cvar95`,
`select_tail_individual`): the ADR-0005 gate first, then capture count and
CVaR95 of the captured correction DV on the validation pool or on the larger
reserved tail-select pool. It is a retro study tool: the trainer's own
finalize always applies the RMS rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from aerocapture.training.cell_eval import FR_DV_TOTAL, is_captured
from aerocapture.training.champion import Champion, revalidate
from aerocapture.training.champion import KnownCandidate as KnownCandidate  # re-export: the rule's public candidate type
from aerocapture.training.evaluate import _SupportsPerSeedEval, constraint_violation_rates, is_feasible
from aerocapture.training.sizing_tail import cvar, dv_tail


class _PerSeedEvaluator(Protocol):
    def evaluate_population_records_per_seed(self, X: npt.NDArray[np.float64], seeds: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


class _RetroEvaluator(_PerSeedEvaluator, _SupportsPerSeedEval, Protocol):
    """What the retro CLI needs on top of the rule: re-validating a checkpoint champion."""

    cost_kwargs: dict[str, Any]


@dataclass
class SelectionResult:
    individual: npt.NDArray[np.float64]
    val_rms: float
    provenance: str
    promoted: bool  # a fresh candidate strictly beat the best known
    winner_index: int | None  # index into the candidates matrix; None when a known candidate won
    n_candidates: int  # fresh rows offered (pre-dedup)
    n_deduped: int  # fresh rows actually simulated
    incumbent_val_rms: float | None = None  # best known (champion) val RMS; None when no champion existed
    candidate_rms: list[dict[str, Any]] = field(default_factory=list)  # [{"provenance", "val_rms", "feasible"?, "tail"?}]
    winner_feasible: bool = True  # False only on the all-infeasible fallback (the RMS rule's: only without a champion)
    criterion: str = "rms"  # "rms" (the rule above) | "cvar95" (`select_tail_individual`)
    tail_pool: dict[str, Any] | None = None  # cvar95 only: {"name", "offset", "n"} of the pool the tail statistic was scored on
    winner_tail: dict[str, Any] | None = None  # cvar95 only: the winner's tail stats on that pool


def _dedup_fresh(candidates: npt.NDArray[np.float64], known: list[KnownCandidate]) -> list[int]:
    """Indices of the fresh rows to simulate: not byte-identical to a known row or to an earlier fresh row."""
    seen: set[bytes] = {np.ascontiguousarray(k.x).tobytes() for k in known}
    kept: list[int] = []
    for i in range(candidates.shape[0]):
        key = np.ascontiguousarray(candidates[i]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)
    return kept


def select_final_individual(
    problem: _PerSeedEvaluator,
    candidates: npt.NDArray[np.float64],
    provenances: list[str],
    known: list[KnownCandidate],
    val_seeds: list[int],
    max_violation_rate: float = 0.0,
    cost_kwargs: dict[str, Any] | None = None,
) -> SelectionResult:
    """The selection rule (spec section 3) over fresh candidates + known champions.

    Winner = lowest val RMS over {FEASIBLE finite fresh candidates} U {known}.
    A fresh candidate displaces the incumbent (lowest-val-RMS known) only with
    a STRICTLY lower val RMS -- ties keep the incumbent, matching the
    in-training validation gate. Fresh rows identical to a known row or to an
    earlier fresh row are deduplicated (never re-simulated).

    Feasibility (ADR-0005): when `cost_kwargs` carries constraint
    limits, a fresh candidate whose validation-pool violation rate exceeds
    `max_violation_rate` on any constraint cannot win. Champions are trusted
    as-validated (their promotion already passed the gate; legacy champions
    predate it -- re-run the retro CLI to re-select if in doubt). If there is
    no champion and every fresh candidate is infeasible, the best-RMS
    infeasible one deploys with a loud warning and `winner_feasible = False`.
    """
    if candidates.shape[0] != len(provenances):
        raise ValueError(f"candidates/provenances length mismatch: {candidates.shape[0]} != {len(provenances)}")
    records: list[dict[str, Any]] = [{"provenance": k.provenance, "val_rms": k.val_rms} for k in known]

    incumbent: KnownCandidate | None = None
    for k in known:
        if incumbent is None or k.val_rms < incumbent.val_rms:
            incumbent = k

    # Pass 1: dedup (identical to a known row or an earlier fresh row).
    kept = _dedup_fresh(candidates, known)
    n_deduped = len(kept)

    # Pass 2: ONE batched evaluation of all kept candidates on the validation
    # pool (previously up to n_pop sequential single-candidate calls, each
    # rebuilding SimData).
    best_fresh_rms = float("inf")
    best_fresh_idx: int | None = None
    best_infeasible_rms = float("inf")
    best_infeasible_idx: int | None = None
    if kept:
        costs, grid = problem.evaluate_population_records_per_seed(candidates[kept], val_seeds)
        rms_all = np.sqrt(np.mean(np.asarray(costs, dtype=np.float64) ** 2, axis=1))
        for row, (i, rms_val) in enumerate(zip(kept, rms_all, strict=True)):
            rms = float(rms_val)
            if not np.isfinite(rms):
                records.append({"provenance": provenances[i], "val_rms": None})
                continue
            rates = constraint_violation_rates(np.asarray(grid[row]), cost_kwargs)
            feasible = is_feasible(rates, max_violation_rate)
            records.append({"provenance": provenances[i], "val_rms": rms, "feasible": feasible, "violation_rates": rates})
            if feasible and rms < best_fresh_rms:
                best_fresh_rms = rms
                best_fresh_idx = i
            elif not feasible and rms < best_infeasible_rms:
                best_infeasible_rms = rms
                best_infeasible_idx = i

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
            incumbent_val_rms=incumbent.val_rms if incumbent is not None else None,
            candidate_rms=records,
        )
    if incumbent is None:
        if best_infeasible_idx is not None:
            print(
                f"WARNING: final selection found NO feasible candidate (ceiling {max_violation_rate:.3%}) "
                f"and no champion - deploying the best-RMS INFEASIBLE candidate {provenances[best_infeasible_idx]!r} "
                f"(rms {best_infeasible_rms:.4g}). Inspect final_selection.json violation_rates before using it."
            )
            return SelectionResult(
                individual=candidates[best_infeasible_idx].copy(),
                val_rms=best_infeasible_rms,
                provenance=provenances[best_infeasible_idx],
                promoted=True,
                winner_index=best_infeasible_idx,
                n_candidates=int(candidates.shape[0]),
                n_deduped=n_deduped,
                incumbent_val_rms=None,
                candidate_rms=records,
                winner_feasible=False,
            )
        raise ValueError("final selection: no finite candidate and no known champion")
    return SelectionResult(
        individual=incumbent.x.copy(),
        val_rms=incumbent.val_rms,
        provenance=incumbent.provenance,
        promoted=False,
        winner_index=None,
        n_candidates=int(candidates.shape[0]),
        n_deduped=n_deduped,
        incumbent_val_rms=incumbent.val_rms,
        candidate_rms=records,
    )


def select_tail_individual(
    problem: _PerSeedEvaluator,
    candidates: npt.NDArray[np.float64],
    provenances: list[str],
    known: list[KnownCandidate],
    val_seeds: list[int],
    tail_seeds: list[int],
    tail_pool: dict[str, Any],
    max_violation_rate: float = 0.0,
    cost_kwargs: dict[str, Any] | None = None,
) -> SelectionResult:
    """The tail-aware criterion of #234, lexicographic over fresh candidates + champions.

    (1) ADR-0005 on the validation pool: a candidate infeasible there cannot win.
    (2) The survivors are scored on `tail_seeds` (the validation pool itself, or the
    reserved tail-select pool; one batched evaluation, skipped when the two pools
    are the same list). (3) Most scenarios captured on the tail pool, then (4) lowest
    CVaR95 of the captured correction DV, then (5) lowest validation RMS. The tail
    pool's violation rates are recorded on every survivor's `tail` block but do not
    exclude: the gate is the validation pool's at its size (ADR-0005: the ceiling is
    a sample-size question; at ceiling 0 a five-times larger pool would reject every
    policy with a 0.1% exceedance rate, #234's first run). Champions are ordinary
    rows here (their recorded RMS pre-empts nothing); dedup is the RMS rule's. With
    no feasible candidate at all the best-RMS infeasible row deploys with a loud
    warning and `winner_feasible = False`, as the RMS rule does without a champion.
    """
    if candidates.shape[0] != len(provenances):
        raise ValueError(f"candidates/provenances length mismatch: {candidates.shape[0]} != {len(provenances)}")
    kept = _dedup_fresh(candidates, known)
    rows: list[npt.NDArray[np.float64]] = [np.asarray(k.x, dtype=np.float64) for k in known] + [candidates[i] for i in kept]
    provs: list[str] = [k.provenance for k in known] + [provenances[i] for i in kept]
    fresh_index: list[int | None] = [None] * len(known) + list(kept)  # index into `candidates`, None for a champion row
    n_deduped = len(kept)
    if not rows:
        raise ValueError("final selection: no candidate and no known champion")
    X = np.vstack(rows)

    # Stage 1: the gate on the validation pool (and the RMS tie-break).
    costs, grid = problem.evaluate_population_records_per_seed(X, val_seeds)
    rms_all = np.sqrt(np.mean(np.asarray(costs, dtype=np.float64) ** 2, axis=1))
    records: list[dict[str, Any]] = []
    survivors: list[int] = []
    for r in range(X.shape[0]):
        rms = float(rms_all[r])
        if not np.isfinite(rms):
            records.append({"provenance": provs[r], "val_rms": None})
            continue
        rates = constraint_violation_rates(np.asarray(grid[r]), cost_kwargs)
        feasible = is_feasible(rates, max_violation_rate)
        records.append({"provenance": provs[r], "val_rms": rms, "feasible": feasible, "violation_rates": rates})
        if feasible:
            survivors.append(r)

    # Stage 2: the tail statistic of the survivors on the tail pool.
    if survivors:
        if list(tail_seeds) == list(val_seeds):
            tail_grid = grid[survivors]
        else:
            _, tail_grid = problem.evaluate_population_records_per_seed(X[survivors], tail_seeds)
    ranked: list[tuple[tuple[int, float, float], int]] = []
    for row, r in enumerate(survivors):
        recs = np.asarray(tail_grid[row])
        cap = is_captured(recs)
        dvc = recs[cap, FR_DV_TOTAL]
        n_captured = int(cap.sum())
        rates = constraint_violation_rates(recs, cost_kwargs)  # the tail pool's: reported, not a gate
        records[r]["tail"] = {"n": int(recs.shape[0]), "n_captured": n_captured, **dv_tail(dvc), "violation_rates": rates}
        # Unrounded (the record's two decimals would tie close CVaRs into the RMS break); no capture ranks last.
        ranked.append(((-n_captured, cvar(dvc, 0.95) if len(dvc) else float("inf"), float(rms_all[r])), r))

    if ranked:
        _, w = min(ranked)
        winner_feasible = True
    else:
        finite = [r for r in range(X.shape[0]) if records[r]["val_rms"] is not None]
        if not finite:
            raise ValueError("final selection: no finite candidate")
        w = min(finite, key=lambda r: float(rms_all[r]))
        winner_feasible = False
        print(
            f"WARNING: tail-aware final selection found NO feasible candidate (ceiling {max_violation_rate:.3%}) "
            f"- deploying the best-RMS INFEASIBLE candidate {provs[w]!r} (rms {float(rms_all[w]):.4g})."
        )
    incumbent_rms = min((k.val_rms for k in known if np.isfinite(k.val_rms)), default=None)
    return SelectionResult(
        individual=X[w].copy(),
        val_rms=float(rms_all[w]),
        provenance=provs[w],
        promoted=not provs[w].endswith("champion"),  # a champion row is the incumbent kept, trusted or not
        winner_index=fresh_index[w],
        n_candidates=int(candidates.shape[0]),
        n_deduped=n_deduped,
        incumbent_val_rms=incumbent_rms,
        candidate_rms=records,
        winner_feasible=winner_feasible,
        criterion="cvar95",
        tail_pool=dict(tail_pool),
        winner_tail=records[w].get("tail"),
    )


def write_final_selection_json(save_dir: Path, result: SelectionResult, n_val_seeds: int) -> None:
    winner: dict[str, Any] = {
        "provenance": result.provenance,
        "val_rms": result.val_rms,
        "promoted": result.promoted,
        "feasible": result.winner_feasible,
    }
    payload: dict[str, Any] = {
        "winner": winner,
        "champion_val_rms": result.incumbent_val_rms,
        "n_candidates": result.n_candidates,
        "n_deduped": result.n_deduped,
        "validation_n_sims": n_val_seeds,
        "candidate_rms": result.candidate_rms,
    }
    if result.criterion != "rms":
        payload["criterion"] = result.criterion
        payload["tail_pool"] = result.tail_pool
        winner["tail"] = result.winner_tail
    with open(save_dir / "final_selection.json", "w") as fp:
        json.dump(payload, fp, indent=2)


def format_selection_summary(result: SelectionResult) -> str:
    finite = [e["val_rms"] for e in result.candidate_rms if e["val_rms"] is not None]
    spread = f", candidate val-rms [{min(finite):.4e}, {max(finite):.4e}]" if finite else ""
    verdict = "PROMOTED" if result.promoted else "champion kept"
    tail = ""
    if result.winner_tail is not None and result.tail_pool is not None:
        pool, wt = result.tail_pool, result.winner_tail
        tail = f"; {result.criterion} on {pool['name']} pool n={pool['n']}: captured {wt['n_captured']}, cvar95={wt.get('dv_cvar95')}"
    return (
        f"  Final selection: {verdict} -> {result.provenance} "
        f"(val_rms={result.val_rms:.4e}; {result.n_deduped}/{result.n_candidates} fresh candidates simulated{spread}{tail})"
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
    champions: dict[str, Champion]  # island name -> its champion record ("" for single); only islands holding one
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
        champions: dict[str, Champion] = {}
        for s in states:
            name = str(s["name"])
            if s.get("pop_X") is not None:
                pop_x = np.asarray(s["pop_X"], dtype=np.float64)
                for j in range(pop_x.shape[0]):
                    rows.append(pop_x[j])
                    provs.append(f"{name}:last_gen[{j}]")
                    row_islands.append(name)
            champ = Champion.from_island_state(s)
            if champ.individual is not None:
                champions[name] = champ
        if not rows:
            raise FileNotFoundError(f"islands checkpoint {islands_npz} has no populations")
        return SelectionState(
            kind="islands",
            save_dir=save_dir,
            population=np.vstack(rows),
            provenances=provs,
            champions=champions,
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
    champ = Champion.from_single(meta, best)
    champions = {"": champ} if champ.individual is not None else {}
    return SelectionState(
        kind="single",
        save_dir=save_dir,
        population=population,
        provenances=[f"last_gen[{i}]" for i in range(population.shape[0])],
        champions=champions,
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
    *,
    feasible: bool = True,
) -> None:
    """Persist the re-selected best into the latest checkpoint (atomic rewrite).

    Without this, a later resume restores the old champion and the next
    checkpoint save silently overwrites the re-selected artifacts. The champion
    record is loaded from the file, `adopt_selection` applied (the recorded
    training cost is kept) and written back under its wire keys; populations /
    costs / RNG state are byte-preserved.
    """
    import pickle  # noqa: PLC0415

    if state.kind == "single":
        assert state.json_path is not None
        meta = json.loads(state.json_path.read_text())
        with np.load(state.npz_path) as data:
            arrays = {k: data[k] for k in data.files}
        champ = Champion.from_single(meta, arrays.get("best_individual"))
        champ.adopt_selection(np.asarray(new_best, dtype=np.float64), new_val_rms, None, feasible=feasible)
        arrays["best_individual"] = champ.individual
        tmp = state.npz_path.with_name(".tmp_" + state.npz_path.name)
        np.savez(tmp, **arrays)
        tmp.rename(state.npz_path)
        meta.update(champ.single_meta())
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
            champ = Champion.from_island_state(s)
            champ.adopt_selection(np.asarray(new_best, dtype=np.float64), new_val_rms, None, feasible=feasible)
            s.update(champ.island_state())
            break
    else:
        raise ValueError(f"island {island_name!r} not found in checkpoint")
    arrays["island_states"] = np.array(pickle.dumps(states), dtype=object)
    tmp = state.npz_path.with_name(".tmp_" + state.npz_path.name)
    np.savez_compressed(tmp, **arrays)
    tmp.rename(state.npz_path)


# ---------------------------------------------------------------------------
# Orchestrator (test seam) + standalone CLI
# ---------------------------------------------------------------------------


def run_final_select(
    training_dir: Path,
    config: Any,  # TrainingConfig (typed Any to avoid heavy import at module load)
    param_specs: list[Any],
    problem: _RetroEvaluator,
    val_seeds: list[int],
    patch: bool = True,
    base_toml: Path | None = None,
    toml_data: dict | None = None,
    criterion: str = "rms",
    tail_seeds: list[int] | None = None,
    tail_pool: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> SelectionResult:
    """Load the latest checkpoint in training_dir, run the selection rule,
    rewrite best artifacts (+ sidecar), and (optionally) patch the checkpoint.

    `criterion="cvar95"` runs `select_tail_individual` on `tail_seeds` (#234); every
    checkpoint champion then competes as an ordinary row. `output_dir` deploys the
    winner and its sidecar there instead of into `training_dir`, and never patches
    the checkpoint: the run's deployed artifacts stay as they are.

    When `base_toml`/`toml_data` are provided (the CLI always passes them),
    non-NN winners also get their optimized_<scheme>.toml rewritten and — for
    joint-ref runs — the winner's reference table regenerated. Without this the
    CLI would leave the PREVIOUS winner's optimized TOML and ref_trajectory.dat
    behind a re-selected best_params.json (gains and reference co-adapt, so a
    mismatched table invalidates every downstream evaluation)."""
    from aerocapture.training.artifacts import deploy_winner  # noqa: PLC0415

    if criterion not in ("rms", "cvar95"):
        raise ValueError(f"unknown selection criterion {criterion!r}")
    if criterion == "cvar95" and (tail_seeds is None or tail_pool is None):
        raise ValueError("criterion cvar95 needs tail_seeds and tail_pool")
    if output_dir is not None and output_dir.resolve() == training_dir.resolve():
        raise ValueError("output_dir is training_dir: drop it to deploy in place (that path also patches the checkpoint)")
    state = load_selection_state(training_dir)
    ceiling = float(config.optimizer.max_violation_rate)
    known: list[KnownCandidate] = []
    population, provenances = state.population, list(state.provenances)
    island_of_row = list(state.island_of_row) if state.island_of_row is not None else None
    for name, champ in state.champions.items():
        label = f"{name}:" if name else ""
        assert champ.individual is not None
        if criterion == "cvar95":
            # The tail rule re-flies every champion as a row (its recorded RMS pre-empts nothing): no
            # re-validation here, and an untrusted one (val_rms inf) goes in under its plain label.
            known.append(KnownCandidate(x=np.asarray(champ.individual, dtype=np.float64), provenance=f"{label}champion", val_rms=float(champ.val_rms)))
            continue
        # Checkpoint champions may predate the gate or a ceiling change: re-validate
        # them before trusting them (`champion.revalidate`, ADR-0005). An untrusted one
        # competes as an ordinary candidate instead.
        revalidate(champ, problem, val_seeds, problem.cost_kwargs, ceiling, verbose=True, label=label)
        if (trusted := champ.known(label)) is not None:
            known.append(trusted)
            continue
        untrusted = champ.untrusted_row(label)
        assert untrusted is not None
        population = np.vstack([population, untrusted[0][None, :]])
        provenances.append(untrusted[1])
        if island_of_row is not None:
            island_of_row.append(name)
    if criterion == "cvar95":
        assert tail_seeds is not None and tail_pool is not None
        sel = select_tail_individual(
            problem,
            population,
            provenances,
            known,
            val_seeds,
            tail_seeds,
            tail_pool,
            max_violation_rate=ceiling,
            cost_kwargs=problem.cost_kwargs,
        )
    else:
        sel = select_final_individual(
            problem,
            population,
            provenances,
            known,
            val_seeds,
            max_violation_rate=ceiling,
            cost_kwargs=problem.cost_kwargs,
        )
    deploy_dir = training_dir if output_dir is None else output_dir
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deploy_winner(sel.individual, config, param_specs, deploy_dir, toml_data=toml_data, base_toml=base_toml, cwd=None)
    if output_dir is not None:
        print(f"  --output-dir: winner deployed to {output_dir}; {training_dir} and its checkpoint untouched.")
    elif patch:
        island_name: str | None = None
        if state.kind == "islands":
            if sel.winner_index is not None:
                assert island_of_row is not None
                island_name = island_of_row[sel.winner_index]
            else:
                island_name = sel.provenance.split(":", 1)[0]
        patch_checkpoint(state, sel.individual, sel.val_rms, island_name=island_name, feasible=sel.winner_feasible)
    else:
        print("  --no-checkpoint-patch: checkpoint untouched; a later resume will revert these artifacts at its next checkpoint save.")
    # Sidecar AFTER the checkpoint patch, as in the trainer's finalize: the campaign
    # runners read "sidecar newer than the latest checkpoint" as "selection done".
    write_final_selection_json(deploy_dir, sel, len(val_seeds))
    print(format_selection_summary(sel))
    return sel


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Re-run end-of-training final selection on an existing training directory.")
    parser.add_argument("training_dir", type=str, help="Directory containing checkpoint_g*.{json,npz} and best artifacts")
    parser.add_argument("--toml", type=str, required=True, help="Training TOML the run used (base inheritance resolved)")
    parser.add_argument("--no-checkpoint-patch", action="store_true", help="Do not write the re-selected best back into the checkpoint")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Per-sim wall-clock timeout (seconds)")
    parser.add_argument(
        "--criterion",
        choices=("rms", "cvar95"),
        default="rms",
        help="rms: the ADR-0002 rule (default). cvar95 (#234): the ADR-0005 gate, then most captured, then lowest CVaR95 of the captured correction DV",
    )
    parser.add_argument(
        "--pool",
        choices=("validation", "tail_select"),
        default=None,
        help="cvar95 only: score the tail on the validation pool (default) or on the reserved tail-select pool (TAIL_SELECT_SEED_OFFSET, --pool-n seeds)",
    )
    parser.add_argument("--pool-n", type=int, default=None, help="cvar95 with --pool tail_select: the tail-select pool size (default 5000)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="deploy the pick (best_model.json / best_params.json, final_selection.json) here instead of into training_dir; the checkpoint is not patched",
    )
    args = parser.parse_args()
    if args.criterion != "cvar95" and (args.pool is not None or args.pool_n is not None):
        parser.error("--pool / --pool-n apply to --criterion cvar95 only")
    if args.pool_n is not None and args.pool != "tail_select":
        parser.error("--pool-n applies to --pool tail_select only")

    from aerocapture.training.chromosome_layout import ChromosomeLayout  # noqa: PLC0415
    from aerocapture.training.cost import build_cost_kwargs  # noqa: PLC0415
    from aerocapture.training.problem import AerocaptureProblem  # noqa: PLC0415
    from aerocapture.training.seeds import TAIL_SELECT_SEED_OFFSET, VALIDATION_SEED_OFFSET, make_reserved_seeds  # noqa: PLC0415
    from aerocapture.training.training_config import run_param_specs  # noqa: PLC0415

    training_dir = Path(args.training_dir)
    config, toml_data, param_specs = run_param_specs(args.toml, training_dir, announce_overlay=True)
    config.sim.sim_timeout_secs = args.sim_timeout

    if config.optimizer.validation_n_sims <= 0:
        raise SystemExit("ERROR: [optimizer] validation_n_sims is 0 -- no validation pool exists to select on. Set validation_n_sims > 0 in the TOML.")

    state = load_selection_state(training_dir)
    base_mc_seed = state.base_mc_seed
    if base_mc_seed is None:
        mc_seed_val = toml_data.get("monte_carlo", {}).get("seed")
        base_mc_seed = int(mc_seed_val) if mc_seed_val is not None else 42
    elif toml_data.get("monte_carlo", {}).get("seed") is not None and int(toml_data["monte_carlo"]["seed"]) != base_mc_seed:
        raise SystemExit(
            f"ERROR: checkpoint base_mc_seed={base_mc_seed} != TOML monte_carlo.seed={toml_data['monte_carlo']['seed']} -- wrong TOML for this training dir?"
        )
    val_seeds = make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, config.optimizer.validation_n_sims)
    tail_seeds: list[int] | None = None
    tail_pool: dict[str, Any] | None = None
    if args.criterion == "cvar95":
        if args.pool == "tail_select":
            pool_n = 5000 if args.pool_n is None else args.pool_n
            tail_seeds = make_reserved_seeds(base_mc_seed, TAIL_SELECT_SEED_OFFSET, pool_n)
            tail_pool = {"name": "tail_select", "offset": TAIL_SELECT_SEED_OFFSET, "n": pool_n}
        else:
            tail_seeds = val_seeds
            tail_pool = {"name": "validation", "offset": VALIDATION_SEED_OFFSET, "n": len(val_seeds)}

    layout = ChromosomeLayout.from_specs(param_specs, config.guidance_type)
    try:
        layout.check_width(state.population.shape[1], "wrong TOML for this training dir?")
    except ValueError as exc:
        raise SystemExit(f"ERROR: checkpoint {exc}") from exc

    problem = AerocaptureProblem(
        param_specs=param_specs,
        toml_path=str(Path(args.toml).resolve()),
        seeds=[base_mc_seed],
        cost_kwargs=build_cost_kwargs(toml_data),
        scheme=config.guidance_type,
        sim_timeout=config.sim.sim_timeout_secs,
        nn_config=config.network if config.guidance_type == "neural_network" else None,
    )

    n_rows = state.population.shape[0] + (len(state.champions) if args.criterion == "cvar95" else 0)
    n_sims_estimate = n_rows * len(val_seeds) + (n_rows * len(tail_seeds) if tail_seeds is not None and tail_seeds != val_seeds else 0)
    print(f"  Final selection ({args.criterion}) over {n_rows} candidates x {len(val_seeds)} validation seeds (<= {n_sims_estimate} sims)")
    if tail_pool is not None and tail_pool["name"] == "tail_select":
        print(f"  Tail statistic of the feasible survivors on the tail-select pool (offset {TAIL_SELECT_SEED_OFFSET}, n={tail_pool['n']})...")
    run_final_select(
        training_dir=training_dir,
        config=config,
        param_specs=param_specs,
        problem=problem,
        val_seeds=val_seeds,
        patch=not args.no_checkpoint_patch,
        base_toml=Path(args.toml),
        toml_data=toml_data,
        criterion=args.criterion,
        tail_seeds=tail_seeds,
        tail_pool=tail_pool,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
