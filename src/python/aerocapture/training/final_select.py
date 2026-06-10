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
    incumbent_val_rms: float | None = None  # best known (champion) val RMS; None when no champion existed
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
            incumbent_val_rms=incumbent.val_rms if incumbent is not None else None,
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
        incumbent_val_rms=incumbent.val_rms,
        candidate_rms=records,
    )


def write_final_selection_json(save_dir: Path, result: SelectionResult, n_val_seeds: int) -> None:
    payload = {
        "winner": {
            "provenance": result.provenance,
            "val_rms": result.val_rms,
            "promoted": result.promoted,
        },
        "champion_val_rms": result.incumbent_val_rms,
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


# ---------------------------------------------------------------------------
# Orchestrator (test seam) + standalone CLI
# ---------------------------------------------------------------------------


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

    from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, make_reserved_seeds  # noqa: PLC0415
    from aerocapture.training.problem import AerocaptureProblem  # noqa: PLC0415
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
        mc_seed_val = toml_data.get("monte_carlo", {}).get("seed")
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
