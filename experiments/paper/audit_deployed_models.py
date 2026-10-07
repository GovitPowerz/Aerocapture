"""Audit the deployed NN model of every run under training_output/ (#170).

Until #170 the trainer also wrote each run's model to the TOML's
`[data] neural_network` path, so an `--output-dir` run of a config that
base-inherits another cell's TOML overwrote that cell's best_model.json. This
script hashes every `training_output/**/best_model.json`, reports byte-identical
duplicates across run dirs and, for every run with a `final_selection.json`,
rebuilds the winner from the final checkpoint (the population row or champion
the sidecar's provenance names, decoded through `artifacts.write_best_artifacts`
into a temp dir) and byte-compares it with the deployed model.

The run's TOML is the one report.py recorded in final_eval.parquet
(`aerocapture.toml_path`); a run without a report falls back to the committed
config whose own `[data] neural_network` lies in the run dir.

Statuses: ok | mismatch | repaired | stale sidecar (the named row is not the
final checkpoint's champion, so the sidecar describes another checkpoint) |
no checkpoint | no toml | no final_selection | error: <reason>. A
"; duplicate of <dir>" suffix names byte-identical models elsewhere; the bundle
column compares the model with its articles/paper/data/runs/ copy.

Read-only by default. --repair rewrites a mismatched best_model.json from the
rebuilt winner (the old file is kept as best_model.json.pre-repair) and prints
the before/after hashes.

Usage (repo root): uv run python experiments/paper/audit_deployed_models.py [--repair]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from aerocapture.training.artifacts import write_best_artifacts
from aerocapture.training.final_select import load_selection_state
from aerocapture.training.training_config import run_param_specs

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "training_output"
sys.path.insert(0, str(REPO / "articles/paper/scripts"))


class StaleSidecar(LookupError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_tomls() -> dict[Path, Path]:
    """Run dir -> the committed config whose own `[data] neural_network` lies in it."""
    out: dict[Path, Path] = {}
    for toml in sorted((REPO / "configs").rglob("*.toml")):
        nn = tomllib.loads(toml.read_text()).get("data", {}).get("neural_network")
        if isinstance(nn, str):
            out.setdefault((REPO / nn).parent.resolve(), toml)
    return out


def run_toml(run: Path, committed: dict[Path, Path]) -> Path | None:
    parquet = run / "final_eval.parquet"
    if parquet.exists():
        recorded = (pq.read_metadata(parquet).metadata or {}).get(b"aerocapture.toml_path")
        if recorded and (toml := REPO / bytes(recorded).decode()).exists():
            return toml
    return committed.get(run.resolve())


def winner_chromosome(run: Path) -> npt.NDArray[np.float64]:
    """The final-checkpoint row the sidecar's provenance names. Final selection
    persists its winner as the checkpoint champion, so a row that is not the
    champion means the sidecar describes another checkpoint (StaleSidecar)."""
    prov = json.loads((run / "final_selection.json").read_text())["winner"]["provenance"].removesuffix("[infeasible]")
    state = load_selection_state(run)
    island, _, name = prov.rpartition(":")
    champion: npt.NDArray[np.float64] | None
    if state.kind == "single":
        with np.load(state.npz_path) as data:
            champion = np.asarray(data["best_individual"], dtype=np.float64) if "best_individual" in data else None
    else:
        champion = next((k.x for k in state.known if k.provenance == f"{island}:champion"), None)
    if name == "champion":
        x = champion
    elif prov in state.provenances:
        x = state.population[state.provenances.index(prov)]
    else:
        raise StaleSidecar(f"{prov} not in {state.npz_path.name}")
    if x is None or champion is None or not np.array_equal(x, champion):
        raise StaleSidecar(f"{prov} is not the champion of {state.npz_path.name}")
    return x


def rebuild(run: Path, toml: Path, dest: Path) -> None:
    config, _toml_data, specs = run_param_specs(toml, run)
    x = winner_chromosome(run)
    if x.shape[0] != len(specs):
        raise ValueError(f"chromosome width {x.shape[0]} != {len(specs)} params under {toml.relative_to(REPO)}")
    write_best_artifacts(x, config, specs, dest, cwd=REPO)


def audit_run(run: Path, committed: dict[Path, Path], repair: bool) -> str:
    if not (run / "final_selection.json").exists():
        return "no final_selection"
    if not any(run.glob("checkpoint_g*.npz")):
        return "no checkpoint"
    toml = run_toml(run, committed)
    if toml is None:
        return "no toml"
    deployed = run / "best_model.json"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            rebuild(run, toml, Path(tmp))
        except StaleSidecar as e:
            return f"stale sidecar ({e})"
        except (Exception, SystemExit) as e:
            return f"error: {type(e).__name__}: {e}"
        rebuilt = Path(tmp) / "best_model.json"
        if rebuilt.read_bytes() == deployed.read_bytes():
            return "ok"
        if not repair:
            return "mismatch"
        before = sha(deployed)
        shutil.copy2(deployed, deployed.with_name("best_model.json.pre-repair"))
        shutil.copyfile(rebuilt, deployed)
    print(f"repaired {run.relative_to(TRAINING)}: {before} -> {sha(deployed)}")
    return "repaired"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit (and optionally repair) every deployed best_model.json under training_output/.")
    parser.add_argument("--repair", action="store_true", help="rewrite each mismatched best_model.json from its rebuilt winner")
    args = parser.parse_args()

    from collect_runs import _run_dirs  # type: ignore[import-not-found]  # articles/paper/scripts via sys.path

    bundle = {src.resolve(): dst / "best_model.json" for src, dst in _run_dirs()}
    committed = committed_tomls()
    runs = sorted(m.parent for m in TRAINING.rglob("best_model.json"))
    status = {run: audit_run(run, committed, args.repair) for run in runs}

    digest = {run: sha(run / "best_model.json") for run in runs}
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for run in runs:
        by_hash[digest[run]].append(run)
    width = max((len(str(run.relative_to(TRAINING))) for run in runs), default=len("run dir"))
    print(f"\n{'run dir':<{width}}  {'bundle':<7}  status")
    for run in runs:
        twins = [str(r.relative_to(TRAINING)) for r in by_hash[digest[run]] if r != run]
        copy = bundle.get(run.resolve())
        in_bundle = "" if copy is None or not copy.exists() else "=" if sha(copy) == digest[run] else "differs"
        line = status[run] + (f"; duplicate of {', '.join(twins)}" if twins else "")
        print(f"{run.relative_to(TRAINING)!s:<{width}}  {in_bundle:<7}  {line}")
    print("\n" + ", ".join(f"{k}: {n}" for k, n in Counter(s.split(" (")[0].split(":")[0] for s in status.values()).most_common()))


if __name__ == "__main__":
    main()
