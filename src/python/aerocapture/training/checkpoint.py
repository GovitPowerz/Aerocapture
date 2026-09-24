"""Single-algorithm checkpoint IO and the resume helpers both adapters share.

`save_checkpoint` / `load_checkpoint` are the paired `checkpoint_g*.{json,npz}`
format (the islands path writes its own npz via `IslandModel.checkpoint`);
`_prune_old_checkpoints`, `_restore_seed_curator` and
`_check_resume_chromosome_shape` serve both paths. A leaf: no trainer or
train import.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.artifacts import write_best_artifacts
from aerocapture.training.config import TrainingConfig
from aerocapture.training.corridor import CorridorAccumulator
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.seed_curator import SeedCurator


def _restore_seed_curator(state: dict, configured: SeedCurator, verbose: bool) -> SeedCurator:
    """Restore curator STATE from a checkpoint, keeping the TOML-configured knobs.

    The checkpoint contributes seed_list / last_curation_gen only; sample_size,
    n_bins, trim_fraction, and bucket_selection come from the freshly-built
    `configured` curator (current [optimizer] TOML values). Restoring them from
    the checkpoint would silently discard a knob edited between runs — legacy
    checkpoints lacking the keys would even reset trim/bucket to 0.0/'random'.
    A notice is printed when the checkpointed knobs differ (mirrors the
    cost_transform change notice).
    """
    restored = SeedCurator.from_dict(state, excluded_seeds=configured.excluded_seeds, rng=configured.rng)
    changed = [
        f"{name} {getattr(restored, name)!r} -> {getattr(configured, name)!r}"
        for name in ("sample_size", "n_bins", "trim_fraction", "bucket_selection")
        if getattr(restored, name) != getattr(configured, name)
    ]
    if changed and verbose:
        print(f"  seed-curator knobs from TOML override checkpoint: {', '.join(changed)}")
    restored.sample_size = configured.sample_size
    restored.n_bins = configured.n_bins
    restored.trim_fraction = configured.trim_fraction
    restored.bucket_selection = configured.bucket_selection
    return restored


def _prune_old_checkpoints(save_dir: Path, keep_last: int | None) -> None:
    """Retain only the `keep_last` most recent checkpoints; no-op when unset.

    Shared by the single-algorithm `save_checkpoint` and the islands path.
    `prune_checkpoints` matches both `checkpoint_g*.json` and `checkpoint_g*.npz`
    (islands writes npz-only), and leaves JSONL logs / best_* / warm_start_* /
    report.pdf untouched, so post-training analysis still works.
    """
    if keep_last is None or keep_last < 1:
        return
    from aerocapture.training.cleanup_checkpoints import prune_checkpoints  # noqa: PLC0415

    prune_checkpoints(save_dir, keep_last=keep_last)


def _check_resume_chromosome_shape(
    saved_population: npt.NDArray[np.float64],
    expected_n_params: int,
) -> None:
    """Fail loudly if a resumed checkpoint's chromosome width disagrees with current ParamSpec count.

    Catches the user flipping `scaffolding` (or `output_parameterization`,
    which changes last-layer width) between training runs.
    """
    saved_n_params = saved_population.shape[1]
    if saved_n_params != expected_n_params:
        msg = (
            f"checkpoint chromosome shape mismatch: saved {saved_n_params} params, "
            f"current ParamSpec list has {expected_n_params}. This usually means "
            f"`[guidance.neural_network] scaffolding` or "
            f"`output_parameterization` was changed since the checkpoint was saved. "
            f"To resume, revert the TOML knob; to start fresh, pass --from-scratch."
        )
        raise ValueError(msg)


def save_checkpoint(
    save_dir: Path,
    generation: int,
    population: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    best_cost: float,
    best_individual: npt.NDArray[np.float64] | None,
    cost_history: list[float],
    rng: np.random.Generator,
    config: TrainingConfig,
    cwd: str | Path | None,
    param_specs: list[ParamSpec],
    seed_curator: SeedCurator | None = None,
    corridor_acc: CorridorAccumulator | None = None,
    best_val_cost: float = np.inf,
    cost_transform: str = "linear",
) -> None:
    """Save full training state for later resumption."""
    prefix = f"checkpoint_g{generation:05d}"

    # Serialize RNG state -- convert large ints to strings for JSON compatibility
    raw_state = rng.bit_generator.state
    rng_state_json = {
        "bit_generator": raw_state["bit_generator"],
        "state": {k: str(v) if isinstance(v, int) and v.bit_length() > 53 else v for k, v in raw_state["state"].items()},
        "has_uint32": raw_state["has_uint32"],
        "uinteger": raw_state["uinteger"],
    }
    meta = {
        "generation": generation,
        "best_cost": best_cost,
        "best_val_cost": best_val_cost,
        "cost_transform": cost_transform,
        "cost_history": [float(c) for c in cost_history],
        "rng_state": rng_state_json,
    }
    if seed_curator is not None:
        meta["seed_curator"] = seed_curator.to_dict()

    arrays: dict[str, npt.NDArray] = {}
    arrays["population"] = population
    arrays["costs"] = costs
    if best_individual is not None:
        arrays["best_individual"] = best_individual
    if corridor_acc is not None:
        for ck, cv in corridor_acc.to_checkpoint().items():
            arrays[ck] = cv

    # Atomic write (tempfile + rename; npz first, json last -- the json is the
    # resume-glob key, so a visible json implies a complete npz). A crash or a
    # second Ctrl+C during the interrupt-save must not leave a truncated pair
    # shadowing the resume path. The `.tmp_` prefix keeps partial files out of
    # the checkpoint_g* globs (same convention as final_select.patch_checkpoint;
    # the islands checkpoint has been atomic all along).
    npz_tmp = save_dir / f".tmp_{prefix}.npz"
    np.savez(npz_tmp, **arrays)  # type: ignore[arg-type]  # mypy vs numpy stubs kwargs issue
    npz_tmp.replace(save_dir / f"{prefix}.npz")
    json_tmp = save_dir / f".tmp_{prefix}.json"
    with open(json_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    json_tmp.replace(save_dir / f"{prefix}.json")

    # Save best model/params (immediately usable by Rust)
    if best_individual is not None:
        write_best_artifacts(best_individual, config, param_specs, save_dir, cwd=cwd, deploy_to_cwd=True)

    # Auto-prune older checkpoints when retention is configured.
    _prune_old_checkpoints(save_dir, config.checkpoints.keep_last)


def load_checkpoint(
    save_dir: Path,
) -> dict | None:
    """Find and load the latest checkpoint from save_dir.

    Returns dict with: generation, population, costs, best_cost,
    best_individual, cost_history, rng_state, cost_transform. Or None if no checkpoint found.
    """
    # Support both new (checkpoint_g*.json) and old (checkpoint_r*_g*.json) naming
    json_files = sorted(save_dir.glob("checkpoint_g*.json"))
    if not json_files:
        json_files = sorted(save_dir.glob("checkpoint_r*_g*.json"))
    if not json_files:
        return None

    # Newest first; fall back past corrupt pairs (pre-atomic-write crashes left
    # truncated files that would otherwise brick auto-resume until hand-deleted).
    for latest in reversed(json_files):
        npz_path = latest.with_suffix(".npz")
        if not npz_path.exists():
            print(f"  Skipping checkpoint {latest.name}: missing {npz_path.name}")
            continue
        try:
            with open(latest) as f:
                meta = json.load(f)
            data = np.load(npz_path)
            if "population" not in data:
                return None  # Incompatible legacy checkpoint; start fresh

            population = data["population"]
            costs = data["costs"]
            best_individual = data.get("best_individual", None)

            # Restore corridor accumulator if present in checkpoint
            corridor_acc_restored: CorridorAccumulator | None = None
            if "corridor_energy_bins" in data:
                corridor_state = {k: data[k] for k in data if k.startswith("corridor_")}
                corridor_acc_restored = CorridorAccumulator.from_checkpoint(corridor_state)

            return {
                "generation": meta["generation"],
                "population": population,
                "costs": costs,
                "best_cost": meta["best_cost"],
                "best_individual": best_individual,
                "cost_history": meta["cost_history"],
                "rng_state": meta.get("rng_state"),
                "best_val_cost": meta.get("best_val_cost", float("inf")),
                "cost_transform": meta.get("cost_transform", None),
                "seed_curator": meta.get("seed_curator"),
                "corridor_acc": corridor_acc_restored,
            }
        except (json.JSONDecodeError, KeyError, ValueError, OSError, EOFError, zipfile.BadZipFile) as e:
            print(f"  Skipping corrupt checkpoint {latest.name}: {e}")
            continue
    return None
