"""The one deploy-side evaluation path: fly a cell on a seed pool.

A cell is a deployed training run (config + output dir). Flying it means: pick the
TOML (`optimized_<scheme>.toml` when the cell baked its tuned params into one, else
the base TOML plus the cell's `best_params.json` scaffolding routed as overrides),
pin the NN to the cell's own `best_model.json` when there is one (the TOML's
`[data] neural_network` path is shared by every `--output-dir` sibling of a config
and can hold a foreign model), then run one sim per seed through
`aerocapture_rs.run_batch`. Every report, demo, comparison and paper number goes
through `evaluate_cell`; `fly_mc` / `fly_nominal` are the same resolution for a
config's own Monte Carlo and for the undispersed nominal.

Training-internal callers (`problem.py` population evaluation, the trainer's
corridor accumulation, reference generation, `sensitivity.run_with_draws`) build
their own overrides and stay on the seam directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerocapture.training.deploy_overrides import resolve_eval_toml
from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
from aerocapture.training.seeds import make_reserved_seeds
from aerocapture.training.toml_utils import load_toml_with_bases

# Final-record columns this module reads (pinned to `final_record_indices()` by tests/test_cell_eval.py).
FR_ECC = 9
FR_IFINAL = 31
FR_DV_TOTAL = 41


def is_captured(final_records: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Canonical captured definition: exited atmosphere (ifinal==3) on a bound orbit (ecc<1)."""
    result: npt.NDArray[np.bool_] = (final_records[:, FR_IFINAL] == 3) & (final_records[:, FR_ECC] < 1.0)
    return result


@dataclass(frozen=True)
class CellResult:
    final_records: npt.NDArray[np.float64]  # (N, 52)
    dispersions: npt.NDArray[np.float64]  # (N, 26)
    trajectories: list[npt.NDArray[np.float64]] | None  # (T_i, 17) each; None unless requested
    seeds: list[int]  # the pool flown; [] for a config's own Monte Carlo
    toml_path: Path  # the TOML flown (resolved)
    overrides: dict[str, object]  # the base override dict every sim shared (seed excluded): provenance

    @property
    def n(self) -> int:
        return int(self.final_records.shape[0])

    @property
    def captured(self) -> npt.NDArray[np.bool_]:
        return is_captured(self.final_records)

    @property
    def dv(self) -> npt.NDArray[np.float64]:
        """Correction delta-v of the captured runs only (unclipped)."""
        result: npt.NDArray[np.float64] = self.final_records[self.captured, FR_DV_TOTAL]
        return result


def reserved_pool(toml_path: Path, offset: int, n: int) -> list[int]:
    """`n` seeds of the pool registered at `offset`, keyed by the TOML's `[monte_carlo] seed`."""
    base_mc_seed = int(load_toml_with_bases(Path(toml_path)).get("monte_carlo", {}).get("seed", 42))
    return make_reserved_seeds(base_mc_seed, offset, n)


def _resolve_cell(cell_dir: Path | None, base_toml: Path, model: Path | None) -> tuple[Path, dict[str, object]]:
    """(TOML to fly, scaffolding + model-pin overrides) for a cell; no cell means the bare base TOML."""
    overrides: dict[str, object] = {}
    eval_toml = Path(base_toml)
    if cell_dir is not None:
        eval_toml, overrides = resolve_eval_toml(eval_toml, Path(cell_dir))
    if model is not None:
        if not Path(model).exists():
            raise FileNotFoundError(f"model {model} does not exist (refusing to fly the TOML's shared [data] neural_network instead)")
        overrides["data.neural_network"] = str(Path(model).resolve())
    elif cell_dir is not None and (Path(cell_dir) / "best_model.json").exists():  # classical cells have none
        overrides["data.neural_network"] = str((Path(cell_dir) / "best_model.json").resolve())
    return eval_toml.resolve(), overrides


def _pack(results: object, seeds: list[int], toml_path: Path, overrides: dict[str, object], include_trajectories: bool) -> CellResult:
    # BatchResults getters rebuild their numpy views per access: materialize each once.
    return CellResult(
        final_records=np.asarray(results.final_records, dtype=np.float64),  # type: ignore[attr-defined]
        dispersions=np.asarray(results.dispersions, dtype=np.float64),  # type: ignore[attr-defined]
        trajectories=[np.asarray(t, dtype=np.float64) for t in results.trajectories] if include_trajectories else None,  # type: ignore[attr-defined]
        seeds=seeds,
        toml_path=toml_path,
        overrides=overrides,
    )


def evaluate_cell(
    cell_dir: Path | None,
    base_toml: Path,
    seeds: Sequence[int] | npt.NDArray[np.integer] | None = None,
    *,
    pool: tuple[int, int] | None = None,
    model: Path | None = None,
    extra_overrides: Mapping[str, object] | None = None,
    per_seed_overrides: Sequence[Mapping[str, object]] | None = None,
    include_trajectories: bool = False,
    sim_timeout_secs: float | None = None,
    n_threads: int | None = None,
) -> CellResult:
    """Fly `cell_dir` on a seed pool, one sim per seed (`aerocapture_rs.run_batch`).

    The pool is `seeds` explicitly, or `pool=(offset, n)`: `n` seeds of the
    reserved pool registered at `offset` (`seeds.py`), keyed by the `[monte_carlo]
    seed` of the TOML actually flown. `model` pins the NN explicitly (a bundle's
    frozen weights, a temp ablated or checkpoint model) instead of
    `<cell_dir>/best_model.json`; `extra_overrides` (noise regime, stress levels,
    a guidance type) win over the cell's scaffolding and pin; `per_seed_overrides[i]`
    is applied to seed `i` only; the seed itself is applied last. `cell_dir=None`
    flies the bare base TOML.
    """
    import aerocapture_rs  # noqa: PLC0415  (lazy: keeps the module importable without the extension)

    if (seeds is None) == (pool is None):
        raise ValueError("give exactly one of seeds= or pool=(offset, n)")
    eval_toml, cell_overrides = _resolve_cell(cell_dir, base_toml, model)
    seed_list = [int(s) for s in seeds] if seeds is not None else reserved_pool(eval_toml, *pool)  # type: ignore[misc]
    if per_seed_overrides is not None and len(per_seed_overrides) != len(seed_list):
        raise ValueError(f"per_seed_overrides has {len(per_seed_overrides)} entries for {len(seed_list)} seeds")
    base: dict[str, object] = {"simulation.n_sims": 1, **cell_overrides, **(extra_overrides or {})}
    overrides_list = [{**base, **(per_seed_overrides[i] if per_seed_overrides is not None else {}), "monte_carlo.seed": s} for i, s in enumerate(seed_list)]
    results = aerocapture_rs.run_batch(
        str(eval_toml),
        overrides_list,
        n_threads=n_threads,
        include_trajectories=include_trajectories,
        sim_timeout_secs=sim_timeout_secs,
    )
    return _pack(results, seed_list, eval_toml, base, include_trajectories)


def fly_mc(
    cell_dir: Path | None,
    base_toml: Path,
    *,
    n_sims: int | None = None,
    model: Path | None = None,
    extra_overrides: Mapping[str, object] | None = None,
    include_trajectories: bool = False,
    sim_timeout_secs: float | None = None,
) -> CellResult:
    """Fly a cell through the config's own Monte Carlo (`aerocapture_rs.run_mc`):
    `n_sims` dispersed sims from the TOML's `[monte_carlo] seed`, not a reserved pool."""
    import aerocapture_rs  # noqa: PLC0415

    eval_toml, cell_overrides = _resolve_cell(cell_dir, base_toml, model)
    base: dict[str, object] = {**({"simulation.n_sims": n_sims} if n_sims is not None else {}), **cell_overrides, **(extra_overrides or {})}
    results = aerocapture_rs.run_mc(str(eval_toml), overrides=base, include_trajectories=include_trajectories, sim_timeout_secs=sim_timeout_secs)
    return _pack(results, [], eval_toml, base, include_trajectories)


def fly_nominal(
    cell_dir: Path | None,
    base_toml: Path,
    *,
    model: Path | None = None,
    extra_overrides: Mapping[str, object] | None = None,
    sim_timeout_secs: float | None = None,
) -> CellResult:
    """The cell's undispersed nominal: one sim with every dispersion domain off, trajectory included."""
    nominal: dict[str, object] = {f"monte_carlo.{d}.level": "off" for d in _MC_DISPERSION_DOMAINS}
    return fly_mc(
        cell_dir,
        base_toml,
        n_sims=1,
        model=model,
        extra_overrides={**nominal, **(extra_overrides or {})},
        include_trajectories=True,
        sim_timeout_secs=sim_timeout_secs,
    )
