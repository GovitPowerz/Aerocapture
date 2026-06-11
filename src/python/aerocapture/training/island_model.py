"""Three-island PSO/GA/DE evolutionary trainer with episodic migration.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm

from aerocapture.training.evaluate import GateStatus, run_validation_gate
from aerocapture.training.metrics import capture_rate as _capture_rate
from aerocapture.training.optimizer import OptimizerConfig, create_algorithm

# A winner whose final-eval rms materially exceeds the validation rms it was
# selected on has overfit the validation corpus (the within-island gate picked
# it to minimize val_rms; a fresh held-out pool revealing a much higher rms means
# that selection didn't generalize). Flag gaps above this relative threshold.
VAL_GENERALIZATION_GAP_THRESHOLD = 0.15


def val_generalization_gap(val_rms: float, final_rms: float) -> tuple[float, bool]:
    """Relative gap between an island's validation rms (its selection metric) and
    its final-eval rms (fresh held-out). Returns (relative_gap, overfit_flag),
    gap = (final_rms - val_rms) / val_rms, flag True when gap exceeds
    VAL_GENERALIZATION_GAP_THRESHOLD. Returns (nan, False) when val_rms is not
    finite/positive (e.g. validation disabled, validation_n_sims=0)."""
    if not math.isfinite(val_rms) or val_rms <= 0.0:
        return float("nan"), False
    gap = (final_rms - val_rms) / val_rms
    return gap, gap > VAL_GENERALIZATION_GAP_THRESHOLD


@dataclass
class MigrationEvent:
    gen: int
    src_island: str
    dst_island: str
    slot_idx: int
    F_migrant: float
    F_displaced: float


@dataclass
class Island:
    """One algorithm-population pair in the 3-island model.

    `algorithm` is a pymoo Algorithm (PSO/GA/DE) whose `.pop` Population is
    mutated in-place by migration.
    """

    name: str
    algorithm: Any  # pymoo Algorithm or test stand-in (must expose .pop)
    last_validated_individual: npt.NDArray[np.float64] | None = None
    best_overall_individual: npt.NDArray[np.float64] | None = None
    best_overall_cost: float = float("inf")
    best_val_cost: float = float("inf")
    stagnation_counter: int = 0
    # Most recent `compute_eval_summary` payload (DV / apoapsis / heat-flux /
    # g-load / heat-load percentiles + violation rates). Carried across gens
    # so the TUI shows continuous per-island shape even when only one island
    # validates per gen. None until the first validation pass.
    latest_val_summary: dict | None = None


def migrate(
    islands: list[Island],
    k_top: int,
    current_gen: int,
    rng: np.random.Generator,
    velocity_scale: float = 0.05,
) -> list[MigrationEvent]:
    """Apply full-mesh top-k -> worst-(k * (n_islands - 1)) migration in-place.

    Snapshots emigrants from every island BEFORE any in-place replacement
    (so a destination's incoming pool is not corrupted by its own outgoing).
    For PSO destinations, also resets V and the personal-best slot
    (pop[slot].X / .F) via `inject_into_pso`.
    """
    # 1. Snapshot top-k emigrants from each island under current F.
    #    Only FINITE-F individuals are eligible: an island whose population is
    #    entirely non-finite (all sims timed out / returned NaN) contributes no
    #    migrants. Without this guard a collapsed island would broadcast its
    #    inf/NaN chromosomes into every healthy destination's worst slots,
    #    destroying finite individuals exactly when the collapsed island most
    #    needs rescue (mirrors the all-inf guard in validate_each).
    emigrants: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = {}
    for src in islands:
        F_src = src.algorithm.pop.get("F").flatten()
        finite_idx = np.flatnonzero(np.isfinite(F_src))
        top_idx = finite_idx[np.argsort(F_src[finite_idx], kind="stable")[:k_top]]
        emigrants[src.name] = [(src.algorithm.pop[int(i)].X.copy(), float(F_src[int(i)])) for i in top_idx]

    # 2. For each destination, apply replacements from all other islands.
    events: list[MigrationEvent] = []
    for dst in islands:
        incoming: list[tuple[npt.NDArray[np.float64], float, str]] = []
        for src in islands:
            if src.name == dst.name:
                continue
            for X_em, F_em in emigrants[src.name]:
                incoming.append((X_em, F_em, src.name))

        n_incoming = len(incoming)
        # Every other island collapsed (no finite emigrants): nothing to inject.
        if n_incoming == 0:
            continue
        pop_size = len(dst.algorithm.pop)
        if n_incoming > pop_size:
            raise ValueError(
                f"migration would overwrite {n_incoming} slots in destination "
                f"'{dst.name}' of size {pop_size}; reduce k_top or increase "
                f"n_pop (k_top * (n_islands - 1) must be <= n_pop).",
            )
        F_dst = dst.algorithm.pop.get("F").flatten()
        # Sort incoming by descending F so the best migrant (smallest F_em)
        # lands in the absolute-worst destination slot (`worst_slots[-1]`),
        # and the worst migrant in the least-bad worst-slot.
        worst_slots = np.argsort(F_dst, kind="stable")[-n_incoming:]
        incoming.sort(key=lambda triple: -triple[1])

        for slot_i, (X_new, F_new, src_name) in zip(worst_slots, incoming, strict=True):
            slot = int(slot_i)
            F_displaced = float(F_dst[slot])
            # Copy: one emigrant snapshot is shared across both destinations,
            # so aliasing it would make two islands' slots reference the same
            # ndarray (the PSO branch below already copies via inject_into_pso).
            dst.algorithm.pop[slot].X = X_new.copy()
            dst.algorithm.pop[slot].F = np.array([F_new])

            if isinstance(dst.algorithm, PSO):
                inject_into_pso(
                    dst.algorithm,
                    slot=slot,
                    X=X_new,
                    F=F_new,
                    velocity_scale=velocity_scale,
                    rng=rng,
                )

            events.append(
                MigrationEvent(
                    gen=current_gen,
                    src_island=src_name,
                    dst_island=dst.name,
                    slot_idx=slot,
                    F_migrant=F_new,
                    F_displaced=F_displaced,
                )
            )

        # Refresh `self.opt` (the social/global attractor used by PSO's
        # `_infill` for the next gen, and by `result()` consumers for GA/DE)
        # so it reflects the post-migration pop. Without this, a migrant
        # whose F is lower than the pre-migration gbest is not picked as
        # the social-best for one full generation.
        if hasattr(dst.algorithm, "_set_optimum"):
            dst.algorithm._set_optimum()

    return events


def compute_migration_origin_stats(
    migration_log: list[MigrationEvent],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-destination, per-source migrant statistics.

    Returns: {dst_name: {src_name: {"wins": int, "mean_F": float, "count": int}}}

    `wins` = number of migration events in which `src_name` supplied the lowest-F
    migrant arriving at `dst_name` (ties broken by first-seen).
    `mean_F` = mean F_migrant across all events from src_name into dst_name.
    `count` = total number of migrants from src_name into dst_name.

    The migration_log is the flat MigrationEvent list maintained by IslandModel.
    Events are grouped by (gen, dst) — that pair identifies one migration event.
    """
    if not migration_log:
        return {}

    # Group by (gen, dst) to recover per-event arrival sets.
    by_event: dict[tuple[int, str], list[MigrationEvent]] = {}
    for ev in migration_log:
        by_event.setdefault((ev.gen, ev.dst_island), []).append(ev)

    # Pre-compute: F lists per (dst, src) for mean.
    fs_per_dst_src: dict[str, dict[str, list[float]]] = {}
    for ev in migration_log:
        fs_per_dst_src.setdefault(ev.dst_island, {}).setdefault(ev.src_island, []).append(
            ev.F_migrant,
        )

    # Count "wins" per (dst, src): per event, which source supplied the lowest F.
    wins: dict[str, dict[str, int]] = {}
    for (_gen, dst), arrivals in by_event.items():
        best = min(arrivals, key=lambda e: e.F_migrant)
        wins.setdefault(dst, {}).setdefault(best.src_island, 0)
        wins[dst][best.src_island] += 1

    # Stitch together.
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for dst, src_map in fs_per_dst_src.items():
        out[dst] = {}
        for src, fs in src_map.items():
            out[dst][src] = {
                "wins": wins.get(dst, {}).get(src, 0),
                "count": len(fs),
                "mean_F": float(sum(fs) / len(fs)),
            }
    return out


def summarize_latest_migration(events: list[MigrationEvent]) -> dict[str, dict]:
    """Per-destination best/worst migrant for one gen's migration events.

    Returns: {dst_name: {"gen", "best": {src, F_migrant, F_displaced},
    "worst": {...}}}. Keeps MigrationEvent field access in this module rather
    than inline in the train loop. Empty input -> empty dict.
    """
    by_dst: dict[str, list[MigrationEvent]] = {}
    for ev in events:
        by_dst.setdefault(ev.dst_island, []).append(ev)
    summary: dict[str, dict] = {}
    for dst_name, dst_events in by_dst.items():
        best = min(dst_events, key=lambda e: e.F_migrant)
        worst = max(dst_events, key=lambda e: e.F_migrant)
        summary[dst_name] = {
            "gen": best.gen,
            "best": {"src": best.src_island, "F_migrant": best.F_migrant, "F_displaced": best.F_displaced},
            "worst": {"src": worst.src_island, "F_migrant": worst.F_migrant, "F_displaced": worst.F_displaced},
        }
    return summary


def inject_into_pso(
    algorithm: Algorithm,
    slot: int,
    X: npt.NDArray[np.float64],
    F: float,
    velocity_scale: float,
    rng: np.random.Generator,
) -> None:
    """Place a migrant into a PSO slot: fresh velocity + pbest + current swarm pos.

    pymoo PSO maintains two Population objects:
    - `algorithm.pop` is the PERSONAL-BEST population (`pop[i].X` is particle i's pbest).
    - `algorithm.particles` is the CURRENT swarm position (identical to `pop` on
      iteration 1, divergent after).

    We must update both so the migrant lands at the slot in both reference frames.
    Velocity gets a small uniform random kick: zero velocity is a trap because a
    collapsed swarm's gbest will pull the migrant in within 2-3 ticks; the kick
    gives 2-3 ticks of independent motion for the migrant to assert its own F.
    """
    n_params = X.shape[0]
    particles = getattr(algorithm, "particles", algorithm.pop)

    # Velocity injection.
    V = particles.get("V")
    V[slot] = rng.uniform(-velocity_scale, velocity_scale, size=n_params)
    particles.set("V", V)

    # Current swarm position (particles[slot]).
    particles[slot].X = X.copy()
    particles[slot].F = np.array([F])

    # Personal best (pop[slot] in pymoo PSO).
    algorithm.pop[slot].X = X.copy()
    algorithm.pop[slot].F = np.array([F])


_ISLAND_NAMES = ("pso", "ga", "de")


def _build_island(
    name: str,
    config: OptimizerConfig,
    n_params: int,
) -> Island:
    sub_config = deepcopy(config)
    sub_config.algorithm = name
    algorithm = create_algorithm(sub_config, n_params=n_params)
    return Island(name=name, algorithm=algorithm)


class IslandModel:
    """Owns 3 islands (PSO, GA, DE) and the migration / validation / final-eval flow."""

    def __init__(
        self,
        config: OptimizerConfig,
        problem: Any,
        n_params: int,
        validation_seeds: list[int],
        final_eval_seeds: list[int],
        base_mc_seed: int,
        rng: np.random.Generator,
    ) -> None:
        # Reject configs that would crash migrate() at the first migration tick
        # because k_top * (n_islands - 1) > n_pop. Done here (vs in
        # IslandSettings.__post_init__) because we need n_pop from the
        # surrounding OptimizerConfig.
        n_other = len(_ISLAND_NAMES) - 1
        n_incoming_per_dst = config.islands.k_top * n_other
        if n_incoming_per_dst > config.n_pop:
            raise ValueError(
                f"islands.k_top * (n_islands - 1) = {config.islands.k_top} * {n_other} = "
                f"{n_incoming_per_dst} exceeds optimizer.n_pop = {config.n_pop}. "
                f"Reduce islands.k_top or increase optimizer.n_pop.",
            )

        self.config = config
        self.problem = problem
        self.n_params = n_params
        self.validation_seeds = validation_seeds
        self.final_eval_seeds = final_eval_seeds
        self.base_mc_seed = base_mc_seed
        self.rng = rng
        self.islands: list[Island] = [_build_island(name, config, n_params) for name in _ISLAND_NAMES]
        self.migration_log: list[MigrationEvent] = []
        # Display-only snapshots, refreshed by the train loop on migration gens
        # and reused (shown stale) between migrations.
        self.latest_migration_summary: dict[str, dict] = {}
        self.latest_migration_gen: int | None = None
        self.origin_stats_cache: dict[str, dict[str, dict[str, float | int]]] = {}

    def step(self, current_gen: int) -> list[MigrationEvent]:
        """Advance every island one generation, then maybe migrate.

        Returns the migration events from this gen (empty list if no migration).
        Validation is intentionally separate — see `validate_each` (Task 6).
        """
        # 1. Advance each island sequentially. Rayon parallelism inside each
        #    algorithm.next() saturates the CPU; three sequential per-island
        #    batches dominate the per-gen wall time.
        for island in self.islands:
            island.algorithm.next()

        # 2. Migration step: every k_period gens, never at gen 0.
        events: list[MigrationEvent] = []
        if self.config.islands.enabled and current_gen > 0 and current_gen % self.config.islands.k_period == 0:
            events = migrate(
                self.islands,
                k_top=self.config.islands.k_top,
                current_gen=current_gen,
                rng=self.rng,
                velocity_scale=self.config.islands.pso_inject_velocity_scale,
            )
            self.migration_log.extend(events)
        return events

    def _cost_transform(self) -> str:
        return str(self.problem.cost_kwargs.get("cost_transform", "linear"))

    def validate_each(self, current_gen: int) -> list[dict[str, Any]]:
        """Run identity-trigger validation per island.

        For each island, if its argmin differs from `last_validated_individual`,
        run validation on the reserved validation seeds; promote `best_overall_*`
        if val_rms < best_val_cost. Returns one summary dict per island for
        logging (includes a "validated" bool flag).
        """
        results: list[dict[str, Any]] = []
        for island in self.islands:
            pop = island.algorithm.pop

            # Shared guarded selection + identity-trigger validation. The gate
            # skips validation when every F is non-finite (e.g. all sims timed
            # out or returned NaN) -- a bare `np.argmin` on an all-inf array
            # silently returns 0, which would promote whatever junk chromosome
            # happens to sit at pop[0] as last_validated / best_overall_individual.
            gate = run_validation_gate(
                pop.get("X"),
                pop.get("F"),
                island.last_validated_individual,
                island.best_val_cost,
                self.problem,
                self.validation_seeds,
            )

            if gate.status is GateStatus.SKIP_ALL_INF:
                island.stagnation_counter += 1
                results.append(
                    {
                        "island": island.name,
                        "validated": False,
                        "promoted": False,
                        "argmin_train_cost": float("inf"),
                        "stagnation": island.stagnation_counter,
                    }
                )
                continue

            if gate.status is GateStatus.SKIP_UNCHANGED:
                island.stagnation_counter += 1
                results.append(
                    {
                        "island": island.name,
                        "validated": False,
                        "promoted": False,
                        "argmin_train_cost": gate.argmin_cost,
                        "stagnation": island.stagnation_counter,
                    }
                )
                continue

            assert gate.individual is not None and gate.val_costs is not None and gate.val_records is not None and gate.val_rms is not None
            val_costs = gate.val_costs
            island.last_validated_individual = gate.individual

            if gate.promoted:
                island.best_val_cost = gate.val_rms
                island.best_overall_individual = gate.individual.copy()
                island.best_overall_cost = gate.argmin_cost
                island.stagnation_counter = 0
            else:
                island.stagnation_counter += 1

            # Rich per-island validation dashboard (DV / apoapsis / heat-flux /
            # g-load / heat-load percentiles + violation rates) mirroring
            # `compute_eval_summary`. Imported here to keep the report module
            # out of the island_model import graph -- it pulls matplotlib.
            from aerocapture.training.report import compute_eval_summary

            val_summary = compute_eval_summary(
                gate.val_records,
                len(self.validation_seeds),
                getattr(self.problem, "cost_kwargs", None),
            )
            island.latest_val_summary = val_summary
            results.append(
                {
                    "island": island.name,
                    "validated": True,
                    "promoted": gate.promoted,
                    "argmin_train_cost": gate.argmin_cost,
                    "val_rms": gate.val_rms,
                    "val_mean": float(np.mean(val_costs)),
                    "val_p95": float(np.percentile(val_costs, 95)),
                    "val_capture_rate": _capture_rate(np.asarray(val_costs), cost_transform=self._cost_transform()),
                    "val_summary": val_summary,
                    "stagnation": island.stagnation_counter,
                }
            )
        return results

    def revalidate_each(self) -> None:
        """Re-validate each island's best_overall_individual under the current config.

        Called once after `from_checkpoint` on resume so each island's
        `best_val_cost` reflects the CURRENT cost_kwargs (notably a changed
        `cost_transform`) rather than the stale value baked into the checkpoint.
        Keeps the individual; only refreshes the metric baseline. Islands with no
        `best_overall_individual` (or with no validation seeds) are skipped.
        """
        if not self.validation_seeds:
            return
        for island in self.islands:
            if island.best_overall_individual is None:
                continue
            val_costs, _ = self.problem.evaluate_individual_records_per_seed(
                island.best_overall_individual,
                self.validation_seeds,
            )
            island.best_val_cost = float(np.sqrt(np.mean(val_costs**2)))
            island.last_validated_individual = island.best_overall_individual.copy()

    def resize_populations(
        self,
        target_n: int,
        rng: np.random.Generator,
        fresh_fraction: float,
        velocity_scale: float,
    ) -> bool:
        """Resize every island's restored population to ``target_n``.

        Grows (clone+jitter + fresh-random) or shrinks (best-N by F) each
        island's pop, re-evaluates the resized pop via ``_run_batch`` (under
        whatever seeds the problem currently holds -- correct for ``fixed`` /
        restored-curator ``adaptive``; for ``rotating`` / bootstrap ``adaptive``
        the first post-resume gen re-evals under proper seeds before any
        selection, so the transient F here is never selection-relevant),
        re-stamps GA/DE ``rank`` via FitnessSurvival, and rebuilds PSO
        ``particles`` (positions = new pop, fresh velocity). Returns True if any
        island changed size.
        """
        from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival  # noqa: PLC0415
        from pymoo.core.population import Population  # noqa: PLC0415

        from aerocapture.training.population import resize_population  # noqa: PLC0415

        # NOTE: resize_population needs no ParamSpec list -- the population is
        # already in normalized [0,1] space, so fresh fill is rng.random.
        any_changed = False
        for island in self.islands:
            pop = island.algorithm.pop
            if pop is None:
                continue
            cur_X = pop.get("X")
            if cur_X.shape[0] == target_n:
                continue
            any_changed = True
            cur_F = pop.get("F").flatten()
            new_X = resize_population(cur_X, cur_F, target_n, rng, fresh_fraction=fresh_fraction)
            # Re-eval the whole resized pop under the CURRENT seeds. On shrink the
            # survivors already had checkpoint-era F, but those were under a
            # possibly-different seed list, so a fresh batch keeps all costs
            # comparable (mirrors re_evaluate_all_populations).
            new_F = self.problem._run_batch(new_X)
            new_pop = Population.new("X", new_X)
            new_pop.set("F", new_F.reshape(-1, 1))
            if not isinstance(island.algorithm, PSO):
                new_pop = FitnessSurvival().do(self.problem, new_pop, n_survive=len(new_pop))
            island.algorithm.pop = new_pop
            if isinstance(island.algorithm, PSO):
                particles = Population.new("X", new_X.copy())
                particles.set("F", new_F.reshape(-1, 1).copy())
                particles.set("V", rng.uniform(-velocity_scale, velocity_scale, size=new_X.shape))
                island.algorithm.particles = particles
            if hasattr(island.algorithm, "_set_optimum"):
                island.algorithm._set_optimum()
        return any_changed

    def pool_top_k_X(self, k: int) -> npt.NDArray[np.float64]:
        """Concatenate all island populations and return the K lowest-F rows.

        Used by the adaptive seed curator: the cost CDF is a search-space-wide
        signal, so the curator probes a representative slice across islands
        rather than per-island top-K (which would silo by algorithm).
        """
        all_X = []
        all_F = []
        for island in self.islands:
            if island.algorithm.pop is None:
                continue
            all_X.append(island.algorithm.pop.get("X"))
            all_F.append(island.algorithm.pop.get("F").flatten())
        if not all_X:
            return np.empty((0, self.n_params), dtype=np.float64)
        X = np.concatenate(all_X, axis=0)
        F = np.concatenate(all_F, axis=0)
        k = min(k, F.shape[0])
        top_idx = np.argsort(F, kind="stable")[:k]
        return X[top_idx]

    def re_evaluate_all_populations(self) -> None:
        """Re-evaluate every island's algorithm.pop under the current seed list.

        Called when the shared seed list changes (rotating strategy or adaptive
        curation). Mirrors the pre-`next()` re-eval block in the single-algorithm
        path in train.py.
        """
        for island in self.islands:
            if island.algorithm.pop is None:
                continue
            parent_X = island.algorithm.pop.get("X")
            fresh_F = self.problem._run_batch(parent_X)
            island.algorithm.pop.set("F", fresh_F.reshape(-1, 1))

    def checkpoint(
        self,
        path: Path,
        generation: int,
        seed_curator_state: dict | None = None,
    ) -> None:
        """Write a v2 atomic .npz checkpoint.

        Atomicity: writes to a tempfile in the same directory, then renames.
        Single file holds all 3 islands' state + migration log + RNG state.

        PSO populations get their `algorithm.particles` (current swarm position
        + velocity) saved separately from `algorithm.pop` (personal-best).
        GA/DE only need `pop_X` / `pop_F`. We deliberately avoid `pop.get("V")`
        / `.get("pbest")` as save guards because pymoo's `Population.get` returns
        an object ndarray of `None` entries — not the literal `None` — when the
        per-individual attribute was never set, so a naive `is not None` check
        would persist junk and a later `pop.set("V", junk)` would corrupt the
        restored algorithm state.
        """
        import pickle  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez_compressed appends .npz automatically; use a .tmp.npz sibling so the
        # atomic rename moves exactly path.stem + ".tmp.npz" -> path.
        tmp = path.with_name(path.stem + ".tmp.npz")

        island_states = []
        for island in self.islands:
            pop = island.algorithm.pop
            n_iter_attr = getattr(island.algorithm, "n_iter", None)
            state: dict[str, Any] = {
                "name": island.name,
                "pop_X": pop.get("X") if pop is not None else None,
                "pop_F": pop.get("F") if pop is not None else None,
                "n_iter": int(n_iter_attr) if n_iter_attr is not None else 1,
                "is_initialized": bool(getattr(island.algorithm, "is_initialized", False)),
                "last_validated_individual": island.last_validated_individual,
                "best_overall_individual": island.best_overall_individual,
                "best_overall_cost": island.best_overall_cost,
                "best_val_cost": island.best_val_cost,
                "stagnation_counter": island.stagnation_counter,
            }
            if isinstance(island.algorithm, PSO):
                particles = getattr(island.algorithm, "particles", None)
                if particles is not None:
                    state["particles_X"] = particles.get("X")
                    state["particles_F"] = particles.get("F")
                    state["particles_V"] = particles.get("V")
            island_states.append(state)

        np.savez_compressed(
            tmp,
            version=2,
            generation=generation,
            base_mc_seed=self.base_mc_seed,
            cost_transform=self._cost_transform(),
            island_states=np.array(pickle.dumps(island_states), dtype=object),
            migration_log=np.array(pickle.dumps(self.migration_log), dtype=object),
            rng_state=np.array(pickle.dumps(self.rng.bit_generator.state), dtype=object),
            seed_curator_state=np.array(pickle.dumps(seed_curator_state), dtype=object),
        )
        tmp.rename(path)

    def from_checkpoint(self, path: Path) -> tuple[int, dict | None, str | None]:
        """Restore from a v2 checkpoint.

        Returns:
            (generation, seed_curator_state, saved_cost_transform) — the saved
            generation, the optional SeedCurator state dict (None if absent or
            if the curator was not in use at save time), and the persisted
            cost_transform string (None for legacy checkpoints that predate
            this field). The caller is responsible for re-hydrating the curator
            via `SeedCurator.from_dict(...)`.

        IMPORTANT: per-island best_overall_* are restored verbatim. The
        resumed population's gen-0 argmin must NOT be allowed to overwrite them
        (cross-gen training-cost incomparability under adaptive/rotating
        seeds -- see project memory project_resume_cost_incomparability).
        """
        import pickle  # noqa: PLC0415

        from pymoo.core.population import Population  # noqa: PLC0415

        with np.load(path, allow_pickle=True) as data:
            version = int(data["version"])
            if version != 2:
                raise ValueError(f"checkpoint version {version} unsupported; expected 2")
            generation = int(data["generation"])
            base_mc_seed = int(data["base_mc_seed"])
            saved_cost_transform = str(data["cost_transform"]) if "cost_transform" in data else None
            island_states = pickle.loads(data["island_states"].item())
            migration_log = pickle.loads(data["migration_log"].item())
            rng_state = pickle.loads(data["rng_state"].item())
            seed_curator_state = pickle.loads(data["seed_curator_state"].item())

        if base_mc_seed != self.base_mc_seed:
            raise ValueError(f"checkpoint base_mc_seed {base_mc_seed} != current {self.base_mc_seed}")

        for island, state in zip(self.islands, island_states, strict=True):
            if island.name != state["name"]:
                raise ValueError(f"checkpoint island order mismatch: {island.name} != {state['name']}")
            island.last_validated_individual = state["last_validated_individual"]
            island.best_overall_individual = state["best_overall_individual"]
            island.best_overall_cost = float(state["best_overall_cost"])
            island.best_val_cost = float(state["best_val_cost"])
            island.stagnation_counter = int(state["stagnation_counter"])

            if state["pop_X"] is not None:
                # Fail loudly when the saved chromosome width disagrees with the
                # current ParamSpec count — the islands analogue of
                # `_check_resume_chromosome_shape` in train.py. Catches the user
                # flipping `scaffolding` / `output_parameterization` /
                # `input_mask` (all change n_params) between runs; without it the
                # old-width pop is restored and later mis-decoded into garbage.
                saved_n_params = state["pop_X"].shape[1]
                if saved_n_params != self.n_params:
                    raise ValueError(
                        f"checkpoint chromosome width {saved_n_params} != current {self.n_params}. "
                        f"This usually means `scaffolding`, `output_parameterization`, "
                        f"or `input_mask` changed since the checkpoint was saved. "
                        f"Revert the TOML knob to resume, or pass --from-scratch.",
                    )
                pop = Population.new("X", state["pop_X"])
                pop.set("F", state["pop_F"])

                # GA/DE's `_advance` ends with `FitnessSurvival.do(...)`
                # which stamps `rank` (and `crowding`) on the pop. DE
                # specifically reads `pop.get("rank") == 0` in `_infill` to
                # pick the "best" target; without rank, `np.where(...)`
                # returns empty and `random_state.choice(empty)` raises
                # ValueError. Re-running FitnessSurvival here leaves the
                # restored pop in the same state as if `_advance` had just
                # completed. Skipped for PSO because FitnessSurvival
                # reorders the population, which would break PSO's
                # slot-stable personal-best invariant (pop[i] is particle
                # i's pbest, paired with particles[i]).
                if not isinstance(island.algorithm, PSO):
                    from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival  # noqa: PLC0415

                    pop = FitnessSurvival().do(self.problem, pop, n_survive=len(pop))
                island.algorithm.pop = pop

                # Restore PSO's particles (current swarm position + velocity).
                # Required for `_infill()` to reproduce the saved swarm
                # dynamics; without this the next `next()` reads a None
                # `particles` and crashes (or worse, silently reinitializes V).
                if isinstance(island.algorithm, PSO) and "particles_X" in state:
                    particles = Population.new("X", state["particles_X"])
                    particles.set("F", state["particles_F"])
                    particles.set("V", state["particles_V"])
                    island.algorithm.particles = particles

                # Bypass pymoo's `_initialize()` on the first post-resume
                # `next()` — without `is_initialized=True` the restored pop
                # would be discarded and replaced with a fresh LHS sample.
                if state.get("is_initialized", False):
                    import time as _time  # noqa: PLC0415

                    island.algorithm.is_initialized = True
                    island.algorithm.n_iter = int(state.get("n_iter", 1))
                    # pymoo's `_initialize()` stamps `start_time`; we bypass
                    # it, so set it explicitly. Without this, the eventual
                    # `result()` call (triggered when internal termination
                    # fires) crashes on `end_time - None`.
                    island.algorithm.start_time = _time.time()
                    island.algorithm._set_optimum()

        self.migration_log = migration_log
        self.rng.bit_generator.state = rng_state
        return generation, seed_curator_state, saved_cost_transform

    def final_eval(self) -> list[dict[str, Any]]:
        """Re-evaluate each island's best_overall on the reserved final-eval seeds.

        Returns one record per island that had a validated best, sorted by rms ascending.
        The lowest-rms record is the winner.
        """
        results: list[dict[str, Any]] = []
        for island in self.islands:
            if island.best_overall_individual is None:
                continue
            costs = self.problem.evaluate_individual_per_seed(
                island.best_overall_individual,
                self.final_eval_seeds,
            )
            rms = float(np.sqrt(np.mean(costs**2)))
            results.append(
                {
                    "island": island.name,
                    "X": island.best_overall_individual.copy(),
                    "rms": rms,
                    # The validation rms this island's best_overall was promoted on
                    # (see validate_each). Compared against `rms` (fresh held-out
                    # final-eval) by val_generalization_gap to flag overfit.
                    "val_rms": island.best_val_cost,
                    "mean": float(np.mean(costs)),
                    "p95": float(np.percentile(costs, 95)),
                    "capture_rate": _capture_rate(np.asarray(costs), cost_transform=self._cost_transform()),
                    "n_sims": len(self.final_eval_seeds),
                }
            )
        results.sort(key=lambda r: r["rms"])
        return results
