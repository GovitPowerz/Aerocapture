"""Three-island PSO/GA/DE evolutionary trainer with episodic migration.

See docs/superpowers/specs/2026-05-28-island-model-pso-ga-de-design.md.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm

from aerocapture.training.metrics import capture_rate as _capture_rate
from aerocapture.training.optimizer import OptimizerConfig, create_algorithm


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
    emigrants: dict[str, list[tuple[npt.NDArray[np.float64], float]]] = {}
    for src in islands:
        F_src = src.algorithm.pop.get("F").flatten()
        top_idx = np.argsort(F_src, kind="stable")[:k_top]
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
        F_dst = dst.algorithm.pop.get("F").flatten()
        worst_slots = np.argsort(F_dst, kind="stable")[-n_incoming:]

        for slot_i, (X_new, F_new, src_name) in zip(worst_slots, incoming, strict=True):
            slot = int(slot_i)
            F_displaced = float(F_dst[slot])
            dst.algorithm.pop[slot].X = X_new
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
        self.config = config
        self.problem = problem
        self.n_params = n_params
        self.validation_seeds = validation_seeds
        self.final_eval_seeds = final_eval_seeds
        self.base_mc_seed = base_mc_seed
        self.rng = rng
        self.islands: list[Island] = [_build_island(name, config, n_params) for name in _ISLAND_NAMES]
        self.migration_log: list[MigrationEvent] = []

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
            X = pop.get("X")
            F = pop.get("F").flatten()
            argmin_idx = int(np.argmin(F))
            argmin_X = X[argmin_idx].copy()
            argmin_cost = float(F[argmin_idx])

            unchanged = island.last_validated_individual is not None and np.array_equal(argmin_X, island.last_validated_individual)
            if unchanged:
                island.stagnation_counter += 1
                results.append(
                    {
                        "island": island.name,
                        "validated": False,
                        "promoted": False,
                        "argmin_train_cost": argmin_cost,
                        "stagnation": island.stagnation_counter,
                    }
                )
                continue

            val_costs = self.problem.evaluate_individual_per_seed(
                argmin_X,
                self.validation_seeds,
            )
            val_rms = float(np.sqrt(np.mean(val_costs**2)))
            island.last_validated_individual = argmin_X

            promoted = val_rms < island.best_val_cost
            if promoted:
                island.best_val_cost = val_rms
                island.best_overall_individual = argmin_X.copy()
                island.best_overall_cost = argmin_cost
                island.stagnation_counter = 0
            else:
                island.stagnation_counter += 1

            results.append(
                {
                    "island": island.name,
                    "validated": True,
                    "promoted": promoted,
                    "argmin_train_cost": argmin_cost,
                    "val_rms": val_rms,
                    "val_mean": float(np.mean(val_costs)),
                    "val_p95": float(np.percentile(val_costs, 95)),
                    "val_capture_rate": _capture_rate(np.asarray(val_costs)),
                    "stagnation": island.stagnation_counter,
                }
            )
        return results

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
            island_states.append(
                {
                    "name": island.name,
                    "pop_X": pop.get("X") if pop is not None else None,
                    "pop_F": pop.get("F") if pop is not None else None,
                    "pop_V": pop.get("V") if pop is not None and pop.get("V") is not None else None,
                    "pop_pbest": pop.get("pbest") if pop is not None and pop.get("pbest") is not None else None,
                    "pop_pbest_F": pop.get("pbest_F") if pop is not None and pop.get("pbest_F") is not None else None,
                    "last_validated_individual": island.last_validated_individual,
                    "best_overall_individual": island.best_overall_individual,
                    "best_overall_cost": island.best_overall_cost,
                    "best_val_cost": island.best_val_cost,
                    "stagnation_counter": island.stagnation_counter,
                }
            )

        np.savez_compressed(
            tmp,
            version=2,
            generation=generation,
            base_mc_seed=self.base_mc_seed,
            island_states=np.array(pickle.dumps(island_states), dtype=object),
            migration_log=np.array(pickle.dumps(self.migration_log), dtype=object),
            rng_state=np.array(pickle.dumps(self.rng.bit_generator.state), dtype=object),
            seed_curator_state=np.array(pickle.dumps(seed_curator_state), dtype=object),
        )
        tmp.rename(path)

    def from_checkpoint(self, path: Path) -> tuple[int, dict | None]:
        """Restore from a v2 checkpoint.

        Returns:
            (generation, seed_curator_state) — the saved generation and the
            optional SeedCurator state dict (None if absent or if the curator
            was not in use at save time). The caller is responsible for
            re-hydrating the curator via `SeedCurator.from_dict(...)`.

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
                pop = Population.new("X", state["pop_X"])
                pop.set("F", state["pop_F"])
                if state["pop_V"] is not None:
                    pop.set("V", state["pop_V"])
                if state["pop_pbest"] is not None:
                    pop.set("pbest", state["pop_pbest"])
                if state["pop_pbest_F"] is not None:
                    pop.set("pbest_F", state["pop_pbest_F"])
                island.algorithm.pop = pop

        self.migration_log = migration_log
        self.rng.bit_generator.state = rng_state
        return generation, seed_curator_state

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
                    "mean": float(np.mean(costs)),
                    "p95": float(np.percentile(costs, 95)),
                    "capture_rate": _capture_rate(np.asarray(costs)),
                    "n_sims": len(self.final_eval_seeds),
                }
            )
        results.sort(key=lambda r: r["rms"])
        return results
